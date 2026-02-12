"""
NeuroKit2-based Preprocessing Pipeline for Strength Training ML.

Handles:
- Signal loading and cleaning
- NeuroKit2-based feature extraction (EMG, ECG, EDA, PPG)
- IMU/Accelerometer preprocessing
- Time-based windowing with configurable overlap

Data Sources:
- Biosignals (EMG, ECG, EDA, PPG, IMU): Model input features
- markers.json: Ground truth labels (start/end times, rep markers)
- joint_data.json: Auxiliary ground truth (skeleton for phase/rep/fatigue)

Note: joint_data is NOT used as model input (only for ground truth label extraction)
"""

import warnings
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import signal as scipy_signal
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore", message=".*ChainedAssignment.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="neurokit2")

import neurokit2 as nk

from ml.v3.config.settings import CONFIG, SIGNALS
from ml.v3.data.validate_data import DataValidator
from ml.v3.utils.logging_utils import get_logger
from ml.v3.utils.constants import (
    BONE_CONNECTIONS,
    EXERCISE_CONFIG,
    JOINT_NAMES,
    REST_VELOCITY_THRESHOLD,
    get_exercise_all_joint_index,
)

logger = get_logger('preprocessing')


@dataclass
class WindowedSignal:
    data: np.ndarray
    start_time: float
    end_time: float
    rep_count: int
    phase: str
    fatigue_score: float


@dataclass
class ExtractedFeatures:
    raw_signal: np.ndarray
    features: Dict[str, float]
    feature_vector: np.ndarray


class SignalPreprocessor:
    """Preprocessor for individual biosignal modalities using NeuroKit2."""

    def __init__(self, config=None):
        self.config = config or CONFIG

    def preprocess_emg(self, emg_signal: np.ndarray, sampling_rate: float, plot: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        logger.debug("Preprocessing EMG")
        emg_processed, info = nk.emg_process(emg_signal, sampling_rate=sampling_rate)
        if plot:
            nk.emg_plot(emg_processed, info)

        cleaned = emg_processed['EMG_Clean'].values
        amplitude = emg_processed['EMG_Amplitude'].values

        features = {
            'emg_rms': np.sqrt(np.mean(cleaned ** 2)),
            'emg_mean_amplitude': np.mean(amplitude),
            'emg_max_amplitude': np.max(amplitude),
            'emg_std': np.std(cleaned),
        }
        features.update(self._compute_emg_frequency_features(cleaned, sampling_rate))
        return cleaned, features

    def preprocess_ecg(self, ecg_signal: np.ndarray, sampling_rate: float, plot: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        logger.debug("Preprocessing ECG")
        processed, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        if plot:
            nk.ecg_plot(processed, info)

        cleaned = processed['ECG_Clean'].values
        r_peaks = info['ECG_R_Peaks']
        features = {}

        if len(r_peaks) >= 3:
            hrv = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)
            features.update({
                'hrv_rmssd': hrv.get('HRV_RMSSD', [0])[0],
                'hrv_sdnn': hrv.get('HRV_SDNN', [0])[0],
                'hrv_mean_hr': hrv.get('HRV_MeanNN', [0])[0],
                'hrv_pnn50': hrv.get('HRV_pNN50', [0])[0],
            })
        features['ecg_hr'] = len(r_peaks) * (60.0 / (len(ecg_signal) / sampling_rate))
        return cleaned, features

    def preprocess_eda(self, eda_signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, Dict[str, float]]:
        logger.debug("Preprocessing EDA")
        processed, info = nk.eda_process(eda_signal, sampling_rate=sampling_rate)
        cleaned = processed['EDA_Clean'].values
        features = {}

        if 'EDA_Tonic' in processed:
            tonic = processed['EDA_Tonic']
            features['eda_scl_mean'] = np.mean(tonic)
            features['eda_scl_std'] = np.std(tonic)

        if 'EDA_Phasic' in processed:
            phasic = processed['EDA_Phasic']
            features['eda_scr_mean'] = np.mean(phasic)
            features['eda_scr_max'] = np.max(phasic)

        if 'SCR_Peaks' in info:
            features['eda_n_scr'] = len(info['SCR_Peaks'])

        return cleaned, features

    def preprocess_ppg(self, ppg_signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, Dict[str, float]]:
        logger.debug("Preprocessing PPG")
        processed, info = nk.ppg_process(ppg_signal, sampling_rate=sampling_rate)
        cleaned = processed['PPG_Clean'].values
        features = {}

        if 'PPG_Peaks' in info:
            peaks = info['PPG_Peaks']
            if len(peaks) > 1:
                ibi = np.diff(peaks) / sampling_rate
                features['ppg_hr'] = 60.0 / np.mean(ibi) if len(ibi) > 0 else 0
                features['ppg_hrv'] = np.std(ibi) * 1000 if len(ibi) > 0 else 0

        return cleaned, features

    def preprocess_accelerometer(self, acc_signal: np.ndarray, sampling_rate: float, apply_filter: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        logger.debug(f"Preprocessing ACC (filter={apply_filter})")
        if apply_filter:
            filtered = nk.signal_filter(
                acc_signal,
                sampling_rate=sampling_rate,
                lowcut=self.config.preprocessing.acc_lowcut,
                highcut=self.config.preprocessing.acc_highcut,
                method='butterworth',
                order=4
            )
        else:
            filtered = acc_signal

        features = {
            'acc_rms': np.sqrt(np.mean(filtered ** 2)),
            'acc_mean': np.mean(filtered),
            'acc_std': np.std(filtered),
            'acc_max': np.max(np.abs(filtered)),
            'acc_range': np.max(filtered) - np.min(filtered),
        }

        zero_cross = np.sum(np.diff(np.signbit(filtered).astype(int)))
        features['acc_zcr'] = zero_cross / len(filtered)

        peaks, _ = scipy_signal.find_peaks(filtered, height=np.std(filtered))
        features['acc_n_peaks'] = len(peaks)

        return filtered, features

    def _compute_emg_frequency_features(self, emg_signal: np.ndarray, sampling_rate: float) -> Dict[str, float]:
        logger.debug("Computing EMG frequency features")
        return compute_emg_frequency_features(emg_signal, sampling_rate)

    def calculate_biosignal_fatigue(
        self,
        window_signals: Dict[str, np.ndarray],
        initial_features: Dict[str, Dict[str, float]],
        sampling_rates: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        component_scores = {}
        weights = {}

        # EMG
        if 'emg' in window_signals and 'emg' in initial_features:
            _, curr = self.preprocess_emg(window_signals['emg'], sampling_rates.get('emg', 2000.0))
            init_mdf = initial_features['emg'].get('emg_median_freq', 0)
            curr_mdf = curr.get('emg_median_freq', 0)
            if init_mdf > 0:
                mdf_ratio = curr_mdf / init_mdf
                emg_fatigue = max(0, min(1, 1 - mdf_ratio))

                init_rms = initial_features['emg'].get('emg_rms', 1)
                curr_rms = curr.get('emg_rms', 1)
                if init_rms > 0:
                    rms_ratio = curr_rms / init_rms
                    rms_fatigue = max(0, min(1, (rms_ratio - 1) / 2))
                    emg_fatigue = 0.7 * emg_fatigue + 0.3 * rms_fatigue

                component_scores['emg_fatigue'] = emg_fatigue
                weights['emg'] = 0.35

        # ECG (forkortet for plass)
        if 'ecg' in window_signals and 'ecg' in initial_features:
            _, curr = self.preprocess_ecg(window_signals['ecg'], sampling_rates.get('ecg', 500.0))
            init_hr = initial_features['ecg'].get('ecg_hr', 70)
            curr_hr = curr.get('ecg_hr', 70)
            if init_hr > 0:
                hr_inc = (curr_hr - init_hr) / init_hr
                hr_fatigue = max(0, min(1, hr_inc / 0.5))

                init_rmssd = initial_features['ecg'].get('hrv_rmssd', 50)
                curr_rmssd = curr.get('hrv_rmssd', 50)
                if init_rmssd > 0:
                    rmssd_ratio = curr_rmssd / init_rmssd
                    hrv_fatigue = max(0, min(1, 1 - rmssd_ratio))
                    ecg_fatigue = 0.4 * hr_fatigue + 0.6 * hrv_fatigue
                else:
                    ecg_fatigue = hr_fatigue

                component_scores['ecg_fatigue'] = ecg_fatigue
                weights['ecg'] = 0.25

        # PPG og ACC (tilsvarende forkortet)

        if not component_scores:
            return 0.0, {}

        total_w = sum(weights.values())
        if total_w == 0:
            return 0.0, component_scores

        score = sum(component_scores.get(f'{k}_fatigue', 0) * w for k, w in weights.items()) / total_w
        return max(0.0, min(1.0, score)), component_scores


def compute_emg_frequency_features(emg_signal: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    # (uendret – behold som er)
    features = {}
    if len(emg_signal) < 10:
        return {'emg_median_freq': 0.0, 'emg_mean_freq': 0.0, 'emg_total_power': 0.0, 'emg_fatigue_ratio': 0.0}

    freqs, psd = welch(emg_signal, fs=sampling_rate, nperseg=min(len(emg_signal), int(sampling_rate)))
    if len(psd) == 0 or np.sum(psd) == 0:
        return features

    cumsum = np.cumsum(psd)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    features['emg_median_freq'] = freqs[min(median_idx, len(freqs)-1)]
    features['emg_mean_freq'] = np.sum(freqs * psd) / np.sum(psd)
    features['emg_total_power'] = np.sum(psd)

    low_mask = freqs < 60
    high_mask = freqs >= 60
    low = np.sum(psd[low_mask])
    high = np.sum(psd[high_mask])
    features['emg_fatigue_ratio'] = high / (low + 1e-10)

    return features
class JointProcessor:
    """
    Processor for skeletal joint data 

    
    """

    def __init__(self, config=None):
        if config is None:
            config = CONFIG
        self.config = config
        # Ensure we have access to signals config for joint channel count
        if not hasattr(self.config, 'signals'):
            self.config.signals = SIGNALS


    @staticmethod
    def _get_exercise_all_joint_index(exercise_type: str) -> List[int]:
        """Return all joint indices (primary + assist) for a given exercise."""
        return get_exercise_all_joint_index(exercise_type)

    @staticmethod
    def _get_exercise_joint_index(exercise_type: str) -> int:
        """Return the primary joint index for a given exercise."""
        from ml.v3.utils.constants import get_exercise_joint_index
        return get_exercise_joint_index(exercise_type)

    @staticmethod
    def _get_exercise_config(exercise_type: str) -> Dict:
        """Return exercise configuration dict from constants."""
        key = exercise_type.lower().strip()
        if key not in EXERCISE_CONFIG:
            raise ValueError(f"Unknown exercise: {exercise_type}")
        return EXERCISE_CONFIG[key]

    def load_joint_data(self, joint_path: Path) -> Optional[Dict]:
        """
        Load joint data from JSON file.

        If all timestamp_usec values are 0 (missing), synthesizes timestamps
        from frame_id using the recording duration from metadata.json or
        the configured joint sampling rate as fallback.

        Args:
            joint_path: Path to joint_data.json

        Returns:
            Dictionary with frames and bone list
        """
        try:
            with open(joint_path, 'r') as f:
                data = json.load(f)

            frames = data.get('frames', [])
            if not frames:
                logger.warning(f"No frames found in {joint_path}")
                return data

            # Check if timestamps are missing (all zeros)
            all_zero = all(f.get('timestamp_usec', 0) == 0 for f in frames)
            if all_zero and len(frames) > 1:
                logger.warning(f"All timestamps in {joint_path} are zero. Synthesizing timestamps from frame_id.")
                # Try to get recording duration from metadata.json
                metadata_path = joint_path.parent / 'metadata.json'
                fps = None
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            meta = json.load(f)
                        # Use EMG duration as reference (most reliable)
                        duration = meta.get('duration_seconds', {}).get('emg', 0)
                        if duration > 0:
                            fps = len(frames) / duration
                    except Exception as e:
                        logger.debug(f"Failed to read metadata for FPS: {e}")

                if fps is None or fps <= 0:
                    fps = self.config.signals.get('joints', None)
                    fps = fps.sampling_rate if fps else 30.0

                # Synthesize timestamps from frame_id
                for frame in frames:
                    frame_id = frame.get('frame_id', 0)
                    frame['timestamp_usec'] = int(frame_id / fps * 1e6)

            return data
        except Exception as e:
            logger.error(f"Error loading joint data: {e}")
            return None

    def get_skeleton_frame(
        self,
        joint_data: Dict,
        target_time: float
    ) -> Optional[Dict]:
        """
        Get a single skeleton frame closest to the target time.

        Args:
            joint_data: Loaded joint data dictionary
            target_time: Target time in seconds (relative to start)

        Returns:
            Dictionary with joint positions and bone connections for visualization
        """
        frames = joint_data.get('frames', [])

        if not frames:
            return None

        # Convert absolute timestamps to relative time
        first_timestamp = frames[0].get('timestamp_usec', 0)

        # Find frame closest to target time
        closest_frame = None
        min_diff = float('inf')

        for frame in frames:
            abs_timestamp = frame.get('timestamp_usec', 0)
            rel_time = (abs_timestamp - first_timestamp) / 1e6

            diff = abs(rel_time - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame

        if closest_frame is None:
            return None

        # Extract joint positions
        bodies = closest_frame.get('bodies', [])
        if not bodies:
            return None

        body = bodies[0]  # Use first detected body
        joint_positions = body.get('joint_positions', [])

        if not joint_positions or len(joint_positions) < len(JOINT_NAMES):
            return None

        # Create Nx3 numpy array for visualization
        joints_array = np.zeros((len(JOINT_NAMES), 3))
        for i, name in enumerate(JOINT_NAMES):
            if i < len(joint_positions):
                joints_array[i, 0] = joint_positions[i][0]  # x
                joints_array[i, 1] = joint_positions[i][1]  # y
                joints_array[i, 2] = joint_positions[i][2] if len(joint_positions[i]) > 2 else 0  # z

        # Convert bone connections from names to indices
        name_to_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
        bone_indices = []
        for start_name, end_name in BONE_CONNECTIONS:
            if start_name in name_to_idx and end_name in name_to_idx:
                bone_indices.append((name_to_idx[start_name], name_to_idx[end_name]))

        return {
            'joints': joints_array,  # Nx3 numpy array
            'joint_names': JOINT_NAMES,
            'bone_connections': bone_indices,  # List of (start_idx, end_idx) tuples
            'timestamp': target_time
        }

    def extract_joint_features(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extract joint features for a time window.

        Args:
            joint_data: Loaded joint data dictionary
            start_time: Window start time in seconds 
            end_time: Window end time in seconds 
            exercise_type: Type of exercise 

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - feature_vector: 
                - extra_features: 
                    - mean_velocity, std_velocity, max_velocity
                    - mean_accel,  std_accel,  max_accel
                    - y_direction
                    - energy
        """
        frames = joint_data.get('frames', [])
        if not frames:
            zero_vec = np.zeros(self.config.signals['joints'].channels)
            empty_dict = {
                'mean_velocity': 0.0, 'std_velocity': 0.0, 'max_velocity': 0.0,
                'mean_accel': 0.0,   'std_accel': 0.0,   'max_accel': 0.0,
                'y_direction': 0.0,  'energy': 0.0
            }
            return zero_vec, empty_dict

        first_timestamp = frames[0].get('timestamp_usec', 0)

        # Gather frames that fall within the specified time window
        window_frames = []
        for frame in frames:
            abs_ts = frame.get('timestamp_usec', 0)
            rel_time = (abs_ts - first_timestamp) / 1e6
            if start_time <= rel_time <= end_time:
                window_frames.append(frame)

        if len(window_frames) < 2:
            logger.warning(f"Not enough joint frames in window ({start_time:.2f}s to {end_time:.2f}s) for exercise {exercise_type}")
            zero_vec = np.zeros(self.config.signals['joints'].channels)
            empty_dict = {
                'mean_velocity': 0.0, 'std_velocity': 0.0, 'max_velocity': 0.0,
                'mean_accel': 0.0,   'std_accel': 0.0,   'max_accel': 0.0,
                'y_direction': 0.0,  'energy': 0.0
            }
            return zero_vec, empty_dict

        # Collect positions – shape: (n_frames, n_joints × 3)
        positions_list = []
        for frame in window_frames:
            bodies = frame.get('bodies', [])
            if not bodies:
                continue
            body = bodies[0]
            joint_pos = body.get('joint_positions', [])
            if len(joint_pos) >= len(JOINT_NAMES):
                positions_list.append(np.array(joint_pos).flatten())

        if len(positions_list) < 2:
            logger.warning(f"Not enough valid joint position frames in window ({start_time:.2f}s to {end_time:.2f}s) for exercise {exercise_type}")
            zero_vec = np.zeros(self.config.signals['joints'].channels)
            empty_dict = {k: 0.0 for k in [
                'mean_velocity', 'std_velocity', 'max_velocity',
                'mean_accel', 'std_accel', 'max_accel',
                'y_direction', 'energy'
            ]}
            return zero_vec, empty_dict

        positions = np.array(positions_list)          # (n_frames, n_coords)
        n_frames, n_coords = positions.shape

        mean_pos = np.mean(positions, axis=0)
        rom = np.max(positions, axis=0) - np.min(positions, axis=0)

        # Compute velocity using timestamps (handles variable frame rates)
        timestamps_rel = np.array([
            (f.get('timestamp_usec', 0) - first_timestamp) / 1e6
            for f in window_frames
        ])
        dt = np.diff(timestamps_rel)
        valid_dt = dt[dt > 0]

        if len(valid_dt) > 0:
            mean_dt = np.mean(valid_dt)
            velocities = np.diff(positions, axis=0) / mean_dt
        else:
            velocities = np.diff(positions, axis=0) / (1/30.0)  # fallback 30 fps

        abs_vel = np.abs(velocities)

        # Build feature vector: mean_pos + mean_abs_velocity + range_of_motion
        features = []
        features.extend(mean_pos)

        if len(velocities) > 0:
            mean_abs_vel = np.mean(abs_vel, axis=0)
            features.extend(mean_abs_vel[:32])
        else:
            features.extend(np.zeros(32))

        features.extend(rom[:32])

        feature_vector = np.array(features)

        # Pad / trim
        target_length = self.config.signals['joints'].channels
        if len(feature_vector) < target_length:
            feature_vector = np.pad(
                feature_vector,
                (0, target_length - len(feature_vector))
            )
        elif len(feature_vector) > target_length:
            feature_vector = feature_vector[:target_length]

        extra_features = {}

        extra_features['mean_velocity'] = float(np.mean(abs_vel))
        extra_features['std_velocity']  = float(np.std(abs_vel))
        extra_features['max_velocity']  = float(np.max(abs_vel))

        # Acceleration
        if len(velocities) >= 2:
            accelerations = np.diff(velocities, axis=0) / mean_dt
            abs_accel = np.abs(accelerations)
            extra_features['mean_accel'] = float(np.mean(abs_accel))
            extra_features['std_accel']  = float(np.std(abs_accel))
            extra_features['max_accel']  = float(np.max(abs_accel))
        else:
            extra_features['mean_accel'] = 0.0
            extra_features['std_accel']  = 0.0
            extra_features['max_accel']  = 0.0

        # Y-direction of movement (exercise-specific)
        ex_key = exercise_type.lower().strip() if exercise_type else ''
        y_idx = EXERCISE_CONFIG.get(ex_key, {}).get("primary_idx", 0) * 3 + 1  # y-koordinat
        if y_idx < n_coords:
            y_series = positions[:, y_idx]
            if len(y_series) >= 5:
                third = max(2, len(y_series) // 3)
                y_start = np.mean(y_series[:third])
                y_end   = np.mean(y_series[-third:])
                delta_y = y_end - y_start

                if abs(delta_y) < self.config.joint.delta_y_threshold:
                    extra_features['y_direction'] = 0.0
                elif delta_y > 0:
                    extra_features['y_direction'] = 1.0   # opp
                else:
                    extra_features['y_direction'] = -1.0  # ned
            else:
                logger.warning(f"Not enough frames to determine Y-direction for exercise {exercise_type}")
                extra_features['y_direction'] = 0.0
        else:
            logger.warning(f"Y-index {y_idx} out of bounds for joint features in exercise {exercise_type}")
            extra_features['y_direction'] = 0.0

        # Energy proxy
        energy_per_frame = np.sum(velocities**2, axis=1)
        extra_features['energy'] = float(np.mean(energy_per_frame)) if len(energy_per_frame) > 0 else 0.0

        return feature_vector, extra_features

    def detect_phase(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str
    ) -> str:
        """
        Detect movement phase from joint data using exercise-specific rules.

        
        Returns:
            Phase label: 'eccentric', 'concentric', 'rest'
        """
        _, phase_features = self.extract_joint_features(joint_data, start_time, end_time, exercise_type)

        y_direction = phase_features.get('y_direction', 0.0)
        velocity = phase_features["mean_velocity"]
        if velocity < REST_VELOCITY_THRESHOLD * 1.5:
            return 'rest'

        if y_direction == 1:
            y = 'y_increase'
        elif y_direction == -1:
            y = 'y_decrease'
        else:
            y = 'transition'
        
        ex_key = exercise_type.lower().strip() if exercise_type else ''
        phase_label = EXERCISE_CONFIG.get(ex_key, {}).get('direction', {}).get(y, 'rest')

        return phase_label


    def calculate_movement_velocity(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str,
        use_concentric_only: bool = True,      
        min_dt: float = 0.005,                 
        velocity_threshold: float = 0.05,      
    ) -> Tuple[float, str]:
        """

        Calculate mean velocity for one or more joints for one window designed for fatigue     
        

        Args:
            joint_data: 
            start_time: 
            end_time: 
            exercise_type: 
            use_concentric_only: 
            min_dt: 
            velocity_threshold:

        Returns:
            Tuple[float, str]: (velocity_m_per_s, direction)
        """
        frames = joint_data.get('frames', [])
        if len(frames) < 2:
            logger.warning("Too few joint_data frames")
            return 0.0, "rest"

        first_timestamp = frames[0].get('timestamp_usec', 0)

        # Get all relevant joint_idx
        joint_indices = self._get_exercise_all_joint_index(exercise_type) 

        # Gather positions and times for all joints
        positions_list: List[List[np.ndarray]] = []  # per frame: [pos1, pos2, ...]
        timestamps = []

        for frame in frames:
            abs_ts = frame.get('timestamp_usec', 0)
            rel_time = (abs_ts - first_timestamp) / 1e6

            if not (start_time <= rel_time <= end_time):
                continue

            bodies = frame.get('bodies', [])
            if not bodies:
                continue

            body = bodies[0]
            joint_pos = body.get('joint_positions', [])

            # Gather positions for all relevant joints in frame
            frame_positions = []
            for idx in joint_indices:
                if idx < len(joint_pos) and len(joint_pos[idx]) >= 3:
                    frame_positions.append(np.array(joint_pos[idx][:3]))  # x, y, z
                else:
                    frame_positions.append(None)  

            if all(p is None for p in frame_positions):
                continue

            positions_list.append(frame_positions)
            timestamps.append(rel_time)

        if len(positions_list) < 2:
            logger.info(f"No valid positions in frame {start_time:.2f}–{end_time:.2f}s")
            return 0.0, "rest"

        positions = np.array(positions_list)  # shape: (n_frames, n_joints, 3)
        timestamps = np.array(timestamps)

        # Time difference
        dt = np.diff(timestamps)
        valid = dt > min_dt

        if not np.any(valid):
            return 0.0, "rest"

        # Calculate 3d velocity for each joint for each timestep
        velocities = np.zeros_like(positions[:-1])  # shape: (n-1, n_joints, 3)
        for i in range(len(positions) - 1):
            if valid[i]:
                delta = positions[i + 1] - positions[i]
                velocities[i] = delta / dt[i]

        # Absolute velocity norms
        abs_vel = np.linalg.norm(velocities, axis=2)  # shape: (n-1, n_joints)

        # Remove noise
        abs_vel = np.where(abs_vel > velocity_threshold, abs_vel, 0.0)

        # Direction
        y_deltas = positions[1:, :, 1] - positions[:-1, :, 1]  # y-change for each joint
        y_direction = np.sign(np.mean(y_deltas, axis=1))       # mean direction

        # Determine concentric direction from exercise config
        cfg = self._get_exercise_config(exercise_type)
        is_concentric_up = cfg["direction"]["y_increase"] == "concentric"

        # Choose concentric only or both
        if use_concentric_only:
            if is_concentric_up:
                mask = y_direction > 0
            else:
                mask = y_direction < 0
        else:
            mask = np.ones_like(y_direction, dtype=bool)

        # Mean velocity for all joints
        valid_vel = abs_vel[mask]
        if valid_vel.size == 0:
            return 0.0, "rest"

        mean_velocity = float(np.mean(valid_vel))

        # Phase
        if mean_velocity < velocity_threshold:
            direction = "rest"
        elif np.mean(y_direction[mask]) > 0:
            direction = "concentric" if is_concentric_up else "eccentric"
        else:
            direction = "eccentric" if is_concentric_up else "concentric"

        logger.debug(f"{exercise_type} velocity: {mean_velocity:.3f} m/s ({direction})")
        return mean_velocity, direction

    def calculate_fatigue_from_velocity(
        self,
        joint_data: Dict,
        current_time: float,
        exercise_start_time: float,
        exercise_type: str,
        window_sec: float = 2.0,
        initial_velocity: float = None
    ) -> float:
        """
        Estimate fatigue score based on velocity degradation over time.

        Compares current velocity to initial velocity.
        Fatigue = 1 - (current_velocity / initial_velocity), clamped to [0, 1].

        Args:
            joint_data: Loaded joint data dictionary
            current_time: Current window end time
            exercise_start_time: When the exercise started
            exercise_type: Type of exercise
            window_sec: Window duration for velocity calculation
            initial_velocity: Pre-computed initial velocity (avoids recomputing every call)

        Returns:
            Fatigue score between 0 (fresh) and 1 (fatigued)
        """
        if initial_velocity is None or initial_velocity <= 0:
            # Try multiple windows from exercise start to find actual movement
            for offset in range(4):
                t_start = exercise_start_time + offset * window_sec
                initial_velocity, _ = self.calculate_movement_velocity(
                    joint_data,
                    t_start,
                    t_start + window_sec * 2,
                    exercise_type
                )
                if initial_velocity > 0:
                    break

        if initial_velocity <= 0:
            return 0.0

        # Calculate current velocity
        current_velocity, _ = self.calculate_movement_velocity(
            joint_data,
            current_time - window_sec,
            current_time,
            exercise_type
        )

        # Fatigue = velocity reduction ratio
        velocity_ratio = current_velocity / initial_velocity

        # Clamp and invert (lower velocity = higher fatigue)
        fatigue = 1.0 - min(1.0, max(0.0, velocity_ratio))

        return fatigue

    def detect_rep_times(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str
    ) -> List[float]:
        """
        Detect repetition times from joint movement.

        Each rep = 1 valley in Y-position (bottom of eccentric).
        Returns the timestamps of each detected rep.

        Args:
            joint_data: Loaded joint data dictionary
            start_time: Start of time range to analyze
            end_time: End of time range to analyze
            exercise_type: Type of exercise

        Returns:
            List of timestamps (seconds) where reps were detected
        """
        frames = joint_data.get('frames', [])

        if not frames or len(frames) < 10:
            return []

        first_timestamp = frames[0].get('timestamp_usec', 0)

        # Extract key joint positions over time (only frames with body detection)
        joint_idx = self._get_exercise_joint_index(exercise_type)
        positions = []
        position_times = []
        for frame in frames:
            abs_timestamp = frame.get('timestamp_usec', 0)
            rel_time = (abs_timestamp - first_timestamp) / 1e6

            if start_time <= rel_time <= end_time:
                bodies = frame.get('bodies', [])
                if bodies:
                    body = bodies[0]
                    joint_positions = body.get('joint_positions', [])
                    if joint_positions and len(joint_positions) > joint_idx:
                        positions.append(joint_positions[joint_idx][1])  # Y-axis
                        position_times.append(rel_time)

        if len(positions) < 10:
            return []

        # Calculate effective sample rate from actual detected positions
        time_span = position_times[-1] - position_times[0]
        if time_span > 0:
            effective_fps = len(positions) / time_span
        else:
            effective_fps = 30.0

        # Minimum distance between reps (~1.5 seconds worth of detected frames)
        min_distance = max(3, int(effective_fps * 1.5))

        # Smooth the signal
        positions = np.array(positions)
        position_times = np.array(position_times)
        smooth_size = min(5, len(positions))
        if smooth_size > 1:
            positions = uniform_filter1d(positions, size=smooth_size)

        # Valleys in Y = bottom of movement = completed rep
        peak_indices, _ = scipy_signal.find_peaks(-positions, distance=min_distance, prominence=0.03)

        return position_times[peak_indices].tolist()

    def count_reps_from_peaks(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str
    ) -> int:
        """Count reps between start_time and end_time."""
        return len(self.detect_rep_times(joint_data, start_time, end_time, exercise_type))


class DataPreprocessor:
    """
    Main preprocessing pipeline: biosignals + joint ground truth.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.signal_processor = SignalPreprocessor(config)
        self.joint_processor = JointProcessor(config)

    def preprocess_session(self, session_path: Path, exercise_type: str) -> Dict[str, Any]:
        result = {
            'signals': {},
            'features': {},
            'markers': None,
            'joint_data': None,
            'windows': [],
            'metadata': {'session_path': str(session_path), 'exercise': exercise_type}
        }

        # Load markers
        if (markers_path := session_path / self.config.data.markers_file).exists():
            with open(markers_path) as f:
                result['markers'] = json.load(f)

        # Load joint data
        if (joint_path := session_path / self.config.data.joints_file).exists():
            result['joint_data'] = self.joint_processor.load_joint_data(joint_path)

        # Process biosignals
        for name, cfg in SIGNALS.items():
            if not cfg.enabled or cfg.file_name.endswith('.json'):
                continue

            path = session_path / cfg.file_name
            if not path.exists():
                logger.debug(f"Missing {name} data in {session_path.name}")
                continue

            try:
                df = pd.read_csv(path)
                time_data = df.iloc[:, 0].values
                raw = df.iloc[:, 1].values
                raw = np.nan_to_num(raw, nan=0.0)

                method = {
                    'emg': self.signal_processor.preprocess_emg,
                    'ecg': self.signal_processor.preprocess_ecg,
                    'eda': self.signal_processor.preprocess_eda,
                    'ppg': self.signal_processor.preprocess_ppg,
                    'ppg_red': self.signal_processor.preprocess_ppg,
                    'ppg_ir': self.signal_processor.preprocess_ppg,
                    'ppg_green': self.signal_processor.preprocess_ppg,
                    'ppg_blue': self.signal_processor.preprocess_ppg,
                    'acc': self.signal_processor.preprocess_accelerometer,
                }.get(name)

                if method:
                    processed, feats = method(raw, cfg.sampling_rate)
                else:
                    processed, feats = raw, {}

                # Normaliser lengde
                expected = int(cfg.sampling_rate * (time_data[-1] - time_data[0] if len(time_data) > 1 else 0))
                if len(processed) != len(time_data):
                    processed = self._normalize_length(processed, len(time_data))

                result['signals'][name] = {'data': processed, 'time': time_data, 'sampling_rate': cfg.sampling_rate}
                result['features'][name] = feats

            except Exception as e:
                logger.error(f"Failed processing {name}: {e}")
                raise

        return result

    @staticmethod
    def _normalize_length(signal: np.ndarray, target_len: int) -> np.ndarray:
        if len(signal) == target_len:
            return signal
        if len(signal) > target_len:
            return signal[:target_len]
        if len(signal) == 0:
            return np.zeros(target_len)
        return np.pad(signal, (0, target_len - len(signal)), mode='edge')

    def create_windows(self, session_data: Dict, exercise_type: str) -> List[Dict]:
        windows = []
        if not (markers := session_data.get('markers', {})) or not (marker_list := markers.get('markers', [])):
            logger.warning("No markers – skipping windows")
            return windows

        if not session_data.get('signals'):
            logger.warning("No signals – skipping")
            return windows

        # Start/end fra markører
        start_marker = next((m for m in marker_list if m.get('label', '').lower() == 'start'), None)
        if not start_marker:
            logger.warning("No 'start' marker – skipping")
            return windows

        start_time = start_marker.get('time', 0)
        end_time = max(m.get('time', 0) for m in marker_list)

        window_sec = self.config.data.time_window_sec
        step_sec = window_sec * (1 - self.config.data.overlap)

        # Cache joint stuff
        joint_rep_times = None
        initial_velocity = None
        if joint_data := session_data.get('joint_data'):
            joint_rep_times = self.joint_processor.detect_rep_times(joint_data, start_time, end_time, exercise_type)
            for offset in range(4):
                t = start_time + offset * window_sec
                v, _ = self.joint_processor.calculate_movement_velocity(joint_data, t, t + window_sec * 2, exercise_type)
                if v > 0.05:
                    initial_velocity = v
                    break

        # Biosignal baseline
        initial_bio_feats = {}
        sampling_rates = {}
        use_bio_fatigue = False

        for name, info in session_data['signals'].items():
            fs = info['sampling_rate']
            sampling_rates[name] = fs
            mask = (info['time'] >= start_time) & (info['time'] <= start_time + window_sec)
            segment = info['data'][mask]
            if len(segment) < 10:
                continue
            try:
                if name == 'emg':
                    _, feats = self.signal_processor.preprocess_emg(segment, fs)
                elif name == 'ecg':
                    _, feats = self.signal_processor.preprocess_ecg(segment, fs)
                elif name.startswith('ppg'):
                    _, feats = self.signal_processor.preprocess_ppg(segment, fs)
                elif name == 'acc':
                    _, feats = self.signal_processor.preprocess_accelerometer(segment, fs)
                else:
                    continue
                initial_bio_feats[name] = feats
                use_bio_fatigue = True
            except Exception as e:
                logger.debug(f"Initial {name} failed: {e}")

        # Generer vinduer
        t = start_time
        idx = 0
        while t + window_sec <= end_time:
            t_end = t + window_sec
            window = {
                'window_idx': idx,
                'start_time': t,
                'end_time': t_end,
                'signals': {},
                'joint_features': None,
                'skeleton_frame': None,
                'phase': 'unknown',
                'rep_count': 0,
                'fatigue_score': 0.0,
                'exercise': exercise_type
            }

            # Signaler
            for name, info in session_data['signals'].items():
                mask = (info['time'] >= t) & (info['time'] <= t_end)
                sig = info['data'][mask]
                expected = int(info['sampling_rate'] * window_sec)
                if len(sig) == 0:
                    sig = np.zeros(expected)
                elif len(sig) < expected:
                    sig = np.pad(sig, (0, expected - len(sig)), 'edge')
                elif len(sig) > expected:
                    sig = sig[:expected]
                window['signals'][name] = sig

            # Reps
            rep_markers = sum(1 for m in marker_list if t <= m.get('time', 0) <= t_end and m.get('label', '').lower() in ['rep', 'valley', 'completion'])
            if rep_markers > 0:
                window['rep_count'] = rep_markers
            elif joint_rep_times:
                window['rep_count'] = sum(t <= rt <= t_end for rt in joint_rep_times)

            # Joint
            if joint_data:
                window['joint_features'] = self.joint_processor.extract_joint_features(joint_data, t, t_end, exercise_type)
                window['skeleton_frame'] = self.joint_processor.get_skeleton_frame(joint_data, (t + t_end) / 2)
                window['phase'] = self.joint_processor.detect_phase(joint_data, t, t_end, exercise_type)

            # Fatigue
            if use_bio_fatigue and initial_bio_feats:
                score, _ = self.signal_processor.calculate_biosignal_fatigue(window['signals'], initial_bio_feats, sampling_rates)
                window['fatigue_score'] = score
            elif joint_data:
                window['fatigue_score'] = self.joint_processor.calculate_fatigue_from_velocity(
                    joint_data, t_end, start_time, exercise_type, window_sec=t_end - t, initial_velocity=initial_velocity
                )

            windows.append(window)
            t += step_sec
            idx += 1

        return windows


# Convenience function
def preprocess_dataset(
    dataset_path: Path = None,
    config=None,
    valid_sessions: List[Tuple[str, str, Path]] = None
) -> List[Dict]:
    """
    Preprocess entire dataset, using only valid sessions.

    Args:
        dataset_path: Path to dataset
        config: Configuration object
        valid_sessions: List of (exercise, session_id, path) tuples from validation.
                       If None, validates and gets valid sessions automatically.

    Returns:
        List of all preprocessed windows with labels
    """
    config = config or CONFIG
    dataset_path = Path(dataset_path or config.data.dataset_path)

    # If valid_sessions not provided, run validation to get them
    if valid_sessions is None:
        from ml.v3.data.validate_data import DataValidator
        validator = DataValidator(dataset_path, config)
        validator.validate_all()
        valid_sessions = validator.get_valid_sessions()

    if not valid_sessions:
        logger.warning("No valid sessions found!")
        return []

    logger.info(f"Processing {len(valid_sessions)} valid sessions...")

    preprocessor = DataPreprocessor(config)

    all_windows = []

    # Group sessions by exercise for organized processing
    sessions_by_exercise = {}
    for exercise, session_id, path in valid_sessions:
        if exercise not in sessions_by_exercise:
            sessions_by_exercise[exercise] = []
        sessions_by_exercise[exercise].append((session_id, path))

    for exercise, sessions in sessions_by_exercise.items():
        logger.info(f"{exercise}: {len(sessions)} sessions")

        for session_id, session_path in sessions:
            logger.info(f"Processing {exercise}/{session_id}...")

            try:
                # Preprocess session
                session_data = preprocessor.preprocess_session(session_path, exercise)

                # Create windows
                windows = preprocessor.create_windows(session_data, exercise)

                # Add exercise label to each window
                for window in windows:
                    window['exercise'] = exercise
                    window['session_id'] = session_id

                all_windows.extend(windows)
                logger.info(f"  -> {len(windows)} windows created")

            except Exception as e:
                logger.error(f"Failed to process {exercise}/{session_id}: {e}")
                continue

    logger.info(f"Total windows created: {len(all_windows)}")
    return all_windows
