"""
NeuroKit2-based Preprocessing Pipeline for Strength Training ML.

Handles:
- Signal loading and cleaning
- NeuroKit2-based feature extraction (EMG, ECG, EDA, PPG)
- IMU/Accelerometer preprocessing
- Time-based windowing with configurable overlap

Data Sources:
- Biosignals (EMG, ECG, EDA, PPG, IMU): Model input features
- markers.json: Ground truth labels (start/end times, rep markers, phases)
- joint_data.json: Auxiliary ground truth (skeleton for phase/rep derivation)

Note: joint_data is NOT used as model input (only for ground truth label extraction)
Features can be combined with raw signals for model input.
"""

import warnings
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d

# Suppress pandas ChainedAssignmentError warnings from neurokit2 internals (CoW compatibility issue)
warnings.filterwarnings("ignore", message=".*ChainedAssignment.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="neurokit2")

import neurokit2 as nk

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG, SIGNALS


@dataclass
class WindowedSignal:
    """Container for a windowed signal segment."""
    data: np.ndarray
    start_time: float
    end_time: float
    rep_count: int
    phase: str  # 'eccentric', 'concentric', or 'unknown'
    fatigue_score: float


@dataclass
class ExtractedFeatures:
    """Container for extracted features from a signal window."""
    raw_signal: np.ndarray
    features: Dict[str, float]
    feature_vector: np.ndarray


class SignalPreprocessor:
    """
    Preprocessor for individual signal modalities using NeuroKit2.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG

    def preprocess_emg(
        self,
        emg_signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Preprocess EMG signal using NeuroKit2.

        Args:
            emg_signal: Raw EMG signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (cleaned signal, extracted features)
        """
        # Clean EMG signal
        emg_cleaned = nk.emg_clean(emg_signal, sampling_rate=sampling_rate)

        # Extract amplitude envelope
        emg_amplitude = nk.emg_amplitude(emg_cleaned)

        # Extract features
        features = {
            'emg_rms': np.sqrt(np.mean(emg_cleaned ** 2)),
            'emg_mean_amplitude': np.mean(emg_amplitude),
            'emg_max_amplitude': np.max(emg_amplitude),
            'emg_std': np.std(emg_cleaned),
        }

        # Frequency domain features for fatigue detection
        freq_features = self._compute_emg_frequency_features(emg_cleaned, sampling_rate)
        features.update(freq_features)

        return emg_cleaned, features

    def preprocess_ecg(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Preprocess ECG signal and extract HRV features using NeuroKit2.

        Args:
            ecg_signal: Raw ECG signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (cleaned signal, extracted features including HRV)
        """
        features = {}

        # Process ECG (clean, detect R-peaks, etc.)
        ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

        ecg_cleaned = ecg_signals['ECG_Clean'].values

        # Extract R-peaks
        r_peaks = info['ECG_R_Peaks']

        if len(r_peaks) >= 3:
            # Compute HRV features
            hrv_indices = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)

            features['hrv_rmssd'] = hrv_indices['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv_indices else 0
            features['hrv_sdnn'] = hrv_indices['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_indices else 0
            features['hrv_mean_hr'] = hrv_indices['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_indices else 0
            features['hrv_pnn50'] = hrv_indices['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv_indices else 0

        features['ecg_hr'] = len(r_peaks) * (60.0 / (len(ecg_signal) / sampling_rate))

        return ecg_cleaned, features

    def preprocess_eda(
        self,
        eda_signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Preprocess EDA signal using NeuroKit2.

        Args:
            eda_signal: Raw EDA signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (cleaned signal, extracted features)
        """
        features = {}

        # Process EDA
        eda_signals, info = nk.eda_process(eda_signal, sampling_rate=sampling_rate)

        eda_cleaned = eda_signals['EDA_Clean'].values

        # Tonic (SCL) and Phasic (SCR) components
        if 'EDA_Tonic' in eda_signals:
            features['eda_scl_mean'] = np.mean(eda_signals['EDA_Tonic'])
            features['eda_scl_std'] = np.std(eda_signals['EDA_Tonic'])

        if 'EDA_Phasic' in eda_signals:
            features['eda_scr_mean'] = np.mean(eda_signals['EDA_Phasic'])
            features['eda_scr_max'] = np.max(eda_signals['EDA_Phasic'])

        # SCR peaks (skin conductance responses)
        if 'SCR_Peaks' in info:
            features['eda_n_scr'] = len(info['SCR_Peaks'])

        return eda_cleaned, features

    def preprocess_ppg(
        self,
        ppg_signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Preprocess PPG signal using NeuroKit2.

        Args:
            ppg_signal: Raw PPG signal
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (cleaned signal, extracted features)
        """
        features = {}

        # Process PPG
        ppg_signals, info = nk.ppg_process(ppg_signal, sampling_rate=sampling_rate)

        ppg_cleaned = ppg_signals['PPG_Clean'].values

        # Extract heart rate from PPG
        if 'PPG_Peaks' in info:
            peaks = info['PPG_Peaks']
            if len(peaks) > 1:
                ibi = np.diff(peaks) / sampling_rate  # Inter-beat intervals
                features['ppg_hr'] = 60.0 / np.mean(ibi) if len(ibi) > 0 else 0
                features['ppg_hrv'] = np.std(ibi) * 1000 if len(ibi) > 0 else 0  # in ms

        return ppg_cleaned, features

    def preprocess_accelerometer(
        self,
        acc_signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Preprocess accelerometer signal using NeuroKit2 filtering.

        Args:
            acc_signal: Combined accelerometer signal (gravity removed)
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (filtered signal, extracted features)
        """
        # Bandpass filter using NeuroKit2
        acc_filtered = nk.signal_filter(
            acc_signal,
            sampling_rate=sampling_rate,
            lowcut=self.config.preprocessing.acc_lowcut,
            highcut=self.config.preprocessing.acc_highcut,
            method='butterworth',
            order=4
        )

        # Extract movement features
        features = {
            'acc_rms': np.sqrt(np.mean(acc_filtered ** 2)),
            'acc_mean': np.mean(acc_filtered),
            'acc_std': np.std(acc_filtered),
            'acc_max': np.max(np.abs(acc_filtered)),
            'acc_range': np.max(acc_filtered) - np.min(acc_filtered),
        }

        # Zero-crossing rate (movement frequency indicator)
        zero_crossings = np.sum(np.diff(np.signbit(acc_filtered).astype(int)))
        features['acc_zcr'] = zero_crossings / len(acc_filtered)

        # Peak detection for repetition-related features
        peaks, _ = scipy_signal.find_peaks(acc_filtered, height=np.std(acc_filtered))
        features['acc_n_peaks'] = len(peaks)

        return acc_filtered, features

    def _compute_emg_frequency_features(
        self,
        emg_signal: np.ndarray,
        sampling_rate: float
    ) -> Dict[str, float]:
        """
        Compute frequency domain features for EMG fatigue detection.

        Key indicators:
        - Median frequency (MDF): decreases with fatigue
        - Mean power frequency (MNF): decreases with fatigue
        """
        features = {}

        # Compute power spectral density
        freqs, psd = scipy_signal.welch(
            emg_signal,
            fs=sampling_rate,
            nperseg=min(len(emg_signal), int(sampling_rate))
        )

        if len(psd) > 0 and np.sum(psd) > 0:
            # Median frequency
            cumsum = np.cumsum(psd)
            median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
            features['emg_median_freq'] = freqs[min(median_idx, len(freqs)-1)]

            # Mean power frequency
            features['emg_mean_freq'] = np.sum(freqs * psd) / np.sum(psd)

            # Total power
            features['emg_total_power'] = np.sum(psd)

            # Fatigue ratio (high freq / low freq power)
            low_freq_mask = freqs < 60
            high_freq_mask = freqs >= 60
            low_power = np.sum(psd[low_freq_mask])
            high_power = np.sum(psd[high_freq_mask])
            features['emg_fatigue_ratio'] = high_power / (low_power + 1e-10)

        return features


class JointProcessor:
    """
    Processor for skeletal joint data from Azure Kinect.
    """

    # Key joint indices for different exercises
    JOINT_GROUPS = {
        'squat': ['PELVIS', 'SPINE_NAVEL', 'HIP_LEFT', 'HIP_RIGHT',
                  'KNEE_LEFT', 'KNEE_RIGHT', 'ANKLE_LEFT', 'ANKLE_RIGHT'],
        'benchpress': ['SHOULDER_LEFT', 'SHOULDER_RIGHT', 'ELBOW_LEFT',
                       'ELBOW_RIGHT', 'WRIST_LEFT', 'WRIST_RIGHT', 'SPINE_CHEST'],
        'deadlift': ['PELVIS', 'SPINE_NAVEL', 'SPINE_CHEST', 'HIP_LEFT',
                     'HIP_RIGHT', 'KNEE_LEFT', 'KNEE_RIGHT'],
        'pullup': ['SHOULDER_LEFT', 'SHOULDER_RIGHT', 'ELBOW_LEFT',
                   'ELBOW_RIGHT', 'WRIST_LEFT', 'WRIST_RIGHT', 'SPINE_CHEST'],
    }

    # Azure Kinect joint names (32 joints)
    JOINT_NAMES = [
        'PELVIS', 'SPINE_NAVEL', 'SPINE_CHEST', 'NECK',
        'CLAVICLE_LEFT', 'SHOULDER_LEFT', 'ELBOW_LEFT', 'WRIST_LEFT',
        'HAND_LEFT', 'HANDTIP_LEFT', 'THUMB_LEFT',
        'CLAVICLE_RIGHT', 'SHOULDER_RIGHT', 'ELBOW_RIGHT', 'WRIST_RIGHT',
        'HAND_RIGHT', 'HANDTIP_RIGHT', 'THUMB_RIGHT',
        'HIP_LEFT', 'KNEE_LEFT', 'ANKLE_LEFT', 'FOOT_LEFT',
        'HIP_RIGHT', 'KNEE_RIGHT', 'ANKLE_RIGHT', 'FOOT_RIGHT',
        'HEAD', 'NOSE', 'EYE_LEFT', 'EAR_LEFT', 'EYE_RIGHT', 'EAR_RIGHT'
    ]

    # Joint index mapping per exercise for phase detection and rep counting
    EXERCISE_JOINT = {
        'squat': 0,       # PELVIS
        'deadlift': 2,    # SPINE_CHEST
        'benchpress': 2,  # SPINE_CHEST
        'pullup': 2,      # SPINE_CHEST
        'pullups': 2,     # SPINE_CHEST
    }

    def __init__(self, config=None):
        self.config = config or CONFIG

    @classmethod
    def _get_exercise_joint_index(cls, exercise_type: str) -> int:
        """Return the joint index to track for a given exercise."""
        return cls.EXERCISE_JOINT.get(exercise_type.lower(), 0)

    # Bone connections for skeleton visualization (Azure Kinect)
    BONE_CONNECTIONS = [
        # Spine
        ('PELVIS', 'SPINE_NAVEL'),
        ('SPINE_NAVEL', 'SPINE_CHEST'),
        ('SPINE_CHEST', 'NECK'),
        ('NECK', 'HEAD'),
        # Left arm
        ('SPINE_CHEST', 'CLAVICLE_LEFT'),
        ('CLAVICLE_LEFT', 'SHOULDER_LEFT'),
        ('SHOULDER_LEFT', 'ELBOW_LEFT'),
        ('ELBOW_LEFT', 'WRIST_LEFT'),
        ('WRIST_LEFT', 'HAND_LEFT'),
        # Right arm
        ('SPINE_CHEST', 'CLAVICLE_RIGHT'),
        ('CLAVICLE_RIGHT', 'SHOULDER_RIGHT'),
        ('SHOULDER_RIGHT', 'ELBOW_RIGHT'),
        ('ELBOW_RIGHT', 'WRIST_RIGHT'),
        ('WRIST_RIGHT', 'HAND_RIGHT'),
        # Left leg
        ('PELVIS', 'HIP_LEFT'),
        ('HIP_LEFT', 'KNEE_LEFT'),
        ('KNEE_LEFT', 'ANKLE_LEFT'),
        ('ANKLE_LEFT', 'FOOT_LEFT'),
        # Right leg
        ('PELVIS', 'HIP_RIGHT'),
        ('HIP_RIGHT', 'KNEE_RIGHT'),
        ('KNEE_RIGHT', 'ANKLE_RIGHT'),
        ('ANKLE_RIGHT', 'FOOT_RIGHT'),
    ]

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
                return data

            # Check if timestamps are missing (all zeros)
            all_zero = all(f.get('timestamp_usec', 0) == 0 for f in frames)
            if all_zero and len(frames) > 1:
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
                    except Exception:
                        pass

                if fps is None or fps <= 0:
                    fps = self.config.signals.get('joints', None)
                    fps = fps.sampling_rate if fps else 30.0

                # Synthesize timestamps from frame_id
                for frame in frames:
                    frame_id = frame.get('frame_id', 0)
                    frame['timestamp_usec'] = int(frame_id / fps * 1e6)

            return data
        except Exception as e:
            print(f"Error loading joint data: {e}")
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

        if not joint_positions or len(joint_positions) < len(self.JOINT_NAMES):
            return None

        # Create Nx3 numpy array for visualization
        joints_array = np.zeros((len(self.JOINT_NAMES), 3))
        for i, name in enumerate(self.JOINT_NAMES):
            if i < len(joint_positions):
                joints_array[i, 0] = joint_positions[i][0]  # x
                joints_array[i, 1] = joint_positions[i][1]  # y
                joints_array[i, 2] = joint_positions[i][2] if len(joint_positions[i]) > 2 else 0  # z

        # Convert bone connections from names to indices
        name_to_idx = {name: i for i, name in enumerate(self.JOINT_NAMES)}
        bone_indices = []
        for start_name, end_name in self.BONE_CONNECTIONS:
            if start_name in name_to_idx and end_name in name_to_idx:
                bone_indices.append((name_to_idx[start_name], name_to_idx[end_name]))

        return {
            'joints': joints_array,  # Nx3 numpy array
            'joint_names': self.JOINT_NAMES,
            'bone_connections': bone_indices,  # List of (start_idx, end_idx) tuples
            'timestamp': target_time
        }

    def extract_joint_features(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str = None
    ) -> np.ndarray:
        """
        Extract joint features for a time window.

        Args:
            joint_data: Loaded joint data dictionary
            start_time: Window start time in seconds (relative)
            end_time: Window end time in seconds (relative)
            exercise_type: Type of exercise for relevant joint selection

        Returns:
            Feature vector for the window
        """
        frames = joint_data.get('frames', [])

        if not frames:
            return np.zeros(self.config.signals['joints'].channels)

        # Convert absolute timestamps to relative time (seconds from first frame)
        first_timestamp = frames[0].get('timestamp_usec', 0)

        # Filter frames within time window using relative timestamps
        window_frames = []
        for frame in frames:
            abs_timestamp = frame.get('timestamp_usec', 0)
            # Convert to relative time in seconds
            rel_time = (abs_timestamp - first_timestamp) / 1e6
            if start_time <= rel_time <= end_time:
                window_frames.append(frame)

        if not window_frames:
            return np.zeros(self.config.signals['joints'].channels)

        # Extract positions from frames
        positions = []
        for frame in window_frames:
            bodies = frame.get('bodies', [])
            if bodies:
                body = bodies[0]  # Use first detected body
                joint_positions = body.get('joint_positions', [])
                if joint_positions:
                    positions.append(np.array(joint_positions).flatten())

        if not positions:
            return np.zeros(self.config.signals['joints'].channels)

        positions = np.array(positions)

        # Compute features
        features = []

        # Mean positions
        features.extend(np.mean(positions, axis=0))

        # Velocity (change in position)
        if len(positions) > 1:
            velocities = np.diff(positions, axis=0)
            features.extend(np.mean(np.abs(velocities), axis=0)[:32])  # Limit to 32 values
        else:
            features.extend(np.zeros(32))

        # Range of motion
        rom = np.max(positions, axis=0) - np.min(positions, axis=0)
        features.extend(rom[:32])

        feature_vector = np.array(features[:self.config.signals['joints'].channels])

        # Pad if necessary
        if len(feature_vector) < self.config.signals['joints'].channels:
            feature_vector = np.pad(
                feature_vector,
                (0, self.config.signals['joints'].channels - len(feature_vector))
            )

        return feature_vector

    def detect_phase(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str
    ) -> str:
        """
        Detect movement phase (eccentric/concentric) from joint data.

        Args:
            joint_data: Loaded joint data dictionary
            start_time: Window start time (relative, in seconds)
            end_time: Window end time (relative, in seconds)
            exercise_type: Type of exercise

        Returns:
            Phase label: 'eccentric', 'concentric', or 'unknown'
        """
        frames = joint_data.get('frames', [])

        if not frames or len(frames) < 2:
            return 'unknown'

        # Convert absolute timestamps to relative time (seconds from first frame)
        first_timestamp = frames[0].get('timestamp_usec', 0)

        # Filter frames within time window using relative timestamps
        window_frames = []
        for frame in frames:
            abs_timestamp = frame.get('timestamp_usec', 0)
            # Convert to relative time in seconds
            rel_time = (abs_timestamp - first_timestamp) / 1e6
            if start_time <= rel_time <= end_time:
                window_frames.append(frame)

        if len(window_frames) < 2:
            return 'unknown'

        # Get key joint positions based on exercise
        # Squat: PELVIS (index 0)
        # Deadlift: SPINE_CHEST (index 2) - chest moves during hip hinge
        # Benchpress: SPINE_CHEST (index 2) - chest reference for bar path
        # Pullups: SPINE_CHEST (index 2) - body moves up/down
        joint_idx = self._get_exercise_joint_index(exercise_type)

        key_positions = []
        for frame in window_frames:
            bodies = frame.get('bodies', [])
            if bodies:
                body = bodies[0]
                joint_positions = body.get('joint_positions', [])
                if joint_positions and len(joint_positions) > joint_idx:
                    if len(joint_positions[joint_idx]) > 1:
                        key_positions.append(joint_positions[joint_idx][1])  # Y-axis

        if len(key_positions) < 2:
            return 'unknown'

        # Determine phase based on movement direction
        # Use first third and last third for more robust comparison
        n = len(key_positions)
        third = max(1, n // 3)
        start_pos = np.mean(key_positions[:third])
        end_pos = np.mean(key_positions[-third:])

        movement = end_pos - start_pos
        threshold = 0.01  # 1cm threshold to avoid noise

        if abs(movement) < threshold:
            return 'unknown'

        # In the joint data coordinate system, Y decreases = concentric (moving down)
        # Squat: concentric = pelvis Y decreases (standing up)
        # Deadlift: concentric = chest Y decreases (standing up)
        # Benchpress: concentric = chest Y decreases (pressing up)
        # Pullups: concentric = chest Y decreases (pulling up)
        return 'concentric' if movement < 0 else 'eccentric'

    def calculate_movement_velocity(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str
    ) -> float:
        """
        Calculate average movement velocity from joint data.
        Used for fatigue estimation (velocity decreases with fatigue).

        Args:
            joint_data: Loaded joint data dictionary
            start_time: Window start time (relative)
            end_time: Window end time (relative)
            exercise_type: Type of exercise

        Returns:
            Average velocity (m/s) or 0 if cannot calculate
        """
        frames = joint_data.get('frames', [])

        if not frames or len(frames) < 2:
            return 0.0

        first_timestamp = frames[0].get('timestamp_usec', 0)

        # Get positions and timestamps within window
        joint_idx = self._get_exercise_joint_index(exercise_type)
        positions = []
        timestamps = []

        for frame in frames:
            abs_timestamp = frame.get('timestamp_usec', 0)
            rel_time = (abs_timestamp - first_timestamp) / 1e6

            if start_time <= rel_time <= end_time:
                bodies = frame.get('bodies', [])
                if bodies:
                    body = bodies[0]
                    joint_positions = body.get('joint_positions', [])
                    if joint_positions and len(joint_positions) > joint_idx:
                        positions.append(joint_positions[joint_idx])
                        timestamps.append(rel_time)

        if len(positions) < 2:
            return 0.0

        # Calculate velocities
        velocities = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                # 3D distance
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                dz = positions[i][2] - positions[i-1][2] if len(positions[i]) > 2 else 0
                distance = np.sqrt(dx**2 + dy**2 + dz**2)
                velocities.append(distance / dt)

        return np.mean(velocities) if velocities else 0.0

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
                initial_velocity = self.calculate_movement_velocity(
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
        current_velocity = self.calculate_movement_velocity(
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
            effective_fps = 15.0

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
    Main data preprocessing pipeline combining all signal processors.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.signal_processor = SignalPreprocessor(config)
        self.joint_processor = JointProcessor(config)

        # Clustering-based phase detector (lazy initialization)
        self._phase_detector = None
        self._phase_detector_initialized = False

    def _get_phase_detector(self):
        """Get or initialize the clustering-based phase detector."""
        if self._phase_detector_initialized:
            return self._phase_detector

        # Check if clustering method is configured
        if not hasattr(self.config, 'phase_detection') or \
           self.config.phase_detection.method != 'clustering':
            self._phase_detector_initialized = True
            return None

        from .phase_clustering import ClusteringPhaseDetector

        # Try to load pre-trained model
        if self.config.phase_detection.pretrained_model_path:
            model_path = Path(self.config.phase_detection.pretrained_model_path)
            if model_path.exists():
                self._phase_detector = ClusteringPhaseDetector(
                    n_clusters=self.config.phase_detection.n_clusters,
                    use_dbscan=self.config.phase_detection.use_dbscan,
                    smoothing_window=self.config.phase_detection.smoothing_window,
                    velocity_threshold=self.config.phase_detection.velocity_threshold
                )
                self._phase_detector.load(model_path)
                print(f"Loaded clustering phase detector from {model_path}")
                self._phase_detector_initialized = True
                return self._phase_detector

        # No pre-trained model - will need to be trained
        self._phase_detector = ClusteringPhaseDetector(
            n_clusters=self.config.phase_detection.n_clusters,
            use_dbscan=self.config.phase_detection.use_dbscan,
            smoothing_window=self.config.phase_detection.smoothing_window,
            velocity_threshold=self.config.phase_detection.velocity_threshold
        )
        self._phase_detector_initialized = True
        return self._phase_detector

    def detect_phase(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str
    ) -> str:
        """
        Detect phase using configured method (rule-based or clustering).

        Args:
            joint_data: Joint data dictionary
            start_time: Window start time
            end_time: Window end time
            exercise_type: Type of exercise

        Returns:
            Phase label: 'eccentric', 'concentric', 'rest', or 'unknown'
        """
        # Check if we should use clustering
        phase_detector = self._get_phase_detector()

        if phase_detector is not None and phase_detector.is_fitted:
            # Use clustering-based detection
            return phase_detector.predict_phase(
                joint_data, start_time, end_time, exercise_type
            )

        # Fall back to rule-based detection
        return self.joint_processor.detect_phase(
            joint_data, start_time, end_time, exercise_type
        )

    def train_phase_detector(
        self,
        valid_sessions: List[Tuple[str, str, Path]]
    ) -> bool:
        """
        Train the clustering-based phase detector on valid sessions.

        Args:
            valid_sessions: List of (exercise, session_id, path) tuples

        Returns:
            True if training successful, False otherwise
        """
        phase_detector = self._get_phase_detector()
        if phase_detector is None:
            print("Phase detection not configured for clustering mode")
            return False

        if phase_detector.is_fitted:
            print("Phase detector already trained")
            return True

        if not hasattr(self.config, 'phase_detection') or \
           not self.config.phase_detection.auto_train:
            print("Auto-training disabled for phase detector")
            return False

        print("\nTraining clustering-based phase detector...")

        all_features = []
        window_sec = self.config.data.time_window_sec
        window_stride = window_sec * (1 - self.config.data.overlap)

        for exercise, session_id, session_path in valid_sessions:
            # Load joint data
            joint_path = session_path / self.config.data.joints_file
            if not joint_path.exists():
                continue

            joint_data = self.joint_processor.load_joint_data(joint_path)
            if not joint_data:
                continue

            frames = joint_data.get('frames', [])
            if len(frames) < 2:
                continue

            # Get time range
            first_ts = frames[0].get('timestamp_usec', 0)
            last_ts = frames[-1].get('timestamp_usec', 0)
            duration = (last_ts - first_ts) / 1e6

            # Extract features for windows
            n_windows = int((duration - window_sec) / window_stride) + 1

            for i in range(n_windows):
                start_time = i * window_stride
                end_time = start_time + window_sec

                features = phase_detector.extract_features(
                    joint_data, start_time, end_time, exercise
                )

                if np.any(features):
                    all_features.append(features)

        if len(all_features) < phase_detector.n_clusters:
            print(f"Not enough features for training: {len(all_features)}")
            return False

        all_features = np.array(all_features)
        print(f"Extracted {len(all_features)} feature vectors from {len(valid_sessions)} sessions")

        # Fit the detector
        phase_detector.fit(all_features)

        # Save if configured
        if self.config.phase_detection.save_trained_model:
            save_path = self.config.output.output_dir / "phase_detector.pkl"
            phase_detector.save(save_path)

        print("Phase detector training complete!")
        return True

    def preprocess_session(
        self,
        session_path: Path,
        exercise_type: str
    ) -> Dict[str, Any]:
        """
        Preprocess all data for a session.

        Args:
            session_path: Path to session folder
            exercise_type: Type of exercise (Squat, Benchpress, Pullups)

        Returns:
            Dictionary with preprocessed signals and metadata
        """
        result = {
            'signals': {},
            'features': {},
            'markers': None,
            'joint_data': None,
            'windows': [],
            'metadata': {
                'session_path': str(session_path),
                'exercise': exercise_type
            }
        }

        # Load and process markers
        markers_path = session_path / self.config.data.markers_file
        if markers_path.exists():
            with open(markers_path, 'r') as f:
                markers_data = json.load(f)
            result['markers'] = markers_data

        # Load joint data
        joint_path = session_path / self.config.data.joints_file
        if joint_path.exists():
            result['joint_data'] = self.joint_processor.load_joint_data(joint_path)

        # Process each signal type
        for signal_name, signal_config in SIGNALS.items():
            if not signal_config.enabled:
                continue

            if signal_config.file_name.endswith('.json'):
                continue  # Joint data handled separately

            signal_path = session_path / signal_config.file_name

            if not signal_path.exists():
                continue

            try:
                # Load signal
                df = pd.read_csv(signal_path)
                signal_data = df.iloc[:, 1].values  # Second column is signal value
                time_data = df.iloc[:, 0].values  # First column is time

                # Handle NaN values
                nan_mask = np.isnan(signal_data)
                if nan_mask.any():
                    signal_data = np.nan_to_num(signal_data, nan=0.0)

                # Preprocess based on signal type
                if signal_name == 'emg':
                    processed, features = self.signal_processor.preprocess_emg(
                        signal_data, signal_config.sampling_rate
                    )
                elif signal_name == 'ecg':
                    processed, features = self.signal_processor.preprocess_ecg(
                        signal_data, signal_config.sampling_rate
                    )
                elif signal_name == 'eda':
                    processed, features = self.signal_processor.preprocess_eda(
                        signal_data, signal_config.sampling_rate
                    )
                elif signal_name.startswith('ppg'):
                    processed, features = self.signal_processor.preprocess_ppg(
                        signal_data, signal_config.sampling_rate
                    )
                elif signal_name == 'acc':
                    processed, features = self.signal_processor.preprocess_accelerometer(
                        signal_data, signal_config.sampling_rate
                    )
                else:
                    processed = signal_data
                    features = {}

                # Ensure processed data has same length as time_data
                if len(processed) != len(time_data):
                    # NeuroKit2 may return slightly different length, truncate/pad
                    if len(processed) > len(time_data):
                        processed = processed[:len(time_data)]
                    elif len(processed) == 0:
                        processed = np.zeros(len(time_data))
                    else:
                        processed = np.pad(processed, (0, len(time_data) - len(processed)), mode='edge')

                result['signals'][signal_name] = {
                    'data': processed,
                    'time': time_data,
                    'sampling_rate': signal_config.sampling_rate
                }
                result['features'][signal_name] = features

            except Exception as e:
                print(f"Error processing {signal_name}: {e}")
                raise

        return result

    def create_windows(
        self,
        session_data: Dict,
        exercise_type: str
    ) -> List[Dict]:
        """
        Create time-based windows from preprocessed session data.

        Args:
            session_data: Output from preprocess_session
            exercise_type: Type of exercise

        Returns:
            List of window dictionaries
        """
        windows = []

        markers = session_data.get('markers', {})
        marker_list = markers.get('markers', [])

        if not marker_list:
            print("      No markers found")
            return windows

        # Check if any signals were loaded
        if not session_data.get('signals'):
            print("      No signals loaded")
            return windows

        # Find start and end times
        start_marker = next(
            (m for m in marker_list if m.get('label', '').lower() == 'start'),
            None
        )

        if not start_marker:
            print("      No 'start' marker found")
            return windows

        start_time = start_marker.get('time', 0)
        end_time = marker_list[-1].get('time', 0)

        # Calculate window parameters
        window_sec = self.config.data.time_window_sec
        overlap = self.config.data.overlap
        step_sec = window_sec * (1 - overlap)

        # Pre-compute rep times from joint data (for per-window lookup)
        joint_rep_times = None
        initial_velocity = None
        if session_data.get('joint_data'):
            joint_rep_times = self.joint_processor.detect_rep_times(
                session_data['joint_data'],
                start_time,
                end_time,
                exercise_type
            )
            # Cache initial velocity once (avoid recomputing for every window)
            for offset in range(4):
                t_start = start_time + offset * window_sec
                initial_velocity = self.joint_processor.calculate_movement_velocity(
                    session_data['joint_data'],
                    t_start,
                    t_start + window_sec * 2,
                    exercise_type
                )
                if initial_velocity > 0:
                    break

        # Generate windows
        window_start = start_time
        window_idx = 0

        while window_start + window_sec <= end_time:
            window_end = window_start + window_sec

            window_data = {
                'window_idx': window_idx,
                'start_time': window_start,
                'end_time': window_end,
                'signals': {},
                'joint_features': None,
                'skeleton_frame': None,  # For visualization
                'phase': 'unknown',
                'rep_count': 0,
                'fatigue_score': 0.0
            }

            # Extract signals for this window
            for signal_name, signal_info in session_data['signals'].items():
                signal_data = signal_info['data']
                time_data = signal_info['time']
                fs = signal_info['sampling_rate']

                # Find indices for this time window
                mask = (time_data >= window_start) & (time_data <= window_end)
                window_signal = signal_data[mask]

                # Ensure correct length
                expected_samples = int(fs * window_sec)
                if len(window_signal) == 0:
                    # Empty window - create zeros
                    window_signal = np.zeros(expected_samples)
                elif len(window_signal) < expected_samples:
                    # Pad with edge values
                    window_signal = np.pad(
                        window_signal,
                        (0, expected_samples - len(window_signal)),
                        mode='edge'
                    )
                elif len(window_signal) > expected_samples:
                    window_signal = window_signal[:expected_samples]

                window_data['signals'][signal_name] = window_signal

            # GROUND TRUTH LABEL EXTRACTION
            #
            # Rep counting: per-window (0 or 1) for live prediction.
            #   Primary: markers.json valley markers in [window_start, window_end]
            #   Secondary: joint_data peak detection if no markers available
            #
            # Phase: from joint_data movement direction
            # Fatigue: velocity degradation over time

            # --- Rep count (per window) ---
            # Primary: count marker valleys within this window
            window_data['rep_count'] = sum(
                1 for m in marker_list
                if window_start <= m.get('time', 0) <= window_end
                and m.get('label', '').lower() == 'valley'
            )

            # If no valley markers exist at all, fall back to joint peak detection
            if not any(m.get('label', '').lower() == 'valley' for m in marker_list):
                if joint_rep_times is not None:
                    window_data['rep_count'] = sum(
                        1 for t in joint_rep_times
                        if window_start <= t <= window_end
                    )

            # --- Phase, joint features, fatigue ---
            if session_data.get('joint_data'):
                window_data['joint_features'] = self.joint_processor.extract_joint_features(
                    session_data['joint_data'],
                    window_start,
                    window_end,
                    exercise_type
                )

                window_mid_time = (window_start + window_end) / 2
                window_data['skeleton_frame'] = self.joint_processor.get_skeleton_frame(
                    session_data['joint_data'],
                    window_mid_time
                )

                window_data['phase'] = self.detect_phase(
                    session_data['joint_data'],
                    window_start,
                    window_end,
                    exercise_type
                )

                window_data['fatigue_score'] = self.joint_processor.calculate_fatigue_from_velocity(
                    session_data['joint_data'],
                    window_end,
                    start_time,
                    exercise_type,
                    window_sec,
                    initial_velocity=initial_velocity
                )
            else:
                # Time-based fatigue estimation (simple progress metric)
                elapsed = end_time - start_time
                window_data['fatigue_score'] = (window_start - start_time) / elapsed if elapsed > 0 else 0.0

            windows.append(window_data)

            window_start += step_sec
            window_idx += 1

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
        from data.validate_data import DataValidator
        validator = DataValidator(dataset_path, config)
        validator.validate_all()
        valid_sessions = validator.get_valid_sessions()

    if not valid_sessions:
        print("No valid sessions found!")
        return []

    print(f"\nProcessing {len(valid_sessions)} valid sessions...")

    preprocessor = DataPreprocessor(config)

    # Train clustering phase detector if configured
    if hasattr(config, 'phase_detection') and config.phase_detection.method == 'clustering':
        preprocessor.train_phase_detector(valid_sessions)

    all_windows = []

    # Group sessions by exercise for organized processing
    sessions_by_exercise = {}
    for exercise, session_id, path in valid_sessions:
        if exercise not in sessions_by_exercise:
            sessions_by_exercise[exercise] = []
        sessions_by_exercise[exercise].append((session_id, path))

    for exercise, sessions in sessions_by_exercise.items():
        print(f"\n  {exercise}: {len(sessions)} sessions")

        for session_id, session_path in sessions:
            print(f"    Processing {exercise}/{session_id}...")

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
                print(f"      -> {len(windows)} windows created")

            except Exception as e:
                print(f"      [ERROR] Failed to process: {e}")
                continue

    print(f"\nTotal windows created: {len(all_windows)}")
    return all_windows


def _find_session_dirs(exercise_path: Path) -> List[Tuple[str, Path]]:
    """Find session directories, supporting nested PersonXX/SetYY layout.

    Returns list of (session_id, path) tuples.
    """
    sessions = []

    for child in sorted(exercise_path.iterdir()):
        if not child.is_dir():
            continue

        # Check if this directory directly contains data files
        has_data = (child / 'markers.json').exists()
        if has_data:
            sessions.append((child.name, child))
        else:
            # Nested layout: Exercise/PersonXX/SetYY/
            for sub in sorted(child.iterdir()):
                if sub.is_dir() and (sub / 'markers.json').exists():
                    session_id = f"{child.name}/{sub.name}"
                    sessions.append((session_id, sub))

    return sessions


def preprocess_dataset_legacy(
    dataset_path: Path = None,
    config=None
) -> List[Dict]:
    """
    Legacy function - processes ALL sessions without validation.
    Use preprocess_dataset() instead for proper validation.
    """
    config = config or CONFIG
    dataset_path = dataset_path or config.data.dataset_path

    preprocessor = DataPreprocessor(config)
    all_windows = []

    active_exercises = config.get_active_exercises() if hasattr(config, 'get_active_exercises') \
        else [e for e in config.data.exercises if e not in getattr(config.data, 'excluded_exercises', [])]

    for exercise in active_exercises:
        exercise_path = dataset_path / exercise

        if not exercise_path.exists():
            continue

        session_dirs = _find_session_dirs(exercise_path)

        for session_id, session_dir in session_dirs:
            print(f"Processing {exercise}/{session_id}...")

            # Preprocess session
            session_data = preprocessor.preprocess_session(session_dir, exercise)

            # Create windows
            windows = preprocessor.create_windows(session_data, exercise)

            # Add exercise label to each window
            for window in windows:
                window['exercise'] = exercise
                window['session_id'] = session_id

            all_windows.extend(windows)

    print(f"Total windows created: {len(all_windows)}")
    return all_windows
