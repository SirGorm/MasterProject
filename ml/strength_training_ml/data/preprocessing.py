"""
NeuroKit2-based Preprocessing Pipeline for Strength Training ML.

Handles:
- Signal loading and cleaning
- NeuroKit2-based feature extraction (EMG, ECG, EDA, PPG)
- IMU/Accelerometer preprocessing
- Joint data processing
- Time-based windowing with configurable overlap

Features can be combined with raw signals for model input.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import signal as scipy_signal
from scipy.ndimage import uniform_filter1d

try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    print("Warning: NeuroKit2 not available. Using fallback preprocessing.")

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
        features = {}

        if NEUROKIT_AVAILABLE:
            try:
                # Clean EMG signal
                emg_cleaned = nk.emg_clean(emg_signal, sampling_rate=sampling_rate)

                # Extract amplitude envelope
                emg_amplitude = nk.emg_amplitude(emg_cleaned)

                # Extract features
                features['emg_rms'] = np.sqrt(np.mean(emg_cleaned ** 2))
                features['emg_mean_amplitude'] = np.mean(emg_amplitude)
                features['emg_max_amplitude'] = np.max(emg_amplitude)
                features['emg_std'] = np.std(emg_cleaned)

                # Frequency domain features for fatigue detection
                freq_features = self._compute_emg_frequency_features(emg_cleaned, sampling_rate)
                features.update(freq_features)

                return emg_cleaned, features

            except Exception as e:
                print(f"NeuroKit2 EMG processing failed: {e}, using fallback")

        # Fallback: simple bandpass filter
        emg_cleaned = self._bandpass_filter(
            emg_signal,
            lowcut=20.0,
            highcut=450.0,
            fs=sampling_rate
        )

        features['emg_rms'] = np.sqrt(np.mean(emg_cleaned ** 2))
        features['emg_std'] = np.std(emg_cleaned)

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

        if NEUROKIT_AVAILABLE:
            try:
                # Process ECG (clean, detect R-peaks, etc.)
                ecg_signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

                ecg_cleaned = ecg_signals['ECG_Clean'].values

                # Extract R-peaks
                r_peaks = info['ECG_R_Peaks']

                if len(r_peaks) >= 3:
                    # Compute HRV features
                    hrv_indices = nk.hrv_time(r_peaks, sampling_rate=sampling_rate)

                    # Key HRV features for fatigue
                    features['hrv_rmssd'] = hrv_indices['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv_indices else 0
                    features['hrv_sdnn'] = hrv_indices['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_indices else 0
                    features['hrv_mean_hr'] = hrv_indices['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_indices else 0
                    features['hrv_pnn50'] = hrv_indices['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv_indices else 0

                features['ecg_hr'] = len(r_peaks) * (60.0 / (len(ecg_signal) / sampling_rate))

                return ecg_cleaned, features

            except Exception as e:
                print(f"NeuroKit2 ECG processing failed: {e}, using fallback")

        # Fallback: simple bandpass filter
        ecg_cleaned = self._bandpass_filter(
            ecg_signal,
            lowcut=0.5,
            highcut=40.0,
            fs=sampling_rate
        )

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

        if NEUROKIT_AVAILABLE:
            try:
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

            except Exception as e:
                print(f"NeuroKit2 EDA processing failed: {e}, using fallback")

        # Fallback: lowpass filter
        eda_cleaned = self._lowpass_filter(eda_signal, cutoff=5.0, fs=sampling_rate)
        features['eda_mean'] = np.mean(eda_cleaned)
        features['eda_std'] = np.std(eda_cleaned)

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

        if NEUROKIT_AVAILABLE:
            try:
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

            except Exception as e:
                print(f"NeuroKit2 PPG processing failed: {e}, using fallback")

        # Fallback: bandpass filter
        ppg_cleaned = self._bandpass_filter(
            ppg_signal,
            lowcut=0.5,
            highcut=4.0,
            fs=sampling_rate
        )

        return ppg_cleaned, features

    def preprocess_accelerometer(
        self,
        acc_signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Preprocess accelerometer signal.

        Args:
            acc_signal: Combined accelerometer signal (gravity removed)
            sampling_rate: Sampling rate in Hz

        Returns:
            Tuple of (filtered signal, extracted features)
        """
        # Bandpass filter to remove noise and drift
        acc_filtered = self._bandpass_filter(
            acc_signal,
            lowcut=self.config.preprocessing.acc_lowcut,
            highcut=self.config.preprocessing.acc_highcut,
            fs=sampling_rate
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

    def _bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
        order: int = 4
    ) -> np.ndarray:
        """Apply bandpass Butterworth filter."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = min(highcut / nyq, 0.99)

        if low >= high:
            return data

        b, a = scipy_signal.butter(order, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, data)

    def _lowpass_filter(
        self,
        data: np.ndarray,
        cutoff: float,
        fs: float,
        order: int = 4
    ) -> np.ndarray:
        """Apply lowpass Butterworth filter."""
        nyq = 0.5 * fs
        normalized_cutoff = min(cutoff / nyq, 0.99)

        b, a = scipy_signal.butter(order, normalized_cutoff, btype='low')
        return scipy_signal.filtfilt(b, a, data)


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
        'pullups': ['SHOULDER_LEFT', 'SHOULDER_RIGHT', 'ELBOW_LEFT',
                    'ELBOW_RIGHT', 'WRIST_LEFT', 'WRIST_RIGHT', 'SPINE_NAVEL']
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

    def __init__(self, config=None):
        self.config = config or CONFIG

    def load_joint_data(self, joint_path: Path) -> Optional[Dict]:
        """
        Load joint data from JSON file.

        Args:
            joint_path: Path to joint_data.json

        Returns:
            Dictionary with frames and bone list
        """
        try:
            with open(joint_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading joint data: {e}")
            return None

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
        key_positions = []
        for frame in window_frames:
            bodies = frame.get('bodies', [])
            if bodies:
                body = bodies[0]
                joint_positions = body.get('joint_positions', [])
                if joint_positions and len(joint_positions) > 0:
                    # Extract relevant joint based on exercise
                    if exercise_type.lower() == 'squat':
                        # Use pelvis Y position (index 0)
                        if len(joint_positions) > 0 and len(joint_positions[0]) > 1:
                            key_positions.append(joint_positions[0][1])
                    elif exercise_type.lower() == 'benchpress':
                        # Use wrist Y position (index 7 = WRIST_LEFT)
                        if len(joint_positions) > 7 and len(joint_positions[7]) > 1:
                            key_positions.append(joint_positions[7][1])
                    elif exercise_type.lower() == 'pullups':
                        # Use spine/chest Y position (index 2 = SPINE_CHEST)
                        if len(joint_positions) > 2 and len(joint_positions[2]) > 1:
                            key_positions.append(joint_positions[2][1])

        if len(key_positions) < 2:
            return 'unknown'

        # Determine phase based on movement direction
        # Use first third and last third for more robust comparison
        n = len(key_positions)
        third = max(1, n // 3)
        start_pos = np.mean(key_positions[:third])
        end_pos = np.mean(key_positions[-third:])

        # Calculate movement threshold (to avoid noise)
        movement = end_pos - start_pos
        threshold = 0.01  # 1cm threshold to avoid noise

        if abs(movement) < threshold:
            # Minimal movement - alternate based on position in sequence
            return 'unknown'

        # Phase determination varies by exercise
        # Y-axis typically points up in camera coordinate system
        if exercise_type.lower() == 'squat':
            # Squat: eccentric = lowering (Y decreases), concentric = rising (Y increases)
            return 'eccentric' if movement < 0 else 'concentric'
        elif exercise_type.lower() == 'benchpress':
            # Bench: eccentric = lowering bar (Y decreases), concentric = pushing up (Y increases)
            return 'eccentric' if movement < 0 else 'concentric'
        elif exercise_type.lower() == 'pullups':
            # Pullups: concentric = pulling up (Y increases), eccentric = lowering (Y decreases)
            return 'eccentric' if movement < 0 else 'concentric'

        return 'unknown'

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
                    if joint_positions and len(joint_positions) > 0:
                        # Get key joint based on exercise
                        if exercise_type.lower() == 'squat':
                            if len(joint_positions) > 0:
                                positions.append(joint_positions[0])  # PELVIS
                                timestamps.append(rel_time)
                        elif exercise_type.lower() == 'benchpress':
                            if len(joint_positions) > 7:
                                positions.append(joint_positions[7])  # WRIST_LEFT
                                timestamps.append(rel_time)
                        elif exercise_type.lower() == 'pullups':
                            if len(joint_positions) > 2:
                                positions.append(joint_positions[2])  # SPINE_CHEST
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
        window_sec: float = 2.0
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

        Returns:
            Fatigue score between 0 (fresh) and 1 (fatigued)
        """
        # Calculate initial velocity (first few seconds)
        initial_velocity = self.calculate_movement_velocity(
            joint_data,
            exercise_start_time,
            exercise_start_time + window_sec * 2,  # First 2 windows
            exercise_type
        )

        # Calculate current velocity
        current_velocity = self.calculate_movement_velocity(
            joint_data,
            current_time - window_sec,
            current_time,
            exercise_type
        )

        if initial_velocity <= 0:
            # Can't calculate, use time-based estimate
            return 0.0

        # Fatigue = velocity reduction ratio
        velocity_ratio = current_velocity / initial_velocity

        # Clamp and invert (lower velocity = higher fatigue)
        fatigue = 1.0 - min(1.0, max(0.0, velocity_ratio))

        return fatigue

    def count_reps_from_peaks(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str
    ) -> int:
        """
        Count repetitions by detecting peaks in joint movement.

        Args:
            joint_data: Loaded joint data dictionary
            start_time: Exercise start time
            end_time: Current time (cumulative count up to this point)
            exercise_type: Type of exercise

        Returns:
            Number of repetitions detected
        """
        frames = joint_data.get('frames', [])

        if not frames or len(frames) < 10:
            return 0

        first_timestamp = frames[0].get('timestamp_usec', 0)

        # Extract key joint positions over time
        positions = []
        for frame in frames:
            abs_timestamp = frame.get('timestamp_usec', 0)
            rel_time = (abs_timestamp - first_timestamp) / 1e6

            if start_time <= rel_time <= end_time:
                bodies = frame.get('bodies', [])
                if bodies:
                    body = bodies[0]
                    joint_positions = body.get('joint_positions', [])
                    if joint_positions:
                        # Get Y position of key joint
                        if exercise_type.lower() == 'squat':
                            if len(joint_positions) > 0:
                                positions.append(joint_positions[0][1])  # PELVIS Y
                        elif exercise_type.lower() == 'benchpress':
                            if len(joint_positions) > 7:
                                positions.append(joint_positions[7][1])  # WRIST Y
                        elif exercise_type.lower() == 'pullups':
                            if len(joint_positions) > 2:
                                positions.append(joint_positions[2][1])  # SPINE_CHEST Y

        if len(positions) < 10:
            return 0

        # Smooth the signal
        positions = np.array(positions)
        if len(positions) > 5:
            positions = uniform_filter1d(positions, size=5)

        # Find peaks (local maxima for pullups, local minima for squat/bench)
        if exercise_type.lower() == 'pullups':
            # For pullups, count peaks (highest points = top of pull)
            peaks, _ = scipy_signal.find_peaks(positions, distance=15, prominence=0.05)
        else:
            # For squat/bench, count valleys (lowest points = bottom of movement)
            peaks, _ = scipy_signal.find_peaks(-positions, distance=15, prominence=0.05)

        return len(peaks)


class DataPreprocessor:
    """
    Main data preprocessing pipeline combining all signal processors.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.signal_processor = SignalPreprocessor(config)
        self.joint_processor = JointProcessor(config)

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
                # Store raw signal as fallback
                try:
                    df = pd.read_csv(signal_path)
                    signal_data = df.iloc[:, 1].values
                    time_data = df.iloc[:, 0].values
                    signal_data = np.nan_to_num(signal_data, nan=0.0)
                    result['signals'][signal_name] = {
                        'data': signal_data,
                        'time': time_data,
                        'sampling_rate': signal_config.sampling_rate
                    }
                    result['features'][signal_name] = {}
                except:
                    pass

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

            # Extract joint features and ground truth labels from joint_data
            if session_data.get('joint_data'):
                window_data['joint_features'] = self.joint_processor.extract_joint_features(
                    session_data['joint_data'],
                    window_start,
                    window_end,
                    exercise_type
                )

                # Detect phase from joint_data (ground truth)
                window_data['phase'] = self.joint_processor.detect_phase(
                    session_data['joint_data'],
                    window_start,
                    window_end,
                    exercise_type
                )

                # Count repetitions from joint_data peaks (ground truth)
                window_data['rep_count'] = self.joint_processor.count_reps_from_peaks(
                    session_data['joint_data'],
                    start_time,  # Count from exercise start
                    window_end,  # Up to current window
                    exercise_type
                )

                # Estimate fatigue from velocity degradation (ground truth)
                window_data['fatigue_score'] = self.joint_processor.calculate_fatigue_from_velocity(
                    session_data['joint_data'],
                    window_end,
                    start_time,
                    exercise_type,
                    window_sec
                )
            else:
                # Fallback to marker-based rep count if no joint_data
                rep_count = sum(
                    1 for m in marker_list
                    if m.get('time', 0) <= window_end and m.get('label', '').lower() != 'start'
                )
                window_data['rep_count'] = rep_count

                # Fallback to time-based fatigue if no joint_data
                progress = (window_start - start_time) / (end_time - start_time)
                window_data['fatigue_score'] = progress

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

    for exercise in config.data.exercises:
        exercise_path = dataset_path / exercise

        if not exercise_path.exists():
            continue

        session_dirs = [
            d for d in sorted(exercise_path.iterdir())
            if d.is_dir() and d.name.isdigit()
        ]

        for session_dir in session_dirs:
            print(f"Processing {exercise}/{session_dir.name}...")

            # Preprocess session
            session_data = preprocessor.preprocess_session(session_dir, exercise)

            # Create windows
            windows = preprocessor.create_windows(session_data, exercise)

            # Add exercise label to each window
            for window in windows:
                window['exercise'] = exercise
                window['session_id'] = session_dir.name

            all_windows.extend(windows)

    print(f"Total windows created: {len(all_windows)}")
    return all_windows
