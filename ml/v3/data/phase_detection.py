"""
Phase detection module with pseudo-label generation for semi-supervised learning.

Workflow:
1. Extract kinematic + biosignal features from labeled windows
2. Train a teacher model (RandomForest) on engineered features + ground truth labels
3. Use teacher to predict pseudo-labels on unlabeled data with confidence filtering
4. (Future) Use pseudo-labels to train a deep learning model on raw time series

Pseudo-labels bridge the gap between rule-based joint kinematics (available only
when skeleton data exists) and biosignal-only inference (needed in production).
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from collections import Counter

from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ml.v3.utils.constants import JOINT_NAMES, JOINT_TO_IDX, PHASES, get_exercise_joints
from ml.v3.utils.logging_utils import get_logger


logger = get_logger('phase')


class PhaseDetection:
    """
    Phase detector with teacher model for pseudo-label generation.

    Modes:
    1. Rule-based: Uses Y-axis velocity direction (fallback when no teacher).
    2. Teacher (supervised): RandomForest trained on engineered features +
       ground truth labels from joint kinematics.
    3. Pseudo-label generation: Teacher predicts on unlabeled windows,
       confidence-filtered outputs serve as training data for a future
       deep learning model on raw time series.
    """

    def __init__(
            self,
            velocity_threshold: float = 0.1,
            use_biosignal_labels: bool = False,
            use_biosignals: bool = True,
            smoothing_signal: bool = False,
            smoothing_window: int = 5
    ):
        """
        Args:
            velocity_threshold: Threshold for rest detection (mm/s).
            use_biosignal_labels: Whether to use biosignal-derived labels.
            use_biosignals: Extract features from ACC/EMG in combined mode.
            smoothing_signal: Whether to smooth velocities before feature extraction.
            smoothing_window: Window size for Savitzky-Golay smoothing (must be odd).
        """
        self.velocity_threshold = velocity_threshold
        self.use_biosignal_labels = use_biosignal_labels
        self.use_biosignals = use_biosignals
        self.smoothing_signal = smoothing_signal
        self.low_velocity_threshold = velocity_threshold
        self.low_energy_threshold = velocity_threshold ** 2
        self.mean_positive_velocity_threshold = velocity_threshold * 2.0
        self.mean_negative_velocity_threshold = velocity_threshold * 2.0
        self.y_direction_threshold = 0.2
        # Ensure odd window for savgol_filter
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.joint_to_idx = JOINT_TO_IDX

        # Teacher model for pseudo-label generation
        self.teacher: Optional[RandomForestClassifier] = None
        self.is_teacher_fitted: bool = False
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(PHASES)  # Pre-fit with known phases

    # -----------------------------------------------------------------
    # Joint index helpers
    # -----------------------------------------------------------------

    def get_joint_index(self, exercise_type: str) -> List[int]:
        """Get joint indices for a specific exercise (primary + assist joints)."""
        joints_config = get_exercise_joints(exercise_type)
        if joints_config is None:
            return list(range(len(JOINT_NAMES)))
        joint_names = [joints_config['primary_joint']] + joints_config.get('assist_joints', [])

        indices = []
        for name in joint_names:
            if name in self.joint_to_idx:
                indices.append(self.joint_to_idx[name])

        return indices if indices else list(range(len(JOINT_NAMES)))

    def get_primary_joint_indices(self, exercise_type: str) -> List[int]:
        """Get index of the primary joint for an exercise."""
        joints_config = get_exercise_joints(exercise_type)
        if joints_config is None:
            return [0]
        primary_name = joints_config['primary_joint']
        if primary_name in self.joint_to_idx:
            return [self.joint_to_idx[primary_name]]
        return [0]

    # -----------------------------------------------------------------
    # Frame / velocity / acceleration
    # -----------------------------------------------------------------

    def extract_frame_positions(
            self,
            joint_data: Dict,
            start_time: float,
            end_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract joint positions for one window from joint_data.

        Returns:
            positions:  (num_frames, num_joints, 3)
            timestamps: (num_frames,)
        """
        frames = joint_data.get('frames', [])
        if not frames:
            logger.warning(f"No frames found in joint data.")
            return np.array([]), np.array([])

        first_timestamp = frames[0].get('timestamp_usec', 0)

        positions = []
        timestamps = []

        for frame in frames:
            abs_timestamp = frame.get('timestamp_usec', 0)
            relative_timestamp = (abs_timestamp - first_timestamp) / 1e6
            if start_time <= relative_timestamp <= end_time:
                bodies = frame.get('bodies', [])
                if bodies:
                    body = bodies[0]
                    joint_positions = body.get('joint_positions', [])

                    if len(joint_positions) >= len(JOINT_NAMES):
                        frame_positions = np.zeros((len(JOINT_NAMES), 3))
                        for i in range(len(JOINT_NAMES)):
                            if i < len(joint_positions):
                                pos = joint_positions[i]
                                frame_positions[i, 0] = pos[0]
                                frame_positions[i, 1] = pos[1]
                                frame_positions[i, 2] = pos[2] if len(pos) > 2 else 0

                        positions.append(frame_positions)
                        timestamps.append(relative_timestamp)

        if positions:
            return np.array(positions), np.array(timestamps)

        logger.warning(
            f"No valid joint positions found in window ({start_time:.2f}s to {end_time:.2f}s)."
        )
        return np.array([]), np.array([])

    def compute_velocities(
            self,
            positions: np.ndarray,
            timestamps: np.ndarray,
            joint_indices: List[int]
    ) -> np.ndarray:
        """
        Compute velocity magnitudes for specified joints.

        Returns:
            velocities: (n_frames-1, n_selected_joints)
        """
        if len(positions) < 2:
            logger.warning("Not enough frames to compute velocities")
            return np.array([])

        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6

        selected_positions = positions[:, joint_indices, :]
        dpos = np.diff(selected_positions, axis=0)
        velocities = np.linalg.norm(dpos, axis=2) / dt[:, np.newaxis]

        return velocities

    def compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute accelerations from velocities."""
        if len(velocities) < 2:
            logger.warning("Not enough velocity data to compute accelerations")
            return np.array([])
        return np.diff(velocities, axis=0)

    def smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay smoothing (with median filter fallback)."""
        if len(signal) < self.smoothing_window:
            logger.warning("Signal too short for smoothing, skipping")
            return signal
        try:
            return savgol_filter(signal, self.smoothing_window, polyorder=2, axis=0)
        except Exception:
            return median_filter(signal, size=(self.smoothing_window, 1))

    # -----------------------------------------------------------------
    # Feature extraction — kinematics
    # -----------------------------------------------------------------

    def extract_features(
            self,
            joint_data: Dict,
            start_time: float,
            end_time: float,
            exercise_type: str = None
    ) -> dict:
        """
        Extract 8-dim kinematic feature vector from a time window.

        Features: mean_vel, std_vel, max_vel, mean_accel, std_accel,
                  max_accel, y_direction, energy.
        """
        positions, timestamps = self.extract_frame_positions(
            joint_data, start_time, end_time
        )

        if len(positions) < 3:
            logger.warning("Not enough joint data in window to extract features")
            return dict.fromkeys([
                'mean_velocity', 'std_velocity', 'max_velocity',
                'mean_accel', 'std_accel', 'max_accel',
                'y_direction', 'energy'
            ], 0.0)

        joint_indices = self.get_joint_index(exercise_type)
        velocities = self.compute_velocities(positions, timestamps, joint_indices)

        if len(velocities) < 2:
            logger.warning("Not enough velocity data to extract features")
            return dict.fromkeys([
                'mean_velocity', 'std_velocity', 'max_velocity',
                'mean_accel', 'std_accel', 'max_accel',
                'y_direction', 'energy'
            ], 0.0)

        if self.smoothing_signal:
            velocities = self.smooth_signal(velocities)

        accelerations = self.compute_accelerations(velocities)

        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)
        max_velocity = np.max(velocities)

        if len(accelerations) > 0:
            mean_accel = np.mean(accelerations)
            std_accel = np.std(accelerations)
            max_accel = np.max(np.abs(accelerations))
        else:
            mean_accel = std_accel = max_accel = 0.0

        # Direction indicator (positive = concentric / upward)
        primary_joints = self.get_primary_joint_indices(exercise_type)
        if primary_joints and len(velocities) > 0:
            dpos = np.diff(positions[:, primary_joints, 1], axis=0)
            y_direction = np.mean(np.sign(dpos))
        else:
            y_direction = 0.0

        energy = np.mean(velocities ** 2)
        logger.debug(f"Extracted features {mean_velocity:.2f}, {std_velocity:.2f}, {max_velocity:.2f}, {mean_accel:.2f}, {y_direction:.2f}, {energy:.2f}")
        return {
            'mean_velocity': mean_velocity,
            'std_velocity': std_velocity,
            'max_velocity': max_velocity,
            'mean_accel': mean_accel,
            'std_accel': std_accel,
            'max_accel': max_accel,
            'y_direction': y_direction,
            'energy': energy
            }

    # -----------------------------------------------------------------
    # Feature extraction — biosignals
    # -----------------------------------------------------------------

    def extract_biosignal_features(
            self,
            window_signals: Dict[str, np.ndarray],
            sampling_rates: Dict[str, float]
    ) -> np.ndarray:
        """
        Extract 15 features from biosignals (7 ACC + 8 EMG).

        Args:
            window_signals: {'acc': np.ndarray, 'emg': np.ndarray}
            sampling_rates: {'acc': float, 'emg': float}
        """
        features: List[float] = []

        from ml.v3.data.preprocessing import SignalPreprocessor
        sp = SignalPreprocessor()

        # Accelerometer features (7)
        if 'acc' in window_signals:
            acc_data = window_signals['acc']
            acc_sr = sampling_rates.get('acc', 50.0)
            features.extend(sp.preprocess_accelerometer(acc_data, acc_sr)[1].values())
        else:
            features.extend([0.0] * 7)
            logger.warning("ACC data missing, using zero features")

        # EMG features (8)
        if 'emg' in window_signals:
            emg_data = window_signals['emg']
            emg_sr = sampling_rates.get('emg', 2000.0)
            features.extend(sp.preprocess_emg(emg_data, emg_sr)[1].values())
        else:
            features.extend([0.0] * 8)
            logger.warning("EMG data missing, using zero features")

        return np.array(features)

    # -----------------------------------------------------------------
    # Phase rules
    # -----------------------------------------------------------------
    def phase_label(self,
            joint_data: Optional[Dict],
            window_signals: Optional[Dict[str, np.ndarray]],
            sampling_rates: Optional[Dict[str, float]],
            start_time: float,
            end_time: float,
            exercise_type: str = None) -> Dict[str, str]:
        """
        Return human-readable phase detection rules.
        Args:
            joint_data: Joint skeleton data dict.
            window_signals: Biosignal data for the window (optional).
            sampling_rates: Sampling rates per signal type (optional).
            start_time: Window start time (seconds).
            end_time: Window end time (seconds).
            exercise_type: Exercise type string.
        Returns:
            Predicted phase string.
        """
        joint_features = self.extract_features(joint_data, start_time, end_time, exercise_type)
        

        if (joint_features['max_velocity'] < self.low_velocity_threshold and 
            joint_features['energy'] < self.low_energy_threshold):
            return 'rest'
        elif (joint_features['max_velocity'] >= self.velocity_threshold and 
             joint_features['mean_velocity'] > self.mean_positive_velocity_threshold and 
             joint_features['y_direction'] > self.y_direction_threshold):
            return 'concentric'
        elif (joint_features['max_velocity'] >= self.velocity_threshold and
              joint_features['mean_velocity'] < self.mean_negative_velocity_threshold and 
              joint_features['y_direction'] < -self.y_direction_threshold):
            return 'eccentric'
        else:
            return 'rest'

    # -----------------------------------------------------------------
    # Combined features (kinematics + biosignals)
    # -----------------------------------------------------------------

    def extract_combined_features(
            self,
            joint_data: Dict,
            window_signals: Optional[Dict[str, np.ndarray]],
            sampling_rates: Optional[Dict[str, float]],
            start_time: float,
            end_time: float,
            exercise_type: str = None
    ) -> np.ndarray:
        """
        Extract combined kinematic + biosignal feature vector for a window.

        Returns:
            Feature vector: 8 kinematic + 15 biosignal = 23 features
        """
        # Kinematic features (8-dim)
        kin_features = self.extract_features(
            joint_data, start_time, end_time, exercise_type
        )
        kin_values = list(kin_features.values())

        # Biosignal features (15-dim: 7 ACC + 8 EMG)
        if self.use_biosignals and window_signals and sampling_rates:
            bio_features = self.extract_biosignal_features(
                window_signals, sampling_rates
            )
        else:
            bio_features = np.zeros(15)

        return np.concatenate([kin_values, bio_features])

    # -----------------------------------------------------------------
    # Teacher model (RandomForest)
    # -----------------------------------------------------------------

    def train_teacher(
            self,
            features: np.ndarray,
            labels: List[str]
    ) -> Dict:
        """
        Train RandomForest teacher model on engineered features + ground truth labels.

        Args:
            features: (N, D) feature matrix from extract_combined_features
            labels: N ground truth phase strings ('rest', 'concentric', 'eccentric')

        Returns:
            Training stats dict with accuracy, distribution, etc.
        """
        if len(features) < 10:
            return {'error': f'Too few samples: {len(features)}'}

        label_counts = Counter(labels)
        logger.info(f"Teacher training data: {len(features)} samples, distribution: {dict(label_counts)}")

        # Encode labels
        encoded_labels = self.label_encoder.transform(labels)

        # Scale features
        X_scaled = self.scaler.fit_transform(features)

        # Train RandomForest
        self.teacher = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=42
        )
        self.teacher.fit(X_scaled, encoded_labels)
        self.is_teacher_fitted = True

        # Evaluate on training data
        train_preds = self.teacher.predict(X_scaled)
        accuracy = np.mean(train_preds == encoded_labels)

        return {
            'accuracy': accuracy,
            'n_samples': len(features),
            'label_distribution': dict(label_counts),
            'feature_dim': features.shape[1],
        }

    def predict_phase(
            self,
            joint_data: Dict,
            start_time: float,
            end_time: float,
            exercise_type: str,
            window_signals: Optional[Dict[str, np.ndarray]] = None,
            sampling_rates: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        """
        Predict phase using the trained teacher model.

        Returns:
            Tuple of (phase_label, confidence)
        """
        if not self.is_teacher_fitted:
            raise RuntimeError("Teacher model not trained. Call train_teacher() first.")

        features = self.extract_combined_features(
            joint_data, window_signals, sampling_rates,
            start_time, end_time, exercise_type
        )

        X_scaled = self.scaler.transform(features.reshape(1, -1))

        proba = self.teacher.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(proba)
        confidence = proba[pred_idx]
        phase_label = self.label_encoder.inverse_transform([pred_idx])[0]

        return phase_label, confidence