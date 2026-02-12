"""
Clustering-based Phase Detection for Strength Training.

Uses unsupervised learning to automatically detect exercise phases
(rest, concentric, eccentric) from joint kinematics.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import pickle

from config import CONFIG


@dataclass
class PhaseResult:
    """Result of phase detection for a window."""
    phase: str                    # 'rest', 'concentric', 'eccentric'
    confidence: float             # 0-1 confidence score
    cluster_id: int               # Raw cluster ID
    features: np.ndarray          # Features used for clustering


class ClusteringPhaseDetector:
    """
    Unsupervised phase detector using clustering on joint kinematics.

    Automatically learns to distinguish between exercise phases based on
    velocity and acceleration patterns in joint movements.
    """

    # Phase names
    PHASES = ['rest', 'concentric', 'eccentric']

    # Joint indices for different exercises (Azure Kinect body tracking)
    # Supports both lowercase and original case names
    EXERCISE_JOINTS = {
        # Original exercises
        'bicep_curl': {
            'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT'],
            'secondary': ['WRIST_RIGHT', 'WRIST_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT']
        },
        'squat': {
            'primary': ['KNEE_RIGHT', 'KNEE_LEFT', 'HIP_RIGHT', 'HIP_LEFT'],
            'secondary': ['ANKLE_RIGHT', 'ANKLE_LEFT', 'PELVIS']
        },
        'Squat': {  # Alias for case-insensitive matching
            'primary': ['KNEE_RIGHT', 'KNEE_LEFT', 'HIP_RIGHT', 'HIP_LEFT'],
            'secondary': ['ANKLE_RIGHT', 'ANKLE_LEFT', 'PELVIS']
        },
        'shoulder_press': {
            'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT'],
            'secondary': ['WRIST_RIGHT', 'WRIST_LEFT']
        },
        'lateral_raise': {
            'primary': ['SHOULDER_RIGHT', 'SHOULDER_LEFT', 'ELBOW_RIGHT', 'ELBOW_LEFT'],
            'secondary': ['WRIST_RIGHT', 'WRIST_LEFT']
        },
        'deadlift': {
            'primary': ['HIP_RIGHT', 'HIP_LEFT', 'KNEE_RIGHT', 'KNEE_LEFT'],
            'secondary': ['PELVIS', 'SPINE_CHEST']
        },
        # Benchpress - uses arm joints
        'benchpress': {
            'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT'],
            'secondary': ['WRIST_RIGHT', 'WRIST_LEFT', 'SPINE_CHEST']
        },
        'Benchpress': {  # Alias
            'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT'],
            'secondary': ['WRIST_RIGHT', 'WRIST_LEFT', 'SPINE_CHEST']
        },
        # Pullups - uses arm and spine joints
        'pullups': {
            'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT'],
            'secondary': ['WRIST_RIGHT', 'WRIST_LEFT', 'SPINE_NAVEL', 'SPINE_CHEST']
        },
        'Pullups': {  # Alias
            'primary': ['ELBOW_RIGHT', 'ELBOW_LEFT', 'SHOULDER_RIGHT', 'SHOULDER_LEFT'],
            'secondary': ['WRIST_RIGHT', 'WRIST_LEFT', 'SPINE_NAVEL', 'SPINE_CHEST']
        }
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

    def __init__(
        self,
        n_clusters: int = 3,
        use_dbscan: bool = False,
        smoothing_window: int = 5,
        velocity_threshold: float = 0.01
    ):
        """
        Initialize the clustering phase detector.

        Args:
            n_clusters: Number of clusters (phases) for K-Means
            use_dbscan: Use DBSCAN instead of K-Means
            smoothing_window: Window size for signal smoothing
            velocity_threshold: Threshold for rest detection
        """
        self.n_clusters = n_clusters
        self.use_dbscan = use_dbscan
        self.smoothing_window = smoothing_window
        self.velocity_threshold = velocity_threshold

        # Models
        self.scaler = StandardScaler()
        self.clusterer = None
        self.cluster_to_phase: Dict[int, str] = {}
        self.is_fitted = False
        self._nn_model = None  # NearestNeighbors for DBSCAN prediction

        # Joint name to index mapping
        self.joint_to_idx = {name: i for i, name in enumerate(self.JOINT_NAMES)}

    def _get_joint_indices(self, exercise_type: str) -> List[int]:
        """Get joint indices for a specific exercise."""
        if exercise_type not in self.EXERCISE_JOINTS:
            # Default: use all joints
            return list(range(len(self.JOINT_NAMES)))

        joints_config = self.EXERCISE_JOINTS[exercise_type]
        joint_names = joints_config['primary'] + joints_config['secondary']

        indices = []
        for name in joint_names:
            if name in self.joint_to_idx:
                indices.append(self.joint_to_idx[name])

        return indices if indices else list(range(len(self.JOINT_NAMES)))

    def _extract_frame_positions(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract joint positions for frames within time window.

        Returns:
            positions: (n_frames, n_joints, 3) array
            timestamps: (n_frames,) array
        """
        frames = joint_data.get('frames', [])
        if not frames:
            return np.array([]), np.array([])

        first_timestamp = frames[0].get('timestamp_usec', 0)

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

                    if len(joint_positions) >= len(self.JOINT_NAMES):
                        frame_positions = np.zeros((len(self.JOINT_NAMES), 3))
                        for i in range(len(self.JOINT_NAMES)):
                            if i < len(joint_positions):
                                pos = joint_positions[i]
                                frame_positions[i, 0] = pos[0]
                                frame_positions[i, 1] = pos[1]
                                frame_positions[i, 2] = pos[2] if len(pos) > 2 else 0

                        positions.append(frame_positions)
                        timestamps.append(rel_time)

        if positions:
            return np.array(positions), np.array(timestamps)
        return np.array([]), np.array([])

    def _compute_velocities(
        self,
        positions: np.ndarray,
        timestamps: np.ndarray,
        joint_indices: List[int]
    ) -> np.ndarray:
        """
        Compute velocities for selected joints.

        Args:
            positions: (n_frames, n_joints, 3) array
            timestamps: (n_frames,) array
            joint_indices: Indices of joints to use

        Returns:
            velocities: (n_frames-1, n_selected_joints) array of velocity magnitudes
        """
        if len(positions) < 2:
            return np.array([])

        # Time differences
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6  # Avoid division by zero

        # Position differences for selected joints
        selected_positions = positions[:, joint_indices, :]  # (n_frames, n_selected, 3)
        dpos = np.diff(selected_positions, axis=0)  # (n_frames-1, n_selected, 3)

        # Velocity magnitudes
        velocities = np.linalg.norm(dpos, axis=2) / dt[:, np.newaxis]  # (n_frames-1, n_selected)

        return velocities

    def _compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute accelerations from velocities."""
        if len(velocities) < 2:
            return np.array([])

        return np.diff(velocities, axis=0)

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce noise."""
        if len(signal) < self.smoothing_window:
            return signal

        # Use Savitzky-Golay filter for smoothing
        try:
            return savgol_filter(signal, self.smoothing_window, polyorder=2, axis=0)
        except:
            return median_filter(signal, size=(self.smoothing_window, 1))

    def extract_features(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str = None
    ) -> np.ndarray:
        """
        Extract features for phase clustering from a time window.

        Args:
            joint_data: Joint data dictionary with 'frames'
            start_time: Window start time (seconds)
            end_time: Window end time (seconds)
            exercise_type: Type of exercise for joint selection

        Returns:
            Feature vector for this window
        """
        # Get joint positions
        positions, timestamps = self._extract_frame_positions(
            joint_data, start_time, end_time
        )

        if len(positions) < 3:
            # Not enough frames, return zero features
            return np.zeros(8)

        # Get relevant joints for this exercise
        joint_indices = self._get_joint_indices(exercise_type)

        # Compute velocities
        velocities = self._compute_velocities(positions, timestamps, joint_indices)

        if len(velocities) < 2:
            return np.zeros(8)

        # Smooth velocities
        velocities = self._smooth_signal(velocities)

        # Compute accelerations
        accelerations = self._compute_accelerations(velocities)

        # Aggregate features across joints
        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)
        max_velocity = np.max(velocities)

        if len(accelerations) > 0:
            mean_accel = np.mean(accelerations)
            std_accel = np.std(accelerations)
            max_accel = np.max(np.abs(accelerations))
        else:
            mean_accel = std_accel = max_accel = 0

        # Direction indicator (positive = upward/concentric, negative = downward/eccentric)
        # Use Y-axis velocity of primary joints
        primary_joints = self._get_primary_joint_indices(exercise_type)
        if primary_joints and len(velocities) > 0:
            # Get Y-component of velocity (vertical movement)
            dpos = np.diff(positions[:, primary_joints, 1], axis=0)
            y_direction = np.mean(np.sign(dpos))
        else:
            y_direction = 0

        # Movement energy (sum of squared velocities)
        energy = np.mean(velocities ** 2)

        features = np.array([
            mean_velocity,
            std_velocity,
            max_velocity,
            mean_accel,
            std_accel,
            max_accel,
            y_direction,
            energy
        ])

        return features

    def _get_primary_joint_indices(self, exercise_type: str) -> List[int]:
        """Get indices of primary joints for an exercise."""
        if exercise_type not in self.EXERCISE_JOINTS:
            return []

        primary_names = self.EXERCISE_JOINTS[exercise_type]['primary']
        indices = []
        for name in primary_names:
            if name in self.joint_to_idx:
                indices.append(self.joint_to_idx[name])
        return indices

    def fit(
        self,
        all_features: np.ndarray,
        labels: np.ndarray = None
    ) -> np.ndarray:
        """
        Fit the clustering model on all extracted features.

        Args:
            all_features: (n_samples, n_features) array of features
            labels: Optional ground truth labels for semi-supervised fitting

        Returns:
            Cluster labels for all samples
        """
        if len(all_features) < self.n_clusters:
            print(f"Warning: Not enough samples ({len(all_features)}) for {self.n_clusters} clusters")
            return np.zeros(len(all_features), dtype=int)

        # Scale features
        features_scaled = self.scaler.fit_transform(all_features)

        # Create clusterer
        if self.use_dbscan:
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)
        else:
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10
            )

        # Fit and predict
        cluster_labels = self.clusterer.fit_predict(features_scaled)

        # For DBSCAN: build NearestNeighbors on core samples for prediction
        if self.use_dbscan:
            core_mask = cluster_labels >= 0  # Exclude noise points (label -1)
            if np.any(core_mask):
                self._nn_model = NearestNeighbors(n_neighbors=1)
                self._nn_model.fit(features_scaled[core_mask])
                self._nn_labels = cluster_labels[core_mask]

        # Auto-map clusters to phase names based on velocity characteristics
        self._auto_map_clusters(all_features, cluster_labels)

        self.is_fitted = True
        return cluster_labels

    def _auto_map_clusters(
        self,
        features: np.ndarray,
        labels: np.ndarray
    ):
        """
        Automatically map cluster IDs to phase names based on kinematics.

        Uses velocity and direction to determine:
        - rest: low velocity
        - concentric: high velocity, positive direction (upward)
        - eccentric: high velocity, negative direction (downward)
        """
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1) from DBSCAN

        cluster_stats = {}
        for c in unique_labels:
            mask = labels == c
            cluster_features = features[mask]

            # Feature indices: [mean_vel, std_vel, max_vel, mean_acc, std_acc, max_acc, y_dir, energy]
            cluster_stats[c] = {
                'mean_velocity': np.mean(cluster_features[:, 0]),
                'y_direction': np.mean(cluster_features[:, 6]),
                'energy': np.mean(cluster_features[:, 7]),
                'count': np.sum(mask)
            }

        # Sort by velocity (low to high)
        sorted_by_velocity = sorted(
            cluster_stats.items(),
            key=lambda x: x[1]['mean_velocity']
        )

        # Assign phases
        if len(sorted_by_velocity) >= 3:
            # Lowest velocity = rest
            rest_cluster = sorted_by_velocity[0][0]

            # For remaining clusters, use direction to distinguish
            remaining = [c for c, _ in sorted_by_velocity[1:]]

            # Higher y_direction = concentric (moving up)
            remaining_sorted = sorted(
                remaining,
                key=lambda c: cluster_stats[c]['y_direction'],
                reverse=True
            )

            concentric_cluster = remaining_sorted[0]
            eccentric_cluster = remaining_sorted[-1] if len(remaining_sorted) > 1 else remaining_sorted[0]

            self.cluster_to_phase = {
                rest_cluster: 'rest',
                concentric_cluster: 'concentric',
                eccentric_cluster: 'eccentric'
            }
        elif len(sorted_by_velocity) == 2:
            # Two clusters: rest and movement
            self.cluster_to_phase = {
                sorted_by_velocity[0][0]: 'rest',
                sorted_by_velocity[1][0]: 'concentric'  # Default to concentric
            }
        else:
            # Single cluster
            self.cluster_to_phase = {sorted_by_velocity[0][0]: 'rest'}

        # Handle any unmapped clusters
        for c in unique_labels:
            if c not in self.cluster_to_phase:
                self.cluster_to_phase[c] = 'unknown'

        print(f"Cluster mapping: {self.cluster_to_phase}")
        print(f"Cluster statistics:")
        for c, stats in cluster_stats.items():
            phase = self.cluster_to_phase.get(c, 'unknown')
            print(f"  Cluster {c} ({phase}): vel={stats['mean_velocity']:.4f}, "
                  f"dir={stats['y_direction']:.2f}, n={stats['count']}")

    def predict(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str = None
    ) -> PhaseResult:
        """
        Predict phase for a single window.

        Args:
            joint_data: Joint data dictionary
            start_time: Window start time
            end_time: Window end time
            exercise_type: Type of exercise

        Returns:
            PhaseResult with predicted phase and confidence
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Extract features
        features = self.extract_features(joint_data, start_time, end_time, exercise_type)

        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict cluster
        if hasattr(self.clusterer, 'predict'):
            # KMeans has predict()
            cluster_id = self.clusterer.predict(features_scaled)[0]
        elif self._nn_model is not None:
            # DBSCAN: use NearestNeighbors trained on core samples
            distances, indices = self._nn_model.kneighbors(features_scaled)
            cluster_id = int(self._nn_labels[indices[0][0]])
        else:
            cluster_id = 0

        # Get phase name
        phase = self.cluster_to_phase.get(cluster_id, 'unknown')

        # Compute confidence (distance to centroid)
        if hasattr(self.clusterer, 'cluster_centers_'):
            centroid = self.clusterer.cluster_centers_[cluster_id]
            distance = np.linalg.norm(features_scaled - centroid)
            confidence = np.exp(-distance)
        elif self._nn_model is not None:
            # Use NN distance for confidence
            distances, _ = self._nn_model.kneighbors(features_scaled)
            confidence = np.exp(-distances[0][0])
        else:
            confidence = 0.5

        return PhaseResult(
            phase=phase,
            confidence=float(confidence),
            cluster_id=int(cluster_id),
            features=features
        )

    def predict_phase(
        self,
        joint_data: Dict,
        start_time: float,
        end_time: float,
        exercise_type: str = None
    ) -> str:
        """
        Simple interface to predict phase as string.

        Returns:
            Phase name: 'rest', 'concentric', or 'eccentric'
        """
        if not self.is_fitted:
            return 'unknown'

        result = self.predict(joint_data, start_time, end_time, exercise_type)
        return result.phase

    def save(self, path: Path):
        """Save the fitted model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'scaler': self.scaler,
            'clusterer': self.clusterer,
            'cluster_to_phase': self.cluster_to_phase,
            'n_clusters': self.n_clusters,
            'is_fitted': self.is_fitted,
            'nn_model': self._nn_model,
            'nn_labels': getattr(self, '_nn_labels', None),
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Saved phase detector to {path}")

    def load(self, path: Path):
        """Load a fitted model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.scaler = model_data['scaler']
        self.clusterer = model_data['clusterer']
        self.cluster_to_phase = model_data['cluster_to_phase']
        self.n_clusters = model_data['n_clusters']
        self.is_fitted = model_data['is_fitted']
        self._nn_model = model_data.get('nn_model')
        self._nn_labels = model_data.get('nn_labels')

        print(f"Loaded phase detector from {path}")


def train_phase_detector(
    dataset_path: Path,
    exercises: List[str] = None,
    window_sec: float = 2.0,
    window_stride_sec: float = 0.5,
    save_path: Path = None
) -> ClusteringPhaseDetector:
    """
    Train a phase detector on a dataset.

    Args:
        dataset_path: Path to dataset directory
        exercises: List of exercises to include
        window_sec: Window size in seconds
        window_stride_sec: Window stride in seconds
        save_path: Path to save the trained model

    Returns:
        Trained ClusteringPhaseDetector
    """
    from .preprocessing import JointProcessor

    dataset_path = Path(dataset_path)
    exercises = exercises or ['bicep_curl', 'squat', 'shoulder_press']

    detector = ClusteringPhaseDetector(n_clusters=3)
    processor = JointProcessor()

    all_features = []

    print("Extracting features from dataset...")

    for exercise in exercises:
        exercise_dir = dataset_path / exercise
        if not exercise_dir.exists():
            continue

        for session_dir in exercise_dir.iterdir():
            if not session_dir.is_dir():
                continue

            joint_path = session_dir / 'joint_data.json'
            if not joint_path.exists():
                continue

            # Load joint data
            try:
                with open(joint_path, 'r') as f:
                    joint_data = json.load(f)
            except Exception as e:
                print(f"Error loading {joint_path}: {e}")
                continue

            if not joint_data:
                continue

            # Get total duration
            frames = joint_data.get('frames', [])
            if len(frames) < 2:
                continue

            first_ts = frames[0].get('timestamp_usec', 0)
            last_ts = frames[-1].get('timestamp_usec', 0)
            duration = (last_ts - first_ts) / 1e6

            # Extract features for each window
            n_windows = int((duration - window_sec) / window_stride_sec) + 1

            for i in range(n_windows):
                start_time = i * window_stride_sec
                end_time = start_time + window_sec

                features = detector.extract_features(
                    joint_data, start_time, end_time, exercise
                )

                if np.any(features):  # Skip zero features
                    all_features.append(features)

    if not all_features:
        print("No features extracted!")
        return detector

    all_features = np.array(all_features)
    print(f"Extracted {len(all_features)} feature vectors")

    # Fit the detector
    print("Fitting clustering model...")
    labels = detector.fit(all_features)

    # Print cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster distribution:")
    for c, count in zip(unique, counts):
        phase = detector.cluster_to_phase.get(c, 'unknown')
        print(f"  {phase} (cluster {c}): {count} samples ({100*count/len(labels):.1f}%)")

    # Save if path provided
    if save_path:
        detector.save(save_path)

    return detector


if __name__ == "__main__":
    # Example usage
    print("Training phase detector...")

    detector = train_phase_detector(
        dataset_path=CONFIG.data.dataset_path,
        exercises=CONFIG.data.exercises,
        save_path=CONFIG.output.output_dir / "phase_detector.pkl"
    )

    print("\nPhase detector ready!")
    print(f"Cluster mapping: {detector.cluster_to_phase}")
