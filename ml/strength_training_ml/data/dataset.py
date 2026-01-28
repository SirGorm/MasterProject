"""
PyTorch Dataset for Strength Training ML Pipeline.

Supports:
- Multi-modal signal input
- Multi-task labels (exercise, phase, reps, fatigue)
- Sliding windows for real-time inference simulation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG, SIGNALS, TASK_SIGNALS
from data.preprocessing import DataPreprocessor, preprocess_dataset


class StrengthTrainingDataset(Dataset):
    """
    PyTorch Dataset for strength training analysis.

    Provides multi-modal signals and multi-task labels.
    """

    def __init__(
        self,
        windows: List[Dict],
        config=None,
        mode: str = 'train',
        scalers: Optional[Dict[str, StandardScaler]] = None,
        fit_scalers: bool = True
    ):
        """
        Initialize dataset.

        Args:
            windows: List of preprocessed window dictionaries
            config: Configuration object
            mode: 'train' or 'val'
            scalers: Pre-fitted scalers for normalization
            fit_scalers: Whether to fit scalers (only for training)
        """
        self.config = config or CONFIG
        self.mode = mode
        self.windows = windows

        # Exercise label mapping
        self.exercise_to_idx = {
            ex: i for i, ex in enumerate(self.config.data.exercises)
        }
        self.idx_to_exercise = {
            i: ex for ex, i in self.exercise_to_idx.items()
        }

        # Phase label mapping
        self.phase_to_idx = {
            'eccentric': 0,
            'concentric': 1,
            'unknown': 0  # Default to eccentric if unknown
        }
        self.idx_to_phase = {
            0: 'eccentric',
            1: 'concentric'
        }

        # Scalers for normalization
        self.scalers = scalers or {}
        if fit_scalers and mode == 'train':
            self._fit_scalers()

    def _fit_scalers(self):
        """Fit scalers on training data (only for enabled signals)."""
        # Only fit scalers for enabled signals (not joints - that's for ground truth only)
        signal_data = {
            name: [] for name, cfg in SIGNALS.items()
            if cfg.enabled
        }

        for window in self.windows:
            for signal_name, signal_array in window.get('signals', {}).items():
                if signal_name in signal_data:
                    signal_data[signal_name].append(signal_array.flatten())

        # Fit scaler for each signal type
        for signal_name, data_list in signal_data.items():
            if data_list:
                all_data = np.concatenate(data_list).reshape(-1, 1)
                self.scalers[signal_name] = StandardScaler()
                self.scalers[signal_name].fit(all_data)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get a sample.

        Returns:
            Tuple of (signals_dict, targets_dict)
        """
        window = self.windows[idx]

        # Prepare signals (only enabled signals, excluding joints which is for ground truth only)
        signals_dict = {}

        for signal_name, signal_config in SIGNALS.items():
            # Skip disabled signals (joints is disabled - used for ground truth labels only)
            if not signal_config.enabled:
                continue

            # Time-series signals
            signal_data = window.get('signals', {}).get(signal_name)
            if signal_data is not None:
                # Normalize
                if signal_name in self.scalers:
                    signal_data = self.scalers[signal_name].transform(
                        signal_data.reshape(-1, 1)
                    ).flatten()

                # Convert to tensor [channels, time]
                signal_tensor = torch.FloatTensor(signal_data).unsqueeze(0)
                signals_dict[signal_name] = signal_tensor
            else:
                # Create zero tensor of expected size
                expected_samples = signal_config.samples_per_window(
                    self.config.data.time_window_sec
                )
                signals_dict[signal_name] = torch.zeros(1, expected_samples)

        # Prepare targets (phase label comes from joint_data ground truth)
        exercise_label = self.exercise_to_idx.get(window.get('exercise', 'Squat'), 0)
        phase_label = self.phase_to_idx.get(window.get('phase', 'unknown'), 0)
        rep_count = window.get('rep_count', 0)
        fatigue_score = window.get('fatigue_score', 0.0)

        targets_dict = {
            'exercise': torch.LongTensor([exercise_label]).squeeze(),
            'phase': torch.LongTensor([phase_label]).squeeze(),
            'reps': torch.FloatTensor([rep_count]).squeeze(),
            'fatigue': torch.FloatTensor([fatigue_score]).squeeze()
        }

        return signals_dict, targets_dict

    def get_num_classes(self) -> Dict[str, int]:
        """Get number of classes for classification tasks."""
        return {
            'exercise': len(self.config.data.exercises),
            'phase': 2  # Eccentric, Concentric
        }

    def get_signal_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get shape information for each enabled signal type."""
        shapes = {}
        for signal_name, signal_config in SIGNALS.items():
            # Only include enabled signals (not joints - that's for ground truth only)
            if not signal_config.enabled:
                continue
            seq_len = signal_config.samples_per_window(
                self.config.data.time_window_sec
            )
            shapes[signal_name] = (signal_config.channels, seq_len)
        return shapes

    def get_class_distribution(self) -> Dict[str, Dict]:
        """Get distribution of labels."""
        from collections import Counter

        exercise_labels = [self.exercise_to_idx.get(w.get('exercise', 'Squat'), 0)
                         for w in self.windows]
        phase_labels = [self.phase_to_idx.get(w.get('phase', 'unknown'), 0)
                       for w in self.windows]

        exercise_dist = Counter(exercise_labels)
        phase_dist = Counter(phase_labels)

        return {
            'exercise': {self.idx_to_exercise[k]: v for k, v in exercise_dist.items()},
            'phase': {self.idx_to_phase[k]: v for k, v in phase_dist.items()}
        }

    def save_scalers(self, filepath: Path):
        """Save fitted scalers to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)

    @staticmethod
    def load_scalers(filepath: Path) -> Dict[str, StandardScaler]:
        """Load scalers from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_dataloaders(
    dataset_path: Path = None,
    config=None,
    batch_size: int = None,
    num_workers: int = 0,
    valid_sessions: List[Tuple[str, str, Path]] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, StandardScaler]]:
    """
    Create train and validation DataLoaders.

    Args:
        dataset_path: Path to dataset
        config: Configuration object
        batch_size: Batch size (default from config)
        num_workers: Number of worker processes
        valid_sessions: List of (exercise, session_id, path) tuples from validation.
                       If None, validates and gets valid sessions automatically.

    Returns:
        Tuple of (train_loader, val_loader, scalers)
    """
    config = config or CONFIG
    dataset_path = Path(dataset_path or config.data.dataset_path)
    batch_size = batch_size or config.training.batch_size

    print("\n" + "="*70)
    print("LOADING AND PREPROCESSING DATASET")
    print("="*70)

    # Preprocess all data (using only valid sessions if provided)
    all_windows = preprocess_dataset(dataset_path, config, valid_sessions=valid_sessions)

    if not all_windows:
        raise ValueError("No windows created from dataset. Check data files.")

    # Split into train/val
    np.random.seed(config.data.random_seed)
    indices = np.random.permutation(len(all_windows))

    n_val = int(len(all_windows) * config.data.train_val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_windows = [all_windows[i] for i in train_indices]
    val_windows = [all_windows[i] for i in val_indices]

    print(f"\nTrain samples: {len(train_windows)}")
    print(f"Val samples:   {len(val_windows)}")

    # Create datasets
    train_dataset = StrengthTrainingDataset(
        windows=train_windows,
        config=config,
        mode='train',
        fit_scalers=True
    )

    val_dataset = StrengthTrainingDataset(
        windows=val_windows,
        config=config,
        mode='val',
        scalers=train_dataset.scalers,
        fit_scalers=False
    )

    # Print class distribution
    print("\nClass distribution (train):")
    train_dist = train_dataset.get_class_distribution()
    print(f"  Exercises: {train_dist['exercise']}")
    print(f"  Phases:    {train_dist['phase']}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nBatch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Print signal shapes
    print("\nSignal shapes:")
    for signal, shape in train_dataset.get_signal_shapes().items():
        print(f"  {signal}: {shape}")

    print("="*70)

    return train_loader, val_loader, train_dataset.scalers


class SlidingWindowInference:
    """
    Sliding window inference for real-time predictions.

    Simulates real-time processing by feeding windows sequentially.
    """

    def __init__(
        self,
        model,
        scalers: Dict[str, StandardScaler],
        config=None,
        device: str = 'cpu'
    ):
        """
        Initialize sliding window inference.

        Args:
            model: Trained model
            scalers: Fitted scalers for normalization
            config: Configuration object
            device: Device for inference ('cpu' or 'cuda')
        """
        self.model = model
        self.scalers = scalers
        self.config = config or CONFIG
        self.device = device
        self.model.to(device)
        self.model.eval()

        # State tracking
        self.current_rep_count = 0
        self.cumulative_fatigue = 0.0
        self.predictions_history = []

    def predict_window(
        self,
        signals: Dict[str, np.ndarray],
        joint_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Make prediction for a single window.

        Args:
            signals: Dictionary of signal arrays
            joint_features: Optional joint feature vector

        Returns:
            Dictionary with predictions
        """
        # Prepare input tensors
        input_dict = {}

        for signal_name, signal_data in signals.items():
            # Normalize
            if signal_name in self.scalers:
                signal_data = self.scalers[signal_name].transform(
                    signal_data.reshape(-1, 1)
                ).flatten()

            # Convert to tensor [batch=1, channels=1, time]
            tensor = torch.FloatTensor(signal_data).unsqueeze(0).unsqueeze(0)
            input_dict[signal_name] = tensor.to(self.device)

        # Handle joint features
        if joint_features is not None:
            if 'joints' in self.scalers:
                joint_features = self.scalers['joints'].transform(
                    joint_features.reshape(1, -1)
                ).flatten()
            input_dict['joints'] = torch.FloatTensor(joint_features).unsqueeze(0).to(self.device)
        else:
            n_joints = self.config.signals['joints'].channels
            input_dict['joints'] = torch.zeros(1, n_joints).to(self.device)

        # Forward pass
        with torch.no_grad():
            exercise_logits, phase_logits, rep_pred, fatigue_delta = self.model(input_dict)

            # Get predictions
            exercise_pred = torch.argmax(exercise_logits, dim=1).item()
            phase_pred = torch.argmax(phase_logits, dim=1).item()
            rep_delta = rep_pred.item()
            fatigue_delta_val = fatigue_delta.item()

        # Update state
        self.current_rep_count = max(0, self.current_rep_count + rep_delta)
        self.cumulative_fatigue += fatigue_delta_val

        # Create result
        result = {
            'exercise': self.config.data.exercises[exercise_pred],
            'phase': 'eccentric' if phase_pred == 0 else 'concentric',
            'rep_count': int(round(self.current_rep_count)),
            'fatigue_level': min(1.0, max(0.0, self.cumulative_fatigue)),
            'raw_predictions': {
                'exercise_logits': exercise_logits.cpu().numpy(),
                'phase_logits': phase_logits.cpu().numpy(),
                'rep_delta': rep_delta,
                'fatigue_delta': fatigue_delta_val
            }
        }

        self.predictions_history.append(result)

        return result

    def reset_state(self):
        """Reset inference state for new exercise session."""
        self.current_rep_count = 0
        self.cumulative_fatigue = 0.0
        self.predictions_history = []

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of predictions."""
        if not self.predictions_history:
            return {}

        exercises = [p['exercise'] for p in self.predictions_history]
        phases = [p['phase'] for p in self.predictions_history]

        from collections import Counter

        return {
            'total_windows': len(self.predictions_history),
            'exercise_distribution': dict(Counter(exercises)),
            'phase_distribution': dict(Counter(phases)),
            'final_rep_count': self.current_rep_count,
            'final_fatigue': self.cumulative_fatigue
        }
