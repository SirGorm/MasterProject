"""
PyTorch Dataset for Strength Training ML Pipeline v2.

Changes from v1:
- Uses sklearn.model_selection.train_test_split instead of manual shuffle+split
- Uses loguru for logging instead of print()
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle

from ml.v3.config import CONFIG, SIGNALS, TASK_SIGNALS
from ml.v3.data.preprocessing import DataPreprocessor, preprocess_dataset
from ml.v3.utils import get_logger

logger = get_logger('data')


def custom_collate_fn(batch):
    """Custom collate function that handles both tensors and metadata."""
    signals_list = [item[0] for item in batch]
    targets_list = [item[1] for item in batch]

    collated_signals = {}
    for key in signals_list[0].keys():
        collated_signals[key] = torch.stack([s[key] for s in signals_list])

    tensor_keys = ['exercise', 'phase', 'reps', 'fatigue']
    metadata_keys = ['session_id', 'exercise_name', 'window_idx', 'start_time', 'end_time', 'phase_name', 'skeleton_frame']

    collated_targets = {}
    for key in tensor_keys:
        if key in targets_list[0]:
            collated_targets[key] = torch.stack([t[key] for t in targets_list])
    for key in metadata_keys:
        if key in targets_list[0]:
            collated_targets[key] = [t[key] for t in targets_list]

    return collated_signals, collated_targets


class StrengthTrainingDataset(Dataset):
    """PyTorch Dataset for strength training analysis."""

    def __init__(
        self,
        windows: List[Dict],
        config=None,
        mode: str = 'train',
        scalers: Optional[Dict[str, StandardScaler]] = None,
        fit_scalers: bool = True
    ):
        self.config = config or CONFIG
        self.mode = mode
        self.windows = windows

        # Get active exercises (excluding any excluded ones)
        if hasattr(self.config, 'get_active_exercises'):
            active_exercises = self.config.get_active_exercises()
        else:
            excluded = getattr(self.config.data, 'excluded_exercises', [])
            active_exercises = [e for e in self.config.data.exercises if e not in excluded]
        self.exercise_to_idx = {ex: i for i, ex in enumerate(active_exercises)}
        self.idx_to_exercise = {i: ex for ex, i in self.exercise_to_idx.items()}

        self.phase_to_idx = {'rest': 0, 'eccentric': 1, 'concentric': 2}
        self.idx_to_phase = {0: 'rest', 1: 'eccentric', 2: 'concentric'}

        self.scalers = scalers or {}
        if fit_scalers and mode == 'train':
            self._fit_scalers()

    def _fit_scalers(self):
        """Fit StandardScalers for each signal and compute feature statistics."""
        signal_data = {
            name: [] for name, cfg in SIGNALS.items() if cfg.enabled
        }
        for window in self.windows:
            for signal_name, signal_array in window.get('signals', {}).items():
                if signal_name in signal_data:
                    signal_data[signal_name].append(signal_array.flatten())

        self.feature_statistics = {}
        for signal_name, data_list in signal_data.items():
            if data_list:
                all_data = np.concatenate(data_list).reshape(-1, 1)
                self.scalers[signal_name] = StandardScaler()
                self.scalers[signal_name].fit(all_data)

                flat_data = all_data.flatten()
                self.feature_statistics[signal_name] = {
                    'min': float(np.min(flat_data)),
                    'max': float(np.max(flat_data)),
                    'mean': float(np.mean(flat_data)),
                    'std': float(np.std(flat_data)),
                    'median': float(np.median(flat_data)),
                    'n_samples': len(flat_data),
                    'n_windows': len(data_list),
                }

        logger.info("Feature statistics per signal modality:")
        for signal_name, stats in self.feature_statistics.items():
            logger.info(f"  {signal_name}: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                       f"mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Return computed feature statistics for MLflow logging."""
        return getattr(self, 'feature_statistics', {})

    def get_class_imbalance_ratio(self) -> Dict[str, float]:
        """Compute class imbalance for rep counting (for pos_weight calculation)."""
        rep_counts = [w.get('rep_count', 0) for w in self.windows]
        n_positive = sum(1 for r in rep_counts if r > 0)
        n_negative = len(rep_counts) - n_positive

        return {
            'n_positive': n_positive,
            'n_negative': n_negative,
            'total': len(rep_counts),
            'positive_ratio': n_positive / len(rep_counts) if rep_counts else 0,
            'pos_weight': n_negative / n_positive if n_positive > 0 else 1.0
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        window = self.windows[idx]
        signals_dict = {}

        for signal_name, signal_config in SIGNALS.items():
            if not signal_config.enabled:
                continue
            signal_data = window.get('signals', {}).get(signal_name)
            if signal_data is not None:
                if signal_name in self.scalers:
                    signal_data = self.scalers[signal_name].transform(
                        signal_data.reshape(-1, 1)
                    ).flatten()
                signal_tensor = torch.FloatTensor(signal_data).unsqueeze(0)
                signals_dict[signal_name] = signal_tensor
            else:
                expected_samples = signal_config.samples_per_window(
                    self.config.data.time_window_sec
                )
                signals_dict[signal_name] = torch.zeros(1, expected_samples)

        exercise_name = window.get('exercise', 'unknown')
        if exercise_name not in self.exercise_to_idx:
            logger.warning(f"Unknown exercise '{exercise_name}' in window {idx}, defaulting to index 0")
        exercise_label = self.exercise_to_idx.get(exercise_name, 0)
        phase_str = window.get('phase', None)
        phase_label = self.phase_to_idx.get(phase_str, 0)
        rep_count = window.get('rep_count', 0)
        fatigue_score = window.get('fatigue_score', 0.0)

        targets_dict = {
            'exercise': torch.LongTensor([exercise_label]).squeeze(),
            'phase': torch.LongTensor([phase_label]).squeeze(),
            'reps': torch.FloatTensor([rep_count]).squeeze(),
            'fatigue': torch.FloatTensor([fatigue_score]).squeeze(),
            'session_id': window.get('session_id', 'unknown'),
            'exercise_name': window.get('exercise', 'unknown'),
            'window_idx': window.get('window_idx', idx),
            'start_time': window.get('start_time', 0.0),
            'end_time': window.get('end_time', 0.0),
            'phase_name': phase_str if phase_str else 'unknown',
            'skeleton_frame': window.get('skeleton_frame', None)
        }

        return signals_dict, targets_dict

    def get_num_classes(self) -> Dict[str, int]:
        return {
            'exercise': len(self.exercise_to_idx),
            'phase': len(self.phase_to_idx)
        }

    def get_signal_shapes(self) -> Dict[str, Tuple[int, ...]]:
        shapes = {}
        for signal_name, signal_config in SIGNALS.items():
            if not signal_config.enabled:
                continue
            seq_len = signal_config.samples_per_window(self.config.data.time_window_sec)
            shapes[signal_name] = (signal_config.channels, seq_len)
        return shapes

    def get_class_distribution(self) -> Dict[str, Dict]:
        from collections import Counter
        exercise_labels = [self.exercise_to_idx.get(w.get('exercise', 'unknown'), 0) for w in self.windows]
        phase_labels = [self.phase_to_idx.get(w.get('phase', 'unknown'), 0) for w in self.windows]
        return {
            'exercise': {self.idx_to_exercise[k]: v for k, v in Counter(exercise_labels).items()},
            'phase': {self.idx_to_phase[k]: v for k, v in Counter(phase_labels).items()}
        }

    def save_scalers(self, filepath: Path):
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)

    @staticmethod
    def load_scalers(filepath: Path) -> Dict[str, StandardScaler]:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_dataloaders(
    dataset_path: Path = None,
    config=None,
    batch_size: int = None,
    num_workers: int = 0,
    valid_sessions: List[Tuple[str, str, Path]] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, StandardScaler]]:
    """Create train and validation DataLoaders."""
    config = config or CONFIG
    dataset_path = Path(dataset_path or config.data.dataset_path)
    batch_size = batch_size or config.training.batch_size

    logger.info("Loading and preprocessing dataset...")

    all_windows = preprocess_dataset(dataset_path, config, valid_sessions=valid_sessions)
    if not all_windows:
        raise ValueError("No windows created from dataset. Check data files.")

    # v2: Use sklearn train_test_split instead of manual shuffle
    train_windows, val_windows = train_test_split(
        all_windows,
        test_size=config.data.train_val_split,
        random_state=config.data.random_seed
    )

    logger.info(f"Train samples: {len(train_windows)}")
    logger.info(f"Val samples:   {len(val_windows)}")

    train_dataset = StrengthTrainingDataset(
        windows=train_windows, config=config, mode='train', fit_scalers=True
    )
    val_dataset = StrengthTrainingDataset(
        windows=val_windows, config=config, mode='val',
        scalers=train_dataset.scalers, fit_scalers=False
    )

    logger.info(f"Class distribution (train): {train_dataset.get_class_distribution()}")

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
        collate_fn=custom_collate_fn
    )

    logger.info(f"Batch size: {batch_size} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    logger.info("Signal shapes:")
    for signal, shape in train_dataset.get_signal_shapes().items():
        logger.info(f"  {signal}: {shape}")

    return train_loader, val_loader, train_dataset.scalers


class SlidingWindowInference:
    """Sliding window inference for real-time predictions."""

    def __init__(self, model, scalers: Dict[str, StandardScaler], config=None, device: str = 'cpu'):
        self.model = model
        self.scalers = scalers
        self.config = config or CONFIG
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.current_rep_count = 0
        self.cumulative_fatigue = 0.0
        self.predictions_history = []

    def predict_window(self, signals: Dict[str, np.ndarray], joint_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        input_dict = {}
        for signal_name, signal_data in signals.items():
            if signal_name in self.scalers:
                signal_data = self.scalers[signal_name].transform(signal_data.reshape(-1, 1)).flatten()
            tensor = torch.FloatTensor(signal_data).unsqueeze(0).unsqueeze(0)
            input_dict[signal_name] = tensor.to(self.device)

        if joint_features is not None:
            if 'joints' in self.scalers:
                joint_features = self.scalers['joints'].transform(joint_features.reshape(1, -1)).flatten()
            input_dict['joints'] = torch.FloatTensor(joint_features).unsqueeze(0).to(self.device)
        else:
            joints_cfg = self.config.signals.get('joints')
            n_joints = joints_cfg.channels if joints_cfg else 96
            input_dict['joints'] = torch.zeros(1, n_joints).to(self.device)

        with torch.no_grad():
            exercise_logits, phase_logits, rep_pred, fatigue_delta = self.model(input_dict)
            exercise_pred = torch.argmax(exercise_logits, dim=1).item()
            phase_pred = torch.argmax(phase_logits, dim=1).item()
            rep_delta = rep_pred.item()
            fatigue_delta_val = fatigue_delta.item()

        self.current_rep_count = max(0, self.current_rep_count + rep_delta)
        self.cumulative_fatigue += fatigue_delta_val

        result = {
            'exercise': self.config.data.exercises[exercise_pred],
            'phase': {0: 'rest', 1: 'eccentric', 2: 'concentric'}.get(phase_pred, 'unknown'),
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
        self.current_rep_count = 0
        self.cumulative_fatigue = 0.0
        self.predictions_history = []

    def get_summary(self) -> Dict[str, Any]:
        if not self.predictions_history:
            return {}
        from collections import Counter
        exercises = [p['exercise'] for p in self.predictions_history]
        phases = [p['phase'] for p in self.predictions_history]
        return {
            'total_windows': len(self.predictions_history),
            'exercise_distribution': dict(Counter(exercises)),
            'phase_distribution': dict(Counter(phases)),
            'final_rep_count': self.current_rep_count,
            'final_fatigue': self.cumulative_fatigue
        }
