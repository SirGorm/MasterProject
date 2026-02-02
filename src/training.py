"""
Training Pipeline for Multi-Task Strength Training Model
=========================================================
This module implements the complete training pipeline including:
- Data loading and augmentation
- Training loop with multi-task loss
- Validation and early stopping
- Model checkpointing
- Logging and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Local imports
from model import MultiTaskStrengthModel, ModelConfig, MultiTaskLoss, create_model
from preprocessing import (
    PreprocessingPipeline, ProcessedSegment, create_dataset_tensors
)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Data
    batch_size: int = 16
    val_split: float = 0.2
    test_split: float = 0.1
    num_workers: int = 0  # Windows compatibility

    # Training
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 15  # Early stopping patience

    # Scheduler
    scheduler_type: str = 'cosine'  # 'cosine' or 'plateau'

    # Augmentation
    use_augmentation: bool = True
    noise_std: float = 0.05
    time_shift_max: int = 20
    scale_range: Tuple[float, float] = (0.9, 1.1)

    # Paths
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'

    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class StrengthDataset(Dataset):
    """
    PyTorch Dataset for strength training data.

    Handles:
    - Loading preprocessed segments
    - On-the-fly augmentation
    - Multi-modal data packaging
    """

    def __init__(self, segments: List[ProcessedSegment],
                 augment: bool = False,
                 config: Optional[TrainingConfig] = None):
        """
        Args:
            segments: List of ProcessedSegment objects
            augment: Whether to apply data augmentation
            config: Training configuration
        """
        self.segments = segments
        self.augment = augment
        self.config = config or TrainingConfig()

        # Convert to tensors
        self.data = create_dataset_tensors(segments, normalize=True)

        # Store for reference
        self.feature_names = self.data.get('feature_names', [])
        self.seq_channels = self.data.get('seq_channels', [])

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        x_seq = self.data['X_seq'][idx].copy()
        x_feat = self.data['X_feat'][idx].copy()
        y_exercise = self.data['y_exercise'][idx]
        y_phase = self.data['y_phase'][idx].copy()
        y_rep_position = self.data['y_rep_position'][idx]

        # Apply augmentation
        if self.augment:
            x_seq, x_feat = self._augment(x_seq, x_feat)

        return {
            'x_seq': torch.from_numpy(x_seq).float(),
            'x_feat': torch.from_numpy(x_feat).float(),
            'y_exercise': torch.tensor(y_exercise).long(),
            'y_phase': torch.from_numpy(y_phase).float(),
            'y_rep_position': torch.tensor(y_rep_position).float(),
        }

    def _augment(self, x_seq: np.ndarray, x_feat: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Gaussian noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.config.noise_std, x_seq.shape)
            x_seq = x_seq + noise.astype(np.float32)

        # Time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-self.config.time_shift_max,
                                       self.config.time_shift_max + 1)
            x_seq = np.roll(x_seq, shift, axis=1)

        # Amplitude scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(*self.config.scale_range)
            x_seq = x_seq * scale

        # Channel dropout (set random channels to zero)
        if np.random.random() < 0.2:
            n_drop = np.random.randint(1, 3)
            drop_channels = np.random.choice(x_seq.shape[0], n_drop, replace=False)
            x_seq[drop_channels] = 0

        return x_seq, x_feat


class Trainer:
    """
    Training manager for multi-task model.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(self, model: MultiTaskStrengthModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 test_loader: Optional[DataLoader] = None):
        """
        Args:
            model: The multi-task model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            test_loader: Optional test data loader
        """
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # Loss function
        self.criterion = MultiTaskLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        if config.scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6
            )
        else:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=5
            )

        # History tracking
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = defaultdict(list)

        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch in pbar:
            # Move to device
            x_seq = batch['x_seq'].to(self.config.device)
            x_feat = batch['x_feat'].to(self.config.device)
            targets = {
                'exercise': batch['y_exercise'].to(self.config.device),
                'phase': batch['y_phase'].to(self.config.device),
                'rep_position': batch['y_rep_position'].to(self.config.device),
            }

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x_seq, x_feat)

            # Compute loss
            loss, loss_dict = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track losses
            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Average losses
        return {key: np.mean(values) for key, values in epoch_losses.items()}

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Run validation."""
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        epoch_losses = defaultdict(list)
        all_preds = {'exercise': [], 'phase': [], 'rep_position': [], 'fatigue': []}
        all_targets = {'exercise': [], 'phase': [], 'rep_position': []}

        for batch in loader:
            x_seq = batch['x_seq'].to(self.config.device)
            x_feat = batch['x_feat'].to(self.config.device)
            targets = {
                'exercise': batch['y_exercise'].to(self.config.device),
                'phase': batch['y_phase'].to(self.config.device),
                'rep_position': batch['y_rep_position'].to(self.config.device),
            }

            outputs = self.model(x_seq, x_feat)
            loss, loss_dict = self.criterion(outputs, targets)

            for key, value in loss_dict.items():
                epoch_losses[key].append(value)

            # Store predictions for metrics
            all_preds['exercise'].append(outputs['exercise_logits'].argmax(dim=1).cpu())
            all_preds['phase'].append((outputs['phase_logits'] > 0).float().cpu())
            all_preds['rep_position'].append(outputs['rep_position'].cpu())
            all_preds['fatigue'].append(outputs['fatigue_score'].cpu())

            all_targets['exercise'].append(batch['y_exercise'])
            all_targets['phase'].append(batch['y_phase'])
            all_targets['rep_position'].append(batch['y_rep_position'])

        # Compute metrics
        metrics = {key: np.mean(values) for key, values in epoch_losses.items()}

        # Exercise accuracy
        exercise_preds = torch.cat(all_preds['exercise'])
        exercise_targets = torch.cat(all_targets['exercise'])
        metrics['exercise_accuracy'] = (exercise_preds == exercise_targets).float().mean().item()

        # Phase accuracy (per-frame)
        phase_preds = torch.cat(all_preds['phase'])
        phase_targets = torch.cat(all_targets['phase'])
        metrics['phase_accuracy'] = ((phase_preds > 0.5) == (phase_targets > 0.5)).float().mean().item()

        # Rep position MAE
        rep_preds = torch.cat(all_preds['rep_position'])
        rep_targets = torch.cat(all_targets['rep_position'])
        metrics['rep_position_mae'] = torch.abs(rep_preds - rep_targets).mean().item()

        return metrics

    def train(self, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Full training loop.

        Args:
            verbose: Whether to print progress

        Returns:
            Training history
        """
        if verbose:
            print(f"Training on {self.config.device}")
            print(f"Training samples: {len(self.train_loader.dataset)}")
            print(f"Validation samples: {len(self.val_loader.dataset)}")
            print("=" * 50)

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_start = time.time()

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['total'])
            else:
                self.scheduler.step()

            # Record history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            # Check for improvement
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.config.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Logging
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch + 1}/{self.config.epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['total']:.4f}")
                print(f"  Val Loss: {val_metrics['total']:.4f} | "
                      f"Acc: {val_metrics['exercise_accuracy']:.3f} | "
                      f"Phase: {val_metrics['phase_accuracy']:.3f} | "
                      f"Rep MAE: {val_metrics['rep_position_mae']:.3f}")

            # Periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch + 1}.pt')

        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {total_time / 60:.1f} minutes")
            print(f"Best validation loss: {self.best_val_loss:.4f}")

        # Save final model
        self.save_checkpoint('final_model.pt')

        return dict(self.history)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': dict(self.history),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = Path(self.config.checkpoint_dir) / filename
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = defaultdict(list, checkpoint.get('history', {}))
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        # Total loss
        ax = axes[0, 0]
        ax.plot(self.history['train_total'], label='Train')
        ax.plot(self.history['val_total'], label='Val')
        ax.set_title('Total Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Exercise classification loss
        ax = axes[0, 1]
        ax.plot(self.history['train_exercise'], label='Train')
        ax.plot(self.history['val_exercise'], label='Val')
        ax.set_title('Exercise Classification Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase detection loss
        ax = axes[0, 2]
        ax.plot(self.history['train_phase'], label='Train')
        ax.plot(self.history['val_phase'], label='Val')
        ax.set_title('Phase Detection Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Exercise accuracy
        ax = axes[1, 0]
        ax.plot(self.history['val_exercise_accuracy'], label='Exercise Acc', color='green')
        ax.plot(self.history['val_phase_accuracy'], label='Phase Acc', color='blue')
        ax.set_title('Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rep position MAE
        ax = axes[1, 1]
        ax.plot(self.history['val_rep_position_mae'], color='orange')
        ax.set_title('Rep Position MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 2]
        ax.plot(self.history['lr'], color='red')
        ax.set_title('Learning Rate')
        ax.set_xlabel('Epoch')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class CrossValidator:
    """
    K-Fold cross-validation for model evaluation.

    Uses leave-one-session-out or stratified k-fold.
    """

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig,
                 n_folds: int = 5):
        self.model_config = model_config
        self.training_config = training_config
        self.n_folds = n_folds
        self.results = []

    def run_cv(self, segments: List[ProcessedSegment],
               by_session: bool = True) -> Dict[str, float]:
        """
        Run cross-validation.

        Args:
            segments: All data segments
            by_session: If True, use leave-one-session-out CV

        Returns:
            Aggregated metrics
        """
        if by_session:
            return self._session_cv(segments)
        else:
            return self._kfold_cv(segments)

    def _session_cv(self, segments: List[ProcessedSegment]) -> Dict[str, float]:
        """Leave-one-session-out cross-validation."""
        # Group by session
        sessions = {}
        for seg in segments:
            key = f"{seg.exercise}_{seg.session_id}"
            if key not in sessions:
                sessions[key] = []
            sessions[key].append(seg)

        session_keys = list(sessions.keys())
        all_metrics = []

        print(f"Running leave-one-session-out CV with {len(session_keys)} sessions")

        for i, test_key in enumerate(session_keys):
            print(f"\nFold {i + 1}/{len(session_keys)}: Testing on {test_key}")

            # Split data
            test_segments = sessions[test_key]
            train_segments = [seg for key, segs in sessions.items()
                             if key != test_key for seg in segs]

            # Create datasets
            train_dataset = StrengthDataset(
                train_segments, augment=True, config=self.training_config
            )
            test_dataset = StrengthDataset(
                test_segments, augment=False, config=self.training_config
            )

            # Split train into train/val
            val_size = int(len(train_dataset) * 0.15)
            train_size = len(train_dataset) - val_size
            train_subset, val_subset = random_split(
                train_dataset, [train_size, val_size]
            )

            # Create loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.training_config.batch_size,
                shuffle=True,
                num_workers=self.training_config.num_workers
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=self.training_config.num_workers
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.training_config.batch_size,
                shuffle=False,
                num_workers=self.training_config.num_workers
            )

            # Train model
            model = create_model(self.model_config)
            trainer = Trainer(
                model, train_loader, val_loader,
                self.training_config, test_loader
            )

            # Reduced epochs for CV
            cv_config = TrainingConfig(**vars(self.training_config))
            cv_config.epochs = min(50, self.training_config.epochs)
            trainer.config = cv_config

            trainer.train(verbose=False)

            # Evaluate on test set
            trainer.load_checkpoint('best_model.pt')
            metrics = trainer.validate(test_loader)
            metrics['fold'] = i + 1
            metrics['test_session'] = test_key
            all_metrics.append(metrics)

            print(f"  Exercise Acc: {metrics['exercise_accuracy']:.3f}")
            print(f"  Phase Acc: {metrics['phase_accuracy']:.3f}")

        # Aggregate results
        aggregated = {}
        metric_keys = [k for k in all_metrics[0].keys()
                       if k not in ['fold', 'test_session']]

        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

        self.results = all_metrics
        return aggregated

    def _kfold_cv(self, segments: List[ProcessedSegment]) -> Dict[str, float]:
        """Standard k-fold cross-validation."""
        from sklearn.model_selection import StratifiedKFold

        # Get labels for stratification
        labels = [seg.labels['exercise'] for seg in segments]
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        all_metrics = []
        indices = np.arange(len(segments))

        for fold, (train_idx, test_idx) in enumerate(skf.split(indices, labels)):
            print(f"\nFold {fold + 1}/{self.n_folds}")

            train_segments = [segments[i] for i in train_idx]
            test_segments = [segments[i] for i in test_idx]

            # Create datasets and train (similar to session CV)
            # ... (abbreviated for space)

        return {}  # Return aggregated metrics


def prepare_data_loaders(segments: List[ProcessedSegment],
                         config: TrainingConfig
                         ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare train/val/test data loaders.

    Args:
        segments: All processed segments
        config: Training configuration

    Returns:
        train_loader, val_loader, test_loader
    """
    # Shuffle segments
    np.random.seed(42)
    indices = np.random.permutation(len(segments))
    segments = [segments[i] for i in indices]

    # Split sizes
    n_total = len(segments)
    n_test = int(n_total * config.test_split)
    n_val = int(n_total * config.val_split)
    n_train = n_total - n_test - n_val

    # Split data
    train_segments = segments[:n_train]
    val_segments = segments[n_train:n_train + n_val]
    test_segments = segments[n_train + n_val:]

    print(f"Data split: Train={n_train}, Val={n_val}, Test={n_test}")

    # Create datasets
    train_dataset = StrengthDataset(train_segments, augment=config.use_augmentation,
                                     config=config)
    val_dataset = StrengthDataset(val_segments, augment=False, config=config)
    test_dataset = StrengthDataset(test_segments, augment=False, config=config)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader


def train_model(dataset_path: str,
                model_config: Optional[ModelConfig] = None,
                training_config: Optional[TrainingConfig] = None,
                verbose: bool = True) -> Tuple[MultiTaskStrengthModel, Dict]:
    """
    Complete training pipeline.

    Args:
        dataset_path: Path to dataset directory
        model_config: Model configuration
        training_config: Training configuration
        verbose: Print progress

    Returns:
        Trained model and training history
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_exploration import DatasetExplorer

    # Default configs
    if model_config is None:
        model_config = ModelConfig()
    if training_config is None:
        training_config = TrainingConfig()

    # Load and preprocess data
    if verbose:
        print("Loading dataset...")
    explorer = DatasetExplorer(dataset_path)
    explorer.discover_exercises()
    explorer.load_all_sessions()

    if verbose:
        print("Preprocessing data...")
    pipeline = PreprocessingPipeline(target_length=model_config.seq_length)

    all_segments = []
    for session in explorer.sessions:
        segments = pipeline.process_session(session)
        all_segments.extend(segments)

    if verbose:
        print(f"Total segments: {len(all_segments)}")

    # Update model config based on data
    if all_segments:
        model_config.n_features = len(all_segments[0].features)

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        all_segments, training_config
    )

    # Create model
    model = create_model(model_config)
    if verbose:
        from model import count_parameters
        print(f"Model parameters: {count_parameters(model):,}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, training_config, test_loader)
    history = trainer.train(verbose=verbose)

    # Final evaluation
    if verbose:
        print("\nFinal evaluation on test set:")
        trainer.load_checkpoint('best_model.pt')
        test_metrics = trainer.validate(test_loader)
        print(f"  Exercise Accuracy: {test_metrics['exercise_accuracy']:.3f}")
        print(f"  Phase Accuracy: {test_metrics['phase_accuracy']:.3f}")
        print(f"  Rep Position MAE: {test_metrics['rep_position_mae']:.3f}")

    # Plot history
    trainer.plot_history(save_path=str(Path(training_config.log_dir) / 'training_history.png'))

    return model, history


if __name__ == "__main__":
    # Run training
    dataset_path = r"C:\MasterProject\VS_Camera\Test_modify\claude\dataset"

    # Configurations
    model_config = ModelConfig(
        n_channels=10,
        seq_length=256,
        n_features=80,  # Will be updated based on data
        backbone_type='hybrid',
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        dropout=0.2,
        n_exercises=3,  # Squat, Benchpress, Pullups (no Deadlift in dataset)
        fatigue_latent_dim=32
    )

    training_config = TrainingConfig(
        batch_size=8,  # Small dataset
        val_split=0.2,
        test_split=0.1,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=15,
        use_augmentation=True,
        checkpoint_dir='checkpoints',
        log_dir='logs'
    )

    # Train
    model, history = train_model(
        dataset_path,
        model_config,
        training_config,
        verbose=True
    )

    print("\nTraining complete!")
