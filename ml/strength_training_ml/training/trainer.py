"""
Training Module for Strength Training ML Pipeline.

Features:
- Multi-task training with adaptive loss weighting
- Early stopping
- GPU support
- Gradient clipping for stability
- Comprehensive logging
- Prediction tracking for debugging and analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import json
import time
import random

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from models import StrengthTrainingModel, MultiTaskLoss, create_model
from utils import get_logger, TrainingLogger
from evaluation import PredictionTracker, visualize_prediction, visualize_prediction_with_skeleton


class Trainer:
    """
    Trainer class for multi-task strength training model.
    """

    def __init__(
        self,
        model: StrengthTrainingModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config=None,
        device: str = None
    ):
        """
        Initialize trainer.

        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config or CONFIG
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Setup device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)

        # Setup logging
        self.logger = get_logger('train')
        self.training_logger = TrainingLogger(
            log_file=self.config.output.logs_dir / self.config.output.training_log_filename
        )

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        self.scheduler = self._create_scheduler()

        # Setup loss function
        self.criterion = MultiTaskLoss(
            init_weights=self.config.training.loss_weights,
            use_uncertainty_weighting=True
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_exercise_acc': [],
            'val_exercise_acc': [],
            'train_phase_acc': [],
            'val_phase_acc': [],
            'train_rep_mae': [],
            'val_rep_mae': [],
            'train_fatigue_mae': [],
            'val_fatigue_mae': [],
            'learning_rates': []
        }

        # Setup prediction tracker
        self.tracker = None
        if self.config.tracking.enabled:
            tracking_dir = self.config.output.output_dir / self.config.tracking.tracking_subdir
            self.tracker = PredictionTracker(
                output_dir=tracking_dir,
                max_records=self.config.tracking.max_records,
                store_signals=self.config.tracking.store_signals,
                store_logits=self.config.tracking.store_logits,
                config=self.config
            )
            self.logger.info(f"Prediction tracking enabled: {tracking_dir}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        cfg = self.config.training

        if cfg.optimizer.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=cfg.learning_rate,
                momentum=0.9,
                weight_decay=cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler based on config."""
        cfg = self.config.training

        if cfg.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=cfg.scheduler_factor,
                patience=cfg.scheduler_patience,
                min_lr=cfg.min_lr
            )
        elif cfg.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.n_epochs,
                eta_min=cfg.min_lr
            )
        elif cfg.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=cfg.scheduler_factor
            )
        else:
            return None

    def _check_for_nan(self, tensor: torch.Tensor, name: str) -> bool:
        """Check if tensor contains NaN or Inf values."""
        if torch.isnan(tensor).any():
            self.logger.warning(f"NaN detected in {name}")
            return True
        if torch.isinf(tensor).any():
            self.logger.warning(f"Inf detected in {name}")
            return True
        return False

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_samples = 0
        correct_exercise = 0
        correct_phase = 0
        total_rep_mae = 0.0
        total_fatigue_mae = 0.0

        nan_batches = 0

        for batch_idx, (signals, targets) in enumerate(self.train_loader):
            # Move to device
            signals = {k: v.to(self.device) for k, v in signals.items()}
            exercise_labels = targets['exercise'].to(self.device)
            phase_labels = targets['phase'].to(self.device)
            rep_labels = targets['reps'].to(self.device)
            fatigue_labels = targets['fatigue'].to(self.device)

            # Check for NaN in inputs
            skip_batch = False
            for name, data in signals.items():
                if self._check_for_nan(data, f"input {name}"):
                    nan_batches += 1
                    skip_batch = True
                    break

            if skip_batch:
                continue

            # Forward pass
            self.optimizer.zero_grad()

            try:
                exercise_logits, phase_logits, rep_pred, fatigue_pred = self.model(signals)

                # Check for NaN in outputs
                if (self._check_for_nan(exercise_logits, "exercise_logits") or
                    self._check_for_nan(phase_logits, "phase_logits") or
                    self._check_for_nan(rep_pred, "rep_pred") or
                    self._check_for_nan(fatigue_pred, "fatigue_pred")):
                    nan_batches += 1
                    continue

                # Compute loss
                predictions = (exercise_logits, phase_logits, rep_pred, fatigue_pred)
                labels = (exercise_labels, phase_labels, rep_labels, fatigue_labels)
                loss, loss_dict = self.criterion(predictions, labels)

                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_batches += 1
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.training.gradient_clip_norm
                )

                # Optimizer step
                self.optimizer.step()

                # Accumulate metrics
                batch_size = exercise_labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Exercise accuracy
                _, pred_exercise = exercise_logits.max(1)
                correct_exercise += pred_exercise.eq(exercise_labels).sum().item()

                # Phase accuracy
                _, pred_phase = phase_logits.max(1)
                correct_phase += pred_phase.eq(phase_labels).sum().item()

                # Rep MAE
                total_rep_mae += torch.abs(rep_pred.squeeze(-1) - rep_labels.float()).sum().item()

                # Fatigue MAE
                total_fatigue_mae += torch.abs(fatigue_pred.squeeze(-1) - fatigue_labels.float()).sum().item()

            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                nan_batches += 1
                continue

        if nan_batches > 0:
            self.logger.warning(f"Skipped {nan_batches} batches due to NaN/Inf")

        if total_samples == 0:
            self.logger.error("No valid samples in epoch!")
            return {'loss': 0, 'exercise_acc': 0, 'phase_acc': 0, 'rep_mae': 0, 'fatigue_mae': 0}

        # Calculate metrics
        metrics = {
            'loss': total_loss / total_samples,
            'exercise_acc': 100.0 * correct_exercise / total_samples,
            'phase_acc': 100.0 * correct_phase / total_samples,
            'rep_mae': total_rep_mae / total_samples,
            'fatigue_mae': total_fatigue_mae / total_samples
        }

        return metrics

    @torch.no_grad()
    def validate(self, track_predictions: bool = True) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            track_predictions: Whether to track predictions this epoch

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        correct_exercise = 0
        correct_phase = 0
        total_rep_mae = 0.0
        total_fatigue_mae = 0.0

        # Collect batches for random sampling
        all_batches = []

        for batch_idx, (signals, targets) in enumerate(self.val_loader):
            # Move to device
            signals = {k: v.to(self.device) for k, v in signals.items()}
            exercise_labels = targets['exercise'].to(self.device)
            phase_labels = targets['phase'].to(self.device)
            rep_labels = targets['reps'].to(self.device)
            fatigue_labels = targets['fatigue'].to(self.device)

            try:
                # Forward pass
                exercise_logits, phase_logits, rep_pred, fatigue_pred = self.model(signals)

                # Compute loss
                predictions = (exercise_logits, phase_logits, rep_pred, fatigue_pred)
                labels = (exercise_labels, phase_labels, rep_labels, fatigue_labels)
                loss, _ = self.criterion(predictions, labels)

                # Accumulate metrics
                batch_size = exercise_labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Exercise accuracy
                _, pred_exercise = exercise_logits.max(1)
                correct_exercise += pred_exercise.eq(exercise_labels).sum().item()

                # Phase accuracy
                _, pred_phase = phase_logits.max(1)
                correct_phase += pred_phase.eq(phase_labels).sum().item()

                # Rep MAE
                total_rep_mae += torch.abs(rep_pred.squeeze(-1) - rep_labels.float()).sum().item()

                # Fatigue MAE
                total_fatigue_mae += torch.abs(fatigue_pred.squeeze(-1) - fatigue_labels.float()).sum().item()

                # Collect batch for tracking
                if track_predictions and self.tracker and self.config.tracking.track_val:
                    # Extract metadata from targets (lists of strings/values)
                    batch_size = exercise_labels.size(0)
                    metadata_list = []
                    for i in range(batch_size):
                        metadata_list.append({
                            'session_id': targets.get('session_id', ['unknown'] * batch_size)[i],
                            'exercise': targets.get('exercise_name', ['unknown'] * batch_size)[i],
                            'window_idx': targets.get('window_idx', [0] * batch_size)[i],
                            'start_time': targets.get('start_time', [0.0] * batch_size)[i],
                            'end_time': targets.get('end_time', [0.0] * batch_size)[i],
                            'skeleton_frame': targets.get('skeleton_frame', [None] * batch_size)[i],
                        })

                    all_batches.append({
                        'signals': {k: v.clone() for k, v in signals.items()},
                        'predictions': (
                            exercise_logits.clone(),
                            phase_logits.clone(),
                            rep_pred.clone(),
                            fatigue_pred.clone()
                        ),
                        'targets': {
                            'exercise': exercise_labels.clone(),
                            'phase': phase_labels.clone(),
                            'reps': rep_labels.clone(),
                            'fatigue': fatigue_labels.clone()
                        },
                        'metadata': metadata_list,
                        'batch_idx': batch_idx
                    })

            except Exception as e:
                self.logger.error(f"Validation error: {e}")
                continue

        # Track random samples from validation
        if track_predictions and self.tracker and self.config.tracking.track_val and all_batches:
            self._track_random_samples(all_batches, split='val')

        if total_samples == 0:
            return {'loss': 0, 'exercise_acc': 0, 'phase_acc': 0, 'rep_mae': 0, 'fatigue_mae': 0}

        # Calculate metrics
        metrics = {
            'loss': total_loss / total_samples,
            'exercise_acc': 100.0 * correct_exercise / total_samples,
            'phase_acc': 100.0 * correct_phase / total_samples,
            'rep_mae': total_rep_mae / total_samples,
            'fatigue_mae': total_fatigue_mae / total_samples
        }

        return metrics

    def _track_random_samples(self, all_batches: List[Dict], split: str = 'val'):
        """
        Track random samples from collected batches.

        Args:
            all_batches: List of batch dictionaries with signals, predictions, targets
            split: Data split ('train', 'val', 'test')
        """
        n_samples = self.config.tracking.n_samples_per_epoch

        # If n_samples is 0, track all samples
        if n_samples == 0:
            # Convert all batches to individual samples with metadata
            batches_to_track = []
            for batch in all_batches:
                batch_size = batch['targets']['exercise'].size(0)
                for idx in range(batch_size):
                    single_batch = {
                        'signals': {k: v[idx:idx+1] for k, v in batch['signals'].items()},
                        'predictions': tuple(p[idx:idx+1] for p in batch['predictions']),
                        'targets': {k: v[idx:idx+1] for k, v in batch['targets'].items()},
                        'metadata': [batch['metadata'][idx]] if 'metadata' in batch else None,
                        'batch_idx': batch['batch_idx']
                    }
                    batches_to_track.append(single_batch)
        else:
            # Calculate total samples available
            total_available = sum(
                batch['targets']['exercise'].size(0) for batch in all_batches
            )

            # Limit n_samples to available samples
            n_samples = min(n_samples, total_available)

            # Randomly select which batches and indices to track
            sample_indices = []
            current_idx = 0
            for batch in all_batches:
                batch_size = batch['targets']['exercise'].size(0)
                for i in range(batch_size):
                    sample_indices.append((batch, i))
                current_idx += batch_size

            # Random selection
            selected = random.sample(sample_indices, n_samples)

            # Group by batch for efficient tracking
            batches_to_track = []
            for batch, idx in selected:
                # Create single-sample batch
                single_batch = {
                    'signals': {k: v[idx:idx+1] for k, v in batch['signals'].items()},
                    'predictions': tuple(p[idx:idx+1] for p in batch['predictions']),
                    'targets': {k: v[idx:idx+1] for k, v in batch['targets'].items()},
                    'metadata': [batch['metadata'][idx]] if 'metadata' in batch else None,
                    'batch_idx': batch['batch_idx']
                }
                batches_to_track.append(single_batch)

        # Track the selected batches
        for batch in batches_to_track:
            self.tracker.track_batch(
                signals=batch['signals'],
                predictions=batch['predictions'],
                targets=batch['targets'],
                metadata=batch.get('metadata'),
                epoch=self.current_epoch,
                batch_idx=batch['batch_idx'],
                split=split
            )

    def train(self) -> Dict[str, Any]:
        """
        Full training loop.

        Returns:
            Training history and best model info
        """
        self.logger.info("="*70)
        self.logger.info("TRAINING STARTED")
        self.logger.info("="*70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config.training.n_epochs}")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        self.logger.info("="*70)

        # Ensure output directory exists
        self.config.output.models_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.config.output.models_dir / self.config.output.model_filename

        start_time = time.time()

        for epoch in range(1, self.config.training.n_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log metrics
            self.training_logger.log_epoch(epoch, train_metrics, val_metrics)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_exercise_acc'].append(train_metrics['exercise_acc'])
            self.history['val_exercise_acc'].append(val_metrics['exercise_acc'])
            self.history['train_phase_acc'].append(train_metrics['phase_acc'])
            self.history['val_phase_acc'].append(val_metrics['phase_acc'])
            self.history['train_rep_mae'].append(train_metrics['rep_mae'])
            self.history['val_rep_mae'].append(val_metrics['rep_mae'])
            self.history['train_fatigue_mae'].append(train_metrics['fatigue_mae'])
            self.history['val_fatigue_mae'].append(val_metrics['fatigue_mae'])
            self.history['learning_rates'].append(current_lr)

            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0

                # Save best model
                self._save_checkpoint(save_path, val_metrics, is_best=True)
                self.training_logger.log_best_model(epoch, val_metrics['loss'], save_path)
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"  No improvement. Patience: {self.patience_counter}/{self.config.training.early_stopping_patience}"
                )

            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                self.training_logger.log_early_stop(epoch, self.config.training.early_stopping_patience)
                break

            epoch_time = time.time() - epoch_start
            self.logger.info(f"  Epoch time: {epoch_time:.1f}s")

        total_time = time.time() - start_time
        self.logger.info("="*70)
        self.logger.info("TRAINING COMPLETE")
        self.logger.info("="*70)
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Model saved to: {save_path}")

        # Save tracked predictions
        tracking_save_path = None
        if self.tracker and self.config.tracking.auto_save:
            tracking_save_path = self._save_tracking_data()

        self.logger.info("="*70)

        return {
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch,
            'save_path': str(save_path),
            'tracking_save_path': tracking_save_path
        }

    def _save_tracking_data(self) -> Optional[str]:
        """
        Save tracked predictions and generate visualizations.

        Returns:
            Path to saved tracking data
        """
        if not self.tracker:
            return None

        # Save tracking data
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"training_predictions_{timestamp}"
        self.tracker.save(filename)

        # Log statistics
        stats = self.tracker.get_statistics()
        self.logger.info("\n" + "="*50)
        self.logger.info("PREDICTION TRACKING SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total tracked predictions: {stats['total_predictions']}")
        self.logger.info(f"Exercise accuracy: {stats['exercise_accuracy']:.2%}")
        self.logger.info(f"Phase accuracy: {stats['phase_accuracy']:.2%}")
        self.logger.info(f"Reps MAE: {stats['reps_mae']:.2f}")
        self.logger.info(f"Fatigue MAE: {stats['fatigue_mae']:.3f}")

        # Generate visualizations if enabled
        if self.config.tracking.save_visualizations:
            self._generate_visualizations()

        return str(self.tracker.output_dir / filename)

    def _generate_visualizations(self, window_indices: List[int] = None):
        """
        Generate visualizations for tracked samples.

        Args:
            window_indices: Optional list of specific window indices to visualize.
                          If None, uses config settings (random or specific mode).
        """
        if not self.tracker:
            return

        records = self.tracker.records

        if not records:
            return

        # Create visualizations directory
        vis_dir = self.tracker.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Determine which records to visualize
        if window_indices is not None:
            # Use provided window indices
            selected_records = [
                r for r in records if r.window_idx in window_indices
            ]
            mode_info = f"specific windows: {window_indices}"
        elif self.config.tracking.visualization_mode == 'specific':
            # Use specific indices from config
            specific_indices = self.config.tracking.specific_window_indices
            if not specific_indices:
                self.logger.warning("visualization_mode='specific' but no specific_window_indices provided")
                return
            selected_records = [
                r for r in records if r.window_idx in specific_indices
            ]
            mode_info = f"specific windows from config: {specific_indices}"
        else:
            # Random mode (default)
            n_vis = self.config.tracking.n_visualizations
            n_vis = min(n_vis, len(records))
            selected_records = random.sample(records, n_vis)
            mode_info = f"random ({n_vis} samples)"

        if not selected_records:
            self.logger.warning("No records found matching the specified window indices")
            return

        self.logger.info(f"\nGenerating visualizations ({mode_info})...")
        self.logger.info(f"Found {len(selected_records)} records to visualize")

        for i, record in enumerate(selected_records):
            record_dict = self.tracker.get_window_with_prediction(record.record_id)
            if record_dict:
                # Create subfolder for each window_idx
                window_dir = vis_dir / f"window_{record.window_idx}"
                window_dir.mkdir(parents=True, exist_ok=True)
                output_path = window_dir / f"record_{record.record_id}.png"
                try:
                    # Use skeleton visualization if skeleton data is available
                    if record_dict.get('skeleton_frame') is not None:
                        visualize_prediction_with_skeleton(
                            record_dict, output_path=output_path, show=False
                        )
                    else:
                        visualize_prediction(record_dict, output_path=output_path, show=False)
                except Exception as e:
                    self.logger.warning(f"Failed to generate visualization {i+1}: {e}")

        # Generate error analysis visualizations
        self._generate_error_visualizations(vis_dir)

        self.logger.info(f"Visualizations saved to: {vis_dir}")

    def _generate_error_visualizations(self, vis_dir: Path):
        """Generate visualizations for incorrectly predicted samples."""
        if not self.tracker:
            return

        # Get incorrect phase predictions
        incorrect_phase = self.tracker.get_records(incorrect_only=True, task='phase')

        if incorrect_phase:
            n_errors = min(5, len(incorrect_phase))
            self.logger.info(f"Generating {n_errors} error visualizations for phase detection...")

            # Create errors subfolder
            errors_dir = vis_dir / "phase_errors"
            errors_dir.mkdir(parents=True, exist_ok=True)

            for i, record in enumerate(incorrect_phase[:n_errors]):
                record_dict = self.tracker.get_window_with_prediction(record.record_id)
                if record_dict:
                    # Group by window_idx
                    window_dir = errors_dir / f"window_{record.window_idx}"
                    window_dir.mkdir(parents=True, exist_ok=True)
                    output_path = window_dir / f"record_{record.record_id}.png"
                    try:
                        # Use skeleton visualization if skeleton data is available
                        if record_dict.get('skeleton_frame') is not None:
                            visualize_prediction_with_skeleton(
                                record_dict, output_path=output_path, show=False
                            )
                        else:
                            visualize_prediction(record_dict, output_path=output_path, show=False)
                    except Exception as e:
                        self.logger.warning(f"Failed to generate error visualization: {e}")

        # Save error analysis summary
        error_analysis = self.tracker.get_error_analysis(task='phase')
        errors_dir = vis_dir / "phase_errors"
        errors_dir.mkdir(parents=True, exist_ok=True)
        analysis_path = errors_dir / "error_analysis.json"
        with open(analysis_path, 'w') as f:
            # Convert tuple keys to strings for JSON serialization
            serializable_analysis = {
                k: v if k != 'confusion_matrix' else {str(ck): cv for ck, cv in v.items()}
                for k, v in error_analysis.items()
            }
            json.dump(serializable_analysis, f, indent=2)

    def _save_checkpoint(
        self,
        path: Path,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': metrics['loss'],
            'val_exercise_acc': metrics['exercise_acc'],
            'val_phase_acc': metrics['phase_acc'],
            'val_rep_mae': metrics['rep_mae'],
            'val_fatigue_mae': metrics['fatigue_mae'],
            'history': self.history,
            'config': self.config.to_dict()
        }

        torch.save(checkpoint, path)

        if is_best:
            # Also save as best_model.pth
            best_path = path.parent / 'best_model.pth'
            torch.save(checkpoint, best_path)


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config=None,
    device: str = None
) -> Tuple[StrengthTrainingModel, Dict[str, Any]]:
    """
    Convenience function to train the model.

    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration object
        device: Device to use

    Returns:
        Trained model and training history
    """
    config = config or CONFIG

    # Create model
    model = create_model(config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    results = trainer.train()

    return model, results
