"""
PyTorch Lightning Module for Strength Training ML Pipeline v2.

Replaces the 799-line manual Trainer class from v1.
Lightning handles: training loop, validation, early stopping,
gradient clipping, checkpointing, LR scheduling, and logging.
"""

import torch
import torch.optim as optim
import pytorch_lightning as pl
from typing import Dict, Tuple, Any, Optional
import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from models import StrengthTrainingModel, MultiTaskLoss
from evaluation import PredictionTracker, visualize_prediction, visualize_prediction_with_skeleton


class StrengthTrainingLitModule(pl.LightningModule):
    """
    Lightning wrapper for multi-task strength training model.

    Replaces Trainer class from v1 (~800 lines -> ~120 lines).
    """

    def __init__(self, config=None):
        super().__init__()
        self.config = config or CONFIG
        self.save_hyperparameters(self.config.to_dict())

        # Model and loss
        self.model = StrengthTrainingModel(self.config)
        self.criterion = MultiTaskLoss(
            init_weights=self.config.training.loss_weights,
            use_uncertainty_weighting=True
        )

        # Metric accumulators
        self._reset_metrics()

        # Prediction tracker
        self.tracker = None
        if self.config.tracking.enabled:
            tracking_dir = self.config.output.output_dir / self.config.tracking.tracking_subdir
            self.tracker = PredictionTracker(
                output_dir=tracking_dir,
                max_records=self.config.tracking.max_records,
                store_signals=self.config.tracking.store_signals,
                store_logits=self.config.tracking.store_logits
            )

    def _reset_metrics(self):
        self._train_metrics = {'loss': 0, 'correct_ex': 0, 'correct_ph': 0,
                               'rep_mae': 0, 'fat_mae': 0, 'n': 0}
        self._val_metrics = {'loss': 0, 'correct_ex': 0, 'correct_ph': 0,
                             'rep_mae': 0, 'fat_mae': 0, 'n': 0}

    def forward(self, signals):
        return self.model(signals)

    def _shared_step(self, batch, metrics_dict):
        signals, targets = batch
        exercise_labels = targets['exercise'].to(self.device)
        phase_labels = targets['phase'].to(self.device)
        rep_labels = targets['reps'].to(self.device)
        fatigue_labels = targets['fatigue'].to(self.device)

        exercise_logits, phase_logits, rep_pred, fatigue_pred = self.model(signals)

        predictions = (exercise_logits, phase_logits, rep_pred, fatigue_pred)
        labels = (exercise_labels, phase_labels, rep_labels, fatigue_labels)
        loss, loss_dict = self.criterion(predictions, labels)

        batch_size = exercise_labels.size(0)
        _, pred_ex = exercise_logits.max(1)
        _, pred_ph = phase_logits.max(1)

        metrics_dict['loss'] += loss.item() * batch_size
        metrics_dict['correct_ex'] += pred_ex.eq(exercise_labels).sum().item()
        metrics_dict['correct_ph'] += pred_ph.eq(phase_labels).sum().item()
        metrics_dict['rep_mae'] += torch.abs(rep_pred.squeeze() - rep_labels.float()).sum().item()
        metrics_dict['fat_mae'] += torch.abs(fatigue_pred.squeeze() - fatigue_labels.float()).sum().item()
        metrics_dict['n'] += batch_size

        return loss, predictions, targets

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, self._train_metrics)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, predictions, targets = self._shared_step(batch, self._val_metrics)
        self.log('val_loss', loss, prog_bar=True)

        # Track predictions if enabled
        if self.tracker and self.config.tracking.track_val:
            signals, targets_full = batch
            exercise_logits, phase_logits, rep_pred, fatigue_pred = predictions
            batch_size = targets_full['exercise'].size(0)
            metadata_list = []
            for i in range(batch_size):
                metadata_list.append({
                    'session_id': targets_full.get('session_id', ['unknown'] * batch_size)[i],
                    'exercise': targets_full.get('exercise_name', ['unknown'] * batch_size)[i],
                    'window_idx': targets_full.get('window_idx', [0] * batch_size)[i],
                    'start_time': targets_full.get('start_time', [0.0] * batch_size)[i],
                    'end_time': targets_full.get('end_time', [0.0] * batch_size)[i],
                    'skeleton_frame': targets_full.get('skeleton_frame', [None] * batch_size)[i],
                })
            self.tracker.track_batch(
                signals=signals,
                predictions=predictions,
                targets={k: targets_full[k] for k in ['exercise', 'phase', 'reps', 'fatigue']},
                metadata=metadata_list,
                epoch=self.current_epoch,
                batch_idx=batch_idx,
                split='val'
            )

        return loss

    def on_train_epoch_end(self):
        m = self._train_metrics
        if m['n'] > 0:
            self.log('train_exercise_acc', 100.0 * m['correct_ex'] / m['n'])
            self.log('train_phase_acc', 100.0 * m['correct_ph'] / m['n'])
            self.log('train_rep_mae', m['rep_mae'] / m['n'])
            self.log('train_fatigue_mae', m['fat_mae'] / m['n'])
        self._train_metrics = {'loss': 0, 'correct_ex': 0, 'correct_ph': 0,
                               'rep_mae': 0, 'fat_mae': 0, 'n': 0}

    def on_validation_epoch_end(self):
        m = self._val_metrics
        if m['n'] > 0:
            self.log('val_exercise_acc', 100.0 * m['correct_ex'] / m['n'])
            self.log('val_phase_acc', 100.0 * m['correct_ph'] / m['n'])
            self.log('val_rep_mae', m['rep_mae'] / m['n'])
            self.log('val_fatigue_mae', m['fat_mae'] / m['n'])
        self._val_metrics = {'loss': 0, 'correct_ex': 0, 'correct_ph': 0,
                             'rep_mae': 0, 'fat_mae': 0, 'n': 0}

    def configure_optimizers(self):
        cfg = self.config.training

        if cfg.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer.lower() == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif cfg.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

        if cfg.scheduler_type == 'plateau':
            scheduler = {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=cfg.scheduler_factor,
                    patience=cfg.scheduler_patience, min_lr=cfg.min_lr
                ),
                'monitor': 'val_loss'
            }
        elif cfg.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.n_epochs, eta_min=cfg.min_lr
            )
        elif cfg.scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=cfg.scheduler_factor)
        else:
            return optimizer

        return [optimizer], [scheduler]

    def save_tracking_data(self) -> Optional[str]:
        """Save tracked predictions after training."""
        if not self.tracker:
            return None
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"training_predictions_{timestamp}"
        self.tracker.save(filename)

        stats = self.tracker.get_statistics()
        print(f"\nTracking: {stats['total_predictions']} predictions | "
              f"Ex acc: {stats['exercise_accuracy']:.2%} | "
              f"Ph acc: {stats['phase_accuracy']:.2%} | "
              f"Rep MAE: {stats['reps_mae']:.2f} | "
              f"Fat MAE: {stats['fatigue_mae']:.3f}")

        if self.config.tracking.save_visualizations:
            self._generate_visualizations()

        return str(self.tracker.output_dir / filename)

    def _generate_visualizations(self):
        if not self.tracker or not self.tracker.records:
            return

        vis_dir = self.tracker.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        records = self.tracker.records
        if self.config.tracking.visualization_mode == 'specific':
            indices = self.config.tracking.specific_window_indices
            selected = [r for r in records if r.window_idx in indices]
        else:
            n_vis = min(self.config.tracking.n_visualizations, len(records))
            selected = random.sample(records, n_vis)

        for record in selected:
            record_dict = self.tracker.get_window_with_prediction(record.record_id)
            if not record_dict:
                continue
            window_dir = vis_dir / f"window_{record.window_idx}"
            window_dir.mkdir(parents=True, exist_ok=True)
            output_path = window_dir / f"record_{record.record_id}.png"
            try:
                if record_dict.get('skeleton_frame') is not None:
                    visualize_prediction_with_skeleton(record_dict, output_path=output_path, show=False)
                else:
                    visualize_prediction(record_dict, output_path=output_path, show=False)
            except Exception as e:
                print(f"Visualization failed: {e}")
