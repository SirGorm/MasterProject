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

        # Training history for evaluation plots
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_exercise_acc': [], 'val_exercise_acc': [],
            'train_phase_acc': [], 'val_phase_acc': [],
            'train_rep_mae': [], 'val_rep_mae': [],
            'train_fatigue_mae': [], 'val_fatigue_mae': [],
            'learning_rates': [],
        }

        # Prediction tracker
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

    def _reset_metrics(self):
        self._train_metrics = {'loss': 0, 'correct_ex': 0, 'correct_ph': 0,
                               'rep_mae': 0, 'fat_mae': 0, 'n': 0}
        self._val_metrics = {'loss': 0, 'correct_ex': 0, 'correct_ph': 0,
                             'rep_mae': 0, 'fat_mae': 0, 'n': 0}

    def forward(self, signals):
        return self.model(signals)

    def _shared_step(self, batch, metrics_dict):
        signals, targets = batch
        signals = {k: v.to(self.device) for k, v in signals.items()}
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

            def _safe_index(d, key, default, idx):
                """Safely index into a batch value, handling None and short lists."""
                val = d.get(key)
                if val is None:
                    return default
                try:
                    return val[idx]
                except (IndexError, TypeError):
                    return default

            metadata_list = []
            for i in range(batch_size):
                metadata_list.append({
                    'session_id': _safe_index(targets_full, 'session_id', 'unknown', i),
                    'exercise': _safe_index(targets_full, 'exercise_name', 'unknown', i),
                    'window_idx': _safe_index(targets_full, 'window_idx', 0, i),
                    'start_time': _safe_index(targets_full, 'start_time', 0.0, i),
                    'end_time': _safe_index(targets_full, 'end_time', 0.0, i),
                    'skeleton_frame': _safe_index(targets_full, 'skeleton_frame', None, i),
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
            train_loss = m['loss'] / m['n']
            train_ex_acc = 100.0 * m['correct_ex'] / m['n']
            train_ph_acc = 100.0 * m['correct_ph'] / m['n']
            train_rep_mae = m['rep_mae'] / m['n']
            train_fat_mae = m['fat_mae'] / m['n']

            self.log('train_exercise_acc', train_ex_acc)
            self.log('train_phase_acc', train_ph_acc)
            self.log('train_rep_mae', train_rep_mae)
            self.log('train_fatigue_mae', train_fat_mae)

            # Store history for evaluation plots
            self.history['train_loss'].append(train_loss)
            self.history['train_exercise_acc'].append(train_ex_acc)
            self.history['train_phase_acc'].append(train_ph_acc)
            self.history['train_rep_mae'].append(train_rep_mae)
            self.history['train_fatigue_mae'].append(train_fat_mae)

            # Store current learning rate
            opt = self.optimizers()
            if opt is not None:
                lr = opt.param_groups[0]['lr'] if hasattr(opt, 'param_groups') else 0
                self.history['learning_rates'].append(lr)

        self._train_metrics = {'loss': 0, 'correct_ex': 0, 'correct_ph': 0,
                               'rep_mae': 0, 'fat_mae': 0, 'n': 0}

    def on_validation_epoch_end(self):
        m = self._val_metrics
        if m['n'] > 0:
            val_loss = m['loss'] / m['n']
            val_ex_acc = 100.0 * m['correct_ex'] / m['n']
            val_ph_acc = 100.0 * m['correct_ph'] / m['n']
            val_rep_mae = m['rep_mae'] / m['n']
            val_fat_mae = m['fat_mae'] / m['n']

            self.log('val_exercise_acc', val_ex_acc)
            self.log('val_phase_acc', val_ph_acc)
            self.log('val_rep_mae', val_rep_mae)
            self.log('val_fatigue_mae', val_fat_mae)

            # Store history for evaluation plots
            self.history['val_loss'].append(val_loss)
            self.history['val_exercise_acc'].append(val_ex_acc)
            self.history['val_phase_acc'].append(val_ph_acc)
            self.history['val_rep_mae'].append(val_rep_mae)
            self.history['val_fatigue_mae'].append(val_fat_mae)

        self._val_metrics = {'loss': 0, 'correct_ex': 0, 'correct_ph': 0,
                             'rep_mae': 0, 'fat_mae': 0, 'n': 0}

        # Save tracking data for this epoch and reset
        if self.tracker and self.config.tracking.auto_save and self.tracker.records:
            filename = f"predictions_epoch_{self.current_epoch:03d}"
            self.tracker.save(filename)
            self.tracker.clear()

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
        """Save any remaining tracked predictions after training and generate visualizations."""
        if not self.tracker:
            return None

        # Save any remaining unsaved records
        if self.tracker.records:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"training_predictions_final_{timestamp}"
            self.tracker.save(filename)

        # Load the last epoch's data for visualizations
        if self.config.tracking.save_visualizations:
            self._generate_final_visualizations()

        return str(self.tracker.output_dir)

    def _generate_final_visualizations(self):
        """Generate visualizations from the last saved epoch's tracking data."""
        if not self.tracker:
            return

        vis_dir = self.tracker.output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # If tracker still has records, use those; otherwise load the last saved file
        if not self.tracker.records:
            import glob
            pkl_files = sorted(glob.glob(str(self.tracker.output_dir / "predictions_epoch_*.pkl")))
            if not pkl_files:
                print("No tracking data found for visualization")
                return
            self.tracker.load(pkl_files[-1])

        records = self.tracker.records
        if not records:
            return

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
