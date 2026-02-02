"""
Evaluation and Inference Module for Strength Training Model
============================================================
This module provides:
- Comprehensive evaluation metrics (F1, precision, recall, confusion matrix)
- ROC curves and precision-recall curves
- Training history visualization
- Fatigue analysis visualization
- Real-time inference utilities
- Model interpretability (attention visualization)
- Complete evaluation dashboard
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    precision_recall_curve, roc_curve, auc, roc_auc_score,
    precision_score, recall_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import json
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

from model import MultiTaskStrengthModel, ModelConfig, create_model
from preprocessing import PreprocessingPipeline, ProcessedSegment


@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation results."""
    # Classification metrics
    exercise_accuracy: float
    exercise_f1_macro: float
    exercise_f1_weighted: float
    exercise_f1_per_class: Dict[str, float]
    exercise_precision_per_class: Dict[str, float]
    exercise_recall_per_class: Dict[str, float]
    confusion_matrix: np.ndarray
    confusion_matrix_normalized: np.ndarray

    # Phase detection metrics
    phase_accuracy: float
    phase_f1: float
    phase_precision: float
    phase_recall: float
    phase_iou: float  # Intersection over Union

    # Rep position metrics
    rep_position_mae: float
    rep_position_rmse: float
    rep_position_r2: float
    rep_position_correlation: float

    # Fatigue metrics
    fatigue_scores: np.ndarray
    fatigue_correlation_with_rep: float
    fatigue_mean: float
    fatigue_std: float

    # Per-sample predictions and probabilities
    predictions: Dict[str, np.ndarray]
    targets: Dict[str, np.ndarray]
    probabilities: Dict[str, np.ndarray] = field(default_factory=dict)

    # Additional metrics
    total_samples: int = 0
    per_exercise_accuracy: Dict[str, float] = field(default_factory=dict)


class MetricsCalculator:
    """
    Utility class for calculating various ML metrics.
    """

    @staticmethod
    def calculate_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Calculate Intersection over Union for binary segmentation."""
        pred_binary = pred > threshold
        target_binary = target > threshold

        intersection = np.logical_and(pred_binary, target_binary).sum()
        union = np.logical_or(pred_binary, target_binary).sum()

        return intersection / (union + 1e-8)

    @staticmethod
    def calculate_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        """Calculate Dice coefficient for binary segmentation."""
        pred_binary = pred > threshold
        target_binary = target > threshold

        intersection = np.logical_and(pred_binary, target_binary).sum()
        return 2 * intersection / (pred_binary.sum() + target_binary.sum() + 1e-8)

    @staticmethod
    def bootstrap_confidence_interval(metric_func, y_true, y_pred,
                                       n_bootstraps: int = 1000,
                                       ci: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for a metric.

        Returns:
            (mean, lower_bound, upper_bound)
        """
        scores = []
        n_samples = len(y_true)

        for _ in range(n_bootstraps):
            indices = np.random.randint(0, n_samples, n_samples)
            score = metric_func(y_true[indices], y_pred[indices])
            scores.append(score)

        scores = np.array(scores)
        mean_score = np.mean(scores)
        lower = np.percentile(scores, (1 - ci) / 2 * 100)
        upper = np.percentile(scores, (1 + ci) / 2 * 100)

        return mean_score, lower, upper


class TrainingVisualizer:
    """
    Visualize training history and metrics.
    """

    def __init__(self, history: Dict[str, List[float]]):
        """
        Args:
            history: Dictionary containing training history
        """
        self.history = history

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot comprehensive training curves."""
        fig = plt.figure(figsize=(20, 15))

        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Total Loss
        ax1 = fig.add_subplot(gs[0, 0])
        if 'train_total' in self.history:
            ax1.plot(self.history['train_total'], 'b-', label='Train', linewidth=2)
        if 'val_total' in self.history:
            ax1.plot(self.history['val_total'], 'r-', label='Val', linewidth=2)
        ax1.set_title('Total Loss', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Exercise Classification Loss
        ax2 = fig.add_subplot(gs[0, 1])
        if 'train_exercise' in self.history:
            ax2.plot(self.history['train_exercise'], 'b-', label='Train', linewidth=2)
        if 'val_exercise' in self.history:
            ax2.plot(self.history['val_exercise'], 'r-', label='Val', linewidth=2)
        ax2.set_title('Exercise Classification Loss', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Phase Detection Loss
        ax3 = fig.add_subplot(gs[0, 2])
        if 'train_phase' in self.history:
            ax3.plot(self.history['train_phase'], 'b-', label='Train', linewidth=2)
        if 'val_phase' in self.history:
            ax3.plot(self.history['val_phase'], 'r-', label='Val', linewidth=2)
        ax3.set_title('Phase Detection Loss', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Rep Position Loss
        ax4 = fig.add_subplot(gs[0, 3])
        if 'train_rep_position' in self.history:
            ax4.plot(self.history['train_rep_position'], 'b-', label='Train', linewidth=2)
        if 'val_rep_position' in self.history:
            ax4.plot(self.history['val_rep_position'], 'r-', label='Val', linewidth=2)
        ax4.set_title('Rep Position Loss', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Fatigue Reconstruction Loss
        ax5 = fig.add_subplot(gs[1, 0])
        if 'train_fatigue_recon' in self.history:
            ax5.plot(self.history['train_fatigue_recon'], 'b-', label='Train', linewidth=2)
        if 'val_fatigue_recon' in self.history:
            ax5.plot(self.history['val_fatigue_recon'], 'r-', label='Val', linewidth=2)
        ax5.set_title('Fatigue VAE Recon Loss', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Fatigue KL Loss
        ax6 = fig.add_subplot(gs[1, 1])
        if 'train_fatigue_kl' in self.history:
            ax6.plot(self.history['train_fatigue_kl'], 'b-', label='Train', linewidth=2)
        if 'val_fatigue_kl' in self.history:
            ax6.plot(self.history['val_fatigue_kl'], 'r-', label='Val', linewidth=2)
        ax6.set_title('Fatigue VAE KL Loss', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Exercise Accuracy
        ax7 = fig.add_subplot(gs[1, 2])
        if 'val_exercise_accuracy' in self.history:
            ax7.plot(self.history['val_exercise_accuracy'], 'g-', linewidth=2, marker='o', markersize=3)
            best_acc = max(self.history['val_exercise_accuracy'])
            best_epoch = self.history['val_exercise_accuracy'].index(best_acc)
            ax7.axhline(y=best_acc, color='r', linestyle='--', alpha=0.7,
                        label=f'Best: {best_acc:.3f} (epoch {best_epoch})')
        ax7.set_title('Validation Exercise Accuracy', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Accuracy')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Phase Accuracy
        ax8 = fig.add_subplot(gs[1, 3])
        if 'val_phase_accuracy' in self.history:
            ax8.plot(self.history['val_phase_accuracy'], 'g-', linewidth=2, marker='o', markersize=3)
            best_acc = max(self.history['val_phase_accuracy'])
            best_epoch = self.history['val_phase_accuracy'].index(best_acc)
            ax8.axhline(y=best_acc, color='r', linestyle='--', alpha=0.7,
                        label=f'Best: {best_acc:.3f} (epoch {best_epoch})')
        ax8.set_title('Validation Phase Accuracy', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Accuracy')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. Rep Position MAE
        ax9 = fig.add_subplot(gs[2, 0])
        if 'val_rep_position_mae' in self.history:
            ax9.plot(self.history['val_rep_position_mae'], 'orange', linewidth=2, marker='o', markersize=3)
            best_mae = min(self.history['val_rep_position_mae'])
            best_epoch = self.history['val_rep_position_mae'].index(best_mae)
            ax9.axhline(y=best_mae, color='r', linestyle='--', alpha=0.7,
                        label=f'Best: {best_mae:.3f} (epoch {best_epoch})')
        ax9.set_title('Validation Rep Position MAE', fontsize=12, fontweight='bold')
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('MAE')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        # 10. Learning Rate
        ax10 = fig.add_subplot(gs[2, 1])
        if 'lr' in self.history:
            ax10.plot(self.history['lr'], 'purple', linewidth=2)
        ax10.set_title('Learning Rate', fontsize=12, fontweight='bold')
        ax10.set_xlabel('Epoch')
        ax10.set_ylabel('LR')
        ax10.set_yscale('log')
        ax10.grid(True, alpha=0.3)

        # 11. Train vs Val Loss Comparison
        ax11 = fig.add_subplot(gs[2, 2])
        if 'train_total' in self.history and 'val_total' in self.history:
            train_loss = np.array(self.history['train_total'])
            val_loss = np.array(self.history['val_total'])
            gap = val_loss - train_loss
            ax11.fill_between(range(len(gap)), 0, gap, alpha=0.3,
                              color='red' if gap.mean() > 0 else 'green',
                              label='Val - Train')
            ax11.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax11.plot(gap, 'k-', linewidth=1)
        ax11.set_title('Generalization Gap', fontsize=12, fontweight='bold')
        ax11.set_xlabel('Epoch')
        ax11.set_ylabel('Val Loss - Train Loss')
        ax11.legend()
        ax11.grid(True, alpha=0.3)

        # 12. Summary Statistics
        ax12 = fig.add_subplot(gs[2, 3])
        ax12.axis('off')

        summary_text = "Training Summary\n" + "=" * 30 + "\n\n"
        if 'val_exercise_accuracy' in self.history:
            summary_text += f"Best Exercise Acc: {max(self.history['val_exercise_accuracy']):.3f}\n"
        if 'val_phase_accuracy' in self.history:
            summary_text += f"Best Phase Acc: {max(self.history['val_phase_accuracy']):.3f}\n"
        if 'val_rep_position_mae' in self.history:
            summary_text += f"Best Rep MAE: {min(self.history['val_rep_position_mae']):.3f}\n"
        if 'val_total' in self.history:
            summary_text += f"Best Val Loss: {min(self.history['val_total']):.4f}\n"
        if 'train_total' in self.history:
            summary_text += f"\nTotal Epochs: {len(self.history['train_total'])}\n"

        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        plt.suptitle('Training History Dashboard', fontsize=16, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        plt.show()

    def plot_loss_components(self, save_path: Optional[str] = None):
        """Plot stacked loss components over training."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        loss_components = ['exercise', 'phase', 'rep_position', 'fatigue_recon', 'fatigue_kl', 'fatigue_contrastive']
        colors = plt.cm.Set2(np.linspace(0, 1, len(loss_components)))

        # Training losses
        ax = axes[0]
        bottom = np.zeros(len(self.history.get('train_total', [])))
        for i, comp in enumerate(loss_components):
            key = f'train_{comp}'
            if key in self.history:
                values = np.array(self.history[key])
                ax.bar(range(len(values)), values, bottom=bottom, label=comp.replace('_', ' ').title(),
                       color=colors[i], alpha=0.8)
                bottom += values
        ax.set_title('Training Loss Components', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right', fontsize=8)

        # Validation losses
        ax = axes[1]
        bottom = np.zeros(len(self.history.get('val_total', [])))
        for i, comp in enumerate(loss_components):
            key = f'val_{comp}'
            if key in self.history:
                values = np.array(self.history[key])
                ax.bar(range(len(values)), values, bottom=bottom, label=comp.replace('_', ' ').title(),
                       color=colors[i], alpha=0.8)
                bottom += values
        ax.set_title('Validation Loss Components', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


class ModelEvaluator:
    """
    Comprehensive model evaluation tool.

    Computes metrics, generates visualizations, and analyzes model behavior.
    """

    EXERCISE_NAMES = ['Squat', 'Benchpress', 'Pullups', 'Deadlift']

    def __init__(self, model: MultiTaskStrengthModel, device: str = 'cpu'):
        """
        Args:
            model: Trained model
            device: Device for inference
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.metrics_calculator = MetricsCalculator()

    @torch.no_grad()
    def evaluate(self, data_loader, exercise_names: Optional[List[str]] = None
                 ) -> EvaluationResults:
        """
        Run comprehensive evaluation.

        Args:
            data_loader: DataLoader with test data
            exercise_names: Names of exercises for reporting

        Returns:
            EvaluationResults with all metrics
        """
        if exercise_names is None:
            exercise_names = self.EXERCISE_NAMES

        all_preds = {
            'exercise': [],
            'exercise_probs': [],
            'phase': [],
            'phase_logits': [],
            'rep_position': [],
            'fatigue': []
        }
        all_targets = {
            'exercise': [],
            'phase': [],
            'rep_position': []
        }

        for batch in data_loader:
            x_seq = batch['x_seq'].to(self.device)
            x_feat = batch['x_feat'].to(self.device)

            outputs = self.model(x_seq, x_feat)

            # Store predictions
            exercise_probs = torch.softmax(outputs['exercise_logits'], dim=1)
            all_preds['exercise'].append(exercise_probs.argmax(dim=1).cpu().numpy())
            all_preds['exercise_probs'].append(exercise_probs.cpu().numpy())
            all_preds['phase'].append(torch.sigmoid(outputs['phase_logits']).cpu().numpy())
            all_preds['phase_logits'].append(outputs['phase_logits'].cpu().numpy())
            all_preds['rep_position'].append(outputs['rep_position'].cpu().numpy())
            all_preds['fatigue'].append(outputs['fatigue_score'].cpu().numpy())

            # Store targets
            all_targets['exercise'].append(batch['y_exercise'].numpy())
            all_targets['phase'].append(batch['y_phase'].numpy())
            all_targets['rep_position'].append(batch['y_rep_position'].numpy())

        # Concatenate
        for key in all_preds:
            all_preds[key] = np.concatenate(all_preds[key])
        for key in all_targets:
            all_targets[key] = np.concatenate(all_targets[key])

        # ==================== Exercise Classification Metrics ====================
        exercise_acc = accuracy_score(all_targets['exercise'], all_preds['exercise'])
        exercise_f1_macro = f1_score(all_targets['exercise'], all_preds['exercise'], average='macro')
        exercise_f1_weighted = f1_score(all_targets['exercise'], all_preds['exercise'], average='weighted')

        # Per-class metrics
        exercise_f1_per_class = f1_score(all_targets['exercise'], all_preds['exercise'], average=None)
        exercise_precision_per_class = precision_score(all_targets['exercise'], all_preds['exercise'], average=None)
        exercise_recall_per_class = recall_score(all_targets['exercise'], all_preds['exercise'], average=None)

        n_classes = len(np.unique(all_targets['exercise']))
        exercise_f1_dict = {
            exercise_names[i] if i < len(exercise_names) else f'Class_{i}': f1
            for i, f1 in enumerate(exercise_f1_per_class) if i < n_classes
        }
        exercise_precision_dict = {
            exercise_names[i] if i < len(exercise_names) else f'Class_{i}': p
            for i, p in enumerate(exercise_precision_per_class) if i < n_classes
        }
        exercise_recall_dict = {
            exercise_names[i] if i < len(exercise_names) else f'Class_{i}': r
            for i, r in enumerate(exercise_recall_per_class) if i < n_classes
        }

        # Confusion matrix
        conf_matrix = confusion_matrix(all_targets['exercise'], all_preds['exercise'])
        conf_matrix_normalized = conf_matrix.astype('float') / (conf_matrix.sum(axis=1, keepdims=True) + 1e-8)

        # Per-exercise accuracy
        per_exercise_acc = {}
        for i in range(n_classes):
            mask = all_targets['exercise'] == i
            if mask.sum() > 0:
                acc = (all_preds['exercise'][mask] == i).mean()
                name = exercise_names[i] if i < len(exercise_names) else f'Class_{i}'
                per_exercise_acc[name] = acc

        # ==================== Phase Detection Metrics ====================
        phase_preds_flat = all_preds['phase'].flatten()
        phase_targets_flat = all_targets['phase'].flatten()

        phase_pred_binary = phase_preds_flat > 0.5
        phase_target_binary = phase_targets_flat > 0.5

        phase_acc = accuracy_score(phase_target_binary, phase_pred_binary)
        phase_f1 = f1_score(phase_target_binary, phase_pred_binary)
        phase_precision = precision_score(phase_target_binary, phase_pred_binary)
        phase_recall = recall_score(phase_target_binary, phase_pred_binary)

        # IoU for phase segmentation
        phase_iou = self.metrics_calculator.calculate_iou(phase_preds_flat, phase_targets_flat)

        # ==================== Rep Position Metrics ====================
        rep_mae = mean_absolute_error(all_targets['rep_position'], all_preds['rep_position'])
        rep_rmse = np.sqrt(mean_squared_error(all_targets['rep_position'], all_preds['rep_position']))
        rep_r2 = r2_score(all_targets['rep_position'], all_preds['rep_position'])
        rep_corr = np.corrcoef(all_targets['rep_position'], all_preds['rep_position'])[0, 1]

        # ==================== Fatigue Metrics ====================
        fatigue_corr = np.corrcoef(all_preds['fatigue'], all_targets['rep_position'])[0, 1]
        fatigue_mean = np.mean(all_preds['fatigue'])
        fatigue_std = np.std(all_preds['fatigue'])

        return EvaluationResults(
            exercise_accuracy=exercise_acc,
            exercise_f1_macro=exercise_f1_macro,
            exercise_f1_weighted=exercise_f1_weighted,
            exercise_f1_per_class=exercise_f1_dict,
            exercise_precision_per_class=exercise_precision_dict,
            exercise_recall_per_class=exercise_recall_dict,
            confusion_matrix=conf_matrix,
            confusion_matrix_normalized=conf_matrix_normalized,
            phase_accuracy=phase_acc,
            phase_f1=phase_f1,
            phase_precision=phase_precision,
            phase_recall=phase_recall,
            phase_iou=phase_iou,
            rep_position_mae=rep_mae,
            rep_position_rmse=rep_rmse,
            rep_position_r2=rep_r2,
            rep_position_correlation=rep_corr,
            fatigue_scores=all_preds['fatigue'],
            fatigue_correlation_with_rep=fatigue_corr,
            fatigue_mean=fatigue_mean,
            fatigue_std=fatigue_std,
            predictions=all_preds,
            targets=all_targets,
            probabilities={'exercise': all_preds['exercise_probs']},
            total_samples=len(all_preds['exercise']),
            per_exercise_accuracy=per_exercise_acc
        )

    def plot_confusion_matrix(self, results: EvaluationResults,
                               exercise_names: Optional[List[str]] = None,
                               save_path: Optional[str] = None,
                               show_counts: bool = True):
        """Plot detailed confusion matrix for exercise classification."""
        if exercise_names is None:
            exercise_names = self.EXERCISE_NAMES[:results.confusion_matrix.shape[0]]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts confusion matrix
        ax = axes[0]
        sns.heatmap(
            results.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=exercise_names,
            yticklabels=exercise_names,
            ax=ax,
            cbar_kws={'label': 'Count'}
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

        # Normalized confusion matrix
        ax = axes[1]
        sns.heatmap(
            results.confusion_matrix_normalized,
            annot=True,
            fmt='.1%',
            cmap='Blues',
            xticklabels=exercise_names,
            yticklabels=exercise_names,
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Percentage'}
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

        plt.suptitle(f'Exercise Classification\nAccuracy: {results.exercise_accuracy:.1%} | '
                     f'Macro F1: {results.exercise_f1_macro:.3f}',
                     fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.show()

    def plot_f1_scores(self, results: EvaluationResults,
                        exercise_names: Optional[List[str]] = None,
                        save_path: Optional[str] = None):
        """Plot F1 scores, precision, and recall per class."""
        if exercise_names is None:
            exercise_names = list(results.exercise_f1_per_class.keys())

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        x = np.arange(len(exercise_names))
        width = 0.6

        # F1 Scores
        ax = axes[0]
        f1_values = [results.exercise_f1_per_class.get(name, 0) for name in exercise_names]
        bars = ax.bar(x, f1_values, width, color='steelblue', alpha=0.8)
        ax.axhline(y=results.exercise_f1_macro, color='red', linestyle='--',
                   label=f'Macro F1: {results.exercise_f1_macro:.3f}')
        ax.set_xlabel('Exercise')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score per Exercise', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exercise_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, f1_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        # Precision
        ax = axes[1]
        precision_values = [results.exercise_precision_per_class.get(name, 0) for name in exercise_names]
        bars = ax.bar(x, precision_values, width, color='forestgreen', alpha=0.8)
        ax.set_xlabel('Exercise')
        ax.set_ylabel('Precision')
        ax.set_title('Precision per Exercise', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exercise_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, precision_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        # Recall
        ax = axes[2]
        recall_values = [results.exercise_recall_per_class.get(name, 0) for name in exercise_names]
        bars = ax.bar(x, recall_values, width, color='coral', alpha=0.8)
        ax.set_xlabel('Exercise')
        ax.set_ylabel('Recall')
        ax.set_title('Recall per Exercise', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(exercise_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, recall_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"F1 scores plot saved to {save_path}")
        plt.show()

    def plot_roc_curves(self, results: EvaluationResults,
                         exercise_names: Optional[List[str]] = None,
                         save_path: Optional[str] = None):
        """Plot ROC curves for multi-class classification (one-vs-rest)."""
        if exercise_names is None:
            exercise_names = self.EXERCISE_NAMES

        n_classes = results.confusion_matrix.shape[0]
        exercise_names = exercise_names[:n_classes]

        # Binarize targets
        y_true_bin = label_binarize(results.targets['exercise'], classes=range(n_classes))
        y_score = results.probabilities['exercise']

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{exercise_names[i]} (AUC = {roc_auc:.3f})')

        # Compute micro-average ROC curve
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, color='navy', linestyle=':', lw=3,
                label=f'Micro-average (AUC = {roc_auc_micro:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        plt.show()

    def plot_precision_recall_curves(self, results: EvaluationResults,
                                      exercise_names: Optional[List[str]] = None,
                                      save_path: Optional[str] = None):
        """Plot precision-recall curves for multi-class classification."""
        if exercise_names is None:
            exercise_names = self.EXERCISE_NAMES

        n_classes = results.confusion_matrix.shape[0]
        exercise_names = exercise_names[:n_classes]

        y_true_bin = label_binarize(results.targets['exercise'], classes=range(n_classes))
        y_score = results.probabilities['exercise']

        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
            ap = np.trapz(precision, recall)  # Average precision
            ax.plot(recall, precision, color=colors[i], lw=2,
                    label=f'{exercise_names[i]} (AP = {ap:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Precision-recall curves saved to {save_path}")
        plt.show()

    def plot_phase_analysis(self, results: EvaluationResults,
                            n_samples: int = 5,
                            save_path: Optional[str] = None):
        """Plot phase detection analysis with metrics."""
        fig = plt.figure(figsize=(16, 10))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        # Sample predictions (left column, spanning 2 rows)
        ax_samples = fig.add_subplot(gs[0:2, 0:2])

        # Select random samples
        n_show = min(n_samples, len(results.predictions['phase']))
        indices = np.random.choice(len(results.predictions['phase']), n_show, replace=False)

        colors = plt.cm.tab10(np.linspace(0, 1, n_show))

        for i, (idx, color) in enumerate(zip(indices, colors)):
            pred = results.predictions['phase'][idx]
            target = results.targets['phase'][idx]
            seq_len = len(pred)
            x = np.arange(seq_len)

            offset = i * 1.2
            ax_samples.fill_between(x, offset, offset + target, alpha=0.3, color=color)
            ax_samples.plot(x, offset + pred, color=color, linewidth=1.5, label=f'Sample {idx}')
            ax_samples.axhline(y=offset + 0.5, color='gray', linestyle='--', alpha=0.3)

        ax_samples.set_xlabel('Time Step')
        ax_samples.set_ylabel('Sample (offset for visibility)')
        ax_samples.set_title('Phase Detection: Predictions vs Ground Truth', fontweight='bold')
        ax_samples.set_yticks([])

        # Phase accuracy histogram
        ax_hist = fig.add_subplot(gs[0, 2])
        # Calculate per-sample accuracy
        per_sample_acc = []
        for pred, target in zip(results.predictions['phase'], results.targets['phase']):
            acc = np.mean((pred > 0.5) == (target > 0.5))
            per_sample_acc.append(acc)
        ax_hist.hist(per_sample_acc, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax_hist.axvline(x=np.mean(per_sample_acc), color='red', linestyle='--',
                        label=f'Mean: {np.mean(per_sample_acc):.3f}')
        ax_hist.set_xlabel('Accuracy')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Per-Sample Phase Accuracy', fontweight='bold')
        ax_hist.legend()

        # Metrics summary
        ax_metrics = fig.add_subplot(gs[1, 2])
        ax_metrics.axis('off')
        metrics_text = (
            f"Phase Detection Metrics\n"
            f"{'='*25}\n\n"
            f"Accuracy:  {results.phase_accuracy:.1%}\n"
            f"F1 Score:  {results.phase_f1:.3f}\n"
            f"Precision: {results.phase_precision:.3f}\n"
            f"Recall:    {results.phase_recall:.3f}\n"
            f"IoU:       {results.phase_iou:.3f}\n"
        )
        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Prediction distribution over time
        ax_time = fig.add_subplot(gs[2, :])
        # Average prediction and ground truth over time
        mean_pred = np.mean(results.predictions['phase'], axis=0)
        mean_target = np.mean(results.targets['phase'], axis=0)
        std_pred = np.std(results.predictions['phase'], axis=0)

        x = np.arange(len(mean_pred))
        ax_time.fill_between(x, mean_pred - std_pred, mean_pred + std_pred,
                             alpha=0.3, color='blue', label='Pred ±σ')
        ax_time.plot(x, mean_pred, 'b-', linewidth=2, label='Mean Prediction')
        ax_time.plot(x, mean_target, 'r--', linewidth=2, label='Mean Ground Truth')
        ax_time.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax_time.set_xlabel('Time Step')
        ax_time.set_ylabel('Phase Probability')
        ax_time.set_title('Average Phase Prediction Over Time', fontweight='bold')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)

        plt.suptitle('Phase Detection Analysis', fontsize=16, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Phase analysis saved to {save_path}")
        plt.show()

    def plot_fatigue_analysis(self, results: EvaluationResults,
                               save_path: Optional[str] = None):
        """Plot comprehensive fatigue estimation analysis."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Fatigue vs Rep Position scatter with regression
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(results.targets['rep_position'], results.fatigue_scores,
                              c=results.targets['exercise'], cmap='Set1', alpha=0.6, s=30)
        z = np.polyfit(results.targets['rep_position'], results.fatigue_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax1.plot(x_line, p(x_line), 'k--', linewidth=2,
                 label=f'Linear fit (r={results.fatigue_correlation_with_rep:.3f})')
        ax1.set_xlabel('Rep Position (normalized)')
        ax1.set_ylabel('Fatigue Score')
        ax1.set_title('Fatigue vs Rep Position', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Fatigue distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(results.fatigue_scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=results.fatigue_mean, color='red', linestyle='--',
                    label=f'Mean: {results.fatigue_mean:.3f}')
        ax2.axvline(x=results.fatigue_mean + results.fatigue_std, color='orange', linestyle=':',
                    label=f'±σ: {results.fatigue_std:.3f}')
        ax2.axvline(x=results.fatigue_mean - results.fatigue_std, color='orange', linestyle=':')
        ax2.set_xlabel('Fatigue Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Fatigue Score Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Fatigue by exercise type (boxplot)
        ax3 = fig.add_subplot(gs[0, 2])
        exercise_types = np.unique(results.targets['exercise'])
        fatigue_by_exercise = []
        labels = []
        for ex in exercise_types:
            mask = results.targets['exercise'] == ex
            fatigue_by_exercise.append(results.fatigue_scores[mask])
            labels.append(self.EXERCISE_NAMES[ex] if ex < len(self.EXERCISE_NAMES) else f'Ex{ex}')

        bp = ax3.boxplot(fatigue_by_exercise, labels=labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax3.set_ylabel('Fatigue Score')
        ax3.set_title('Fatigue by Exercise', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Fatigue progression (binned)
        ax4 = fig.add_subplot(gs[1, 0])
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(results.targets['rep_position'], bins)
        fatigue_means = []
        fatigue_stds = []
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for i in range(1, len(bins)):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                fatigue_means.append(np.mean(results.fatigue_scores[mask]))
                fatigue_stds.append(np.std(results.fatigue_scores[mask]))
            else:
                fatigue_means.append(np.nan)
                fatigue_stds.append(np.nan)

        ax4.errorbar(bin_centers, fatigue_means, yerr=fatigue_stds,
                     fmt='o-', capsize=5, color='purple', linewidth=2, markersize=8)
        ax4.set_xlabel('Rep Position (binned)')
        ax4.set_ylabel('Mean Fatigue Score')
        ax4.set_title('Fatigue Progression Through Set', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. Fatigue heatmap by exercise and rep position
        ax5 = fig.add_subplot(gs[1, 1])
        heatmap_data = np.zeros((len(exercise_types), 10))
        for i, ex in enumerate(exercise_types):
            ex_mask = results.targets['exercise'] == ex
            for j in range(10):
                rep_mask = (results.targets['rep_position'] >= j/10) & (results.targets['rep_position'] < (j+1)/10)
                combined_mask = ex_mask & rep_mask
                if combined_mask.sum() > 0:
                    heatmap_data[i, j] = np.mean(results.fatigue_scores[combined_mask])

        sns.heatmap(heatmap_data, ax=ax5, cmap='YlOrRd',
                    xticklabels=[f'{i*10}-{(i+1)*10}%' for i in range(10)],
                    yticklabels=labels,
                    annot=True, fmt='.2f', cbar_kws={'label': 'Fatigue'})
        ax5.set_xlabel('Rep Position Range')
        ax5.set_ylabel('Exercise')
        ax5.set_title('Fatigue Heatmap', fontweight='bold')

        # 6. Fatigue percentiles
        ax6 = fig.add_subplot(gs[1, 2])
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = np.percentile(results.fatigue_scores, percentiles)
        bars = ax6.bar(range(len(percentiles)), percentile_values, color='teal', alpha=0.7)
        ax6.set_xticks(range(len(percentiles)))
        ax6.set_xticklabels([f'P{p}' for p in percentiles])
        ax6.set_ylabel('Fatigue Score')
        ax6.set_title('Fatigue Percentiles', fontweight='bold')
        for bar, val in zip(bars, percentile_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')

        # 7-9. Fatigue vs other metrics
        ax7 = fig.add_subplot(gs[2, 0])
        # Early vs Late reps
        early_mask = results.targets['rep_position'] < 0.33
        late_mask = results.targets['rep_position'] > 0.66
        early_fatigue = results.fatigue_scores[early_mask]
        late_fatigue = results.fatigue_scores[late_mask]

        parts = ax7.violinplot([early_fatigue, late_fatigue], positions=[0, 1], showmeans=True)
        for pc in parts['bodies']:
            pc.set_facecolor('steelblue')
            pc.set_alpha(0.6)
        ax7.set_xticks([0, 1])
        ax7.set_xticklabels(['Early Reps\n(0-33%)', 'Late Reps\n(66-100%)'])
        ax7.set_ylabel('Fatigue Score')
        ax7.set_title('Early vs Late Rep Fatigue', fontweight='bold')
        ax7.grid(True, alpha=0.3)

        # Summary text
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        summary_text = (
            f"Fatigue Estimation Summary\n"
            f"{'='*40}\n\n"
            f"Total Samples:        {len(results.fatigue_scores)}\n"
            f"Mean Fatigue:         {results.fatigue_mean:.3f}\n"
            f"Std Fatigue:          {results.fatigue_std:.3f}\n"
            f"Min Fatigue:          {results.fatigue_scores.min():.3f}\n"
            f"Max Fatigue:          {results.fatigue_scores.max():.3f}\n\n"
            f"Correlation with Rep Position: {results.fatigue_correlation_with_rep:.3f}\n\n"
            f"Early Rep Mean:       {np.mean(early_fatigue):.3f}\n"
            f"Late Rep Mean:        {np.mean(late_fatigue):.3f}\n"
            f"Fatigue Increase:     {np.mean(late_fatigue) - np.mean(early_fatigue):.3f}\n"
        )
        ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.suptitle('Fatigue Estimation Analysis (Unsupervised)', fontsize=16, fontweight='bold', y=1.02)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Fatigue analysis saved to {save_path}")
        plt.show()

    def plot_rep_position_analysis(self, results: EvaluationResults,
                                    save_path: Optional[str] = None):
        """Plot comprehensive rep position prediction analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 1. Scatter plot with perfect line
        ax = axes[0, 0]
        ax.scatter(results.targets['rep_position'], results.predictions['rep_position'],
                   alpha=0.5, s=30, c='steelblue')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')

        # Add regression line
        z = np.polyfit(results.targets['rep_position'], results.predictions['rep_position'], 1)
        p = np.poly1d(z)
        ax.plot([0, 1], [p(0), p(1)], 'g-', linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')

        ax.set_xlabel('True Rep Position')
        ax.set_ylabel('Predicted Rep Position')
        ax.set_title(f'Rep Position Prediction\nMAE={results.rep_position_mae:.3f}, R²={results.rep_position_r2:.3f}',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Error distribution
        ax = axes[0, 1]
        errors = results.predictions['rep_position'] - results.targets['rep_position']
        ax.hist(errors, bins=30, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=np.mean(errors), color='blue', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(errors):.3f}')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Count')
        ax.set_title(f'Error Distribution\nStd={np.std(errors):.3f}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Error vs true position
        ax = axes[0, 2]
        abs_errors = np.abs(errors)
        ax.scatter(results.targets['rep_position'], abs_errors, alpha=0.5, s=30, c='purple')
        # Binned mean error
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_errors = []
        for i in range(len(bins) - 1):
            mask = (results.targets['rep_position'] >= bins[i]) & (results.targets['rep_position'] < bins[i+1])
            if mask.sum() > 0:
                bin_errors.append(np.mean(abs_errors[mask]))
            else:
                bin_errors.append(np.nan)
        ax.plot(bin_centers, bin_errors, 'ro-', linewidth=2, markersize=8, label='Binned Mean')
        ax.set_xlabel('True Rep Position')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error vs Rep Position', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Error by exercise type
        ax = axes[1, 0]
        exercise_types = np.unique(results.targets['exercise'])
        error_by_exercise = []
        labels = []
        for ex in exercise_types:
            mask = results.targets['exercise'] == ex
            error_by_exercise.append(abs_errors[mask])
            labels.append(self.EXERCISE_NAMES[ex] if ex < len(self.EXERCISE_NAMES) else f'Ex{ex}')

        bp = ax.boxplot(error_by_exercise, labels=labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel('Absolute Error')
        ax.set_title('Error by Exercise Type', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 5. Prediction density
        ax = axes[1, 1]
        h = ax.hist2d(results.targets['rep_position'], results.predictions['rep_position'],
                      bins=20, cmap='Blues', cmin=1)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2)
        plt.colorbar(h[3], ax=ax, label='Count')
        ax.set_xlabel('True Rep Position')
        ax.set_ylabel('Predicted Rep Position')
        ax.set_title('Prediction Density', fontweight='bold')

        # 6. Summary metrics
        ax = axes[1, 2]
        ax.axis('off')
        summary_text = (
            f"Rep Position Metrics\n"
            f"{'='*30}\n\n"
            f"MAE:          {results.rep_position_mae:.4f}\n"
            f"RMSE:         {results.rep_position_rmse:.4f}\n"
            f"R² Score:     {results.rep_position_r2:.4f}\n"
            f"Correlation:  {results.rep_position_correlation:.4f}\n\n"
            f"Error Statistics:\n"
            f"  Mean:       {np.mean(errors):.4f}\n"
            f"  Std:        {np.std(errors):.4f}\n"
            f"  Min:        {np.min(errors):.4f}\n"
            f"  Max:        {np.max(errors):.4f}\n\n"
            f"Absolute Error:\n"
            f"  Mean:       {np.mean(abs_errors):.4f}\n"
            f"  Median:     {np.median(abs_errors):.4f}\n"
            f"  95th %ile:  {np.percentile(abs_errors, 95):.4f}\n"
        )
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

        plt.suptitle('Rep Position Prediction Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Rep position analysis saved to {save_path}")
        plt.show()

    def plot_complete_dashboard(self, results: EvaluationResults,
                                 exercise_names: Optional[List[str]] = None,
                                 save_path: Optional[str] = None):
        """Generate a complete evaluation dashboard."""
        if exercise_names is None:
            exercise_names = self.EXERCISE_NAMES[:results.confusion_matrix.shape[0]]

        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)

        # 1. Confusion Matrix (normalized)
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(results.confusion_matrix_normalized, annot=True, fmt='.1%',
                    cmap='Blues', xticklabels=exercise_names, yticklabels=exercise_names, ax=ax1)
        ax1.set_title(f'Confusion Matrix\nAcc: {results.exercise_accuracy:.1%}', fontweight='bold')

        # 2. F1 Scores per class
        ax2 = fig.add_subplot(gs[0, 1])
        f1_values = [results.exercise_f1_per_class.get(name, 0) for name in exercise_names]
        bars = ax2.barh(exercise_names, f1_values, color='steelblue', alpha=0.8)
        ax2.axvline(x=results.exercise_f1_macro, color='red', linestyle='--',
                    label=f'Macro: {results.exercise_f1_macro:.3f}')
        ax2.set_xlabel('F1 Score')
        ax2.set_title('F1 Score per Exercise', fontweight='bold')
        ax2.legend()
        ax2.set_xlim(0, 1)

        # 3. Phase metrics summary
        ax3 = fig.add_subplot(gs[0, 2])
        phase_metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'IoU']
        phase_values = [results.phase_accuracy, results.phase_f1,
                        results.phase_precision, results.phase_recall, results.phase_iou]
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax3.bar(phase_metrics, phase_values, color=colors, alpha=0.8)
        ax3.set_ylabel('Score')
        ax3.set_title('Phase Detection Metrics', fontweight='bold')
        ax3.set_ylim(0, 1)
        for bar, val in zip(bars, phase_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # 4. Rep position scatter
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.scatter(results.targets['rep_position'], results.predictions['rep_position'],
                    alpha=0.4, s=20, c='steelblue')
        ax4.plot([0, 1], [0, 1], 'r--', linewidth=2)
        ax4.set_xlabel('True')
        ax4.set_ylabel('Predicted')
        ax4.set_title(f'Rep Position\nMAE: {results.rep_position_mae:.3f}', fontweight='bold')

        # 5. Fatigue vs Rep Position
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.scatter(results.targets['rep_position'], results.fatigue_scores,
                    alpha=0.5, s=30, c='purple')
        z = np.polyfit(results.targets['rep_position'], results.fatigue_scores, 1)
        p = np.poly1d(z)
        ax5.plot([0, 1], [p(0), p(1)], 'r--', linewidth=2)
        ax5.set_xlabel('Rep Position')
        ax5.set_ylabel('Fatigue Score')
        ax5.set_title(f'Fatigue Correlation\nr={results.fatigue_correlation_with_rep:.3f}', fontweight='bold')

        # 6. Fatigue distribution
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.hist(results.fatigue_scores, bins=25, color='teal', alpha=0.7, edgecolor='black')
        ax6.axvline(x=results.fatigue_mean, color='red', linestyle='--')
        ax6.set_xlabel('Fatigue Score')
        ax6.set_ylabel('Count')
        ax6.set_title('Fatigue Distribution', fontweight='bold')

        # 7. Fatigue by exercise
        ax7 = fig.add_subplot(gs[1, 2])
        exercise_types = np.unique(results.targets['exercise'])
        fatigue_by_ex = [results.fatigue_scores[results.targets['exercise'] == ex] for ex in exercise_types]
        labels = [exercise_names[ex] if ex < len(exercise_names) else f'Ex{ex}' for ex in exercise_types]
        bp = ax7.boxplot(fatigue_by_ex, labels=labels, patch_artist=True)
        colors_box = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, c in zip(bp['boxes'], colors_box):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax7.set_ylabel('Fatigue Score')
        ax7.set_title('Fatigue by Exercise', fontweight='bold')

        # 8. Error distribution
        ax8 = fig.add_subplot(gs[1, 3])
        errors = results.predictions['rep_position'] - results.targets['rep_position']
        ax8.hist(errors, bins=25, color='coral', alpha=0.7, edgecolor='black')
        ax8.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax8.set_xlabel('Prediction Error')
        ax8.set_ylabel('Count')
        ax8.set_title('Rep Position Error', fontweight='bold')

        # 9-12. Phase predictions (sample)
        for i in range(4):
            ax = fig.add_subplot(gs[2, i])
            if i < len(results.predictions['phase']):
                idx = np.random.randint(0, len(results.predictions['phase']))
                pred = results.predictions['phase'][idx]
                target = results.targets['phase'][idx]
                x = np.arange(len(pred))
                ax.fill_between(x, 0, target, alpha=0.3, color='blue', label='True')
                ax.plot(x, pred, 'r-', linewidth=1.5, label='Pred')
                ax.set_ylim(-0.1, 1.1)
                ax.set_title(f'Phase Sample {idx}', fontweight='bold')
                if i == 0:
                    ax.legend(fontsize=8)

        # 13-16. Summary statistics and metrics
        ax13 = fig.add_subplot(gs[3, :2])
        ax13.axis('off')

        # Classification report
        class_report = classification_report(
            results.targets['exercise'],
            results.predictions['exercise'],
            target_names=exercise_names,
            output_dict=False
        )
        ax13.text(0.05, 0.95, f"Classification Report:\n\n{class_report}",
                  transform=ax13.transAxes, fontsize=9, verticalalignment='top',
                  fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

        ax14 = fig.add_subplot(gs[3, 2:])
        ax14.axis('off')

        summary = (
            f"{'='*50}\n"
            f"EVALUATION SUMMARY\n"
            f"{'='*50}\n\n"
            f"EXERCISE CLASSIFICATION\n"
            f"  Accuracy:        {results.exercise_accuracy:.1%}\n"
            f"  Macro F1:        {results.exercise_f1_macro:.3f}\n"
            f"  Weighted F1:     {results.exercise_f1_weighted:.3f}\n\n"
            f"PHASE DETECTION\n"
            f"  Accuracy:        {results.phase_accuracy:.1%}\n"
            f"  F1 Score:        {results.phase_f1:.3f}\n"
            f"  IoU:             {results.phase_iou:.3f}\n\n"
            f"REP POSITION\n"
            f"  MAE:             {results.rep_position_mae:.4f}\n"
            f"  RMSE:            {results.rep_position_rmse:.4f}\n"
            f"  R²:              {results.rep_position_r2:.4f}\n\n"
            f"FATIGUE ESTIMATION\n"
            f"  Mean:            {results.fatigue_mean:.3f}\n"
            f"  Correlation:     {results.fatigue_correlation_with_rep:.3f}\n\n"
            f"Total Samples:     {results.total_samples}\n"
            f"{'='*50}"
        )
        ax14.text(0.05, 0.95, summary, transform=ax14.transAxes, fontsize=10,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        plt.suptitle('Complete Evaluation Dashboard', fontsize=18, fontweight='bold', y=1.01)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Complete dashboard saved to {save_path}")
        plt.show()

    def generate_report(self, results: EvaluationResults,
                        exercise_names: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> str:
        """Generate a comprehensive text report."""
        if exercise_names is None:
            exercise_names = list(results.exercise_f1_per_class.keys())

        report = []
        report.append("=" * 70)
        report.append("              COMPREHENSIVE MODEL EVALUATION REPORT")
        report.append("=" * 70)
        report.append("")

        # Exercise classification
        report.append("1. EXERCISE CLASSIFICATION")
        report.append("-" * 50)
        report.append(f"   Accuracy:           {results.exercise_accuracy:.1%}")
        report.append(f"   Macro F1:           {results.exercise_f1_macro:.4f}")
        report.append(f"   Weighted F1:        {results.exercise_f1_weighted:.4f}")
        report.append("")
        report.append("   Per-class Metrics:")
        report.append(f"   {'Exercise':<15} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Accuracy':>10}")
        report.append("   " + "-" * 55)
        for name in exercise_names:
            f1 = results.exercise_f1_per_class.get(name, 0)
            prec = results.exercise_precision_per_class.get(name, 0)
            rec = results.exercise_recall_per_class.get(name, 0)
            acc = results.per_exercise_accuracy.get(name, 0)
            report.append(f"   {name:<15} {f1:>10.4f} {prec:>10.4f} {rec:>10.4f} {acc:>10.1%}")
        report.append("")

        # Phase detection
        report.append("2. PHASE DETECTION (Eccentric/Concentric)")
        report.append("-" * 50)
        report.append(f"   Accuracy:           {results.phase_accuracy:.1%}")
        report.append(f"   F1 Score:           {results.phase_f1:.4f}")
        report.append(f"   Precision:          {results.phase_precision:.4f}")
        report.append(f"   Recall:             {results.phase_recall:.4f}")
        report.append(f"   IoU:                {results.phase_iou:.4f}")
        report.append("")

        # Rep position
        report.append("3. REP POSITION ESTIMATION")
        report.append("-" * 50)
        report.append(f"   MAE:                {results.rep_position_mae:.4f}")
        report.append(f"   RMSE:               {results.rep_position_rmse:.4f}")
        report.append(f"   R² Score:           {results.rep_position_r2:.4f}")
        report.append(f"   Correlation:        {results.rep_position_correlation:.4f}")
        report.append("")

        # Fatigue
        report.append("4. FATIGUE ESTIMATION (Unsupervised)")
        report.append("-" * 50)
        report.append(f"   Mean Score:         {results.fatigue_mean:.4f}")
        report.append(f"   Std Score:          {results.fatigue_std:.4f}")
        report.append(f"   Score Range:        [{results.fatigue_scores.min():.4f}, {results.fatigue_scores.max():.4f}]")
        report.append(f"   Correlation w/Rep:  {results.fatigue_correlation_with_rep:.4f}")
        report.append("")

        # Summary
        report.append("5. DATASET SUMMARY")
        report.append("-" * 50)
        report.append(f"   Total Samples:      {results.total_samples}")
        unique_ex = np.unique(results.targets['exercise'])
        report.append(f"   Exercise Classes:   {len(unique_ex)}")
        for ex in unique_ex:
            count = (results.targets['exercise'] == ex).sum()
            name = exercise_names[ex] if ex < len(exercise_names) else f'Class_{ex}'
            report.append(f"     - {name}: {count} samples")
        report.append("")

        report.append("=" * 70)
        report.append("                      END OF REPORT")
        report.append("=" * 70)

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")

        return report_text


class RealTimeInference:
    """
    Real-time inference class for streaming sensor data.
    """

    def __init__(self, model: MultiTaskStrengthModel,
                 pipeline: PreprocessingPipeline,
                 window_size: int = 256,
                 stride: int = 64,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.pipeline = pipeline
        self.window_size = window_size
        self.stride = stride
        self.device = device

        self.buffer = {
            'emg': [], 'ecg': [], 'eda': [],
            'ppg_ir': [], 'ppg_blue': [], 'ppg_red': [], 'ppg_green': [],
            'acc_x': [], 'acc_y': [], 'acc_z': []
        }

        self.rep_count = 0
        self.current_exercise = None
        self.fatigue_history = []

    def add_samples(self, samples: Dict[str, np.ndarray]):
        for key, values in samples.items():
            if key in self.buffer:
                self.buffer[key].extend(values if isinstance(values, list) else [values])

    def process_window(self) -> Optional[Dict[str, float]]:
        min_samples = min(len(v) for v in self.buffer.values())
        if min_samples < self.window_size:
            return None

        window = {key: np.array(self.buffer[key][-self.window_size:]) for key in self.buffer}

        segment = {
            'signals': window,
            'rep_number': 0,
            'start_time': 0,
            'end_time': 1,
            'duration': 1
        }

        features = self.pipeline.extract_all_features(segment)
        sequences = self.pipeline.prepare_sequence_data(segment)

        x_seq = np.stack([sequences[ch] for ch in
                         ['emg', 'ecg', 'eda', 'ppg_ir', 'ppg_blue',
                          'ppg_red', 'ppg_green', 'acc_x', 'acc_y', 'acc_z']])
        x_seq = torch.from_numpy(x_seq).float().unsqueeze(0).to(self.device)

        x_feat = np.array([features.get(name, 0) for name in sorted(features.keys())])
        x_feat = torch.from_numpy(x_feat).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(x_seq, x_feat)

        exercise_probs = torch.softmax(outputs['exercise_logits'], dim=1)[0].cpu().numpy()
        exercise_pred = int(np.argmax(exercise_probs))
        phase_pred = (outputs['phase_logits'][0] > 0).float().cpu().numpy()
        current_phase = 'concentric' if phase_pred[-1] > 0.5 else 'eccentric'
        fatigue_score = outputs['fatigue_score'][0].cpu().item()

        self.fatigue_history.append(fatigue_score)
        self.current_exercise = exercise_pred

        boundary_scores = outputs['boundary_scores'][0].cpu().numpy()
        if boundary_scores.max() > 0.7:
            self.rep_count += 1

        for key in self.buffer:
            if len(self.buffer[key]) > self.window_size:
                self.buffer[key] = self.buffer[key][-self.window_size + self.stride:]

        return {
            'exercise': exercise_pred,
            'exercise_confidence': float(exercise_probs[exercise_pred]),
            'current_phase': current_phase,
            'fatigue_score': fatigue_score,
            'rep_count': self.rep_count,
        }

    def reset(self):
        for key in self.buffer:
            self.buffer[key] = []
        self.rep_count = 0
        self.current_exercise = None
        self.fatigue_history = []


def load_model_for_inference(checkpoint_path: str,
                              model_config: Optional[ModelConfig] = None,
                              device: str = 'cpu') -> MultiTaskStrengthModel:
    """Load a trained model for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model_config is None:
        model_config = ModelConfig()

    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def run_full_evaluation(model, data_loader, history: Dict = None,
                        output_dir: str = 'evaluation_results',
                        exercise_names: List[str] = None):
    """
    Run complete evaluation pipeline with all visualizations.

    Args:
        model: Trained model
        data_loader: Test data loader
        history: Training history dictionary
        output_dir: Directory to save results
        exercise_names: List of exercise names
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    evaluator = ModelEvaluator(model, device=str(device))

    print("Running evaluation...")
    results = evaluator.evaluate(data_loader, exercise_names)

    # Generate all plots
    print("\nGenerating visualizations...")

    # 1. Confusion Matrix
    evaluator.plot_confusion_matrix(
        results, exercise_names,
        save_path=str(output_path / 'confusion_matrix.png')
    )

    # 2. F1 Scores
    evaluator.plot_f1_scores(
        results, exercise_names,
        save_path=str(output_path / 'f1_scores.png')
    )

    # 3. ROC Curves
    evaluator.plot_roc_curves(
        results, exercise_names,
        save_path=str(output_path / 'roc_curves.png')
    )

    # 4. Precision-Recall Curves
    evaluator.plot_precision_recall_curves(
        results, exercise_names,
        save_path=str(output_path / 'precision_recall_curves.png')
    )

    # 5. Phase Analysis
    evaluator.plot_phase_analysis(
        results,
        save_path=str(output_path / 'phase_analysis.png')
    )

    # 6. Fatigue Analysis
    evaluator.plot_fatigue_analysis(
        results,
        save_path=str(output_path / 'fatigue_analysis.png')
    )

    # 7. Rep Position Analysis
    evaluator.plot_rep_position_analysis(
        results,
        save_path=str(output_path / 'rep_position_analysis.png')
    )

    # 8. Complete Dashboard
    evaluator.plot_complete_dashboard(
        results, exercise_names,
        save_path=str(output_path / 'complete_dashboard.png')
    )

    # 9. Training History (if provided)
    if history:
        visualizer = TrainingVisualizer(history)
        visualizer.plot_training_curves(
            save_path=str(output_path / 'training_curves.png')
        )
        visualizer.plot_loss_components(
            save_path=str(output_path / 'loss_components.png')
        )

    # 10. Generate Text Report
    report = evaluator.generate_report(
        results, exercise_names,
        save_path=str(output_path / 'evaluation_report.txt')
    )
    print("\n" + report)

    # Save results as JSON
    results_dict = {
        'exercise_accuracy': results.exercise_accuracy,
        'exercise_f1_macro': results.exercise_f1_macro,
        'exercise_f1_weighted': results.exercise_f1_weighted,
        'exercise_f1_per_class': results.exercise_f1_per_class,
        'phase_accuracy': results.phase_accuracy,
        'phase_f1': results.phase_f1,
        'phase_iou': results.phase_iou,
        'rep_position_mae': results.rep_position_mae,
        'rep_position_r2': results.rep_position_r2,
        'fatigue_correlation': results.fatigue_correlation_with_rep,
        'total_samples': results.total_samples
    }
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nAll results saved to: {output_path}")

    return results


if __name__ == "__main__":
    print("Evaluation module loaded.")
    print("\nAvailable functions:")
    print("  - ModelEvaluator: Comprehensive model evaluation")
    print("  - TrainingVisualizer: Training history visualization")
    print("  - run_full_evaluation: Complete evaluation pipeline")
    print("  - RealTimeInference: Streaming inference")
