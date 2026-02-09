"""
Evaluation Module for Strength Training ML Pipeline v2.

Changes from v1:
- Confusion matrices use seaborn.heatmap (replaces ~40 lines of manual imshow)
- NumpyEncoder class replaces recursive convert_numpy function
- Uses loguru logger
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, mean_absolute_error,
    mean_squared_error, r2_score, ConfusionMatrixDisplay,
    precision_score, recall_score, precision_recall_curve, auc
)
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
import json

from ml.v3.config import CONFIG
from ml.v3.models import StrengthTrainingModel, create_model
from ml.v3.utils import get_logger

logger = get_logger('eval')


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


class ModelEvaluator:
    """Comprehensive model evaluator for multi-task strength training model."""

    def __init__(self, model: StrengthTrainingModel, config=None, device: str = None):
        self.config = config or CONFIG
        self.model = model

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.exercise_names = self.config.get_active_exercises() if hasattr(self.config, 'get_active_exercises') \
            else self.config.data.exercises
        self.phase_names = ['Eccentric', 'Concentric']

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Run full evaluation on a dataset."""
        exercise_true, exercise_pred = [], []
        phase_true, phase_pred = [], []
        rep_true, rep_pred = [], []
        fatigue_true, fatigue_pred = [], []

        for signals, targets in data_loader:
            signals = {k: v.to(self.device) for k, v in signals.items()}
            exercise_logits, phase_logits, rep_output, fatigue_output = self.model(signals)

            _, exercise_preds = exercise_logits.max(1)
            _, phase_preds = phase_logits.max(1)

            exercise_true.extend(targets['exercise'].cpu().numpy())
            exercise_pred.extend(exercise_preds.cpu().numpy())
            phase_true.extend(targets['phase'].cpu().numpy())
            phase_pred.extend(phase_preds.cpu().numpy())
            rep_true.extend(targets['reps'].cpu().numpy())
            rep_pred.extend(rep_output.squeeze().cpu().numpy())
            fatigue_true.extend(targets['fatigue'].cpu().numpy())
            fatigue_pred.extend(fatigue_output.squeeze().cpu().numpy())

        results = {
            'predictions': {
                'exercise_true': np.array(exercise_true),
                'exercise_pred': np.array(exercise_pred),
                'phase_true': np.array(phase_true),
                'phase_pred': np.array(phase_pred),
                'rep_true': np.array(rep_true),
                'rep_pred': np.array(rep_pred),
                'fatigue_true': np.array(fatigue_true),
                'fatigue_pred': np.array(fatigue_pred)
            },
            'exercise_metrics': self._classification_metrics(
                np.array(exercise_true), np.array(exercise_pred), self.exercise_names
            ),
            'phase_metrics': self._classification_metrics(
                np.array(phase_true), np.array(phase_pred), self.phase_names
            ),
            'rep_metrics': self._regression_metrics(
                np.array(rep_true), np.array(rep_pred), is_reps=True
            ),
            'fatigue_metrics': self._regression_metrics(
                np.array(fatigue_true), np.array(fatigue_pred)
            )
        }
        return results

    def _classification_metrics(self, y_true, y_pred, class_names) -> Dict[str, Any]:
        labels = list(range(len(class_names)))
        accuracy = accuracy_score(y_true, y_pred) * 100
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        try:
            report = classification_report(
                y_true, y_pred, labels=labels, target_names=class_names,
                output_dict=True, zero_division=0
            )
        except ValueError:
            report = {name: {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
                      for name in class_names}

        return {
            'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(class_names, f1_per_class)),
            'confusion_matrix': cm, 'classification_report': report,
            'n_samples': len(y_true),
            'unique_labels_in_data': np.unique(np.concatenate([y_true, y_pred])).tolist()
        }

    def _regression_metrics(self, y_true, y_pred, is_reps=False) -> Dict[str, float]:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        if np.var(y_true) > 0:
            r2 = r2_score(y_true, y_pred)
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            r2 = correlation = 0.0

        metrics = {
            'mae': mae, 'mse': mse, 'rmse': rmse,
            'r2': r2 if not np.isnan(r2) else 0.0,
            'correlation': correlation if not np.isnan(correlation) else 0.0,
            'n_samples': len(y_true)
        }

        if is_reps:
            metrics['within_1_rep'] = np.sum(np.abs(y_true - y_pred) <= 1) / len(y_true) * 100
            metrics['within_2_reps'] = np.sum(np.abs(y_true - y_pred) <= 2) / len(y_true) * 100

            # Binary classification metrics for rep detection
            # Model outputs logits, apply sigmoid to get probabilities
            y_pred_proba = 1 / (1 + np.exp(-np.clip(y_pred, -500, 500)))
            y_pred_binary = (y_pred_proba >= 0.5).astype(int)
            y_true_binary = (y_true > 0).astype(int)

            if len(np.unique(y_true_binary)) > 1:
                metrics['precision'] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics['recall'] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics['f1'] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics['accuracy_binary'] = float(accuracy_score(y_true_binary, y_pred_binary) * 100)

                # Precision-Recall AUC
                precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_pred_proba)
                metrics['pr_auc'] = float(auc(recall_curve, precision_curve))

                # Confusion matrix counts
                tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
                metrics['true_positives'] = int(tp)
                metrics['false_positives'] = int(fp)
                metrics['true_negatives'] = int(tn)
                metrics['false_negatives'] = int(fn)
            else:
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0
                metrics['accuracy_binary'] = 0.0
                metrics['pr_auc'] = 0.0

        return metrics

    def print_results(self, results: Dict[str, Any]):
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)

        ex = results['exercise_metrics']
        print(f"\nEXERCISE CLASSIFICATION")
        print(f"  Accuracy: {ex['accuracy']:.2f}% | F1 (macro): {ex['f1_macro']:.4f}")
        for name, f1 in ex['f1_per_class'].items():
            print(f"    {name}: {f1:.4f}")

        ph = results['phase_metrics']
        print(f"\nPHASE DETECTION")
        print(f"  Accuracy: {ph['accuracy']:.2f}% | F1 (macro): {ph['f1_macro']:.4f}")

        rep = results['rep_metrics']
        print(f"\nREPETITION COUNTING")
        print(f"  MAE: {rep['mae']:.2f} | RMSE: {rep['rmse']:.2f} | R2: {rep['r2']:.4f}")
        if 'within_1_rep' in rep:
            print(f"  Within +/-1: {rep['within_1_rep']:.1f}% | Within +/-2: {rep['within_2_reps']:.1f}%")
        if 'f1' in rep:
            print(f"  Binary Classification:")
            print(f"    Precision: {rep['precision']:.4f} | Recall: {rep['recall']:.4f} | F1: {rep['f1']:.4f}")
            print(f"    PR-AUC: {rep['pr_auc']:.4f} | Accuracy: {rep['accuracy_binary']:.1f}%")
            if 'true_positives' in rep:
                print(f"    TP: {rep['true_positives']} | FP: {rep['false_positives']} | "
                      f"TN: {rep['true_negatives']} | FN: {rep['false_negatives']}")

        fat = results['fatigue_metrics']
        print(f"\nFATIGUE ESTIMATION")
        print(f"  MAE: {fat['mae']:.4f} | RMSE: {fat['rmse']:.4f} | R2: {fat['r2']:.4f}")
        print("="*70)

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics = {
            'exercise': {k: v for k, v in results['exercise_metrics'].items()
                        if k not in ['confusion_matrix', 'classification_report']},
            'phase': {k: v for k, v in results['phase_metrics'].items()
                     if k not in ['confusion_matrix', 'classification_report']},
            'repetition': results['rep_metrics'],
            'fatigue': results['fatigue_metrics']
        }

        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Results saved to {output_dir}")


class PlotGenerator:
    """Generate evaluation plots using seaborn."""

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.figsize = (10, 8)
        self.dpi = 150

    def plot_confusion_matrix(self, cm, class_names, title, output_path):
        """Plot confusion matrix with seaborn heatmap (replaces ~40 lines from v1)."""
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax
        )
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        fig.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_training_history(self, history, output_path):
        """Plot training history curves."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        plots = [
            (axes[0, 0], 'train_loss', 'val_loss', 'Loss', 'Training Loss'),
            (axes[0, 1], 'train_exercise_acc', 'val_exercise_acc', 'Accuracy (%)', 'Exercise Accuracy'),
            (axes[0, 2], 'train_phase_acc', 'val_phase_acc', 'Accuracy (%)', 'Phase Accuracy'),
            (axes[1, 0], 'train_rep_mae', 'val_rep_mae', 'MAE (reps)', 'Rep Count MAE'),
            (axes[1, 1], 'train_fatigue_mae', 'val_fatigue_mae', 'MAE', 'Fatigue MAE'),
        ]

        for ax, train_key, val_key, ylabel, title in plots:
            if train_key in history and history[train_key]:
                ax.plot(history[train_key], label='Train', linewidth=2)
                ax.plot(history[val_key], label='Validation', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        if 'learning_rates' in history and history['learning_rates']:
            ax.plot(history['learning_rates'], linewidth=2, color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_regression_analysis(self, y_true, y_pred, title, output_path):
        """Plot regression analysis with seaborn."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter
        axes[0].scatter(y_true, y_pred, alpha=0.5, edgecolors='none')
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                     'r--', linewidth=2, label='Perfect fit')
        axes[0].set(xlabel='True', ylabel='Predicted', title=f'{title}: Predicted vs True')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals
        residuals = y_pred - y_true
        axes[1].scatter(y_true, residuals, alpha=0.5, edgecolors='none')
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set(xlabel='True', ylabel='Residuals', title=f'{title}: Residuals')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_rep_detection_binary(self, y_true, y_pred, output_path):
        """
        Plot binary classification analysis for rep detection.

        Since rep counting is per-window binary (0 or 1), this shows:
        1. Confusion matrix (binary)
        2. Precision-Recall curve
        3. Prediction distribution
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Convert to binary: model outputs logits, apply sigmoid
        y_pred_proba = 1 / (1 + np.exp(-np.clip(y_pred, -500, 500)))
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
        y_true_binary = (y_true > 0).astype(int)

        # 1. Confusion Matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Rep', 'Rep'], yticklabels=['No Rep', 'Rep'],
            ax=axes[0]
        )
        axes[0].set_title('Rep Detection Confusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # Calculate and display metrics
        if len(np.unique(y_true_binary)) > 1:
            prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            acc = accuracy_score(y_true_binary, y_pred_binary)
            axes[0].text(0.5, -0.15, f'Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | Acc: {acc:.1%}',
                        transform=axes[0].transAxes, ha='center', fontsize=10)

        # 2. Precision-Recall Curve
        if len(np.unique(y_true_binary)) > 1:
            precision_curve, recall_curve, thresholds = precision_recall_curve(y_true_binary, y_pred_proba)
            pr_auc_score = auc(recall_curve, precision_curve)

            axes[1].plot(recall_curve, precision_curve, 'b-', linewidth=2, label=f'PR curve (AUC={pr_auc_score:.3f})')
            axes[1].axhline(y=np.mean(y_true_binary), color='r', linestyle='--', label='Random classifier')
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_title('Precision-Recall Curve')
            axes[1].legend(loc='lower left')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim([0, 1])
            axes[1].set_ylim([0, 1.05])
        else:
            axes[1].text(0.5, 0.5, 'Only one class in data', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Precision-Recall Curve')

        # 3. Prediction Distribution
        axes[2].hist(y_pred_proba[y_true_binary == 0], bins=30, alpha=0.6, label='No Rep (true)', color='blue')
        axes[2].hist(y_pred_proba[y_true_binary == 1], bins=30, alpha=0.6, label='Rep (true)', color='orange')
        axes[2].axvline(x=0.5, color='r', linestyle='--', linewidth=2, label='Threshold (0.5)')
        axes[2].set_xlabel('Predicted Probability')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Prediction Distribution by True Class')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def generate_all_plots(self, results, history, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if history:
            self.plot_training_history(history, output_dir / 'training_history.png')

        exercise_names = self.config.get_active_exercises() if hasattr(self.config, 'get_active_exercises') \
            else self.config.data.exercises
        self.plot_confusion_matrix(
            results['exercise_metrics']['confusion_matrix'],
            exercise_names,
            'Exercise Classification Confusion Matrix',
            output_dir / 'exercise_confusion_matrix.png'
        )
        self.plot_confusion_matrix(
            results['phase_metrics']['confusion_matrix'],
            ['Eccentric', 'Concentric'],
            'Phase Detection Confusion Matrix',
            output_dir / 'phase_confusion_matrix.png'
        )

        # Rep detection: use binary classification plot (confusion matrix + PR curve)
        # since rep counting is per-window binary (0 or 1)
        self.plot_rep_detection_binary(
            results['predictions']['rep_true'], results['predictions']['rep_pred'],
            output_dir / 'rep_detection_binary.png'
        )

        # Fatigue: keep regression plot (continuous 0-1 value)
        self.plot_regression_analysis(
            results['predictions']['fatigue_true'], results['predictions']['fatigue_pred'],
            'Fatigue Estimation', output_dir / 'fatigue_analysis.png'
        )
        print(f"Plots saved to {output_dir}")


def evaluate_model(
    model_path: Path,
    data_loader: DataLoader,
    config=None,
    output_dir: Path = None,
    history: Dict[str, List[float]] = None
) -> Dict[str, Any]:
    """Evaluate a trained model."""
    config = config or CONFIG
    output_dir = output_dir or config.output.results_dir

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    evaluator = ModelEvaluator(model, config)
    results = evaluator.evaluate(data_loader)
    evaluator.print_results(results)
    evaluator.save_results(results, output_dir)

    plot_gen = PlotGenerator(config)
    if history is None and 'history' in checkpoint:
        history = checkpoint['history']
    plot_gen.generate_all_plots(results, history, output_dir)

    return results
