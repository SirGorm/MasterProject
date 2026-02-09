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
    mean_squared_error, r2_score, ConfusionMatrixDisplay
)
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from torch.utils.data import DataLoader
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from models import StrengthTrainingModel, create_model
from utils import get_logger

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
        self.plot_regression_analysis(
            results['predictions']['rep_true'], results['predictions']['rep_pred'],
            'Repetition Count', output_dir / 'repetition_analysis.png'
        )
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

    checkpoint = torch.load(model_path, map_location='cpu')
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
