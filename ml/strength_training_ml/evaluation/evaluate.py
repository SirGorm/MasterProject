"""
Comprehensive Evaluation Module for Strength Training ML Pipeline.

Generates:
- Exercise classification metrics (Accuracy, F1, Confusion Matrix)
- Phase detection metrics (Accuracy, F1)
- Repetition counting metrics (MAE, Counting Error)
- Fatigue estimation metrics (MAE, MSE, R²)
- Training curves
- Evaluation plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
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


class ModelEvaluator:
    """
    Comprehensive model evaluator for multi-task strength training model.
    """

    def __init__(
        self,
        model: StrengthTrainingModel,
        config=None,
        device: str = None
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            config: Configuration object
            device: Device for evaluation
        """
        self.config = config or CONFIG
        self.model = model
        self.logger = get_logger('eval')

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Class names
        self.exercise_names = self.config.data.exercises
        self.phase_names = ['Eccentric', 'Concentric']

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Run full evaluation on a dataset.

        Args:
            data_loader: DataLoader to evaluate

        Returns:
            Dictionary with all predictions and metrics
        """
        # Collect predictions
        exercise_true = []
        exercise_pred = []
        phase_true = []
        phase_pred = []
        rep_true = []
        rep_pred = []
        fatigue_true = []
        fatigue_pred = []

        for signals, targets in data_loader:
            # Move to device
            signals = {k: v.to(self.device) for k, v in signals.items()}

            # Forward pass
            exercise_logits, phase_logits, rep_output, fatigue_output = self.model(signals)

            # Get predictions
            _, exercise_preds = exercise_logits.max(1)
            _, phase_preds = phase_logits.max(1)

            # Collect results
            exercise_true.extend(targets['exercise'].cpu().numpy())
            exercise_pred.extend(exercise_preds.cpu().numpy())
            phase_true.extend(targets['phase'].cpu().numpy())
            phase_pred.extend(phase_preds.cpu().numpy())
            rep_true.extend(targets['reps'].cpu().numpy())
            rep_pred.extend(rep_output.squeeze().cpu().numpy())
            fatigue_true.extend(targets['fatigue'].cpu().numpy())
            fatigue_pred.extend(fatigue_output.squeeze().cpu().numpy())

        # Convert to numpy arrays
        exercise_true = np.array(exercise_true)
        exercise_pred = np.array(exercise_pred)
        phase_true = np.array(phase_true)
        phase_pred = np.array(phase_pred)
        rep_true = np.array(rep_true)
        rep_pred = np.array(rep_pred)
        fatigue_true = np.array(fatigue_true)
        fatigue_pred = np.array(fatigue_pred)

        # Calculate metrics
        results = {
            'predictions': {
                'exercise_true': exercise_true,
                'exercise_pred': exercise_pred,
                'phase_true': phase_true,
                'phase_pred': phase_pred,
                'rep_true': rep_true,
                'rep_pred': rep_pred,
                'fatigue_true': fatigue_true,
                'fatigue_pred': fatigue_pred
            },
            'exercise_metrics': self._calculate_classification_metrics(
                exercise_true, exercise_pred, self.exercise_names, 'Exercise'
            ),
            'phase_metrics': self._calculate_classification_metrics(
                phase_true, phase_pred, self.phase_names, 'Phase'
            ),
            'rep_metrics': self._calculate_regression_metrics(
                rep_true, rep_pred, 'Repetition'
            ),
            'fatigue_metrics': self._calculate_regression_metrics(
                fatigue_true, fatigue_pred, 'Fatigue'
            )
        }

        return results

    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        task_name: str
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        # Get unique labels present in the data
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(class_names)

        # Create label list for all expected classes
        labels = list(range(n_classes))

        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred) * 100
        f1_macro = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)

        # Per-class F1
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

        # Confusion matrix with all labels
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Classification report with labels parameter
        try:
            report = classification_report(
                y_true, y_pred,
                labels=labels,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
        except ValueError:
            # Fallback if still fails
            report = {}
            for i, name in enumerate(class_names):
                report[name] = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(class_names, f1_per_class)),
            'confusion_matrix': cm,
            'classification_report': report,
            'n_samples': len(y_true),
            'unique_labels_in_data': unique_labels.tolist()
        }

    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_name: str
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        # R² score (handle edge cases)
        if np.var(y_true) > 0:
            r2 = r2_score(y_true, y_pred)
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
        else:
            r2 = 0.0
            correlation = 0.0

        # Additional metrics for repetition counting
        if task_name == 'Repetition':
            # Counting accuracy (within ±1)
            within_1 = np.sum(np.abs(y_true - y_pred) <= 1) / len(y_true) * 100
            within_2 = np.sum(np.abs(y_true - y_pred) <= 2) / len(y_true) * 100

            return {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2 if not np.isnan(r2) else 0.0,
                'correlation': correlation if not np.isnan(correlation) else 0.0,
                'within_1_rep': within_1,
                'within_2_reps': within_2,
                'n_samples': len(y_true)
            }

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2 if not np.isnan(r2) else 0.0,
            'correlation': correlation if not np.isnan(correlation) else 0.0,
            'n_samples': len(y_true)
        }

    def print_results(self, results: Dict[str, Any]):
        """Print evaluation results to console."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)

        # Exercise Classification
        ex = results['exercise_metrics']
        print("\n" + "-"*70)
        print("EXERCISE CLASSIFICATION")
        print("-"*70)
        print(f"  Accuracy:        {ex['accuracy']:.2f}%")
        print(f"  F1 Score (macro): {ex['f1_macro']:.4f}")
        print(f"  F1 Score (weighted): {ex['f1_weighted']:.4f}")
        print("\n  Per-class F1:")
        for name, f1 in ex['f1_per_class'].items():
            print(f"    {name}: {f1:.4f}")

        # Phase Detection
        ph = results['phase_metrics']
        print("\n" + "-"*70)
        print("PHASE DETECTION (Eccentric/Concentric)")
        print("-"*70)
        print(f"  Accuracy:        {ph['accuracy']:.2f}%")
        print(f"  F1 Score (macro): {ph['f1_macro']:.4f}")

        # Repetition Counting
        rep = results['rep_metrics']
        print("\n" + "-"*70)
        print("REPETITION COUNTING")
        print("-"*70)
        print(f"  MAE:             {rep['mae']:.2f} reps")
        print(f"  RMSE:            {rep['rmse']:.2f} reps")
        print(f"  R²:              {rep['r2']:.4f}")
        if 'within_1_rep' in rep:
            print(f"  Within ±1 rep:   {rep['within_1_rep']:.1f}%")
            print(f"  Within ±2 reps:  {rep['within_2_reps']:.1f}%")

        # Fatigue Estimation
        fat = results['fatigue_metrics']
        print("\n" + "-"*70)
        print("FATIGUE ESTIMATION")
        print("-"*70)
        print(f"  MAE:             {fat['mae']:.4f}")
        print(f"  MSE:             {fat['mse']:.4f}")
        print(f"  RMSE:            {fat['rmse']:.4f}")
        print(f"  R²:              {fat['r2']:.4f}")
        print(f"  Correlation:     {fat['correlation']:.4f}")

        print("\n" + "="*70)

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics = {
            'exercise': {k: v for k, v in results['exercise_metrics'].items()
                        if k not in ['confusion_matrix', 'classification_report']},
            'phase': {k: v for k, v in results['phase_metrics'].items()
                     if k not in ['confusion_matrix', 'classification_report']},
            'repetition': results['rep_metrics'],
            'fatigue': results['fatigue_metrics']
        }

        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        metrics = convert_numpy(metrics)

        with open(output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save detailed text report
        with open(output_dir / 'evaluation_report.txt', 'w') as f:
            f.write("="*70 + "\n")
            f.write("EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")

            f.write("EXERCISE CLASSIFICATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy: {results['exercise_metrics']['accuracy']:.2f}%\n")
            f.write(f"F1 Score (macro): {results['exercise_metrics']['f1_macro']:.4f}\n\n")

            f.write("PHASE DETECTION\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy: {results['phase_metrics']['accuracy']:.2f}%\n")
            f.write(f"F1 Score (macro): {results['phase_metrics']['f1_macro']:.4f}\n\n")

            f.write("REPETITION COUNTING\n")
            f.write("-"*40 + "\n")
            f.write(f"MAE: {results['rep_metrics']['mae']:.2f} reps\n")
            f.write(f"RMSE: {results['rep_metrics']['rmse']:.2f} reps\n\n")

            f.write("FATIGUE ESTIMATION\n")
            f.write("-"*40 + "\n")
            f.write(f"MAE: {results['fatigue_metrics']['mae']:.4f}\n")
            f.write(f"R²: {results['fatigue_metrics']['r2']:.4f}\n")

        self.logger.info(f"Results saved to {output_dir}")


class PlotGenerator:
    """
    Generate evaluation plots.
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.figsize = (10, 8)
        self.dpi = 150

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str,
        output_path: Path
    ):
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=self.figsize)

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel='True Label',
            xlabel='Predicted Label'
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        output_path: Path
    ):
        """Plot training history curves."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Loss
        ax = axes[0, 0]
        ax.plot(history['train_loss'], label='Train', linewidth=2)
        ax.plot(history['val_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Exercise Accuracy
        ax = axes[0, 1]
        ax.plot(history['train_exercise_acc'], label='Train', linewidth=2)
        ax.plot(history['val_exercise_acc'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Exercise Classification Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Phase Accuracy
        ax = axes[0, 2]
        ax.plot(history['train_phase_acc'], label='Train', linewidth=2)
        ax.plot(history['val_phase_acc'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Phase Detection Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Rep MAE
        ax = axes[1, 0]
        ax.plot(history['train_rep_mae'], label='Train', linewidth=2)
        ax.plot(history['val_rep_mae'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (reps)')
        ax.set_title('Repetition Count MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Fatigue MAE
        ax = axes[1, 1]
        ax.plot(history['train_fatigue_mae'], label='Train', linewidth=2)
        ax.plot(history['val_fatigue_mae'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Fatigue Estimation MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Learning Rate
        ax = axes[1, 2]
        if 'learning_rates' in history:
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

    def plot_regression_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str,
        output_path: Path
    ):
        """Plot regression analysis (scatter + residuals)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='none')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
               'r--', linewidth=2, label='Perfect fit')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{title}: Predicted vs True')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residual plot
        ax = axes[1]
        residuals = y_pred - y_true
        ax.scatter(y_true, residuals, alpha=0.5, edgecolors='none')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Residuals (Pred - True)')
        ax.set_title(f'{title}: Residuals')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

    def generate_all_plots(
        self,
        results: Dict[str, Any],
        history: Dict[str, List[float]],
        output_dir: Path
    ):
        """Generate all evaluation plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        if history:
            self.plot_training_history(
                history,
                output_dir / 'training_history.png'
            )

        # Exercise confusion matrix
        self.plot_confusion_matrix(
            results['exercise_metrics']['confusion_matrix'],
            self.config.data.exercises,
            'Exercise Classification Confusion Matrix',
            output_dir / 'exercise_confusion_matrix.png'
        )

        # Phase confusion matrix
        self.plot_confusion_matrix(
            results['phase_metrics']['confusion_matrix'],
            ['Eccentric', 'Concentric'],
            'Phase Detection Confusion Matrix',
            output_dir / 'phase_confusion_matrix.png'
        )

        # Repetition regression
        self.plot_regression_analysis(
            results['predictions']['rep_true'],
            results['predictions']['rep_pred'],
            'Repetition Count',
            output_dir / 'repetition_analysis.png'
        )

        # Fatigue regression
        self.plot_regression_analysis(
            results['predictions']['fatigue_true'],
            results['predictions']['fatigue_pred'],
            'Fatigue Estimation',
            output_dir / 'fatigue_analysis.png'
        )

        print(f"Plots saved to {output_dir}")


def evaluate_model(
    model_path: Path,
    data_loader: DataLoader,
    config=None,
    output_dir: Path = None,
    history: Dict[str, List[float]] = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model.

    Args:
        model_path: Path to saved model checkpoint
        data_loader: DataLoader for evaluation
        config: Configuration object
        output_dir: Directory for output files
        history: Training history for plotting

    Returns:
        Evaluation results
    """
    config = config or CONFIG
    output_dir = output_dir or config.output.results_dir

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create evaluator
    evaluator = ModelEvaluator(model, config)

    # Run evaluation
    results = evaluator.evaluate(data_loader)

    # Print results
    evaluator.print_results(results)

    # Save results
    evaluator.save_results(results, output_dir)

    # Generate plots
    plot_gen = PlotGenerator(config)

    # Get history from checkpoint if not provided
    if history is None and 'history' in checkpoint:
        history = checkpoint['history']

    plot_gen.generate_all_plots(results, history, output_dir)

    return results


def run_standalone_evaluation(model_path: Path = None):
    """
    Run evaluation as a standalone script.

    Automatically loads data and evaluates the model.

    Args:
        model_path: Path to model checkpoint. If None, uses default from config.
    """
    from data import DataValidator
    from data.dataset import create_dataloaders

    print("\n" + "="*70)
    print("STANDALONE MODEL EVALUATION")
    print("="*70)

    # Determine model path
    if model_path is None:
        model_path = CONFIG.output.models_dir / CONFIG.output.model_filename
    else:
        model_path = Path(model_path)

    print(f"\nModel path: {model_path}")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Train a model first with: python main.py --epochs 50")
        return None

    # Validate and load data
    print("\nLoading data...")
    validator = DataValidator(CONFIG.data.dataset_path, CONFIG)
    validation_passed = validator.validate_all()

    if not validation_passed:
        print("ERROR: No valid sessions found in dataset")
        return None

    valid_sessions = validator.get_valid_sessions()
    print(f"Found {len(valid_sessions)} valid sessions")

    # Create dataloaders
    try:
        _, val_loader, _ = create_dataloaders(
            CONFIG.data.dataset_path,
            CONFIG,
            valid_sessions=valid_sessions
        )
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return None

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_model(
        model_path=model_path,
        data_loader=val_loader,
        config=CONFIG,
        output_dir=CONFIG.output.results_dir
    )

    print(f"\nResults saved to: {CONFIG.output.results_dir}")
    print(f"Plots saved to: {CONFIG.output.results_dir}")

    return results


if __name__ == "__main__":
    import sys

    # Check if arguments were provided
    if len(sys.argv) > 1 and sys.argv[1] not in ['-h', '--help']:
        # Model path provided as argument
        model_path = Path(sys.argv[1])
        run_standalone_evaluation(model_path)
    else:
        # No arguments - use defaults
        run_standalone_evaluation()
