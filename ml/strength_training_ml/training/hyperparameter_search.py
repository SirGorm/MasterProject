"""
Hyperparameter Search for Strength Training ML.

Uses Optuna for efficient Bayesian optimization.
"""

import optuna
from optuna.trial import Trial
import torch
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG
from data.dataset import create_dataloaders
from models import create_model, MultiTaskLoss
from training.trainer import Trainer


def objective(trial: Trial, n_epochs: int = 10) -> float:
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        n_epochs: Number of epochs per trial (use fewer for faster search)

    Returns:
        Validation loss (lower is better)
    """
    # Sample hyperparameters
    config = CONFIG

    # Learning rate (log scale)
    config.training.learning_rate = trial.suggest_float(
        'learning_rate', 1e-5, 1e-2, log=True
    )

    # Batch size
    config.training.batch_size = trial.suggest_categorical(
        'batch_size', [8, 16, 32, 64]
    )

    # Optimizer
    config.training.optimizer = trial.suggest_categorical(
        'optimizer', ['adam', 'adamw']
    )

    # Weight decay
    config.training.weight_decay = trial.suggest_float(
        'weight_decay', 1e-6, 1e-2, log=True
    )

    # Model architecture
    config.model.hidden_dim = trial.suggest_categorical(
        'hidden_dim', [64, 128, 256]
    )

    config.model.lstm_hidden = trial.suggest_categorical(
        'lstm_hidden', [64, 128, 256]
    )

    config.model.lstm_layers = trial.suggest_int(
        'lstm_layers', 1, 3
    )

    config.model.dropout = trial.suggest_float(
        'dropout', 0.1, 0.5
    )

    # Loss weights
    config.training.exercise_weight = trial.suggest_float(
        'exercise_weight', 0.5, 2.0
    )

    config.training.phase_weight = trial.suggest_float(
        'phase_weight', 0.5, 2.0
    )

    config.training.rep_weight = trial.suggest_float(
        'rep_weight', 0.1, 1.0
    )

    config.training.fatigue_weight = trial.suggest_float(
        'fatigue_weight', 0.1, 1.0
    )

    # Create data loaders
    try:
        train_loader, val_loader, scalers = create_dataloaders(config=config)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise optuna.TrialPruned()

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(config).to(device)

    # Create trainer with reduced epochs
    original_epochs = config.training.n_epochs
    config.training.n_epochs = n_epochs
    config.tracking.enabled = False  # Disable tracking for speed

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    try:
        results = trainer.train()
        val_loss = results['best_val_loss']
    except Exception as e:
        print(f"Training failed: {e}")
        raise optuna.TrialPruned()
    finally:
        config.training.n_epochs = original_epochs
        config.tracking.enabled = True

    # Report intermediate values for pruning
    trial.report(val_loss, n_epochs)

    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_loss


def run_hyperparameter_search(
    n_trials: int = 50,
    n_epochs_per_trial: int = 10,
    study_name: str = "strength_training_optimization",
    storage: Optional[str] = None,
    load_if_exists: bool = True
) -> Dict[str, Any]:
    """
    Run hyperparameter search.

    Args:
        n_trials: Number of trials to run
        n_epochs_per_trial: Epochs per trial (fewer = faster search)
        study_name: Name for the study
        storage: SQLite database path (optional, for persistence)
        load_if_exists: Resume previous study if exists

    Returns:
        Dictionary with best parameters and study results
    """
    # Create study
    if storage:
        storage = f"sqlite:///{storage}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",  # Minimize validation loss
        storage=storage,
        load_if_exists=load_if_exists,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, n_epochs=n_epochs_per_trial),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # Get results
    best_params = study.best_params
    best_value = study.best_value

    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("="*60)

    # Save results
    output_dir = CONFIG.output.output_dir / "hyperparameter_search"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': len(study.trials),
        'study_name': study_name
    }

    with open(output_dir / f"{study_name}_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Generate visualization if optuna-dashboard or plotly available
    try:
        import optuna.visualization as vis

        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))

        # Parameter relationships
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(str(output_dir / "parallel_coordinate.html"))

        print(f"\nVisualizations saved to: {output_dir}")
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")

    return results


def apply_best_params(params: Dict[str, Any]):
    """
    Apply best hyperparameters to CONFIG.

    Args:
        params: Dictionary of hyperparameters from search
    """
    config = CONFIG

    # Training params
    if 'learning_rate' in params:
        config.training.learning_rate = params['learning_rate']
    if 'batch_size' in params:
        config.training.batch_size = params['batch_size']
    if 'optimizer' in params:
        config.training.optimizer = params['optimizer']
    if 'weight_decay' in params:
        config.training.weight_decay = params['weight_decay']

    # Model params
    if 'hidden_dim' in params:
        config.model.hidden_dim = params['hidden_dim']
    if 'lstm_hidden' in params:
        config.model.lstm_hidden = params['lstm_hidden']
    if 'lstm_layers' in params:
        config.model.lstm_layers = params['lstm_layers']
    if 'dropout' in params:
        config.model.dropout = params['dropout']

    # Loss weights
    if 'exercise_weight' in params:
        config.training.exercise_weight = params['exercise_weight']
    if 'phase_weight' in params:
        config.training.phase_weight = params['phase_weight']
    if 'rep_weight' in params:
        config.training.rep_weight = params['rep_weight']
    if 'fatigue_weight' in params:
        config.training.fatigue_weight = params['fatigue_weight']

    print("Applied best hyperparameters to CONFIG")


def quick_search(n_trials: int = 20):
    """
    Quick hyperparameter search with fewer trials and epochs.
    Good for initial exploration.
    """
    return run_hyperparameter_search(
        n_trials=n_trials,
        n_epochs_per_trial=5,
        study_name="quick_search"
    )


def full_search(n_trials: int = 100):
    """
    Full hyperparameter search with more trials.
    Use after quick_search to refine.
    """
    return run_hyperparameter_search(
        n_trials=n_trials,
        n_epochs_per_trial=15,
        study_name="full_search",
        storage="optuna_study.db"  # Persist for resume
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help="Search mode: quick (20 trials, 5 epochs) or full (100 trials, 15 epochs)")
    parser.add_argument('--trials', type=int, default=None,
                       help="Override number of trials")

    args = parser.parse_args()

    if args.mode == 'quick':
        n_trials = args.trials or 20
        results = quick_search(n_trials)
    else:
        n_trials = args.trials or 100
        results = full_search(n_trials)

    print("\nTo use the best parameters, run:")
    print("  from training.hyperparameter_search import apply_best_params")
    print(f"  apply_best_params({results['best_params']})")
