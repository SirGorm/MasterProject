"""
Hyperparameter Search for Strength Training ML v2.

Uses Optuna + PyTorch Lightning.
"""

import copy

import optuna
from optuna.trial import Trial
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pathlib import Path
from typing import Dict, Any, Optional
import json

from ml.v3.config import CONFIG
from ml.v3.data.dataset import create_dataloaders
from ml.v3.training.lightning_module import StrengthTrainingLitModule


def objective(trial: Trial, n_epochs: int = 10) -> float:
    """Objective function for Optuna optimization."""
    config = copy.deepcopy(CONFIG)

    config.training.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    config.training.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    config.training.optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    config.training.weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    config.model.dropout = trial.suggest_float('dropout', 0.1, 0.5)

    config.training.loss_weights = {
        'exercise': trial.suggest_float('exercise_weight', 0.5, 2.0),
        'phase': trial.suggest_float('phase_weight', 0.5, 2.0),
        'reps': trial.suggest_float('rep_weight', 0.1, 1.0),
        'fatigue': trial.suggest_float('fatigue_weight', 0.1, 1.0),
    }

    try:
        train_loader, val_loader, _ = create_dataloaders(config=config)
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        raise optuna.TrialPruned()

    # Disable tracking for speed
    config.tracking.enabled = False

    lit_module = StrengthTrainingLitModule(config)

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator='auto',
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')],
        gradient_clip_val=config.training.gradient_clip_norm,
        logger=False,
    )

    try:
        trainer.fit(lit_module, train_loader, val_loader)
        val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
        val_loss = val_loss.item() if hasattr(val_loss, 'item') else float(val_loss)
    except Exception as e:
        print(f"Training failed: {e}")
        raise optuna.TrialPruned()

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
    """Run hyperparameter search."""
    if storage:
        storage = f"sqlite:///{storage}"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage,
        load_if_exists=load_if_exists,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3)
    )

    study.optimize(
        lambda trial: objective(trial, n_epochs=n_epochs_per_trial),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_value = study.best_value

    print(f"\nBest validation loss: {best_value:.4f}")
    print(f"Best hyperparameters: {best_params}")

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

    try:
        import optuna.visualization as vis
        vis.plot_param_importances(study).write_html(str(output_dir / "param_importances.html"))
        vis.plot_optimization_history(study).write_html(str(output_dir / "optimization_history.html"))
        vis.plot_parallel_coordinate(study).write_html(str(output_dir / "parallel_coordinate.html"))
    except ImportError:
        print("Install plotly for visualizations: pip install plotly")

    return results


def quick_search(n_trials: int = 20):
    return run_hyperparameter_search(n_trials=n_trials, n_epochs_per_trial=5, study_name="quick_search")


def full_search(n_trials: int = 100):
    return run_hyperparameter_search(
        n_trials=n_trials, n_epochs_per_trial=15,
        study_name="full_search", storage="optuna_study.db"
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter search (v2 Lightning)")
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick')
    parser.add_argument('--trials', type=int, default=None)
    args = parser.parse_args()

    if args.mode == 'quick':
        results = quick_search(args.trials or 20)
    else:
        results = full_search(args.trials or 100)
