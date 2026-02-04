#!/usr/bin/env python3
"""
Main Entry Point for Strength Training ML Pipeline v2.

Changes from v1:
- Uses PyTorch Lightning Trainer (replaces manual training loop)
- Uses Lightning callbacks for early stopping, checkpointing, LR monitoring
- Uses loguru for logging
- Same CLI interface

Usage:
    python main.py --epochs 50
    python main.py --validate
    python main.py --evaluate
"""

import argparse
import sys
import pickle
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor
)

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description='Strength Training ML Pipeline v2 (Lightning)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --epochs 50          Train with 50 epochs
    python main.py --validate           Validate dataset only
    python main.py --evaluate           Evaluate trained model
    python main.py --epochs 100 --eval  Train and evaluate
        """
    )
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--evaluate', '--eval', action='store_true')
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--summary', action='store_true')
    args = parser.parse_args()

    from config import CONFIG, set_epochs
    from utils import setup_logging, get_logger

    setup_logging(
        log_dir=CONFIG.output.logs_dir,
        log_level=CONFIG.output.log_level,
        verbose_console=CONFIG.output.verbose_console
    )
    logger = get_logger('main')

    print("\n" + "="*70)
    print("STRENGTH TRAINING ML PIPELINE v2 (Lightning)")
    print("="*70)

    if args.epochs is not None:
        set_epochs(args.epochs)
        logger.info(f"Epochs set to: {args.epochs}")

    if args.summary:
        CONFIG.print_summary()
        return 0

    errors, warnings = CONFIG.validate()
    if errors:
        for error in errors:
            logger.error(f"  - {error}")
        return 1
    for warning in warnings:
        logger.warning(f"  - {warning}")

    if args.validate:
        return run_validation()
    if args.evaluate and args.model_path:
        return run_evaluation(args.model_path)
    return run_training(evaluate_after=args.evaluate)


def run_validation() -> int:
    from data import validate_dataset
    from config import CONFIG
    print("\nRunning data validation...")
    try:
        passed = validate_dataset(CONFIG.data.dataset_path, stop_on_error=False)
        return 0 if passed else 1
    except Exception as e:
        print(f"Validation error: {e}")
        return 1


def run_training(evaluate_after: bool = False) -> int:
    from config import CONFIG
    from data import DataValidator
    from data.dataset import create_dataloaders
    from training import StrengthTrainingLitModule
    from evaluation import evaluate_model
    from utils import get_logger

    logger = get_logger('main')

    # Step 1: Validate data
    print("\n" + "="*70)
    print("STEP 1: DATA VALIDATION")
    print("="*70)

    validator = DataValidator(CONFIG.data.dataset_path, CONFIG)
    if not validator.validate_all():
        logger.error("Data validation failed: No usable sessions found")
        return 1

    valid_sessions = validator.get_valid_sessions()
    logger.info(f"Proceeding with {len(valid_sessions)} valid sessions")

    # Step 2: Load data
    print("\n" + "="*70)
    print("STEP 2: DATA LOADING")
    print("="*70)

    try:
        train_loader, val_loader, scalers = create_dataloaders(
            CONFIG.data.dataset_path, CONFIG, valid_sessions=valid_sessions
        )
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        import traceback; traceback.print_exc()
        return 1

    # Step 3: Train with Lightning
    print("\n" + "="*70)
    print("STEP 3: TRAINING (Lightning)")
    print("="*70)

    CONFIG.print_summary()

    lit_module = StrengthTrainingLitModule(CONFIG)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG.training.early_stopping_patience,
            mode='min',
            verbose=True
        ),
        ModelCheckpoint(
            dirpath=str(CONFIG.output.models_dir),
            filename='best_model',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            verbose=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    trainer = pl.Trainer(
        max_epochs=CONFIG.training.n_epochs,
        accelerator='auto',
        callbacks=callbacks,
        gradient_clip_val=CONFIG.training.gradient_clip_norm,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    try:
        trainer.fit(lit_module, train_loader, val_loader)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback; traceback.print_exc()
        return 1

    # Step 4: Save artifacts
    print("\n" + "="*70)
    print("STEP 4: SAVING ARTIFACTS")
    print("="*70)

    # Save model in v1-compatible format for evaluate_model()
    save_path = CONFIG.output.models_dir / CONFIG.output.model_filename
    checkpoint = {
        'model_state_dict': lit_module.model.state_dict(),
        'config': CONFIG.to_dict(),
    }
    import torch
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to: {save_path}")

    try:
        scaler_path = CONFIG.output.models_dir / CONFIG.output.scaler_filename
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        logger.info(f"Scalers saved to: {scaler_path}")

        config_path = CONFIG.output.models_dir / 'config.json'
        CONFIG.save(config_path)
        logger.info(f"Config saved to: {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save artifacts: {e}")

    # Save prediction tracking data
    if lit_module.tracker and CONFIG.tracking.auto_save:
        lit_module.save_tracking_data()

    # Step 5: Evaluate
    if evaluate_after:
        print("\n" + "="*70)
        print("STEP 5: EVALUATION")
        print("="*70)

        try:
            evaluate_model(
                model_path=save_path,
                data_loader=val_loader,
                config=CONFIG,
                output_dir=CONFIG.output.results_dir,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Model saved to: {CONFIG.output.models_dir}")
    print(f"Logs saved to: {CONFIG.output.logs_dir}")
    if evaluate_after:
        print(f"Results saved to: {CONFIG.output.results_dir}")
    print("="*70)

    return 0


def run_evaluation(model_path: str) -> int:
    from config import CONFIG
    from data.dataset import create_dataloaders
    from evaluation import evaluate_model
    from utils import get_logger

    logger = get_logger('main')
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1

    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    try:
        _, val_loader, _ = create_dataloaders(CONFIG.data.dataset_path, CONFIG)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return 1

    try:
        evaluate_model(
            model_path=model_path, data_loader=val_loader,
            config=CONFIG, output_dir=CONFIG.output.results_dir
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback; traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
