#!/usr/bin/env python3
"""
Main Entry Point for Strength Training ML Pipeline.

Usage:
    python main.py --epochs 50    # Train with 50 epochs
    python main.py --validate     # Only validate data
    python main.py --evaluate     # Evaluate existing model

From command line, only epochs can be specified.
All other parameters are in config/settings.py
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is in the path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Strength Training ML Pipeline - CNN-LSTM Multi-task Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --epochs 50          Train with 50 epochs
    python main.py --validate           Validate dataset only
    python main.py --evaluate           Evaluate trained model
    python main.py --epochs 100 --eval  Train and evaluate
        """
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (only parameter from CLI)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Only validate dataset (no training)'
    )

    parser.add_argument(
        '--evaluate', '--eval',
        action='store_true',
        help='Evaluate model after training (or existing model)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to existing model for evaluation'
    )

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print configuration summary and exit'
    )

    args = parser.parse_args()

    # Import after parsing to avoid slow imports for --help
    from config import CONFIG, set_epochs
    from utils import setup_logging, get_logger

    # Setup logging
    setup_logging(
        log_dir=CONFIG.output.logs_dir,
        log_level=CONFIG.output.log_level,
        verbose_console=CONFIG.output.verbose_console
    )
    logger = get_logger('main')

    # Print header
    print("\n" + "="*70)
    print("STRENGTH TRAINING ML PIPELINE")
    print("CNN-LSTM Multi-task Learning for Exercise Analysis")
    print("="*70)

    # Set epochs if provided
    if args.epochs is not None:
        set_epochs(args.epochs)
        logger.info(f"Epochs set to: {args.epochs}")

    # Print summary if requested
    if args.summary:
        CONFIG.print_summary()
        return 0

    # Validate configuration
    errors, warnings = CONFIG.validate()
    if errors:
        logger.error("Configuration validation failed!")
        for error in errors:
            logger.error(f"  - {error}")
        return 1

    if warnings:
        for warning in warnings:
            logger.warning(f"  - {warning}")

    # Data validation mode
    if args.validate:
        return run_validation()

    # Evaluate existing model
    if args.evaluate and args.model_path:
        return run_evaluation(args.model_path)

    # Full training pipeline
    return run_training(evaluate_after=args.evaluate)


def run_validation() -> int:
    """Run data validation only."""
    from data import validate_dataset
    from config import CONFIG

    print("\nRunning data validation...")

    try:
        passed = validate_dataset(
            CONFIG.data.dataset_path,
            stop_on_error=False
        )
        return 0 if passed else 1
    except Exception as e:
        print(f"Validation error: {e}")
        return 1


def run_training(evaluate_after: bool = False) -> int:
    """Run full training pipeline."""
    from config import CONFIG
    from data import DataValidator
    from data.dataset import create_dataloaders
    from training import train_model
    from evaluation import evaluate_model
    from utils import get_logger

    logger = get_logger('main')

    # Step 1: Validate data
    print("\n" + "="*70)
    print("STEP 1: DATA VALIDATION")
    print("="*70)

    validator = DataValidator(CONFIG.data.dataset_path, CONFIG)
    validation_passed = validator.validate_all()

    if not validation_passed:
        logger.error("Data validation failed: No usable sessions found")
        return 1

    # Get valid sessions to use for training
    valid_sessions = validator.get_valid_sessions()
    logger.info(f"Proceeding with {len(valid_sessions)} valid sessions")

    # Step 2: Load and preprocess data
    print("\n" + "="*70)
    print("STEP 2: DATA LOADING")
    print("="*70)

    try:
        train_loader, val_loader, scalers = create_dataloaders(
            CONFIG.data.dataset_path,
            CONFIG,
            valid_sessions=valid_sessions
        )
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Train model
    print("\n" + "="*70)
    print("STEP 3: TRAINING")
    print("="*70)

    CONFIG.print_summary()

    try:
        model, results = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            config=CONFIG
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Save scalers
    print("\n" + "="*70)
    print("STEP 4: SAVING ARTIFACTS")
    print("="*70)

    try:
        import pickle
        scaler_path = CONFIG.output.models_dir / CONFIG.output.scaler_filename
        with open(scaler_path, 'wb') as f:
            pickle.dump(scalers, f)
        logger.info(f"Scalers saved to: {scaler_path}")

        # Save config
        config_path = CONFIG.output.models_dir / 'config.json'
        CONFIG.save(config_path)
        logger.info(f"Config saved to: {config_path}")
    except Exception as e:
        logger.warning(f"Failed to save artifacts: {e}")

    # Step 5: Evaluate if requested
    if evaluate_after:
        print("\n" + "="*70)
        print("STEP 5: EVALUATION")
        print("="*70)

        model_path = CONFIG.output.models_dir / CONFIG.output.model_filename

        try:
            eval_results = evaluate_model(
                model_path=model_path,
                data_loader=val_loader,
                config=CONFIG,
                output_dir=CONFIG.output.results_dir,
                history=results.get('history')
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # Print final summary
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
    """Run evaluation on existing model."""
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

    # Load data
    try:
        _, val_loader, _ = create_dataloaders(
            CONFIG.data.dataset_path,
            CONFIG
        )
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return 1

    # Evaluate
    try:
        evaluate_model(
            model_path=model_path,
            data_loader=val_loader,
            config=CONFIG,
            output_dir=CONFIG.output.results_dir
        )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
