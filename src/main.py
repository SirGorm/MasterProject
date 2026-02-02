"""
Main Entry Point for Strength Training Multi-Task Model
========================================================
This script provides a complete workflow for:
1. Data exploration and analysis
2. Preprocessing and feature extraction
3. Model training
4. Evaluation and visualization

Usage:
    python main.py --mode explore    # Explore and analyze dataset
    python main.py --mode train      # Train the model
    python main.py --mode evaluate   # Evaluate trained model
    python main.py --mode all        # Run complete pipeline
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_exploration import DatasetExplorer
from preprocessing import PreprocessingPipeline, create_dataset_tensors
from model import ModelConfig, create_model, count_parameters
from training import TrainingConfig, train_model, prepare_data_loaders, Trainer, StrengthDataset
from evaluation import ModelEvaluator, EvaluationResults, RealTimeInference


def explore_dataset(dataset_path: str, output_dir: str = 'analysis'):
    """
    Run comprehensive dataset exploration.

    Args:
        dataset_path: Path to dataset
        output_dir: Directory for output files
    """
    print("=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize explorer
    explorer = DatasetExplorer(dataset_path)
    explorer.discover_exercises()
    explorer.load_all_sessions()

    # Generate report
    report = explorer.generate_report()
    print(report)

    # Save report
    report_path = output_path / 'dataset_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Plot examples from each exercise
    for exercise in explorer.exercise_types:
        sessions = [s for s in explorer.sessions if s.exercise == exercise]
        if sessions:
            best_session = max(sessions, key=lambda s: len(s.markers))
            explorer.plot_signal_examples(
                best_session,
                save_path=str(output_path / f'signals_{exercise.lower()}.png')
            )

    # Cross-exercise comparison
    explorer.plot_exercise_comparison(
        save_path=str(output_path / 'exercise_comparison.png')
    )

    # Fatigue analysis
    fatigue_df = explorer.analyze_fatigue_indicators()
    fatigue_df.to_csv(output_path / 'fatigue_indicators.csv', index=False)
    print(f"Fatigue analysis saved to {output_path / 'fatigue_indicators.csv'}")

    # Statistics
    stats_df = explorer.compute_statistics()
    stats_df.to_csv(output_path / 'dataset_statistics.csv', index=False)
    print(f"Statistics saved to {output_path / 'dataset_statistics.csv'}")

    print("\nExploration complete!")

    return explorer


def train_pipeline(dataset_path: str,
                   model_config: ModelConfig = None,
                   training_config: TrainingConfig = None,
                   output_dir: str = 'output'):
    """
    Run the complete training pipeline.

    Args:
        dataset_path: Path to dataset
        model_config: Model configuration
        training_config: Training configuration
        output_dir: Directory for output files
    """
    print("=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default configurations
    if model_config is None:
        model_config = ModelConfig(
            n_channels=10,
            seq_length=256,
            n_features=80,
            backbone_type='hybrid',
            hidden_dim=128,
            n_layers=4,
            n_heads=4,
            dropout=0.2,
            n_exercises=3,  # Squat, Benchpress, Pullups
            fatigue_latent_dim=32
        )

    if training_config is None:
        training_config = TrainingConfig(
            batch_size=8,
            val_split=0.2,
            test_split=0.1,
            epochs=100,
            learning_rate=1e-3,
            weight_decay=1e-4,
            patience=15,
            use_augmentation=True,
            checkpoint_dir=str(output_path / 'checkpoints'),
            log_dir=str(output_path / 'logs')
        )

    # Load and preprocess data
    print("\nLoading dataset...")
    explorer = DatasetExplorer(dataset_path)
    explorer.discover_exercises()
    explorer.load_all_sessions()

    print("\nPreprocessing data...")
    pipeline = PreprocessingPipeline(target_length=model_config.seq_length)

    all_segments = []
    for session in explorer.sessions:
        segments = pipeline.process_session(session)
        all_segments.extend(segments)

    print(f"Total segments: {len(all_segments)}")

    # Update model config based on data
    if all_segments:
        model_config.n_features = len(all_segments[0].features)
        # Update n_exercises based on actual data
        unique_exercises = set(seg.labels['exercise'] for seg in all_segments)
        model_config.n_exercises = len(unique_exercises)

    print(f"\nModel configuration:")
    print(f"  Channels: {model_config.n_channels}")
    print(f"  Sequence length: {model_config.seq_length}")
    print(f"  Features: {model_config.n_features}")
    print(f"  Exercises: {model_config.n_exercises}")
    print(f"  Backbone: {model_config.backbone_type}")

    # Create data loaders
    from torch.utils.data import DataLoader

    train_loader, val_loader, test_loader = prepare_data_loaders(
        all_segments, training_config
    )

    # Create model
    model = create_model(model_config)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, training_config, test_loader)
    history = trainer.train(verbose=True)

    # Save training config
    config_path = output_path / 'config.json'
    config_data = {
        'model_config': {
            'n_channels': model_config.n_channels,
            'seq_length': model_config.seq_length,
            'n_features': model_config.n_features,
            'backbone_type': model_config.backbone_type,
            'hidden_dim': model_config.hidden_dim,
            'n_layers': model_config.n_layers,
            'n_heads': model_config.n_heads,
            'dropout': model_config.dropout,
            'n_exercises': model_config.n_exercises,
            'fatigue_latent_dim': model_config.fatigue_latent_dim,
        },
        'training_config': {
            'batch_size': training_config.batch_size,
            'epochs': training_config.epochs,
            'learning_rate': training_config.learning_rate,
            'weight_decay': training_config.weight_decay,
        },
        'dataset': {
            'total_segments': len(all_segments),
            'exercises': list(set(seg.exercise for seg in all_segments)),
        },
        'training_date': datetime.now().isoformat(),
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    # Plot training history
    trainer.plot_history(save_path=str(output_path / 'training_history.png'))

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {training_config.checkpoint_dir}")

    return model, trainer, test_loader


def evaluate_model(model, test_loader, output_dir: str = 'output',
                   device: str = 'cpu', history: Optional[Dict] = None):
    """
    Run comprehensive model evaluation with all visualizations.

    Args:
        model: Trained model
        test_loader: Test data loader
        output_dir: Directory for output files
        device: Device for evaluation
        history: Optional training history for visualization
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exercise_names = ['Squat', 'Benchpress', 'Pullups']

    # Use the comprehensive evaluation function
    from evaluation import run_full_evaluation, TrainingVisualizer
    from visualizations import PerformanceDashboard

    # Run full evaluation (generates all plots)
    results = run_full_evaluation(
        model=model,
        data_loader=test_loader,
        history=history,
        output_dir=str(output_path),
        exercise_names=exercise_names
    )

    # Additional training dashboard if history is provided
    if history:
        print("\nGenerating training dashboard...")
        PerformanceDashboard.create_training_dashboard(
            history,
            save_path=str(output_path / 'training_dashboard.png')
        )

    print("\nEvaluation complete!")
    print(f"Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Exercise Accuracy:     {results.exercise_accuracy:.1%}")
    print(f"  Exercise Macro F1:     {results.exercise_f1_macro:.4f}")
    print(f"  Phase Accuracy:        {results.phase_accuracy:.1%}")
    print(f"  Phase F1:              {results.phase_f1:.4f}")
    print(f"  Phase IoU:             {results.phase_iou:.4f}")
    print(f"  Rep Position MAE:      {results.rep_position_mae:.4f}")
    print(f"  Rep Position R²:       {results.rep_position_r2:.4f}")
    print(f"  Fatigue Correlation:   {results.fatigue_correlation_with_rep:.4f}")
    print("=" * 60)

    return results


def run_full_pipeline(dataset_path: str, output_dir: str = 'output'):
    """
    Run the complete pipeline: explore, train, evaluate.

    Args:
        dataset_path: Path to dataset
        output_dir: Directory for all output files
    """
    print("=" * 60)
    print("FULL PIPELINE: EXPLORATION -> TRAINING -> EVALUATION")
    print("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Explore
    print("\n" + "=" * 40)
    print("STEP 1: DATA EXPLORATION")
    print("=" * 40)
    explorer = explore_dataset(dataset_path, str(output_path / 'analysis'))

    # Step 2: Train
    print("\n" + "=" * 40)
    print("STEP 2: MODEL TRAINING")
    print("=" * 40)
    model, trainer, test_loader = train_pipeline(
        dataset_path,
        output_dir=str(output_path)
    )

    # Load best model for evaluation
    trainer.load_checkpoint('best_model.pt')

    # Step 3: Evaluate
    print("\n" + "=" * 40)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 40)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Get training history from trainer
    history = dict(trainer.history) if hasattr(trainer, 'history') else None

    results = evaluate_model(model, test_loader, str(output_path), device, history)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_path}")

    # List generated files
    print("\nGenerated files:")
    for f in sorted(output_path.rglob('*')):
        if f.is_file():
            print(f"  - {f.relative_to(output_path)}")

    return explorer, model, results


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Strength Training Multi-Task Model Pipeline'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['explore', 'train', 'evaluate', 'all'],
        default='all',
        help='Pipeline mode: explore, train, evaluate, or all'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=r'C:\MasterProject\VS_Camera\Test_modify\claude\dataset',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (for evaluate mode)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size'
    )
    parser.add_argument(
        '--backbone',
        type=str,
        choices=['transformer', 'cnn_lstm', 'hybrid'],
        default='hybrid',
        help='Model backbone type'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.mode == 'explore':
        explore_dataset(args.dataset, str(output_path / 'analysis'))

    elif args.mode == 'train':
        model_config = ModelConfig(
            backbone_type=args.backbone
        )
        training_config = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_dir=str(output_path / 'checkpoints'),
            log_dir=str(output_path / 'logs')
        )
        train_pipeline(args.dataset, model_config, training_config, str(output_path))

    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            checkpoint = output_path / 'checkpoints' / 'best_model.pt'
        else:
            checkpoint = Path(args.checkpoint)

        if not checkpoint.exists():
            print(f"Error: Checkpoint not found at {checkpoint}")
            print("Please train a model first or provide a valid checkpoint path.")
            return

        # Load model config from checkpoint or use default
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = ModelConfig(backbone_type=args.backbone)

        from evaluation import load_model_for_inference
        model = load_model_for_inference(str(checkpoint), config, device)

        # Load test data
        explorer = DatasetExplorer(args.dataset)
        explorer.discover_exercises()
        explorer.load_all_sessions()

        pipeline = PreprocessingPipeline(target_length=config.seq_length)
        all_segments = []
        for session in explorer.sessions:
            segments = pipeline.process_session(session)
            all_segments.extend(segments)

        training_config = TrainingConfig()
        _, _, test_loader = prepare_data_loaders(all_segments, training_config)

        evaluate_model(model, test_loader, str(output_path), device)

    elif args.mode == 'all':
        run_full_pipeline(args.dataset, str(output_path))


if __name__ == "__main__":
    main()
