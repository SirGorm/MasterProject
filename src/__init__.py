"""
Strength Training Multi-Task Model Package
==========================================
A comprehensive deep learning system for analyzing strength training exercises
using multi-modal sensor data.

Components:
- data_exploration: Dataset analysis and visualization
- preprocessing: Signal processing and feature extraction
- model: Multi-task neural network architecture
- training: Training pipeline and utilities
- evaluation: Model evaluation and inference

Usage:
    from src.data_exploration import DatasetExplorer
    from src.preprocessing import PreprocessingPipeline
    from src.model import MultiTaskStrengthModel, ModelConfig, create_model
    from src.training import train_model, TrainingConfig
    from src.evaluation import ModelEvaluator, RealTimeInference
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .data_exploration import DatasetExplorer, SessionData
from .preprocessing import (
    PreprocessingPipeline,
    ProcessedSegment,
    SignalProcessor,
    FeatureExtractor,
    PhaseDetector,
    create_dataset_tensors
)
from .model import (
    MultiTaskStrengthModel,
    ModelConfig,
    MultiTaskLoss,
    create_model,
    count_parameters
)
from .training import (
    TrainingConfig,
    StrengthDataset,
    Trainer,
    train_model,
    prepare_data_loaders
)
from .evaluation import (
    ModelEvaluator,
    EvaluationResults,
    RealTimeInference,
    load_model_for_inference
)

__all__ = [
    # Data exploration
    'DatasetExplorer',
    'SessionData',
    # Preprocessing
    'PreprocessingPipeline',
    'ProcessedSegment',
    'SignalProcessor',
    'FeatureExtractor',
    'PhaseDetector',
    'create_dataset_tensors',
    # Model
    'MultiTaskStrengthModel',
    'ModelConfig',
    'MultiTaskLoss',
    'create_model',
    'count_parameters',
    # Training
    'TrainingConfig',
    'StrengthDataset',
    'Trainer',
    'train_model',
    'prepare_data_loaders',
    # Evaluation
    'ModelEvaluator',
    'EvaluationResults',
    'RealTimeInference',
    'load_model_for_inference',
]
