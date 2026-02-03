"""
Data module for Strength Training ML Pipeline.
"""

from .validate_data import (
    DataValidator,
    validate_dataset,
    ValidationResult,
    SessionValidation,
)

from .preprocessing import (
    SignalPreprocessor,
    JointProcessor,
    DataPreprocessor,
    preprocess_dataset,
    WindowedSignal,
    ExtractedFeatures,
)

from .phase_clustering import (
    ClusteringPhaseDetector,
    PhaseResult,
    train_phase_detector,
)

__all__ = [
    # Validation
    'DataValidator',
    'validate_dataset',
    'ValidationResult',
    'SessionValidation',

    # Preprocessing
    'SignalPreprocessor',
    'JointProcessor',
    'DataPreprocessor',
    'preprocess_dataset',
    'WindowedSignal',
    'ExtractedFeatures',

    # Phase Detection
    'ClusteringPhaseDetector',
    'PhaseResult',
    'train_phase_detector',
]
