"""Data module for Strength Training ML Pipeline v2."""

from .validate_data import (
    DataValidator, validate_dataset, ValidationResult, SessionValidation,
)
from .preprocessing import (
    SignalPreprocessor, JointProcessor, DataPreprocessor,
    preprocess_dataset, WindowedSignal, ExtractedFeatures,
)
from .phase_clustering import (
    ClusteringPhaseDetector, PhaseResult, train_phase_detector,
)

__all__ = [
    'DataValidator', 'validate_dataset', 'ValidationResult', 'SessionValidation',
    'SignalPreprocessor', 'JointProcessor', 'DataPreprocessor',
    'preprocess_dataset', 'WindowedSignal', 'ExtractedFeatures',
    'ClusteringPhaseDetector', 'PhaseResult', 'train_phase_detector',
]
