"""
Strength Training ML Pipeline

A modular CNN-LSTM multi-task learning system for real-time strength training analysis.

Features:
- Exercise Classification (Squat, Bench Press, Pull-ups)
- Phase Detection (Eccentric/Concentric)
- Repetition Counting
- Fatigue Estimation

Usage:
    python main.py --epochs 50    # Train with 50 epochs
    python main.py --validate     # Validate data only
    python main.py --evaluate     # Evaluate existing model
"""

__version__ = "1.0.0"
__author__ = "Master Thesis Project"

from config import CONFIG, get_config, set_epochs
from models import StrengthTrainingModel, create_model
from training import Trainer, train_model
from evaluation import ModelEvaluator, evaluate_model

__all__ = [
    'CONFIG',
    'get_config',
    'set_epochs',
    'StrengthTrainingModel',
    'create_model',
    'Trainer',
    'train_model',
    'ModelEvaluator',
    'evaluate_model',
]
