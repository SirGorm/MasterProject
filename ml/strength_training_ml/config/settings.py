"""
Centralized Configuration Module for Strength Training ML Pipeline.

All paths, hyperparameters, signal configurations, and sampling rates are defined here.
Only epochs can be overridden via command line.

Author: Master Thesis Project
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json


# =============================================================================
# BASE PATHS - Adjust these for your system
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Dataset path - CHANGE THIS to your dataset location
DATASET_PATH = Path("C:\MasterProject\VS_Camera\Test_modify\dataset")

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = OUTPUT_DIR / "logs"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"


# =============================================================================
# SIGNAL CONFIGURATION
# =============================================================================

@dataclass
class SignalConfig:
    """Configuration for a single signal modality."""
    name: str
    sampling_rate: float
    channels: int
    file_name: str
    enabled: bool = True

    def samples_per_window(self, window_sec: float) -> int:
        """Calculate number of samples for a given time window."""
        return int(self.sampling_rate * window_sec)


# Signal definitions with sampling rates
SIGNALS: Dict[str, SignalConfig] = {
    'emg': SignalConfig(
        name='emg',
        sampling_rate=2000.0,
        channels=1,
        file_name='biopoint_emg.csv'
    ),
    'ecg': SignalConfig(
        name='ecg',
        sampling_rate=500.0,
        channels=1,
        file_name='biopoint_ecg.csv'
    ),
    'eda': SignalConfig(
        name='eda',
        sampling_rate=50.0,
        channels=1,
        file_name='biopoint_eda.csv'
    ),
    'acc': SignalConfig(
        name='acc',
        sampling_rate=50.0,
        channels=1,
        file_name='biopoint_a_combined.csv'
    ),
    'ppg_ir': SignalConfig(
        name='ppg_ir',
        sampling_rate=50.0,
        channels=1,
        file_name='biopoint_ppg_ir.csv'
    ),
    'ppg_red': SignalConfig(
        name='ppg_red',
        sampling_rate=50.0,
        channels=1,
        file_name='biopoint_ppg_red.csv'
    ),
    'ppg_green': SignalConfig(
        name='ppg_green',
        sampling_rate=50.0,
        channels=1,
        file_name='biopoint_ppg_green.csv'
    ),
    'ppg_blue': SignalConfig(
        name='ppg_blue',
        sampling_rate=50.0,
        channels=1,
        file_name='biopoint_ppg_blue.csv'
    ),
    # NOTE: joints is DISABLED as model input - used only for ground truth labels
    'joints': SignalConfig(
        name='joints',
        sampling_rate=30.0,
        channels=96,  # 32 joints x 3 coordinates
        file_name='joint_data.json',
        enabled=False  # Ground truth only, not model input
    )
}


# =============================================================================
# TASK-SIGNAL MAPPING
# =============================================================================

# Which signals are used for each prediction task
# NOTE: joint_data is used for GROUND TRUTH labels only (phase detection), NOT as model input
TASK_SIGNALS: Dict[str, List[str]] = {
    # Exercise classification uses movement + physiological data
    'exercise': ['emg', 'ecg', 'acc', 'ppg_ir'],

    # Repetition counting uses movement-based signals
    'reps': ['acc', 'emg'],

    # Phase detection uses movement + muscle signals (ground truth from joint_data)
    'phase': ['acc', 'emg'],

    # Fatigue estimation uses physiological indicators
    'fatigue': ['emg', 'ecg', 'ppg_ir', 'eda']
}


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    # Dataset settings
    dataset_path: Path = DATASET_PATH
    exercises: List[str] = field(default_factory=lambda: ['Squat', 'Benchpress', 'Pullups'])

    # Time windowing (in seconds, not samples)
    time_window_sec: float = 2.0
    overlap: float = 0.5  # 50% overlap for sliding windows

    # Train/validation split
    train_val_split: float = 0.2
    random_seed: int = 42

    # JSON files
    markers_file: str = 'markers.json'
    joints_file: str = 'joint_data.json'

    # Data augmentation
    augmentation_enabled: bool = False
    noise_factor: float = 0.02
    time_shift_max: int = 50
    scaling_factor: float = 0.1


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Neural network architecture configuration."""

    # CNN configuration
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 128])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [5, 5, 3])

    # LSTM configuration
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True

    # Fusion configuration
    fusion_dim: int = 256
    attention_heads: int = 4
    attention_dropout: float = 0.1

    # General
    dropout: float = 0.3

    # Output heads
    n_exercises: int = 3
    n_phases: int = 2  # Eccentric, Concentric


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Basic training
    n_epochs: int = 50  # This can be overridden via CLI
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 1e-5

    # Optimizer
    optimizer: str = 'adamw'

    # Early stopping
    early_stopping_patience: int = 10

    # Learning rate scheduler
    scheduler_type: str = 'plateau'
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_lr: float = 1e-6

    # Gradient clipping
    gradient_clip_norm: float = 1.0

    # Multi-task loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'exercise': 1.0,
        'reps': 0.5,
        'phase': 1.0,
        'fatigue': 1.0
    })

    # Device
    device: str = 'auto'  # 'auto', 'cuda', or 'cpu'


# =============================================================================
# PREPROCESSING CONFIGURATION (NeuroKit2)
# =============================================================================

@dataclass
class PreprocessingConfig:
    """NeuroKit2-based preprocessing configuration."""

    # EMG preprocessing
    emg_lowcut: float = 20.0
    emg_highcut: float = 450.0
    emg_method: str = 'biosppy'

    # ECG preprocessing
    ecg_method: str = 'neurokit'
    ecg_clean_method: str = 'neurokit'

    # EDA preprocessing
    eda_method: str = 'neurokit'

    # PPG preprocessing
    ppg_method: str = 'elgendi'

    # IMU/Accelerometer preprocessing
    acc_lowcut: float = 0.5
    acc_highcut: float = 20.0

    # Feature extraction
    extract_hrv_features: bool = True
    extract_emg_frequency_features: bool = True

    # Raw signal + features combination
    use_raw_signals: bool = True
    use_extracted_features: bool = True


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""

    # Which evaluations to run
    evaluate_exercise: bool = True
    evaluate_phase: bool = True
    evaluate_reps: bool = True
    evaluate_fatigue: bool = True

    # Plot types to generate
    plot_types: List[str] = field(default_factory=lambda: [
        'training_history',
        'confusion_matrix',
        'phase_confusion_matrix',
        'rep_counting_analysis',
        'fatigue_regression',
        'attention_weights'
    ])


# =============================================================================
# OUTPUT CONFIGURATION
# =============================================================================

@dataclass
class OutputConfig:
    """Output paths and logging configuration."""

    # Directories
    output_dir: Path = OUTPUT_DIR
    logs_dir: Path = LOGS_DIR
    results_dir: Path = RESULTS_DIR
    plots_dir: Path = PLOTS_DIR
    models_dir: Path = MODELS_DIR

    # File names
    model_filename: str = 'best_model.pth'
    scaler_filename: str = 'scalers.pkl'
    metadata_filename: str = 'metadata.json'
    training_log_filename: str = 'training.log'

    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_to_console: bool = True
    verbose_console: bool = False  # Only show essential info in terminal

    def ensure_directories(self):
        """Create output directories if they don't exist."""
        for dir_path in [self.output_dir, self.logs_dir, self.results_dir,
                         self.plots_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FATIGUE CONFIGURATION
# =============================================================================

@dataclass
class FatigueConfig:
    """Fatigue detection configuration."""

    # Window for fatigue analysis (different from main window)
    fatigue_window_sec: float = 20.0

    # EMG fatigue indicators
    emg_window_samples: int = 1000  # For frequency analysis

    # Fatigue output type
    continuous_output: bool = True  # True = regression, False = classification
    n_fatigue_levels: int = 3  # Only used if continuous_output = False

    # Thresholds for classification
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low_moderate': 0.33,
        'moderate_high': 0.67
    })


# =============================================================================
# MASTER CONFIGURATION CLASS
# =============================================================================

@dataclass
class Config:
    """Master configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    fatigue: FatigueConfig = field(default_factory=FatigueConfig)

    # Signal configurations (copied from module level)
    signals: Dict[str, SignalConfig] = field(default_factory=lambda: SIGNALS.copy())
    task_signals: Dict[str, List[str]] = field(default_factory=lambda: TASK_SIGNALS.copy())

    def __post_init__(self):
        """Initialize and validate configuration."""
        self.output.ensure_directories()

    def get_device(self) -> str:
        """Get the device to use for training."""
        import torch
        if self.training.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.training.device

    def get_signal_config(self, signal_name: str) -> Optional[SignalConfig]:
        """Get configuration for a specific signal."""
        return self.signals.get(signal_name)

    def get_enabled_signals(self) -> List[str]:
        """Get list of enabled signal names."""
        return [name for name, cfg in self.signals.items() if cfg.enabled]

    def validate(self) -> bool:
        """Validate configuration settings."""
        errors = []
        warnings = []

        # Check dataset path exists
        if not self.data.dataset_path.exists():
            errors.append(f"Dataset path does not exist: {self.data.dataset_path}")

        # Check signals used in tasks are defined
        for task, signal_list in self.task_signals.items():
            for signal_name in signal_list:
                if signal_name not in self.signals:
                    errors.append(f"Task '{task}' uses undefined signal '{signal_name}'")

        # Check time window
        if self.data.time_window_sec <= 0:
            errors.append("time_window_sec must be positive")

        # Check sampling rates
        for signal_name, signal_cfg in self.signals.items():
            if signal_cfg.sampling_rate <= 0:
                errors.append(f"Signal '{signal_name}' has invalid sampling rate")

        # Warnings
        if self.data.time_window_sec < 2.0:
            warnings.append("Very short time window may limit temporal modeling")

        if self.training.batch_size > 32:
            warnings.append("Large batch size may affect training stability")

        if self.training.learning_rate > 0.01:
            warnings.append("High learning rate may cause training instability")

        return errors, warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        from dataclasses import asdict

        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, SignalConfig):
                return {
                    'name': obj.name,
                    'sampling_rate': obj.sampling_rate,
                    'channels': obj.channels,
                    'file_name': obj.file_name,
                    'enabled': obj.enabled
                }
            return obj

        result = {}
        for field_name in ['data', 'model', 'training', 'preprocessing',
                          'evaluation', 'output', 'fatigue']:
            field_value = getattr(self, field_name)
            result[field_name] = asdict(field_value)
            # Convert Path objects
            for key, value in result[field_name].items():
                result[field_name][key] = convert(value)

        result['signals'] = {k: convert(v) for k, v in self.signals.items()}
        result['task_signals'] = self.task_signals

        return result

    def save(self, filepath: Path):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: Path) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct config
        config = cls()

        # Update data config
        for key, value in data.get('data', {}).items():
            if hasattr(config.data, key):
                if key == 'dataset_path':
                    value = Path(value)
                setattr(config.data, key, value)

        # Update training config
        for key, value in data.get('training', {}).items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)

        return config

    def print_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "="*70)
        print("CONFIGURATION SUMMARY")
        print("="*70)

        print(f"\nDataset: {self.data.dataset_path}")
        print(f"Exercises: {', '.join(self.data.exercises)}")

        print(f"\nTime Window: {self.data.time_window_sec} seconds")
        print(f"Overlap: {self.data.overlap * 100}%")

        print("\nSignals Configured:")
        for name, cfg in self.signals.items():
            if cfg.enabled:
                samples = cfg.samples_per_window(self.data.time_window_sec)
                print(f"  {name:12s}: {cfg.sampling_rate:>7.1f} Hz, "
                      f"{cfg.channels} ch, {samples:>6d} samples/window")

        print("\nTask-Signal Mapping:")
        for task, signals in self.task_signals.items():
            print(f"  {task:10s}: {', '.join(signals)}")

        print(f"\nModel Architecture:")
        print(f"  CNN filters: {self.model.cnn_filters}")
        print(f"  LSTM hidden: {self.model.lstm_hidden_size}")
        print(f"  LSTM layers: {self.model.lstm_num_layers}")
        print(f"  Bidirectional: {self.model.lstm_bidirectional}")
        print(f"  Fusion dim: {self.model.fusion_dim}")
        print(f"  Attention heads: {self.model.attention_heads}")

        print(f"\nTraining:")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Epochs: {self.training.n_epochs}")
        print(f"  Early stopping: {self.training.early_stopping_patience}")
        print(f"  Device: {self.get_device()}")

        print("="*70)


# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================

# Create default configuration
CONFIG = Config()


def get_config() -> Config:
    """Get the default configuration instance."""
    return CONFIG


def set_epochs(n_epochs: int):
    """Set number of training epochs (CLI override)."""
    CONFIG.training.n_epochs = n_epochs
