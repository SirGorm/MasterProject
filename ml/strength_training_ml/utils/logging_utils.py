"""
Logging utilities for Strength Training ML Pipeline.

Provides structured logging with separate handlers for:
- Console output (minimal, essential information only)
- File output (detailed logging for debugging)
- Signal-specific logs
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


class MinimalConsoleFormatter(logging.Formatter):
    """Minimal formatter for essential console output."""

    def format(self, record):
        # Only show message for INFO, add prefix for warnings/errors
        if record.levelno == logging.INFO:
            return record.getMessage()
        elif record.levelno == logging.WARNING:
            return f"[WARNING] {record.getMessage()}"
        elif record.levelno >= logging.ERROR:
            return f"[ERROR] {record.getMessage()}"
        return record.getMessage()


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = 'INFO',
    verbose_console: bool = False
) -> Dict[str, logging.Logger]:
    """
    Setup logging system with multiple loggers.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        verbose_console: If True, show detailed output in console

    Returns:
        Dictionary of loggers for different purposes
    """
    from config import LOGS_DIR

    log_dir = log_dir or LOGS_DIR
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define loggers
    loggers = {}
    logger_configs = {
        'main': 'main.log',
        'train': 'training.log',
        'eval': 'evaluation.log',
        'data': 'data.log',
        'signals': 'signals.log',
        'validation': 'validation.log'
    }

    for logger_name, log_filename in logger_configs.items():
        logger = logging.getLogger(f'strength_ml.{logger_name}')
        logger.setLevel(getattr(logging, log_level))
        logger.handlers = []  # Clear existing handlers

        # File handler - detailed output
        file_handler = logging.FileHandler(
            log_dir / f"{timestamp}_{log_filename}",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler - minimal output for main, train, eval
        if logger_name in ['main', 'train', 'eval']:
            console_handler = logging.StreamHandler(sys.stdout)
            if verbose_console:
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(ColorFormatter(
                    '%(levelname)s | %(message)s'
                ))
            else:
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(MinimalConsoleFormatter())
            logger.addHandler(console_handler)

        loggers[logger_name] = logger

    return loggers


def get_logger(name: str = 'main') -> logging.Logger:
    """
    Get a logger by name.

    Args:
        name: Logger name (main, train, eval, data, signals, validation)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(f'strength_ml.{name}')
    if not logger.handlers:
        # Logger not set up yet, create a basic one
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(MinimalConsoleFormatter())
        logger.addHandler(handler)
    return logger


class ProgressLogger:
    """Simple progress logger for training epochs."""

    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.current = 0
        self.desc = desc
        self.logger = get_logger('train')

    def update(self, metrics: Dict[str, float] = None):
        """Update progress with optional metrics."""
        self.current += 1
        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"{self.desc} [{self.current}/{self.total}] | {metrics_str}")
        else:
            self.logger.info(f"{self.desc} [{self.current}/{self.total}]")

    def finish(self, final_metrics: Dict[str, float] = None):
        """Mark progress as complete."""
        if final_metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in final_metrics.items()])
            self.logger.info(f"{self.desc} Complete | {metrics_str}")
        else:
            self.logger.info(f"{self.desc} Complete")


class TrainingLogger:
    """Logger specifically for training metrics."""

    def __init__(self, log_file: Path = None):
        self.logger = get_logger('train')
        self.log_file = log_file
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_exercise_acc': [],
            'val_exercise_acc': [],
            'train_phase_acc': [],
            'val_phase_acc': [],
            'train_rep_mae': [],
            'val_rep_mae': [],
            'train_fatigue_mae': [],
            'val_fatigue_mae': []
        }

        # Write CSV header if log file specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'w') as f:
                f.write("Epoch,Train Loss,Val Loss,Train Ex Acc,Val Ex Acc,"
                       "Train Phase Acc,Val Phase Acc,Train Rep MAE,Val Rep MAE,"
                       "Train Fat MAE,Val Fat MAE\n")

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log metrics for an epoch."""
        # Store in history
        for key, value in train_metrics.items():
            history_key = f'train_{key}'
            if history_key in self.history:
                self.history[history_key].append(value)

        for key, value in val_metrics.items():
            history_key = f'val_{key}'
            if history_key in self.history:
                self.history[history_key].append(value)

        # Log to console
        self.logger.info(f"Epoch {epoch}")
        self.logger.info(
            f"  Train | Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Ex Acc: {train_metrics.get('exercise_acc', 0):.1f}% | "
            f"Phase Acc: {train_metrics.get('phase_acc', 0):.1f}% | "
            f"Rep MAE: {train_metrics.get('rep_mae', 0):.2f} | "
            f"Fat MAE: {train_metrics.get('fatigue_mae', 0):.4f}"
        )
        self.logger.info(
            f"  Val   | Loss: {val_metrics.get('loss', 0):.4f} | "
            f"Ex Acc: {val_metrics.get('exercise_acc', 0):.1f}% | "
            f"Phase Acc: {val_metrics.get('phase_acc', 0):.1f}% | "
            f"Rep MAE: {val_metrics.get('rep_mae', 0):.2f} | "
            f"Fat MAE: {val_metrics.get('fatigue_mae', 0):.4f}"
        )

        # Write to CSV file
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(
                    f"{epoch},"
                    f"{train_metrics.get('loss', 0):.4f},"
                    f"{val_metrics.get('loss', 0):.4f},"
                    f"{train_metrics.get('exercise_acc', 0):.2f},"
                    f"{val_metrics.get('exercise_acc', 0):.2f},"
                    f"{train_metrics.get('phase_acc', 0):.2f},"
                    f"{val_metrics.get('phase_acc', 0):.2f},"
                    f"{train_metrics.get('rep_mae', 0):.2f},"
                    f"{val_metrics.get('rep_mae', 0):.2f},"
                    f"{train_metrics.get('fatigue_mae', 0):.4f},"
                    f"{val_metrics.get('fatigue_mae', 0):.4f}\n"
                )

    def log_best_model(self, epoch: int, val_loss: float, save_path: Path):
        """Log when a new best model is saved."""
        self.logger.info(f"  New best model! Val Loss: {val_loss:.4f} | Saved to: {save_path}")

    def log_early_stop(self, epoch: int, patience: int):
        """Log early stopping."""
        self.logger.info(f"Early stopping triggered at epoch {epoch} (patience: {patience})")

    def get_history(self) -> Dict[str, list]:
        """Get training history."""
        return self.history
