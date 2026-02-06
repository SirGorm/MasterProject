"""
Logging utilities for Strength Training ML Pipeline v2.

Replaces the 261-line v1 logging_utils.py with loguru + tqdm + csv.DictWriter.
Same public API: setup_logging, get_logger, TrainingLogger.
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from loguru import logger
from tqdm import tqdm


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = 'INFO',
    verbose_console: bool = False
) -> Dict:
    """Setup logging with loguru. Returns dict of logger references (for API compat)."""
    from config import LOGS_DIR

    log_dir = Path(log_dir or LOGS_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Remove default handler and add custom ones
    logger.remove()

    # Console: minimal format
    if verbose_console:
        logger.add(sys.stdout, level="DEBUG",
                    format="<level>{level:<8}</level> | {message}")
    else:
        logger.add(sys.stdout, level=log_level,
                    format="{message}", filter=lambda r: r["level"].name == "INFO")
        logger.add(sys.stdout, level="WARNING",
                    format="[{level}] {message}")

    # File: detailed format
    logger.add(
        log_dir / f"{timestamp}_main.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}",
        encoding='utf-8'
    )

    return {'main': logger, 'train': logger, 'eval': logger,
            'data': logger, 'signals': logger, 'validation': logger}


def get_logger(name: str = 'main'):
    """Get logger instance (returns loguru logger, name is for API compat)."""
    return logger


class TrainingLogger:
    """Logger for training metrics. Uses csv.DictWriter instead of manual f-strings.

    Deprecated: Not used by the Lightning module (StrengthTrainingLitModule),
    which handles metric logging via Lightning's built-in self.log().
    Kept for backwards compatibility with non-Lightning training scripts.
    """

    FIELDS = [
        'Epoch', 'Train Loss', 'Val Loss',
        'Train Ex Acc', 'Val Ex Acc',
        'Train Phase Acc', 'Val Phase Acc',
        'Train Rep MAE', 'Val Rep MAE',
        'Train Fat MAE', 'Val Fat MAE'
    ]

    def __init__(self, log_file: Path = None):
        self.log_file = log_file
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_exercise_acc': [], 'val_exercise_acc': [],
            'train_phase_acc': [], 'val_phase_acc': [],
            'train_rep_mae': [], 'val_rep_mae': [],
            'train_fatigue_mae': [], 'val_fatigue_mae': []
        }

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        for key, value in train_metrics.items():
            history_key = f'train_{key}'
            if history_key in self.history:
                self.history[history_key].append(value)
        for key, value in val_metrics.items():
            history_key = f'val_{key}'
            if history_key in self.history:
                self.history[history_key].append(value)

        logger.info(f"Epoch {epoch}")
        logger.info(
            f"  Train | Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Ex Acc: {train_metrics.get('exercise_acc', 0):.1f}% | "
            f"Phase Acc: {train_metrics.get('phase_acc', 0):.1f}% | "
            f"Rep MAE: {train_metrics.get('rep_mae', 0):.2f} | "
            f"Fat MAE: {train_metrics.get('fatigue_mae', 0):.4f}"
        )
        logger.info(
            f"  Val   | Loss: {val_metrics.get('loss', 0):.4f} | "
            f"Ex Acc: {val_metrics.get('exercise_acc', 0):.1f}% | "
            f"Phase Acc: {val_metrics.get('phase_acc', 0):.1f}% | "
            f"Rep MAE: {val_metrics.get('rep_mae', 0):.2f} | "
            f"Fat MAE: {val_metrics.get('fatigue_mae', 0):.4f}"
        )

        if self.log_file:
            row = {
                'Epoch': epoch,
                'Train Loss': f"{train_metrics.get('loss', 0):.4f}",
                'Val Loss': f"{val_metrics.get('loss', 0):.4f}",
                'Train Ex Acc': f"{train_metrics.get('exercise_acc', 0):.2f}",
                'Val Ex Acc': f"{val_metrics.get('exercise_acc', 0):.2f}",
                'Train Phase Acc': f"{train_metrics.get('phase_acc', 0):.2f}",
                'Val Phase Acc': f"{val_metrics.get('phase_acc', 0):.2f}",
                'Train Rep MAE': f"{train_metrics.get('rep_mae', 0):.2f}",
                'Val Rep MAE': f"{val_metrics.get('rep_mae', 0):.2f}",
                'Train Fat MAE': f"{train_metrics.get('fatigue_mae', 0):.4f}",
                'Val Fat MAE': f"{val_metrics.get('fatigue_mae', 0):.4f}",
            }
            with open(self.log_file, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)

    def log_best_model(self, epoch: int, val_loss: float, save_path: Path):
        logger.info(f"  New best model! Val Loss: {val_loss:.4f} | Saved to: {save_path}")

    def log_early_stop(self, epoch: int, patience: int):
        logger.info(f"Early stopping triggered at epoch {epoch} (patience: {patience})")

    def get_history(self) -> Dict[str, list]:
        return self.history
