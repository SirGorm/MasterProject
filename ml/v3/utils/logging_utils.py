"""
Logging utilities for Strength Training ML Pipeline v3.

Uses loguru for logging. TrainingLogger removed as Lightning handles
metric logging via built-in self.log().
"""
import sys
from typing import Dict, Optional
from pathlib import Path    
from datetime import datetime
from loguru import logger
def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: tuple = ('INFO', 'ERROR'),
    verbose_console: bool = False
) -> Dict:
    """Setup logging with loguru. Returns dict of logger references (for API compat)."""
    
    

    #from loguru import logger
    from ml.v3.config import LOGS_DIR

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
                    format="{message}", filter=lambda r: r["level"].name in log_level)
        logger.add(sys.stdout, level="WARNING",
                    format="[{level}] {message}")

    # File: detailed format
    logger.add(
        log_dir / f"{timestamp}_main.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name} | {message}",
        encoding='utf-8'
    )

    return {'main': logger, 'train': logger, 'eval': logger, 'phase': logger,
            'data': logger, 'signals': logger, 'preprocessing': logger}


def get_logger(name: str = 'main'):
    """Get logger instance (returns loguru logger, name is for API compat)."""
    from loguru import logger
    return logger.bind(name=name)
