"""
Central logging setup with rotation to keep log sizes bounded.
"""
import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(
    log_dir: str = None,
    filename: str = "rl-coordinator.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    level: int = logging.INFO,
):
    """Configure root logger with a RotatingFileHandler and console handler.

    - log_dir: directory to write logs into (created if missing). If None, only console logging is configured.
    - filename: log file name within log_dir
    - max_bytes: rotate when file exceeds this size
    - backup_count: number of rotated files to keep
    - level: logging level
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if setup is called multiple times
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s'))
        root.addHandler(console)

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, filename)
            rotating = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=backup_count)
            rotating.setLevel(level)
            rotating.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s'))
            root.addHandler(rotating)
        except Exception as e:
            # Fall back silently to console-only logging
            logging.getLogger(__name__).warning(f"Failed to setup rotating file logs: {e}")


