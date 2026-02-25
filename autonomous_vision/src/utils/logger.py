"""
logger.py — Structured logging with colored console output.
"""

import logging
import sys
from pathlib import Path

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""

    if HAS_COLORAMA:
        COLORS = {
            logging.DEBUG:    Fore.CYAN,
            logging.INFO:     Fore.GREEN,
            logging.WARNING:  Fore.YELLOW,
            logging.ERROR:    Fore.RED,
            logging.CRITICAL: Fore.RED + Style.BRIGHT,
        }
        RESET = Style.RESET_ALL
    else:
        COLORS = {}
        RESET = ""

    ICONS = {
        logging.DEBUG:    "🔍",
        logging.INFO:     "✅",
        logging.WARNING:  "⚠️",
        logging.ERROR:    "❌",
        logging.CRITICAL: "🔴",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        icon = self.ICONS.get(record.levelno, "")
        record.msg = f"{color}{icon} {record.msg}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "autonomous_vision",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """
    Set up a logger with colored console output and optional file output.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).
        log_file: Optional path to log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = ColoredFormatter(
        fmt="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (plain text, no colors)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger
