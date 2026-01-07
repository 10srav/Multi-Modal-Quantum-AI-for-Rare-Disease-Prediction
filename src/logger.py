"""
Structured logging configuration for HGPS Prediction System.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    name: str = "hgps",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    json_output: bool = False,
) -> logging.Logger:
    """
    Set up logging with console and file handlers.

    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        json_output: Use JSON format for file logs

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(logging.DEBUG)

        if json_output:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s"
                )
            )

        logger.addHandler(file_handler)

        # Error log file
        error_handler = logging.FileHandler(log_dir / f"{name}_errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s\n%(exc_info)s"
            )
        )
        logger.addHandler(error_handler)

    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with extra context."""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str, **context) -> logging.LoggerAdapter:
    """
    Get a logger with optional context.

    Args:
        name: Logger name
        **context: Additional context to include in logs

    Returns:
        Logger adapter with context
    """
    logger = logging.getLogger(f"hgps.{name}")
    return LoggerAdapter(logger, context)


# Initialize default logger
from src.config import settings

default_logger = setup_logging(
    name="hgps",
    level=settings.log_level,
    log_dir=settings.paths.logs_dir,
    json_output=settings.is_production(),
)
