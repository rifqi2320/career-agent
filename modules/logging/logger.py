"""Logger setup helpers for per-level file logging."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import logging

_LEVEL_FILE_NAMES: dict[int, str] = {
    logging.DEBUG: "debug",
    logging.INFO: "info",
    logging.WARNING: "warning",
    logging.ERROR: "error",
    logging.CRITICAL: "critical",
}


class _ExactLevelFilter(logging.Filter):
    """Allow only records with the exact configured level."""

    def __init__(self, *, level: int) -> None:
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self._level


def _backup_path(base_path: Path, timestamp: str) -> Path:
    """Return a unique backup path for an existing level log file."""
    stem = base_path.stem
    suffix = base_path.suffix
    candidate = base_path.with_name(f"{stem}.{timestamp}{suffix}")
    index = 1
    while candidate.exists():
        candidate = base_path.with_name(f"{stem}.{timestamp}.{index}{suffix}")
        index += 1
    return candidate


def _rotate_existing_level_logs(*, logs_dir: Path, timestamp: str) -> None:
    """Move existing level logs to timestamped backups."""
    for level_name in _LEVEL_FILE_NAMES.values():
        log_path = logs_dir / f"{level_name}.log"
        if not log_path.exists():
            continue
        destination = _backup_path(log_path, timestamp)
        log_path.replace(destination)


def configure_level_file_logger(
    *,
    name: str = "career-agent",
    logs_dir: str | Path = "logs",
    level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure a logger that writes each level to `logs/{level}.log`."""
    logger = logging.getLogger(name)
    if getattr(logger, "_is_level_file_logger_configured", False):
        return logger

    destination_dir = Path(logs_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    _rotate_existing_level_logs(logs_dir=destination_dir, timestamp=timestamp)

    for existing_handler in logger.handlers.copy():
        logger.removeHandler(existing_handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for level_number, level_name in _LEVEL_FILE_NAMES.items():
        file_handler = logging.FileHandler(
            filename=destination_dir / f"{level_name}.log",
            encoding="utf-8",
            mode="a",
        )
        file_handler.setLevel(level_number)
        file_handler.addFilter(_ExactLevelFilter(level=level_number))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level)
    logger.propagate = False
    setattr(logger, "_is_level_file_logger_configured", True)
    return logger
