"""
Centralized structured logging module for the project.
"""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "covid_xray", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured standard library logger.

    :param name: Logger name (typically __name__).
    :param level: Logging level (default INFO).
    :return: Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger
