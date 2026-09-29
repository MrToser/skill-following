# LOCKED: false
"""Logging boundary shared by Retool modules."""

from __future__ import annotations

import logging
from typing import Any


class my_logger:
    """Small logger wrapper kept consistent with other my_work release code."""

    # Data: logger name and level. Algorithm: initialize stdlib logging once and expose compact methods.
    def __init__(self, name: str = "retool_skill_generate", level: int = logging.INFO) -> None:
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._logger = logging.getLogger(name)

    # Data: info message. Algorithm: delegate to stdlib logger.
    def info(self, message: str, *args: Any) -> None:
        self._logger.info(message, *args)

    # Data: warning message. Algorithm: delegate to stdlib logger.
    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(message, *args)


LOGGER = my_logger()

