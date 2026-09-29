# LOCKED: false
"""统一运行时使用的轻量日志封装。"""

from __future__ import annotations

import logging
from typing import Any


class my_logger:
    """保持 my_work 新增代码一致的具名 logger。"""

    # 数据：logger 名称和日志级别。算法：初始化标准库 logging 并保存具名实例。
    def __init__(self, name: str, level: int = logging.INFO) -> None:
        assert name.strip(), "logger name must not be empty"
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._logger = logging.getLogger(name)

    # 数据：info 消息与格式化参数。算法：交给标准库 logger 输出。
    def info(self, message: str, *args: Any) -> None:
        self._logger.info(message, *args)

    # 数据：warning 消息与格式化参数。算法：交给标准库 logger 输出。
    def warning(self, message: str, *args: Any) -> None:
        self._logger.warning(message, *args)

    # 数据：error 消息与格式化参数。算法：交给标准库 logger 输出。
    def error(self, message: str, *args: Any) -> None:
        self._logger.error(message, *args)
