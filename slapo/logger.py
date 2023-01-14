# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logging."""
import sys
import logging
from logging import getLevelName

import torch.distributed as dist

FORMATTER = logging.Formatter(
    "[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setFormatter(FORMATTER)

LOGGER_TABLE = {}

# Syntax suger.
CRITICAL = logging.CRITICAL
FATAL = logging.FATAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = logging.WARN
INFO = logging.INFO
DEBUG = logging.DEBUG
NOTSET = logging.NOTSET


def get_logger(name="Slapo", level=INFO):
    """Attach to the default logger."""
    if name in LOGGER_TABLE:
        logger = LOGGER_TABLE[name]
        if logger.level != level:
            logger.warning(
                f"Logger {name} already exists with {getLevelName(logger.level)}. "
                f"The new level {getLevelName(level)} will be ignored."
            )
        return logger

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)
    orig_log = logger._log

    def wrapper(level, msg, *args, **kwargs):
        """Log when distributed is not initialized or when the rank is in the list.
        Note that ranks=None means all ranks.
        """

        ranks = kwargs.pop("ranks", None)
        group = kwargs.pop("group", None)

        # Always log when distributed is not initialized or ranks are not specified.
        should_log = True
        if dist.is_initialized():
            my_rank = dist.get_rank(group)
            rank_info = f"[Rank {my_rank}] "
            if ranks is not None:
                # Only log when the current rank is in the list.
                ranks = ranks if isinstance(ranks, (list, tuple)) else [ranks]
                should_log = my_rank in set(ranks)
        else:
            rank_info = ""

        if should_log:
            orig_log(
                level,
                f"{rank_info}{msg}",
                *args,
                **kwargs,
            )

    logger._log = wrapper
    LOGGER_TABLE[name] = logger
    return logger
