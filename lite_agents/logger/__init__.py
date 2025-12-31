from __future__ import annotations

import logging
import sys

import loguru

def setup_logger_logging(
    name: str = __name__, datetime_level: bool = False
) -> logging.Logger:
    """Set up a standard logging logger.

    Args:
        name (str, optional): Name of the logger. Defaults to __name__.
        datetime_level (bool, optional): Whether to include datetime in the log format. Defaults to False.
    Returns:
        logging.Logger: Configured logging logger instance.
    """
    # Remove AWS Lambda's default logging handler
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if datetime_level is True:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s - %(filename)s - %(funcName)s - %(message)s",
        )

    return logging.getLogger(name)


def setup_logger(datetime_level: bool = False) -> loguru.Logger:
    """Change the level when calling logger.<level>(...) to get different colors.

    logurus levels are: TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL

    Args:
        datetime_level (bool, optional): Whether to include datetime in the log format. Defaults to False.
    Returns:
        loguru.Logger: Configured loguru logger instance.
    """

    loguru_format = [
        "<yellow>{time:YYYY-MM-DD HH:mm:ss} </yellow>",
        "<level>{level}</level>",
        "<ly>{module}</ly>",
        "<cyan>{function}:{line}</cyan>",
        "<level>{message}</level>",
    ]
    if not datetime_level:
        loguru_format = loguru_format[1:]

    logger = loguru.logger
    logger.remove()
    logger.add(sys.stderr, format=" | ".join(loguru_format))
    return logger
