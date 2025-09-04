import logging
import os
import sys
from typing import Optional

_LOGGER: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER

    logger = logging.getLogger("iso55001")
    logger.propagate = False  # avoid duplicate through root/uvicorn
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.set_name("json_stream")
        # Minimal formatter so our message is just the JSON string
        handler.setFormatter(logging.Formatter("%(message)s"))
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        logger.addHandler(handler)

    _LOGGER = logger
    return logger

