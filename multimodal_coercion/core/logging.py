import logging
import sys


def setup_logging(level="INFO"):
    logger = logging.getLogger()
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger

