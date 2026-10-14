"""
Central logging configuration for the ANPR platform.

Writes to both the console (so `docker logs` shows it) and a rotating
log file under LOG_DIR (default: ./logs/app.log) so the file can be
mounted as a Docker volume and tailed / shipped to a monitoring system.
"""
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

os.makedirs(LOG_DIR, exist_ok=True)

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging() -> logging.Logger:
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. reloaded by uvicorn --reload)
        return logging.getLogger("anpr")

    root.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger("uvicorn.access").handlers = [console_handler, file_handler]

    return logging.getLogger("anpr")


logger = setup_logging()
