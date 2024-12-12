import logging
import os
from deepgram import DepgramClient, SpeakOptions

def setup_logging():
    """
    The `setup_logging` function configures logging for an application, setting
    up both console and file handlers with specific levels and formatting.
    :return: The `setup_logging` function returns a logger object that is
    configured to log messages to both the console and a file named "app.log".
    The logger is set to log messages at the INFO level for the console handler
    and at the DEBUG level for the file handler. The logger includes a
    formatter that specifies the format of the log messages.
    """
    logger = logging.getLogger(__name__)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler("app.log", maxBytes=10000000, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s -  %(levelname)s - %(message)s - Line: %(lineno)d"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()
