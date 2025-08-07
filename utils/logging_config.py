import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, 'app.log')

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a rotating file handler
    handler = RotatingFileHandler(
        log_file_path, maxBytes=10*1024*1024, backupCount=5
    )

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
    )
    handler.setFormatter(formatter)

    # Create a stream handler to log to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add both handlers to the logger
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(stream_handler)

    logging.info(f"Logging configured. Log file at: {os.path.abspath(log_file_path)}")
