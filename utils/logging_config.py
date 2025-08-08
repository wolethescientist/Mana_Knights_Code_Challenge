import logging
import logging.handlers
import json
import os
from datetime import datetime

class JsonFormatter(logging.Formatter):
    """
    Formats log records as JSON strings.
    """
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if hasattr(record, 'extra_data'):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

def setup_logging():
    """
    Sets up a comprehensive logging system with structured JSON output,
    multiple log files, and log rotation.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Set the lowest level to capture everything

    # Create a JSON formatter
    json_formatter = JsonFormatter()

    # Handlers for different log files
    
    # Application Log Handler (INFO and above)
    app_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "app.log"), maxBytes=5*1024*1024, backupCount=5
    )
    app_handler.setLevel(logging.INFO)
    app_handler.setFormatter(json_formatter)

    # Error Log Handler (ERROR and above)
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "error.log"), maxBytes=5*1024*1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter)

    # Performance Log Handler
    performance_logger = logging.getLogger('performance')
    performance_logger.setLevel(logging.INFO)
    performance_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "performance.log"), maxBytes=5*1024*1024, backupCount=5
    )
    performance_handler.setFormatter(json_formatter)
    performance_logger.addHandler(performance_handler)
    performance_logger.propagate = False
    
    # API Log Handler
    api_logger = logging.getLogger('api')
    api_logger.setLevel(logging.INFO)
    api_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "api.log"), maxBytes=5*1024*1024, backupCount=5
    )
    api_handler.setFormatter(json_formatter)
    api_logger.addHandler(api_handler)
    api_logger.propagate = False

    # Console Handler for development (DEBUG and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    # Add handlers to the root logger
    root_logger.addHandler(app_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

