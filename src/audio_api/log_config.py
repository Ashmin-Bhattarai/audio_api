import sys
from pathlib import Path
from loguru import logger

def setup_logging():
    """
    Configures Loguru to sink logs to both stderr and a rotating file.
    """
    # Create the logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Define the log file path
    log_file_path = log_dir / "audio_api.log"

    # Remove the default handler to prevent duplicate console logs
    logger.remove()

    # Add a handler for stderr (console) with a specific format and level
    # This is useful for development and seeing logs in Docker.
    logger.add(
        sys.stderr,
        level="INFO",  # Log INFO level and above to the console
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # Add a handler for the log file
    # This will create a new log file when it reaches 10 MB or at midnight.
    # It will keep up to 5 old log files.
    logger.add(
        log_file_path,
        level="DEBUG",  # Log DEBUG level and above to the file
        rotation="10 MB",  # Rotate the log file when it reaches 10 MB
        retention=5, # Keep up to 5 old log files
        enqueue=True,      # Make logging non-blocking (important for async)
        backtrace=True,    # Show full stack traces for exceptions
        diagnose=True,     # Add exception variable values for easier debugging
        format="{time} {level} {message}" # A simpler format for file logs
    )

    logger.info("Logging configured: Sinking to console (INFO) and file (DEBUG).")