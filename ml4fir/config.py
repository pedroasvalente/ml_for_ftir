import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
RESULTS_DIR = PROJ_ROOT / "results"
EXPERIMENTS_DIR = PROJ_ROOT / "experiments"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAINING_DATA_FILENAME = os.environ.get("TRAINING_DATA_FILENAME", "training_data.csv")
PROCESSED_TRAINING_DATA_FILEPATH = PROCESSED_DATA_DIR / TRAINING_DATA_FILENAME

random_seed = int(os.environ.get("RANDOM_SEED", 52))
global_threshold = int(os.environ.get("GLOBAL_THRESHOLD", 70))
main_metric = str(os.environ.get("MAIN_METRIC", "acc"))

roc_plot_path = os.path.join(FIGURES_DIR, "ROC")
confusion_matrix_plot_path = os.path.join(FIGURES_DIR, "Confusion_Matrix")
principal_wavenumber_path = os.path.join(FIGURES_DIR, "Principal_Wavenumber")

MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"
TESTS_DIR = PROJ_ROOT / "tests"

os.makedirs(roc_plot_path, exist_ok=True)
os.makedirs(confusion_matrix_plot_path, exist_ok=True)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass


try:
    import mlflow

    mlflow_logger = logging.getLogger("mlflow")

    # Create a custom handler to redirect MLflow logs to Loguru
    class LoguruHandler(logging.Handler):
        def emit(self, record):
            # Convert the LogRecord to a Loguru-compatible message
            log_entry = self.format(record)
            level = record.levelname.lower()
            mlflow_logging_level = os.environ.get("MLFLOW_LOGGING_LEVEL", None)
            if mlflow_logging_level is not None:
                if level == "warning":
                    logger.warning(log_entry)
                elif level == "error":
                    logger.error(log_entry)
                elif level == "critical":
                    logger.critical(log_entry)
                elif level == "info":
                    logger.info(log_entry)
                elif level == "debug":
                    logger.debug(log_entry)
                else:
                    logger.info(log_entry)
                # mlflow_logger.setLevel(logging.DEBUG)  # Set the desired logging level

    # Attach the custom handler to the MLflow logger
    mlflow_logger.addHandler(LoguruHandler())
    # Remove all default handlers from the MLflow logger
    for handler in mlflow_logger.handlers[:]:
        if not isinstance(handler, LoguruHandler):  # Keep only the LoguruHandler
            mlflow_logger.removeHandler(handler)


except ModuleNotFoundError:
    pass
