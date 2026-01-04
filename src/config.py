from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BASE_DIR / "reports"

DEFAULT_INPUT = RAW_DIR / "sample.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "clean.csv"

SEED = 42
TEST_SIZE = 0.2
