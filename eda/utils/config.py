"""Configuration for Numerai EDA Dashboard."""

import os
from pathlib import Path

DATASET_VERSION = os.environ.get("ALPHAPULSE_DATASET_VERSION", "v5.2")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(
    os.environ.get("ALPHAPULSE_DATA_DIR", str(ROOT / "data" / DATASET_VERSION))
)
TRAIN_DATA_PATH = DATA_DIR / "train.parquet"
FEATURES_JSON_PATH = DATA_DIR / "features.json"

APP_TITLE = f"Numerai {DATASET_VERSION} - Analiza Eksploracyjna Danych"
APP_ICON = "📊"
LAYOUT = "wide"

FEATURE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
