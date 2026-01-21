"""Configuration file for Numerai EDA Dashboard"""

from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent.parent
TRAIN_DATA_PATH = ROOT / "data" / "v5.2" / "train.parquet"
FEATURES_JSON_PATH = ROOT / "data" / "v5.2" / "features.json"

FEATURE_SETS = {
    "small": None,
    "medium": None,
    "all": None,
}

APP_TITLE = "Numerai v5.2 - Analiza Eksploracyjna Danych"
APP_ICON = "📊"
LAYOUT = "wide"

FEATURE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
