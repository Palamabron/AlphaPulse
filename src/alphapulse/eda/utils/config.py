"""Configuration file for Numerai EDA Dashboard"""

from pathlib import Path

# Data paths
ROOT = Path(__file__).parent.parent.parent.parent.parent
TRAIN_DATA_PATH = ROOT / "data" / "v5.2" / "train.parquet"
FEATURES_JSON_PATH = ROOT / "data" / "v5.2" / "features.json"

# Feature sets - define manually since get_feature_metadata doesn't exist
FEATURE_SETS = {
    "small": None,  # Will be loaded from features.json
    "medium": None,
    "all": None,
}

# App configuration
APP_TITLE = "Numerai v5.2 - Analiza Eksploracyjna Danych"
APP_ICON = "📊"
LAYOUT = "wide"

# Numerai specific constants
FEATURE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
