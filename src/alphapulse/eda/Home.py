"""
Numerai v5.0 EDA Dashboard - Home Page
Main entry point for the multipage Streamlit application
"""

import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import streamlit as st
from utils.config import (
    APP_ICON,
    APP_TITLE,
    FEATURES_JSON_PATH,
    LAYOUT,
    TRAIN_DATA_PATH,
)
from utils.data_loader import load_numerai_data

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


lang = st.sidebar.radio("🌐 Language / Język", ["English", "Polski"], index=1)


def t(en: str, pl: str) -> str:
    """Translation helper function"""
    return en if lang == "English" else pl


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1 {
        color: #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_feature_sets() -> dict[str, list[str]]:
    """Load feature sets with sizes"""
    try:
        with open(FEATURES_JSON_PATH, encoding="utf-8") as f:
            features_json = json.load(f)
            if not features_json:
                return {}
            feature_sets: dict[str, list[str]] = features_json.get("feature_sets", {})
            return feature_sets
    except Exception:
        return {}


feature_sets_data: dict[str, list[str]] = load_feature_sets()

if feature_sets_data is None:
    st.stop()

n_small = len(feature_sets_data.get("small", []))
n_medium = len(feature_sets_data.get("medium", []))
n_big = len(feature_sets_data.get("all", feature_sets_data.get("big", [])))


@st.cache_data
def get_available_targets(data_path: str) -> list[str]:
    """Load sample and detect all target columns using PyArrow metadata"""
    try:
        pf = pq.ParquetFile(data_path)
        all_cols = pf.schema.names
        target_cols = sorted([col for col in all_cols if col.startswith("target")])
        return target_cols
    except Exception as e:
        st.warning(f"Error loading targets: {e}")
        return ["target"]


available_targets = get_available_targets(str(TRAIN_DATA_PATH))

st.sidebar.header("⚙️ Konfiguracja Datasetu")

feature_set_choice = st.sidebar.selectbox(
    t("Select feature set:", "Wybierz zestaw cech:"),
    ["small", "medium", "all"],
    index=1,
    help=t(
        f"Small: {n_small} features,"
        "Medium: {n_medium} features, All: {n_big} features",
        f"Mały: {n_small} cech, Średni: {n_medium} cech, Wszystkie: {n_big} cech",
    ),
)

st.sidebar.subheader("🎯 Wybór Aktywnego Targetu")

selected_target = st.sidebar.selectbox(
    t("Select target to analyze:", "Wybierz target do analizy:"),
    available_targets,
    index=0,
    help=t(
        "Choose which target to work with in the analysis",
        "Wybierz który target ma być aktywny do analizy",
    ),
)

st.sidebar.success(f"📊 Aktywny target: **{selected_target}**")

cache_key = f"{feature_set_choice}_{selected_target}"

if (
    "data_loaded" not in st.session_state
    or st.session_state.get("cache_key") != cache_key
):
    with st.spinner(
        t("Loading Numerai v5.0 data...", "Ładowanie danych Numerai v5.0...")
    ):
        try:
            train, feature_set, messages = load_numerai_data(
                str(TRAIN_DATA_PATH),
                feature_set_name=feature_set_choice,
                subsample_eras=True,
                selected_target=selected_target,
            )

            for msg in messages["info"]:
                st.info(msg)
            for msg in messages["warning"]:
                st.warning(msg)
            for msg in messages["error"]:
                st.error(msg)

            if selected_target not in train.columns:
                st.error(f"❌ Target '{selected_target}' nie istnieje w danych!")
                available_in_data = [
                    col for col in train.columns if col.startswith("target")
                ]
                st.warning(f"Dostępne targety: {', '.join(available_in_data)}")
                st.stop()

            train["target"] = train[selected_target].copy()

            st.session_state["train"] = train
            st.session_state["feature_set"] = feature_set
            st.session_state["cache_key"] = cache_key
            st.session_state["data_loaded"] = True
            st.session_state["selected_target"] = selected_target

            all_targets_in_data = sorted(
                [col for col in train.columns if col.startswith("target")]
            )
            st.session_state["all_targets"] = all_targets_in_data

            st.success(f"✅ Załadowano dane z targetem: {selected_target}")

        except KeyError as e:
            st.error(f"❌ Błąd: Kolumna '{e}' nie znaleziona w danych")
            st.info(
                "💡 Upewnij się, że plik parquet zawiera wszystkie kolumny targetów"
            )
            st.stop()
        except Exception as e:
            st.error(
                t(
                    f"❌ Error loading data: {e}",
                    f"❌ Błąd podczas ładowania danych: {e}",
                )
            )
            st.info(
                t(
                    "💡 Make sure data file exists at: ",
                    "💡 Upewnij się, że plik danych znajduje się w: ",
                )
                + str(TRAIN_DATA_PATH)
            )
            st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]
selected_target = st.session_state.get("selected_target", "target")
all_targets = st.session_state.get("all_targets", available_targets)


def sanity_check_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    """Check if features are discrete (only values 0, 0.25, 0.5, 0.75, 1.0)"""
    continuous = []
    for feat in features:
        if feat in df.columns:
            unique_vals = df[feat].unique()
            if not set(unique_vals).issubset({0, 0.25, 0.5, 0.75, 1.0}):
                continuous.append(feat)
    return continuous


continuous_features = sanity_check_features(train, feature_set)


def get_data_size_mb(path: str) -> float:
    """Get file size in MB"""
    return os.path.getsize(path) / (1024**2)


data_size = get_data_size_mb(str(TRAIN_DATA_PATH))

st.title(f"{APP_ICON} {APP_TITLE}")

st.markdown(
    t(
        """
---
## 👋 Welcome to Numerai v5.0 EDA Dashboard!

This interactive dashboard allows comprehensive exploratory data analysis
of Numerai tournament data using **encrypted stock market data**.

### 📊 Data loaded successfully!
""",
        """
---
## 👋 Witaj w Dashboard EDA dla Numerai v5.0!

Ten interaktywny dashboard pozwala na kompleksową analizę eksploracyjną danych
z turnieju Numerai wykorzystując **zaszyfrowane dane giełdowe**.

### 📊 Dane załadowane pomyślnie!
""",
    )
)

st.subheader(t("📊 Dataset Overview", "📊 Przegląd Datasetu"))

rows_col, features_col, missing_data_col = st.columns(3)

with rows_col:
    st.metric(t("📝 Rows", "📝 Liczba Wierszy"), f"{len(train):,}")

with features_col:
    st.metric(
        t("🔢 Features (Current Set)", "🔢 Liczba Cech (Bieżący Zestaw)"),
        len(feature_set),
    )

with missing_data_col:
    missing_pct = (train.isnull().sum().sum() / (len(train) * len(train.columns))) * 100
    st.metric(t("❓ Missing Data", "❓ Brakujące Dane"), f"{missing_pct:.2f}%")

st.divider()

st.subheader("🎯 Informacja o Target")

rows_col, features_col, missing_data_col = st.columns(3)

with rows_col:
    st.metric(t("Active Target", "Aktywny Target"), selected_target)

with features_col:
    st.metric(
        t("Available Targets", "Dostępne Targety"),
        len(all_targets),
        help=t(
            "Total number of targets available in the dataset",
            "Całkowita liczba targetów dostępnych w datasecie",
        ),
    )

with missing_data_col:
    st.metric(t("Target Mean", "Średnia Target"), f"{train['target'].mean():.6f}")

with st.expander(
    t("📋 Show all available targets", "📋 Pokaż wszystkie dostępne targety")
):
    cols = st.columns(3)
    for idx, target in enumerate(all_targets):
        is_active = "✅" if target == selected_target else "⭕"
        cols[idx % 3].write(f"{is_active} {target}")

    st.info(
        t(
            "✅ = Currently active target |"
            " ⭕ = Available targets (select in sidebar to switch)",
            "✅ = Aktywny target |"
            " ⭕ = Dostępne targety (wybierz w sidebaru aby zmienić)",
        )
    )

st.divider()

st.subheader(t("📦 Feature Set Sizes", "📦 Rozmiary Zestawów Cech"))

rows_col, features_col, missing_data_col = st.columns(3)

with rows_col:
    st.metric(t("Small Set (20%)", "Zestaw Mały (20%)"), f"{n_small:,}")

with features_col:
    st.metric(t("Medium Set (50%)", "Zestaw Średni (50%)"), f"{n_medium:,}")

with missing_data_col:
    st.metric(t("Big/All Set (100%)", "Zestaw Duży/Wszystkie (100%)"), f"{n_big:,}")

st.divider()

st.metric(t("💾 Data Size (MB)", "💾 Rozmiar Danych (MB)"), f"{data_size:.2f}")

st.divider()

st.subheader(t("🔍 Sanity Check", "🔍 Sanity Check"))

discrete_features = [f for f in feature_set if f not in continuous_features]

rows_col, features_col = st.columns(2)

with rows_col:
    st.metric(
        t("✅ Discrete Features", "✅ Cechy Dyskretne"),
        len(discrete_features),
        help=t(
            "Features with only values: 0, 0.25, 0.5, 0.75, 1.0",
            "Cechy zawierające tylko wartości: 0, 0.25, 0.5, 0.75, 1.0",
        ),
    )

with features_col:
    st.metric(
        t("⚠️ Continuous Features", "⚠️ Cechy Ciągłe"),
        len(continuous_features),
        help=t(
            "Features with values outside discrete set",
            "Cechy zawierające wartości poza zestawem dyskretnym",
        ),
    )

if continuous_features:
    st.warning(
        t(
            f"⚠️ Warning: {len(continuous_features)} continuous "
            "features detected (expected discrete: 0, 0.25, 0.5, 0.75, 1.0)",
            f"⚠️ Uwaga: Wykryto {len(continuous_features)} "
            "cech ciągłych (oczekiwane dyskretne: 0, 0.25, 0.5, 0.75, 1.0)",
        )
    )

    with st.expander(t("Show continuous features", "Pokaż cechy ciągłe")):
        for feat in continuous_features[:20]:
            st.write(f"- {feat}")
        if len(continuous_features) > 20:
            st.write(
                t(
                    f"... and {len(continuous_features) - 20} more",
                    f"... i {len(continuous_features) - 20} więcej",
                )
            )
else:
    st.success(
        t(
            "✅ All features are discrete "
            "with expected values (0, 0.25, 0.5, 0.75, 1.0)",
            "✅ Wszystkie cechy są dyskretne "
            "z oczekiwanymi wartościami (0, 0.25, 0.5, 0.75, 1.0)",
        )
    )

with st.expander(t("Show discrete features", "Pokaż cechy dyskretne")):
    if discrete_features:
        st.info(
            t(
                f"✅ {len(discrete_features)} features are properly discrete",
                f"✅ {len(discrete_features)} cech jest poprawnie dyskretnych",
            )
        )

        cols = st.columns(3)
        for idx, feat in enumerate(discrete_features[:50]):
            cols[idx % 3].write(f"- {feat}")

        if len(discrete_features) > 50:
            st.write(
                t(
                    f"... and {len(discrete_features) - 50} more discrete features",
                    f"... i {len(discrete_features) - 50} więcej cech dyskretnych",
                )
            )
    else:
        st.error(
            t("❌ No discrete features found!", "❌ Nie znaleziono cech dyskretnych!")
        )

st.divider()

st.subheader(t("🧭 Navigation", "🧭 Nawigacja"))

st.markdown(
    t(
        """
Use the **sidebar** to navigate to different analysis sections:

1. **📋 Data Overview** - Basic statistics and dataset overview
2. **🎯 Target Analysis** - Detailed analysis of target variable
3. **🔍 Feature Analysis** - Individual feature analysis
4. **🔗 Correlations** - Correlation analysis between features and target
5. **⏰ Era Analysis** - Temporal data analysis over time
6. **📊 Feature Distributions** - Detailed feature value distributions
7. **⭐ Feature Importance** - Feature importance ranking
8. **🔬 Clustering** - Feature clustering and grouping

### 🚀 Start exploring!

Select a page from the sidebar to begin analysis.
""",
        """
Użyj **menu bocznego** aby przejść do różnych sekcji analizy:

1. **📋 Przegląd Danych** - Podstawowe statystyki i przegląd datasetu
2. **🎯 Analiza Target** - Szczegółowa analiza zmiennej docelowej
3. **🔍 Analiza Cech** - Indywidualna analiza wybranych cech
4. **🔗 Korelacje** - Analiza korelacji między cechami i targetem
5. **⏰ Analiza Era** - Analiza temporalna danych w czasie
6. **📊 Rozkłady Cech** - Szczegółowe rozkłady wartości cech
7. **⭐ Feature Importance** - Ranking ważności cech
8. **🔬 Clustering** - Klasteryzacja i grupowanie cech

### 🚀 Rozpocznij eksplorację!

Wybierz stronę z menu bocznego, aby rozpocząć analizę.
""",
    )
)

st.divider()

with st.expander(
    t("🔍 Data preview (first 10 rows)", "🔍 Podgląd danych (pierwsze 10 wierszy)")
):
    st.dataframe(train.head(10), width="stretch")

with st.expander(t("📊 Quick statistics", "📊 Szybkie statystyki")):
    st.dataframe(train.describe(), width="stretch")

st.markdown(
    t(
        f"""
---
**Numerai v5.0 EDA Dashboard** | Dataset: `r1105_v5_0_train.parquet`
📊 *Dataset contains encrypted stock market data*
**Currently Analyzing:** {selected_target}
""",
        f"""
---
**Numerai v5.0 EDA Dashboard** | Dataset: `r1105_v5_0_train.parquet`
📊 *Dataset zawiera zaszyfrowane dane giełdowe*
**Aktualnie Analizuję:** {selected_target}
""",
    )
)
