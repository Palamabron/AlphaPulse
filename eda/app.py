"""Numerai EDA Dashboard — main entry point.

Run with:  ``streamlit run eda/app.py``
"""

import json
import os

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from eda.utils import get_translations
from eda.utils.config import (
    APP_ICON,
    APP_TITLE,
    DATASET_VERSION,
    FEATURES_JSON_PATH,
    LAYOUT,
    TRAIN_DATA_PATH,
)
from eda.utils.data_loader import load_numerai_data

# set_page_config MUST be the first Streamlit call
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state="expanded",
)

lang = st.sidebar.radio("Language / Język", ["English", "Polski"], index=0)
st.session_state["lang"] = lang

# Get translations accessor for current language (new YAML-based API)
t = get_translations(lang)


st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; }
    h1 { color: #1f77b4; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_feature_sets() -> dict[str, list[str]]:
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

n_small = len(feature_sets_data.get("small", []))
n_medium = len(feature_sets_data.get("medium", []))
n_big = len(feature_sets_data.get("all", feature_sets_data.get("big", [])))


@st.cache_data
def get_available_targets(data_path: str) -> list[str]:
    try:
        pf = pq.ParquetFile(data_path)
        all_cols = pf.schema.names
        return sorted([col for col in all_cols if col.startswith("target")])
    except Exception as e:
        st.warning(f"Error loading targets: {e}")
        return ["target"]


available_targets = get_available_targets(str(TRAIN_DATA_PATH))

# Dataset Configuration Section (using new YAML-based translations)
st.sidebar.header(t["dataset.header"])

feature_set_choice = st.sidebar.selectbox(
    t["dataset.feature_set_label"],
    ["small", "medium", "all"],
    index=1,
    help=t.format(
        "dataset.feature_set_help", small=n_small, medium=n_medium, all=n_big
    ),
)

st.sidebar.subheader(t["dataset.active_target"])

selected_target = st.sidebar.selectbox(
    t["dataset.select_target"],
    available_targets,
    index=0,
    help=t["dataset.select_target_help"],
)

st.sidebar.success(t.format("dataset.active_label", target=selected_target))

cache_key = f"{feature_set_choice}_{selected_target}"

if (
    "data_loaded" not in st.session_state
    or st.session_state.get("cache_key") != cache_key
):
    with st.spinner(t.format("dataset.loading_data", version=DATASET_VERSION)):
        try:
            train, feature_set, messages = load_numerai_data(
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
                st.error(t.format("dataset.target_not_found", target=selected_target))
                available_in_data = [
                    col for col in train.columns if col.startswith("target")
                ]
                targets_str = ", ".join(available_in_data)
                st.warning(t.format("dataset.available_targets", targets=targets_str))
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

            st.success(t.format("dataset.data_loaded", target=selected_target))

        except KeyError as e:
            st.error(t.format("errors.column_not_found", column=str(e)))
            st.info(t["errors.check_parquet_targets"])
            st.stop()
        except Exception as e:
            st.error(t.format("errors.data_loading_failed", error=str(e)))
            st.info(t.format("errors.data_path_expected", path=str(TRAIN_DATA_PATH)))
            st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]
selected_target = st.session_state.get("selected_target", "target")
all_targets = st.session_state.get("all_targets", available_targets)


def sanity_check_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    continuous = []
    for feat in features:
        if feat in df.columns:
            unique_vals = df[feat].unique()
            if not set(unique_vals).issubset({0, 0.25, 0.5, 0.75, 1.0}):
                continuous.append(feat)
    return continuous


continuous_features = sanity_check_features(train, feature_set)


def get_data_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024**2)


data_size = get_data_size_mb(str(TRAIN_DATA_PATH))

st.title(f"{APP_ICON} {APP_TITLE}")

st.markdown(t.format("app.subtitle", version=DATASET_VERSION))
st.divider()

# ── Dataset overview ────────────────────────────────────────────────────────

st.subheader(t["overview.header"])

rows_col, features_col, missing_data_col = st.columns(3)

with rows_col:
    st.metric(t["overview.rows"], f"{len(train):,}")

with features_col:
    st.metric(t["overview.features"], len(feature_set))

with missing_data_col:
    missing_pct = (train.isnull().sum().sum() / (len(train) * len(train.columns))) * 100
    st.metric(t["overview.missing_data"], f"{missing_pct:.2f}%")

st.divider()

# ── Target information ───────────────────────────────────────────────────────

st.subheader(t["target_info.header"])

rows_col, features_col, missing_data_col = st.columns(3)

with rows_col:
    st.metric(t["target_info.active_target"], selected_target)

with features_col:
    st.metric(
        t["target_info.available_targets"],
        len(all_targets),
        help=t["target_info.available_targets_help"],
    )

with missing_data_col:
    st.metric(t["target_info.target_mean"], f"{train['target'].mean():.6f}")

with st.expander(t["target_info.show_all_targets"]):
    cols = st.columns(3)
    for idx, target in enumerate(all_targets):
        marker = ">" if target == selected_target else " "
        cols[idx % 3].write(f"{marker} {target}")

    st.caption(t["target_info.currently_active"])

st.divider()

# ── Feature set sizes ────────────────────────────────────────────────────────

st.subheader(t["feature_sets.header"])

rows_col, features_col, missing_data_col = st.columns(3)

with rows_col:
    st.metric(t["feature_sets.small"], f"{n_small:,}")

with features_col:
    st.metric(t["feature_sets.medium"], f"{n_medium:,}")

with missing_data_col:
    st.metric(t["feature_sets.all"], f"{n_big:,}")

st.divider()

st.metric(t["feature_sets.file_size"], f"{data_size:.2f}")

st.divider()

# ── Sanity check ─────────────────────────────────────────────────────────────

st.subheader(t["sanity.header"])

discrete_features = [f for f in feature_set if f not in continuous_features]

rows_col, features_col = st.columns(2)

with rows_col:
    st.metric(
        t["sanity.discrete_features"],
        len(discrete_features),
        help=t["sanity.discrete_help"],
    )

with features_col:
    st.metric(
        t["sanity.continuous_features"],
        len(continuous_features),
        help=t["sanity.continuous_help"],
    )

if continuous_features:
    st.warning(t.format("sanity.continuous_detected", count=len(continuous_features)))

    with st.expander(t["sanity.show_continuous"]):
        for feat in continuous_features[:20]:
            st.write(f"- {feat}")
        if len(continuous_features) > 20:
            st.write(t.format("sanity.and_more", count=len(continuous_features) - 20))
else:
    st.success(t["sanity.all_discrete"])

with st.expander(t["sanity.show_discrete"]):
    if discrete_features:
        st.caption(t.format("sanity.discrete_count", count=len(discrete_features)))
        cols = st.columns(3)
        for idx, feat in enumerate(discrete_features[:50]):
            cols[idx % 3].write(f"- {feat}")

        if len(discrete_features) > 50:
            st.write(t.format("sanity.and_more", count=len(discrete_features) - 50))
    else:
        st.error(t["sanity.no_discrete"])

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────

st.subheader(t["navigation.header"])

st.markdown(t["navigation.description"])

st.divider()

# ── Data preview ──────────────────────────────────────────────────────────────

with st.expander(t["preview.data_preview"]):
    st.dataframe(train.head(10), use_container_width=True)

with st.expander(t["preview.quick_stats"]):
    st.dataframe(train.describe(), use_container_width=True)

st.caption(t.format("preview.footer", version=DATASET_VERSION, target=selected_target))
