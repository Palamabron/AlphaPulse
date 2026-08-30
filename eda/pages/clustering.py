"""Clustering Page — Hierarchical clustering to identify feature groups."""

import json
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from eda.utils import get_translations
from eda.utils.config import FEATURES_JSON_PATH

st.set_page_config(page_title="Clustering", page_icon="🔬", layout="wide")

t = get_translations()

if "data_loaded" not in st.session_state:
    st.warning(t["errors.data_not_loaded"])
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]

st.title(t["clustering.title"])
st.markdown(t["clustering.description"])


@st.cache_data
def load_feature_groups() -> dict[str, list[str]]:
    try:
        encodings = ["utf-8", "latin1", "cp1250", "iso-8859-2", "windows-1250"]
        features_json = None

        for encoding in encodings:
            try:
                with open(FEATURES_JSON_PATH, encoding=encoding) as f:
                    features_json = json.load(f)
                st.success(t.format("clustering.load_success", encoding=encoding))
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

        if features_json is None:
            raise ValueError(t["clustering.load_corrupted"])

    except FileNotFoundError:
        st.warning(t.format("clustering.load_not_found", path=FEATURES_JSON_PATH))
        return {}
    except Exception as e:
        st.error(t.format("clustering.load_error", error=e))
        return {}

    feature_to_group: dict[str, str] = {}

    if "feature_stats" in features_json:
        for feature_name, stats in features_json["feature_stats"].items():
            if "feature_group" in stats:
                feature_to_group[feature_name] = stats["feature_group"]

    if not feature_to_group:
        featuresets = features_json.get("feature_sets", {})
        all_features = []
        for features_list in featuresets.values():
            all_features.extend(features_list)

        for feat in set(all_features):
            parts = feat.split("_", 1)
            if len(parts) >= 2:
                feature_to_group[feat] = parts[0]
            else:
                feature_to_group[feat] = "other"

    groups: dict[str, list[str]] = {}
    for feature, group in feature_to_group.items():
        if group not in groups:
            groups[group] = []
        groups[group].append(feature)

    return groups


feature_to_group_map = load_feature_groups()


def get_feature_group(feature_name: str) -> str:
    if feature_name in feature_to_group_map:
        group = feature_to_group_map[feature_name]
        return group if isinstance(group, str) else "other"

    try:
        parts = feature_name.split("_")
        if len(parts) >= 2:
            return parts[1]
        return "other"
    except (IndexError, ValueError, AttributeError):
        return "other"


def get_group_color(group_name: str | None) -> str:
    if group_name is None:
        group_name = "other"
    color_map = {
        "charisma": "#FF6B6B",
        "intelligence": "#4ECDC4",
        "strength": "#45B7D1",
        "dexterity": "#FFA07A",
        "constitution": "#98D8C8",
        "wisdom": "#F7DC6F",
        "sunshine": "#FFD93D",
        "rain": "#6BCB77",
        "jerome": "#4D96FF",
        "charles": "#FF6B9D",
        "other": "#95A5A6",
    }
    return color_map.get(group_name, "#95A5A6")


st.sidebar.header(t["clustering.sidebar_header"])

num_features = st.sidebar.slider(
    t["clustering.num_features_label"],
    min_value=20,
    max_value=min(200, len(feature_set)),
    value=50,
    step=10,
)

linkage_method: Literal[
    "single", "complete", "average", "weighted", "centroid", "median", "ward"
] = "ward"

selected_features = feature_set[:num_features]
feature_groups = [get_feature_group(f) for f in selected_features]

unique_groups = set(feature_groups)

all_targets = st.session_state.get("all_targets", [])

if len(all_targets) > 0:
    st.sidebar.subheader(t["clustering.target_header"])
    selected_target = st.sidebar.selectbox(
        t["clustering.target_select"],
        all_targets,
        index=0,
        help=t["clustering.target_help"],
    )
else:
    selected_target = None

st.divider()

try:
    with st.spinner(t["clustering.computing"]):
        corr_matrix = train[selected_features].corr()
        distance_matrix = 1 - corr_matrix.abs()
        distance_condensed = squareform(distance_matrix, checks=False)
        linkage_matrix = hierarchy.linkage(distance_condensed, method=linkage_method)
except KeyError as e:
    st.error(t.format("errors.column_not_found", column=str(e)))
    st.stop()
except ValueError as e:
    st.error(t.format("errors.computation_error", error=str(e)))
    st.stop()
except Exception as e:
    st.error(t.format("errors.unexpected_clustering_error", error=str(e)))
    st.stop()

feature_count_col, avg_corr_col, max_corr_col, method_col = st.columns(4)
with feature_count_col:
    st.metric(t["clustering.num_features_metric"], num_features)
with avg_corr_col:
    avg_corr = (
        corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    )
    st.metric(t["clustering.avg_abs_corr"], f"{avg_corr:.4f}")
with max_corr_col:
    max_corr = (
        corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max()
    )
    st.metric(t["clustering.max_abs_corr"], f"{max_corr:.4f}")
with method_col:
    st.metric(t["clustering.method"], linkage_method)

st.divider()

st.header(t["clustering.dendrogram_header"])
st.markdown(t["clustering.dendrogram_description"])

fig_dendro, ax = plt.subplots(figsize=(16, 8))

dendro = hierarchy.dendrogram(
    linkage_matrix,
    labels=selected_features,
    ax=ax,
    orientation="bottom",
    color_threshold=0.7,
    above_threshold_color="gray",
)

ax.set_xlabel(t["clustering.dendrogram_xlabel"], fontsize=12)
ax.set_ylabel(t["clustering.dendrogram_ylabel"], fontsize=12)
ax.set_title(t["clustering.dendrogram_title"], fontsize=14, fontweight="bold")

ordered_labels = dendro["ivl"]

xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    feature_name = lbl.get_text()
    group = get_feature_group(feature_name)
    color = get_group_color(group)
    lbl.set_color(color)
    lbl.set_rotation(90)
    lbl.set_fontsize(8)

plt.tight_layout()

st.pyplot(fig_dendro)
plt.close()

st.divider()

st.header(t["clustering.legend_header"])
st.markdown(t["clustering.legend_description"])

legend_data = []

for group in sorted(unique_groups):
    color = get_group_color(group)
    count = feature_groups.count(group)
    legend_data.append(
        {
            "Group": group,
            "Feature_Count": count,
            "Percent": f"{(count / len(selected_features) * 100):.1f}%",
            "Color": color,
        }
    )

legend_df = pd.DataFrame(legend_data)

st.markdown(t["clustering.legend_color_mapping"])

col_count = 3
cols = st.columns(col_count)

for idx, (group, color, count) in enumerate(
    zip(
        legend_df["Group"], legend_df["Color"], legend_df["Feature_Count"], strict=False
    )
):
    pct = count / len(selected_features) * 100
    with cols[idx % col_count]:
        legend_item_text = t.format(
            "clustering.legend_item", count=count, pct=f"{pct:.1f}"
        )
        st.markdown(
            f"""
            <div style="
                padding: 10px;
                border-radius: 8px;
                background-color: {color}20;
                border-left: 5px solid {color};
                margin: 5px 0;
            ">
                <strong style="color: {color}">&#9632; {group}</strong><br>
                <small>{legend_item_text}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(t["clustering.legend_table_title"])

st.dataframe(
    legend_df.style.apply(
        lambda row: ["", "", "", f"background-color: {row['Color']}"], axis=1
    ),
    width="stretch",
    use_container_width=True,
)

st.divider()

st.header(t["clustering.heatmap_header"])
st.markdown(t["clustering.heatmap_description"])

dendro_leaves = dendro["leaves"]

clustered_corr = corr_matrix.iloc[dendro_leaves, dendro_leaves]

show_numbers = num_features <= 30

fig = px.imshow(
    clustered_corr,
    title=t.format("clustering.heatmap_title", count=num_features),
    labels={"color": "Correlation"},
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    text_auto=".2f" if show_numbers else False,
)

fig.update_xaxes(side="bottom", tickangle=90, showticklabels=num_features <= 60)
fig.update_yaxes(showticklabels=num_features <= 60)
fig.update_layout(height=max(700, num_features * 10))

st.plotly_chart(fig, width="stretch")

if show_numbers:
    st.success(t.format("clustering.heatmap_values_visible", count=num_features))
else:
    st.info(t.format("clustering.heatmap_values_hidden", count=num_features))

st.divider()

st.header(t["clustering.redundancy_header"])
st.markdown(t["clustering.redundancy_description"])

threshold = st.slider(t["clustering.threshold_label"], 0.5, 0.95, 0.8, 0.05)

redundant_pairs = []
for i in range(len(selected_features)):
    for j in range(i + 1, len(selected_features)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val >= threshold:
            redundant_pairs.append(
                {
                    "Feature_1": selected_features[i],
                    "Feature_2": selected_features[j],
                    "Group_1": get_feature_group(selected_features[i]),
                    "Group_2": get_feature_group(selected_features[j]),
                    "Correlation": corr_matrix.iloc[i, j],
                    "Abs_Correlation": corr_val,
                }
            )

if redundant_pairs:
    redundant_df = pd.DataFrame(redundant_pairs).sort_values(
        "Abs_Correlation", ascending=False
    )

    st.success(
        t.format(
            "clustering.redundant_found", count=len(redundant_df), threshold=threshold
        )
    )

    feature_count_col, avg_corr_col = st.columns([3, 1])

    with feature_count_col:
        st.subheader(t["clustering.redundant_subheader"])

        def highlight_same_group(row: pd.Series) -> list[str]:
            if row["Group_1"] == row["Group_2"]:
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        st.dataframe(
            redundant_df.head(50)
            .style.apply(highlight_same_group, axis=1)
            .background_gradient(
                subset=["Correlation"], cmap="RdBu_r", vmin=-1, vmax=1
            ),
            width="stretch",
            height=500,
        )

        st.info(t["clustering.same_group_info"])

    with avg_corr_col:
        st.subheader(t["clustering.stats_subheader"])
        st.metric(t["clustering.num_pairs"], len(redundant_df))
        st.metric(
            t["clustering.avg_abs_corr_label"],
            f"{redundant_df['Abs_Correlation'].mean():.4f}",
        )
        st.metric(t["clustering.max_corr"], f"{redundant_df['Correlation'].max():.4f}")
        st.metric(t["clustering.min_corr"], f"{redundant_df['Correlation'].min():.4f}")

    csv = redundant_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=t["clustering.download_redundant"],
        data=csv,
        file_name="redundant_feature_pairs.csv",
        mime="text/csv",
    )
else:
    st.warning(t.format("clustering.no_redundant", threshold=threshold))

st.divider()
