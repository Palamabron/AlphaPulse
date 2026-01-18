"""
Clustering Page - Hierarchical clustering to identify feature groups
"""

import json
import os
import sys
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from utils.config import FEATURES_JSON_PATH

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

st.set_page_config(page_title="Clustering", page_icon="🔬", layout="wide")

if "data_loaded" not in st.session_state:
    st.warning("⚠️ Dane nie zostały załadowane. Przejdź do strony głównej.")
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]

st.title("🔬 Analiza Klasteryzacji Cech")
st.markdown("""
Grupowanie podobnych cech na podstawie \
    korelacji - identyfikacja grup redundantnych cech.

**Cel:** Wykryć cechy, które są silnie skorelowane ze sobą i mogą być redundantne.
""")

# ============================================================================
# LOAD FEATURE GROUPS FROM features.json
# ============================================================================


@st.cache_data
def load_feature_groups() -> dict[str, list[str]]:
    """Load feature group mappings from features.json"""
    try:
        # Automatyczne wykrywanie kodowania
        encodings = ["utf-8", "latin1", "cp1250", "iso-8859-2", "windows-1250"]
        features_json = None

        for encoding in encodings:
            try:
                with open(FEATURES_JSON_PATH, encoding=encoding) as f:
                    features_json = json.load(f)
                st.success(f"✓ Załadowano features.json (kodowanie: {encoding})")
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

        if features_json is None:
            raise ValueError("Plik features.json uszkodzony lub nie istnieje")

    except FileNotFoundError:
        st.warning(f"features.json nie istnieje: {FEATURES_JSON_PATH}")
        return {}
    except Exception as e:
        st.error(f"❌ Błąd ładowania features.json: {e}")
        return {}

    # Parsowanie grup cech
    feature_to_group: dict[str, str] = {}

    # 1. Z feature_stats
    if "feature_stats" in features_json:
        for feature_name, stats in features_json["feature_stats"].items():
            if "feature_group" in stats:
                feature_to_group[feature_name] = stats["feature_group"]

    # 2. Fallback - ekstrakcja z featuresets
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

    # Konwersja na grupy (lista cech per grupa)
    groups: dict[str, list[str]] = {}
    for feature, group in feature_to_group.items():
        if group not in groups:
            groups[group] = []
        groups[group].append(feature)

    return groups


feature_to_group_map = load_feature_groups()


def get_feature_group(feature_name: str) -> str:
    """Get group for a feature"""
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
    """Assign color based on feature group"""
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


# ============================================================================
# SETTINGS
# ============================================================================
st.sidebar.header("⚙️ Ustawienia Klasteryzacji")

num_features = st.sidebar.slider(
    "Liczba cech do klasteryzacji:",
    min_value=20,
    max_value=min(200, len(feature_set)),
    value=50,
    step=10,
)

linkage_method: Literal[
    "single", "complete", "average", "weighted", "centroid", "median", "ward"
] = "ward"  # lub inna metoda z listy

selected_features = feature_set[:num_features]
feature_groups = [get_feature_group(f) for f in selected_features]

# Show unique groups
unique_groups = set(feature_groups)


# ============================================================================
# TARGET SELECTION (OPTIONAL - IF YOU WANT TO ANALYZE BY TARGET)
# ============================================================================
all_targets = st.session_state.get("all_targets", [])

if len(all_targets) > 0:
    st.sidebar.subheader("🎯 Wybór Target (opcjonalnie)")
    selected_target = st.sidebar.selectbox(
        "Analizuj korelacje z targetem:",
        all_targets,
        index=0,
        help="Możesz wybrać target do korelacji (jeśli potrzebne)",
    )
else:
    selected_target = None

st.divider()

# ============================================================================
# COMPUTE CORRELATION AND DISTANCE
# ============================================================================
with st.spinner("Obliczanie macierzy korelacji i linkage..."):
    corr_matrix = train[selected_features].corr()
    distance_matrix = 1 - corr_matrix.abs()
    distance_condensed = squareform(distance_matrix, checks=False)
    linkage_matrix = hierarchy.linkage(distance_condensed, method=linkage_method)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Liczba Cech", num_features)
with col2:
    avg_corr = (
        corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    )
    st.metric("Średnia |Korelacja|", f"{avg_corr:.4f}")
with col3:
    max_corr = (
        corr_matrix.abs().values[np.triu_indices_from(corr_matrix.values, k=1)].max()
    )
    st.metric("Max |Korelacja|", f"{max_corr:.4f}")
with col4:
    st.metric("Metoda", linkage_method)

st.divider()

# ============================================================================
# DENDROGRAM WITH COLORED LABELS USING MATPLOTLIB
# ============================================================================
st.header("🌳 Dendrogram Hierarchicznej Klasteryzacji")

st.markdown("""
Dendrogram pokazuje hierarchię podobieństw między cechami.

**Nazwy cech są kolorowane według grup z features.json.**

Legenda poniżej pokazuje kolor każdej grupy.
""")

# Create matplotlib dendrogram with colored labels
fig_dendro, ax = plt.subplots(figsize=(16, 8))

dendro = hierarchy.dendrogram(
    linkage_matrix,
    labels=selected_features,
    ax=ax,
    orientation="bottom",
    color_threshold=0.7,
    above_threshold_color="gray",
)

ax.set_xlabel("Cechy (kolorowane według grup z features.json)", fontsize=12)
ax.set_ylabel("Odległość (1 - |Correlation|)", fontsize=12)
ax.set_title(
    "Dendrogram Hierarchicznej Klasteryzacji Cech", fontsize=14, fontweight="bold"
)

# Get ordered labels from dendrogram
ordered_labels = dendro["ivl"]

# Color the x-axis labels
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    feature_name = lbl.get_text()
    group = get_feature_group(feature_name)
    color = get_group_color(group)
    lbl.set_color(color)
    lbl.set_rotation(90)
    lbl.set_fontsize(8)

plt.tight_layout()

# Display in Streamlit
st.pyplot(fig_dendro)
plt.close()

st.divider()

# ============================================================================
# LEGEND - COLOR TO GROUP MAPPING
# ============================================================================
st.header("📋 Legenda - Kolor = Grupa")

st.markdown("""
Poniżej znajduje się legenda mapująca kolory etykiet w dendrogramie do grup cech.
""")

# Create legend as a visual color bar with group names
legend_data = []

for group in sorted(unique_groups):
    color = get_group_color(group)
    count = feature_groups.count(group)
    legend_data.append(
        {
            "Grupa": group,
            "Liczba_Cech": count,
            "Procent": f"{(count / len(selected_features) * 100):.1f}%",
            "Kolor": color,
        }
    )

legend_df = pd.DataFrame(legend_data)

# Display as styled HTML
st.markdown("### Mapowanie Kolor → Grupa")

col_count = 3
cols = st.columns(col_count)

for idx, (group, color, count) in enumerate(
    zip(legend_df["Grupa"], legend_df["Kolor"], legend_df["Liczba_Cech"], strict=False)
):
    with cols[idx % col_count]:
        # Create HTML box with colored square
        st.markdown(
            f"""
            <div style="
                padding: 10px;
                border-radius: 8px;
                background-color: {color}20;
                border-left: 5px solid {color};
                margin: 5px 0;
            ">
                <strong style="color: {color}">■ {group}</strong><br>
                <small>{count} cech
                ({(count / len(selected_features) * 100):.1f}%)</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Alternative: table with colors
st.markdown("### Tabela Legendy")

st.dataframe(
    legend_df.style.apply(
        lambda row: ["", "", "", f"background-color: {row['Kolor']}"], axis=1
    ),
    width="stretch",
    use_container_width=True,
)

st.divider()

# ============================================================================
# CLUSTERED CORRELATION HEATMAP (WITH NUMBERS)
# ============================================================================
st.header("🗺️ Klasterowana Mapa Cieplna Korelacji")

st.markdown("""
Macierz korelacji uporządkowana według struktury dendrogramu.
**Wartości korelacji są wyświetlane w komórkach (dla ≤30 cech).**
""")

# Get dendrogram leaves order
dendro_leaves = dendro["leaves"]

# Reorder correlation matrix
clustered_corr = corr_matrix.iloc[dendro_leaves, dendro_leaves]

# Show numbers only if ≤30 features
show_numbers = num_features <= 30

fig = px.imshow(
    clustered_corr,
    title=f"Macierz Korelacji - Uporządkowana ({num_features} cech)",
    labels={"color": "Korelacja"},
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
    st.success(f"✅ Wartości korelacji wyświetlane (liczba cech: {num_features} ≤ 30)")
else:
    st.info(
        f"💡 Wartości ukryte (zbyt wiele cech: {num_features}).\
        Zmniejsz do ≤30 aby zobaczyć liczby."
    )

st.divider()

# ============================================================================
# REDUNDANCY ANALYSIS
# ============================================================================
st.header("🔍 Analiza Redundancji")

st.markdown("""
Identyfikacja par cech z wysoką korelacją (potencjalna redundancja).
**Pary z tej samej grupy są podświetlone na czerwono.**
""")

threshold = st.slider("Próg korelacji dla redundancji:", 0.5, 0.95, 0.8, 0.05)

redundant_pairs = []
for i in range(len(selected_features)):
    for j in range(i + 1, len(selected_features)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val >= threshold:
            redundant_pairs.append(
                {
                    "Cecha_1": selected_features[i],
                    "Cecha_2": selected_features[j],
                    "Grupa_1": get_feature_group(selected_features[i]),
                    "Grupa_2": get_feature_group(selected_features[j]),
                    "Korelacja": corr_matrix.iloc[i, j],
                    "Abs_Korelacja": corr_val,
                }
            )

if redundant_pairs:
    redundant_df = pd.DataFrame(redundant_pairs).sort_values(
        "Abs_Korelacja", ascending=False
    )

    st.success(
        f"✅ Znaleziono **{len(redundant_df)}** par cech z |korelacją| ≥ {threshold}"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Redundantne Pary Cech")

        def highlight_same_group(row: pd.Series) -> list[str]:  # Zmień na list[str]
            if row["Grupa_1"] == row["Grupa_2"]:
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        st.dataframe(
            redundant_df.head(50)
            .style.apply(highlight_same_group, axis=1)  # axis=1 zamiast int
            .background_gradient(subset=["Korelacja"], cmap="RdBu_r", vmin=-1, vmax=1),
            width="stretch",
            height=500,
        )

        st.info("🎨 Czerwone tło = cechy z tej samej grupy (według features.json)")

    with col2:
        st.subheader("Statystyki")
        st.metric("Liczba Par", len(redundant_df))
        st.metric("Średnia |Korelacja|", f"{redundant_df['Abs_Korelacja'].mean():.4f}")
        st.metric("Max Korelacja", f"{redundant_df['Korelacja'].max():.4f}")
        st.metric("Min Korelacja", f"{redundant_df['Korelacja'].min():.4f}")

    csv = redundant_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Pobierz redundantne pary",
        data=csv,
        file_name="redundant_feature_pairs.csv",
        mime="text/csv",
    )
else:
    st.warning(f"⚠️ Nie znaleziono par z |korelacją| ≥ {threshold}. Zmniejsz próg.")

st.divider()
