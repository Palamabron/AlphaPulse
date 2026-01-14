"""
Feature Importance Page - Ranking and analysis of feature importance
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

st.set_page_config(page_title="Feature Importance", page_icon="⭐", layout="wide")

if "data_loaded" not in st.session_state:
    st.warning("⚠️ Dane nie zostały załadowane. Przejdź do strony głównej.")
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]

st.title("⭐ Ważność Cech")
st.markdown("""
Ranking ważności cech na podstawie korelacji z Target (znormalizowany zwrot z akcji).
""")


# ============================================================================
# COMPUTE IMPORTANCE METRICS
# ============================================================================
@st.cache_data
def compute_importance(_train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    importance_data = []
    target_col = st.session_state.get("selected_target", "target")

    for feature in features:
        corr = _train[feature].corr(_train[target_col])
        abs_corr = abs(float(corr))
        importance_data.append(
            {
                "Cecha": feature,
                "Korelacja": corr,
                "Abs_Korelacja": abs_corr,  # <-- DODAJ TĘ KOLUMNĘ!
            }
        )

    return pd.DataFrame(importance_data).sort_values("Abs_Korelacja", ascending=False)


with st.spinner("Obliczanie ważności cech..."):
    importance_df = compute_importance(train, feature_set)

# ============================================================================
# SUMMARY METRICS
# ============================================================================
st.header("📊 Podsumowanie")

col1, col2, col3 = st.columns(3)

with col1:
    top_corr = importance_df.iloc[0]
    st.metric(
        "Top Cecha", top_corr["Cecha"][:30] + "...", f"{top_corr['Abs_Korelacja']:.6f}"
    )

with col2:
    st.metric("Średnia |Korelacja|", f"{importance_df['Abs_Korelacja'].mean():.6f}")

with col3:
    above_threshold = (importance_df["Abs_Korelacja"] > 0.01).sum()
    st.metric("Cechy z |r| > 0.01", above_threshold)

st.divider()

# REMOVED: "Wizualizacje Ważności" section completely

# ============================================================================
# DETAILED TABLE
# ============================================================================
st.header("📋 Ranking Ważności Cech")

# Filter options
col1, col2 = st.columns(2)

with col1:
    filter_type = st.selectbox(
        "Filtruj według:", ["Wszystkie", "Top 50", "Top 100", "Powyżej progu"]
    )

with col2:
    if filter_type == "Powyżej progu":
        threshold = st.number_input(
            "Próg |Korelacji|:", min_value=0.0, max_value=1.0, value=0.01, format="%.4f"
        )

# Apply filters
if filter_type == "Top 50":
    display_df = importance_df.head(50)
elif filter_type == "Top 100":
    display_df = importance_df.head(100)
elif filter_type == "Powyżej progu":
    display_df = importance_df[importance_df["Abs_Korelacja"] >= threshold]
else:
    display_df = importance_df

st.dataframe(
    display_df.style.background_gradient(
        subset=["Korelacja"], cmap="RdYlGn", vmin=-0.05, vmax=0.05
    ),
    width="stretch",
    height=600,
)

# Download
csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Pobierz jako CSV",
    data=csv,
    file_name="feature_importance.csv",
    mime="text/csv",
)

st.divider()

# ============================================================================
# TOP 20 FEATURES - SIMPLE BAR CHARTS
# ============================================================================
st.header("🏆 Top 20 Cech")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dodatnie Korelacje")
    top_pos = importance_df.nlargest(20, "Korelacja")

    fig = px.bar(
        top_pos,
        y="Cecha",
        x="Korelacja",
        orientation="h",
        title="Top 20 Dodatnie Korelacje",
        text="Korelacja",
        color="Korelacja",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
    fig.update_layout(height=600)
    fig.update_xaxes(title_text="Korelacja z Target")
    fig.update_yaxes(title_text="Cecha")
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("Ujemne Korelacje")
    top_neg = importance_df.nsmallest(20, "Korelacja")

    fig = px.bar(
        top_neg,
        y="Cecha",
        x="Korelacja",
        orientation="h",
        title="Top 20 Ujemne Korelacje",
        text="Korelacja",
        color="Korelacja",
        color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
    fig.update_layout(height=600)
    fig.update_xaxes(title_text="Korelacja z Target")
    fig.update_yaxes(title_text="Cecha")
    st.plotly_chart(fig, width="stretch")

# Footer
st.divider()
