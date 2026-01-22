"""
Feature Distributions Page - Detailed distribution analysis
"""

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

st.set_page_config(page_title="Rozkłady Cech", page_icon="📊", layout="wide")

if "data_loaded" not in st.session_state:
    st.warning("⚠️ Dane nie zostały załadowane. Przejdź do strony głównej.")
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]

st.title("📊 Rozkłady Cech i Target")
st.markdown("""
**Target** = Znormalizowany zwrot z akcji (wartość do przewidzenia).

Szczegółowa analiza rozkładów pomaga zrozumieć dystrybucję wartości.
""")

st.header("📈 Podsumowanie Rozkładów Cech")

num_features_display = st.slider(
    "Liczba cech do analizy:",
    min_value=10,
    max_value=min(100, len(feature_set)),
    value=30,
    step=10,
)

sample_features = feature_set[:num_features_display]

with st.spinner("Obliczanie statystyk rozkładów..."):
    dist_stats = []

    for feature in sample_features:
        feature_data = train[feature]
        value_counts = feature_data.value_counts(normalize=True).sort_index()

        entropy = -(value_counts * np.log2(value_counts + 1e-10)).sum()

        mode_val = feature_data.mode()[0] if len(feature_data.mode()) > 0 else np.nan

        dist_stats.append(
            {
                "Cecha": feature,
                "Mean": feature_data.mean(),
                "Median": feature_data.median(),
                "Mode": mode_val,
                "Std": feature_data.std(),
                "Skewness": feature_data.skew(),
                "Kurtosis": feature_data.kurtosis(),
                "Entropy": entropy,
                "IQR": feature_data.quantile(0.75) - feature_data.quantile(0.25),
            }
        )

    dist_df = pd.DataFrame(dist_stats)

st.dataframe(
    dist_df.style.background_gradient(subset=["Entropy"], cmap="YlOrRd"),
    width="stretch",
    height=400,
)

csv = dist_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Pobierz statystyki jako CSV",
    data=csv,
    file_name="feature_distributions.csv",
    mime="text/csv",
)

st.divider()

st.header("🔄 Porównanie Rozkładów Cech")

compare_features = st.multiselect(
    "Wybierz cechy do porównania (max 5):",
    feature_set,
    default=feature_set[:3] if len(feature_set) >= 3 else feature_set,
)

if len(compare_features) > 5:
    st.warning("Wybrano zbyt wiele cech. Wyświetlane będą tylko pierwsze 5.")
    compare_features = compare_features[:5]

if len(compare_features) > 0:
    st.subheader("Porównanie rozkładów wartości")

    comparison_data = []
    for feature in compare_features:
        value_counts = train[feature].value_counts(normalize=True).sort_index()
        for value, pct in value_counts.items():
            comparison_data.append(
                {"Cecha": feature, "Wartość": value, "Procent": pct * 100}
            )

    comp_df = pd.DataFrame(comparison_data)

    fig = px.bar(
        comp_df,
        x="Wartość",
        y="Procent",
        color="Cecha",
        barmode="group",
        title="Porównanie rozkładów wartości (0.0-1.0)",
        labels={"Wartość": "Wartość", "Procent": "Procent (%)"},
        height=500,
    )
    fig.update_layout(xaxis_title="Wartość cechy", yaxis_title="Procent (%)")

    st.plotly_chart(fig, width="stretch")

    st.subheader("Szczegółowe porównanie statystyk")

    compare_stats = dist_df[dist_df["Cecha"].isin(compare_features)].copy()

    st.dataframe(
        compare_stats.style.background_gradient(
            subset=["Mean", "Std", "Entropy"], cmap="YlOrRd"
        ),
        width="stretch",
    )
else:
    st.info("👆 Wybierz cechy z listy powyżej, aby rozpocząć porównanie.")

st.divider()

st.header("📐 Test Chi-Kwadrat (Równomierność)")

st.markdown("""
Test hipotezy: Czy rozkład wartości cech jest równomierny (20% dla każdej wartości)?
""")

chi_square_results: list[dict[str, Any]] = []
for feature in sample_features[:20]:
    observed = train[feature].value_counts().sort_index().values
    expected = np.full(5, len(train) / 5)

    result = stats.chisquare(observed, expected)
    chi_stat = result.statistic
    p_val = result.pvalue

    chi_square_results.append(
        {
            "Cecha": feature,
            "Chi-Square": chi_stat,
            "P-Value": p_val,
            "Równomierny": "Tak" if p_val > 0.05 else "Nie",
        }
    )


chi_df = pd.DataFrame(chi_square_results).sort_values("P-Value")

st.dataframe(
    chi_df.style.background_gradient(subset=["P-Value"], cmap="RdYlGn"), width="stretch"
)

uniform_count_col, avg_p_value_col = st.columns(2)

with uniform_count_col:
    uniform_count = (chi_df["P-Value"] > 0.05).sum()
    st.metric(
        "Cechy z Równomiernym Rozkładem",
        f"{uniform_count} / {len(chi_df)}",
        f"{uniform_count / len(chi_df) * 100:.1f}%",
    )

with avg_p_value_col:
    avg_p_value = chi_df["P-Value"].mean()
    st.metric("Średnia P-Value", f"{avg_p_value:.4f}")

st.divider()
