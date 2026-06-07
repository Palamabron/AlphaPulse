"""Feature Distributions Page — Detailed distribution analysis."""

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats

from eda.utils import get_translations

st.set_page_config(page_title="Feature Distributions", page_icon="📊", layout="wide")

t = get_translations()

if "data_loaded" not in st.session_state:
    st.warning(t["errors.data_not_loaded"])
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]

st.title(t["feature_distributions.title"])
st.markdown(t["feature_distributions.description"])

st.header(t["feature_distributions.summary_header"])

num_features_display = st.slider(
    t["feature_distributions.num_features_label"],
    min_value=10,
    max_value=min(100, len(feature_set)),
    value=30,
    step=10,
)

sample_features = feature_set[:num_features_display]

with st.spinner(t["feature_distributions.computing"]):
    dist_stats = []

    for feature in sample_features:
        feature_data = train[feature]
        value_counts = feature_data.value_counts(normalize=True).sort_index()

        entropy = -(value_counts * np.log2(value_counts + 1e-10)).sum()

        mode_val = feature_data.mode()[0] if len(feature_data.mode()) > 0 else np.nan

        dist_stats.append(
            {
                "Feature": feature,
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
    label=t["feature_distributions.download_stats"],
    data=csv,
    file_name="feature_distributions.csv",
    mime="text/csv",
)

st.divider()

st.header(t["feature_distributions.comparison_header"])

compare_features = st.multiselect(
    t["feature_distributions.comparison_select"],
    feature_set,
    default=feature_set[:3] if len(feature_set) >= 3 else feature_set,
)

if len(compare_features) > 5:
    st.warning(t["feature_distributions.too_many_features"])
    compare_features = compare_features[:5]

if len(compare_features) > 0:
    st.subheader(t["feature_distributions.comparison_subheader"])

    comparison_data = []
    for feature in compare_features:
        value_counts = train[feature].value_counts(normalize=True).sort_index()
        for value, pct in value_counts.items():
            comparison_data.append(
                {"Feature": feature, "Value": value, "Percent": pct * 100}
            )

    comp_df = pd.DataFrame(comparison_data)

    fig = px.bar(
        comp_df,
        x="Value",
        y="Percent",
        color="Feature",
        barmode="group",
        title=t["feature_distributions.comparison_title"],
        labels={"Value": "Value", "Percent": "Percent (%)"},
        height=500,
    )
    fig.update_layout(xaxis_title="Feature value", yaxis_title="Percent (%)")

    st.plotly_chart(fig, width="stretch")

    st.subheader(t["feature_distributions.comparison_stats_subheader"])

    compare_stats = dist_df[dist_df["Feature"].isin(compare_features)].copy()

    st.dataframe(
        compare_stats.style.background_gradient(
            subset=["Mean", "Std", "Entropy"], cmap="YlOrRd"
        ),
        width="stretch",
    )
else:
    st.info(t["feature_distributions.select_prompt"])

st.divider()

st.header(t["feature_distributions.chi_square_header"])
st.markdown(t["feature_distributions.chi_square_description"])

chi_square_results: list[dict[str, Any]] = []
for feature in sample_features[:20]:
    observed = train[feature].value_counts().sort_index().values
    expected = np.full(5, len(train) / 5)

    result = stats.chisquare(observed, expected)
    chi_stat = result.statistic
    p_val = result.pvalue

    chi_square_results.append(
        {
            "Feature": feature,
            "Chi-Square": chi_stat,
            "P-Value": p_val,
            "Uniform": t["feature_distributions.uniform_yes"]
            if p_val > 0.05
            else t["feature_distributions.uniform_no"],
        }
    )


chi_df = pd.DataFrame(chi_square_results).sort_values("P-Value")

st.dataframe(
    chi_df.style.background_gradient(subset=["P-Value"], cmap="RdYlGn"),
    width="stretch",
)

uniform_count_col, avg_p_value_col = st.columns(2)

with uniform_count_col:
    uniform_count = (chi_df["P-Value"] > 0.05).sum()
    st.metric(
        t["feature_distributions.uniform_count"],
        f"{uniform_count} / {len(chi_df)}",
        f"{uniform_count / len(chi_df) * 100:.1f}%",
    )

with avg_p_value_col:
    avg_p_value = chi_df["P-Value"].mean()
    st.metric(t["feature_distributions.avg_p_value"], f"{avg_p_value:.4f}")

csv_chi = chi_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label=t["common.download_csv"],
    data=csv_chi,
    file_name="chi_square_uniformity_test.csv",
    mime="text/csv",
    key="download_chi_square",
)

st.divider()
