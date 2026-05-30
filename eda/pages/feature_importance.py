"""Feature Importance — ranking features by correlation or LightGBM gain."""

import pandas as pd
import plotly.express as px
import streamlit as st

from eda.utils import get_translations

st.set_page_config(page_title="Feature Importance", page_icon="⭐", layout="wide")

t = get_translations()

if "data_loaded" not in st.session_state:
    st.warning(t["errors.data_not_loaded"])
    st.stop()

train = st.session_state["train"]
feature_set = st.session_state["feature_set"]
target_col = st.session_state.get("selected_target", "target")

st.title(t["feature_importance.title"])
st.markdown(t["feature_importance.description"])

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header(t["common.settings"])

_method_options = [t["feature_importance.method_pearson"], "LightGBM"]
importance_method = st.sidebar.radio(
    t["feature_importance.method_label"],
    _method_options,
    index=0,
    help=t["feature_importance.method_help"],
)
lgbm_subsample = st.sidebar.slider(
    t["feature_importance.lgbm_subsample"],
    min_value=5,
    max_value=100,
    value=30,
    step=5,
    help=t["feature_importance.lgbm_subsample_help"],
)

# ── Cached computation ────────────────────────────────────────────────────────


@st.cache_data
def compute_correlation_importance(
    _train: pd.DataFrame, features: list[str], target: str
) -> pd.DataFrame:
    rows = []
    for feat in features:
        corr = float(_train[feat].corr(_train[target]))
        rows.append(
            {"feature": feat, "correlation": corr, "abs_correlation": abs(corr)}
        )
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


@st.cache_data
def compute_lgbm_importance(
    _train: pd.DataFrame, features: list[str], target: str, subsample_frac: float
) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except ImportError:
        st.error(t["feature_importance.lgbm_not_installed"])
        return pd.DataFrame()

    df = _train[features + [target]].dropna()
    if subsample_frac < 1.0:
        df = df.sample(frac=subsample_frac, random_state=42)

    X, y = df[features], df[target]
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=4,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=50,
        verbose=-1,
        random_state=42,
    )
    model.fit(X, y)

    importances = model.feature_importances_
    result = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=False
    )
    total = result["importance"].sum()
    result["importance_norm"] = result["importance"] / total if total > 0 else 0.0
    return result


# ── Compute ───────────────────────────────────────────────────────────────────

use_pearson = importance_method == _method_options[0]

if use_pearson:
    with st.spinner(t["common.computing"]):
        importance_df = compute_correlation_importance(train, feature_set, target_col)
    score_col = "abs_correlation"
    display_col = "correlation"
    method_label = t["feature_importance.pearson_label"]
else:
    spinner_text = t.format("feature_importance.lgbm_fitting", percent=lgbm_subsample)
    with st.spinner(spinner_text):
        importance_df = compute_lgbm_importance(
            train, feature_set, target_col, lgbm_subsample / 100.0
        )
    if importance_df.empty:
        st.stop()
    score_col = "importance_norm"
    display_col = "importance_norm"
    method_label = t["feature_importance.lgbm_label"]

# ── Summary metrics ───────────────────────────────────────────────────────────

st.subheader(t["common.summary"])

top_col, avg_col, threshold_col = st.columns(3)

with top_col:
    top_row = importance_df.iloc[0]
    label = top_row["feature"]
    label = label[:30] + "..." if len(label) > 30 else label
    st.metric(t["feature_importance.top_feature"], label, f"{top_row[score_col]:.6f}")

with avg_col:
    st.metric(
        t.format("feature_importance.mean_score", score=score_col),
        f"{importance_df[score_col].mean():.6f}",
    )

with threshold_col:
    if use_pearson:
        above = int((importance_df["abs_correlation"] > 0.01).sum())
        st.metric(t["feature_importance.features_above_threshold"], above)
    else:
        top10_pct = importance_df[score_col].head(10).sum() * 100
        st.metric(t["feature_importance.top10_share"], f"{top10_pct:.1f}%")

st.caption(f"{t['common.method']}: {method_label}")
st.divider()

# ── Full ranking table ────────────────────────────────────────────────────────

st.subheader(t["feature_importance.ranking_title"])

filter_col, threshold_col = st.columns(2)
_filter_options = [
    t["common.all"],
    "Top 50",
    "Top 100",
    t["feature_importance.above_threshold"],
]
with filter_col:
    filter_type = st.selectbox(t["common.show"], _filter_options)
threshold = 0.01
with threshold_col:
    if filter_type == _filter_options[-1]:
        threshold = st.number_input(
            f"{t['common.threshold']} ({score_col}):",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            format="%.4f",
        )

if filter_type == "Top 50":
    display_df = importance_df.head(50)
elif filter_type == "Top 100":
    display_df = importance_df.head(100)
elif filter_type == _filter_options[-1]:
    display_df = importance_df[importance_df[score_col] >= threshold]
else:
    display_df = importance_df

gradient_kwargs = (
    {"cmap": "RdYlGn", "vmin": -0.05, "vmax": 0.05}
    if use_pearson
    else {"cmap": "Blues"}
)
st.dataframe(
    display_df.style.background_gradient(subset=[display_col], **gradient_kwargs),
    use_container_width=True,
    height=600,
)

st.download_button(
    label=t["common.download_csv"],
    data=display_df.to_csv(index=False).encode("utf-8"),
    file_name="feature_importance.csv",
    mime="text/csv",
)

st.divider()

st.subheader(t["feature_importance.top20_title"])

if use_pearson:
    pos_col, neg_col = st.columns(2)

    with pos_col:
        st.markdown(t["feature_importance.positive_correlations"])
        top_pos = importance_df.nlargest(20, "correlation")
        fig = px.bar(
            top_pos,
            y="feature",
            x="correlation",
            orientation="h",
            text="correlation",
            color="correlation",
            color_continuous_scale="Blues",
            labels={
                "feature": t["common.feature"],
                "correlation": t["common.correlation"],
            },
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with neg_col:
        st.markdown(t["feature_importance.negative_correlations"])
        top_neg = importance_df.nsmallest(20, "correlation")
        fig = px.bar(
            top_neg,
            y="feature",
            x="correlation",
            orientation="h",
            text="correlation",
            color="correlation",
            color_continuous_scale="Reds_r",
            labels={
                "feature": t["common.feature"],
                "correlation": t["common.correlation"],
            },
        )
        fig.update_traces(texttemplate="%{text:.6f}", textposition="outside")
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

else:
    top20 = importance_df.head(20)
    fig = px.bar(
        top20,
        y="feature",
        x="importance_norm",
        orientation="h",
        text="importance_norm",
        color="importance_norm",
        color_continuous_scale="Viridis",
        labels={
            "feature": t["common.feature"],
            "importance_norm": t["feature_importance.importance_normalized"],
        },
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
