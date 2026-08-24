import pandas as pd


def align_series_to_frame(
    X: pd.DataFrame,
    values: pd.Series,
    *,
    name: str = "target",
) -> pd.Series:
    if X.index.has_duplicates:
        raise ValueError("feature index must not contain duplicate row IDs")
    if values.index.has_duplicates:
        raise ValueError(f"{name} index must not contain duplicate row IDs")
    missing = X.index.difference(values.index)
    extra = values.index.difference(X.index)
    if len(missing) > 0 or len(extra) > 0:
        raise ValueError(f"feature and {name} row IDs must exactly match")
    return values.reindex(X.index)
