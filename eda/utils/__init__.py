from importlib import import_module
from typing import Any

__all__ = [
    "APP_ICON",
    "APP_TITLE",
    "FEATURE_VALUES",
    "LAYOUT",
    "get_era_statistics",
    "get_feature_correlations",
    "load_numerai_data",
    "get_translations",
    "translate",
    "Translations",
    "compute_feature_target_correlations",
    "compute_era_statistics",
    "create_download_button",
    "get_plotly_theme",
    "apply_chart_layout",
]

_EXPORT_MODULES = {
    "APP_ICON": ".config",
    "APP_TITLE": ".config",
    "FEATURE_VALUES": ".config",
    "LAYOUT": ".config",
    "get_era_statistics": ".data_loader",
    "get_feature_correlations": ".data_loader",
    "load_numerai_data": ".data_loader",
    "get_translations": ".i18n",
    "translate": ".i18n",
    "Translations": ".i18n",
    "compute_feature_target_correlations": ".common",
    "compute_era_statistics": ".common",
    "create_download_button": ".common",
    "get_plotly_theme": ".common",
    "apply_chart_layout": ".common",
}


def __getattr__(name: str) -> Any:
    """Load Streamlit-dependent helpers only when they are requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
