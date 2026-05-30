"""Translation management for the EDA dashboard.

The selected language is persisted in ``st.session_state["lang"]`` by the
main ``app.py`` page and read here so that any sub-page can access translations
without re-rendering the language switcher.

Usage:
    t = get_translations()
    text = t["common.download_csv"]
    formatted = t.format("dataset.data_loaded", target="target_cyrus_v4_20")

Translation files are located in ``eda/locales/`` as YAML files (en.yaml, pl.yaml).
Organized by category with dot-notation keys
(e.g., "common.download_csv", "dataset.header").
"""

from pathlib import Path
from typing import Any

import streamlit as st
import yaml

_DEFAULT_LANG = "English"
_LANG_CODE_MAP = {"English": "en", "Polski": "pl"}


@st.cache_data
def _load_yaml_translations(lang_code: str) -> dict[str, Any]:
    """Load translations from YAML file.

    Args:
        lang_code: Language code ("en" or "pl")

    Returns:
        Nested dict of translations
    """
    locales_dir = Path(__file__).parent.parent / "locales"
    yaml_path = locales_dir / f"{lang_code}.yaml"

    if not yaml_path.exists():
        return {}

    try:
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_nested_value(data: dict[str, Any], key: str, default: str = "") -> str:
    """Get value from nested dict using dot notation.

    Args:
        data: Nested dictionary
        key: Dot-separated key path (e.g., "common.download_csv")
        default: Default value if key not found

    Returns:
        Value from dict or default
    """
    keys = key.split(".")
    value: Any = data

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default

    return str(value) if value is not None else default


class Translations:
    """Translation accessor with dict-like interface."""

    def __init__(self, lang: str = "English"):
        """Initialize translations for given language.

        Args:
            lang: Language name ("English" or "Polski")
        """
        self.lang = lang
        self.lang_code = _LANG_CODE_MAP.get(lang, "en")
        self._data = _load_yaml_translations(self.lang_code)

    def __getitem__(self, key: str) -> str:
        """Get translation by dot-notation key.

        Args:
            key: Dot-separated key (e.g., "common.download_csv")

        Returns:
            Translated string

        Example:
            t = Translations("English")
            text = t["common.download_csv"]  # "Download CSV"
        """
        return _get_nested_value(self._data, key, default=key)

    def get(self, key: str, default: str = "") -> str:
        """Get translation with optional default.

        Args:
            key: Dot-separated key
            default: Default value if key not found

        Returns:
            Translated string or default
        """
        return _get_nested_value(self._data, key, default=default)

    def format(self, key: str, **kwargs: Any) -> str:
        """Get translation and format with variables.

        Args:
            key: Dot-separated key
            **kwargs: Variables for string formatting

        Returns:
            Formatted translated string

        Example:
            t.format("dataset.data_loaded", target="target_cyrus_v4_20")
        """
        text = self[key]
        return text.format(**kwargs)


def get_translations(lang: str | None = None) -> Translations:
    """Get translations accessor for current language.

    Args:
        lang: Language name ("English" or "Polski").
              If None, uses session state.

    Returns:
        Translations accessor

    Example:
        t = get_translations()
        st.title(t["app.title"])
        st.write(t.format("dataset.data_loaded", target="target"))
    """
    if lang is None:
        lang = str(st.session_state.get("lang", _DEFAULT_LANG))
    return Translations(lang)


# Legacy translate() function removed - use get_translations() instead
