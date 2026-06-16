from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _reset_matplotlib_state() -> Iterator[None]:
    yield
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        plt.close("all")
        mpl.use("Agg", force=True)
    except ImportError:
        pass
