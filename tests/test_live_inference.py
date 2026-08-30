from pathlib import Path

import cloudpickle
import pandas as pd
from scripts.live_inference import main


def _benchmark_predict(
    live_features: pd.DataFrame,
    live_benchmark_models: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {"prediction": live_benchmark_models["v2_equivalent_return"]},
        index=live_features.index,
    )


def test_live_inference_loads_and_aligns_benchmark_file(tmp_path: Path) -> None:
    ids = pd.Index(["a", "b", "c"], name="id")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"feature_x": [0.0, 1.0, 2.0]}, index=ids).to_parquet(
        data_dir / "live.parquet"
    )
    pd.DataFrame(
        {"v2_equivalent_return": [0.3, 0.1, 0.2]},
        index=pd.Index(["c", "a", "b"], name="id"),
    ).to_parquet(data_dir / "live_benchmark_models.parquet")

    model_path = tmp_path / "predict.pkl"
    with open(model_path, "wb") as file:
        cloudpickle.dump(_benchmark_predict, file)

    output_path = tmp_path / "predictions.csv"
    main(model_path=model_path, data_dir=data_dir, output_path=output_path)

    output = pd.read_csv(output_path)
    assert output.to_dict(orient="list") == {
        "id": ["a", "b", "c"],
        "prediction": [0.1, 0.2, 0.3],
    }
