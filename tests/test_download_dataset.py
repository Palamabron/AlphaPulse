from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from scripts import download_dataset
from scripts.download_dataset import DownloadConfig


class _NumerAPI:
    def __init__(
        self,
        *args: Any,
        available: list[str] | None = None,
        list_error: Exception | None = None,
        download_error: Exception | None = None,
        write_partial_before_error: bool = False,
        write_corrupt_success: bool = False,
        **kwargs: Any,
    ) -> None:
        self.available = available or []
        self.list_error = list_error
        self.download_error = download_error
        self.write_partial_before_error = write_partial_before_error
        self.write_corrupt_success = write_corrupt_success

    def list_datasets(self) -> list[str]:
        if self.list_error is not None:
            raise self.list_error
        return self.available

    def download_dataset(self, remote_path: str, destination: str) -> None:
        if self.download_error is not None:
            if self.write_partial_before_error:
                Path(destination).write_bytes(b"partial")
            raise self.download_error
        if self.write_corrupt_success:
            Path(destination).write_bytes(b"not parquet")
            return
        pd.DataFrame({"value": [1.0]}).to_parquet(destination)


def test_download_config_does_not_materialize_credentials_in_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NUMERAI_PUBLIC_API_KEY", "public-value-that-must-stay-hidden")
    monkeypatch.setenv("NUMERAI_PRIVATE_API_KEY", "secret-value-that-must-stay-hidden")

    config = DownloadConfig(files=["train.parquet"])

    assert config.public_id is None
    assert config.secret_key is None


def test_download_dataset_resolves_environment_credentials_at_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("NUMERAI_PUBLIC_API_KEY", "public-runtime-value")
    monkeypatch.setenv("NUMERAI_PRIVATE_API_KEY", "secret-runtime-value")

    def factory(**kwargs: Any) -> _NumerAPI:
        captured.update(kwargs)
        return _NumerAPI(list_error=OSError("offline"))

    monkeypatch.setattr(download_dataset.numerapi, "NumerAPI", factory)

    with pytest.raises(RuntimeError, match="Failed to list datasets"):
        download_dataset.main(
            DownloadConfig(output_dir=tmp_path, files=["train.parquet"])
        )

    assert captured["public_id"] == "public-runtime-value"
    assert captured["secret_key"] == "secret-runtime-value"  # noqa: S105


def test_download_dataset_fails_when_listing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        download_dataset.numerapi,
        "NumerAPI",
        lambda **kwargs: _NumerAPI(list_error=OSError("offline")),
    )

    with pytest.raises(RuntimeError, match="Failed to list datasets"):
        download_dataset.main(
            DownloadConfig(output_dir=tmp_path, files=["train.parquet"])
        )


def test_download_dataset_fails_on_partial_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = "v5.2/train.parquet"
    monkeypatch.setattr(
        download_dataset.numerapi,
        "NumerAPI",
        lambda **kwargs: _NumerAPI(
            available=[remote],
            download_error=OSError("network"),
            write_partial_before_error=True,
        ),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        download_dataset.main(
            DownloadConfig(output_dir=tmp_path, files=["train.parquet"])
        )
    destination = tmp_path / "v5.2" / "train.parquet"
    assert not destination.exists()
    assert not destination.with_name("train.parquet.part").exists()


def test_download_dataset_writes_requested_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = "v5.2/train.parquet"
    monkeypatch.setattr(
        download_dataset.numerapi,
        "NumerAPI",
        lambda **kwargs: _NumerAPI(available=[remote]),
    )

    download_dataset.main(DownloadConfig(output_dir=tmp_path, files=["train.parquet"]))

    downloaded = pd.read_parquet(tmp_path / "v5.2" / "train.parquet")
    assert downloaded["value"].tolist() == [1.0]


def test_download_dataset_replaces_preexisting_empty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = "v5.2/train.parquet"
    destination = tmp_path / "v5.2" / "train.parquet"
    destination.parent.mkdir()
    destination.write_bytes(b"")
    client = _NumerAPI(available=[remote])
    monkeypatch.setattr(
        download_dataset.numerapi,
        "NumerAPI",
        lambda **kwargs: client,
    )

    download_dataset.main(DownloadConfig(output_dir=tmp_path, files=["train.parquet"]))

    assert pd.read_parquet(destination)["value"].tolist() == [1.0]


def test_download_dataset_replaces_corrupt_nonempty_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = "v5.2/train.parquet"
    destination = tmp_path / "v5.2" / "train.parquet"
    destination.parent.mkdir()
    destination.write_bytes(b"partial")
    monkeypatch.setattr(
        download_dataset.numerapi,
        "NumerAPI",
        lambda **kwargs: _NumerAPI(available=[remote]),
    )

    download_dataset.main(DownloadConfig(output_dir=tmp_path, files=["train.parquet"]))

    assert pd.read_parquet(destination)["value"].tolist() == [1.0]


def test_download_dataset_rejects_corrupt_successful_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = "v5.2/train.parquet"
    destination = tmp_path / "v5.2" / "train.parquet"
    destination.parent.mkdir()
    destination.write_bytes(b"old corrupt artifact")
    client = _NumerAPI(available=[remote], write_corrupt_success=True)
    monkeypatch.setattr(
        download_dataset.numerapi,
        "NumerAPI",
        lambda **kwargs: client,
    )

    with pytest.raises(RuntimeError, match="download failed"):
        download_dataset.main(
            DownloadConfig(output_dir=tmp_path, files=["train.parquet"])
        )

    assert destination.read_bytes() == b"old corrupt artifact"
