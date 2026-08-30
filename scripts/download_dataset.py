import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numerapi
import tyro
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def _artifact_is_valid(path: Path, *, expected_suffix: str | None = None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        suffix = expected_suffix or path.suffix
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            pq.ParquetFile(path)
        elif suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


@dataclass
class DownloadConfig:
    """Configuration parameters for the Numerai dataset downloader.

    Attributes:
        dataset_version (str): The version of the dataset to download (e.g., 'v5.2').
            Defaults to 'v5.2'.
        output_dir (Path): The directory where downloaded files will be saved.
            Defaults to './data'.
        files (Optional[List[str]]): A list of specific filenames to download.
            If None or empty, downloads all standard dataset files.
        public_id (Optional[str]): Numerai Public API Key. Defaults to
            NUMERAI_PUBLIC_API_KEY environment variable.
        secret_key (Optional[str]): Numerai Private API Key. Defaults to
            NUMERAI_PRIVATE_API_KEY environment variable.
    """

    dataset_version: str = "v5.2"
    output_dir: Path = Path("data")
    files: list[str] | None = None
    # Keep credentials out of dataclass defaults. Tyro displays defaults in
    # ``--help``, so resolving environment variables here would leak API keys
    # into terminal output and logs.
    public_id: str | None = None
    secret_key: str | None = None

    def __post_init__(self) -> None:
        if not self.files:
            self.files = [
                "train.parquet",
                "validation.parquet",
                "validation_example_preds.parquet",
                "live.parquet",
                "live_example_preds.parquet",
                "features.json",
                "train_benchmark_models.parquet",
                "validation_benchmark_models.parquet",
                "live_benchmark_models.parquet",
                "meta_model.parquet",
            ]


def main(config: DownloadConfig) -> None:
    """Downloads requested Numerai dataset files using the provided configuration.


    Initializes the NumerAPI client, verifies file availability, and downloads
    files to the specified output directory. Skips files that are not found
    in the remote repository or already exist locally.


    Args:
        config (DownloadConfig): Configuration object containing API credentials,
            target version, output path, and file list.
    """
    versioned_output_dir = config.output_dir / config.dataset_version
    versioned_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing NumerAPI for dataset version: {config.dataset_version}")

    public_id = config.public_id or os.getenv("NUMERAI_PUBLIC_API_KEY")
    secret_key = config.secret_key or os.getenv("NUMERAI_PRIVATE_API_KEY")
    napi = numerapi.NumerAPI(
        public_id=public_id, secret_key=secret_key, verbosity="warning"
    )

    try:
        available_files = napi.list_datasets()
    except Exception as exc:
        raise RuntimeError("Failed to list datasets from Numerai API") from exc

    logger.info(f"Starting download to: {versioned_output_dir.absolute()}")

    failures: list[str] = []
    for filename in config.files or []:
        full_remote_path = f"{config.dataset_version}/{filename}"
        dest_path = versioned_output_dir / filename

        if full_remote_path not in available_files:
            logger.warning(f"File not found on remote: {full_remote_path}. Skipping.")
            failures.append(f"not available: {full_remote_path}")
            continue

        if _artifact_is_valid(dest_path):
            logger.info(f"File already exists, skipping download: {filename}")
            continue
        if dest_path.exists():
            logger.warning(f"Existing file is invalid; downloading again: {filename}")

        part_path = dest_path.with_name(f"{dest_path.name}.part")
        part_path.unlink(missing_ok=True)
        try:
            logger.info(f"Downloading: {full_remote_path}")
            napi.download_dataset(full_remote_path, str(part_path))
            if not _artifact_is_valid(part_path, expected_suffix=dest_path.suffix):
                raise RuntimeError(f"downloaded artifact is invalid: {filename}")
            part_path.replace(dest_path)
            logger.success(f"Successfully downloaded: {filename}")
        except Exception as exc:
            logger.error(f"Failed to download {filename}: {exc}")
            failures.append(f"download failed: {filename}")
        finally:
            part_path.unlink(missing_ok=True)

    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"Dataset download incomplete: {joined}")

    logger.success("Download process completed.")


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
        sys.exit(0)
