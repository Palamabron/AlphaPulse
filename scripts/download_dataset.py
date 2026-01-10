import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


import numerapi
import tyro
from dotenv import load_dotenv
from loguru import logger


load_dotenv()


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
    files: Optional[List[str]] = None
    public_id: Optional[str] = field(
        default_factory=lambda: os.getenv("NUMERAI_PUBLIC_API_KEY")
    )
    secret_key: Optional[str] = field(
        default_factory=lambda: os.getenv("NUMERAI_PRIVATE_API_KEY")
    )

    def __post_init__(self):
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

    napi = numerapi.NumerAPI(
        public_id=config.public_id, secret_key=config.secret_key, verbosity="warning"
    )

    try:
        available_files = napi.list_datasets()
    except Exception as e:
        logger.error(f"Failed to list datasets from Numerai API: {e}")
        return

    logger.info(f"Starting download to: {versioned_output_dir.absolute()}")

    for filename in config.files:
        full_remote_path = f"{config.dataset_version}/{filename}"
        dest_path = versioned_output_dir / filename

        if full_remote_path not in available_files:
            logger.warning(f"File not found on remote: {full_remote_path}. Skipping.")
            continue

        if dest_path.exists():
            logger.info(f"File already exists, skipping download: {filename}")
            continue

        try:
            logger.info(f"Downloading: {full_remote_path}")
            napi.download_dataset(full_remote_path, str(dest_path))
            logger.success(f"Successfully downloaded: {filename}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")

    logger.success("Download process completed.")


if __name__ == "__main__":
    tyro.cli(main)
