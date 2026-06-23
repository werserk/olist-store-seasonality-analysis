#!/usr/bin/env python3
"""Download and extract the Olist Kaggle dataset registered for this project.

Requires Kaggle credentials via ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "data" / "datasets.json"
ARCHIVE_DIR = ROOT / "data" / "raw" / "archives"
RAW_DIR = ROOT / "data" / "raw"


@dataclass(frozen=True)
class Dataset:
    id: str
    title: str
    slug: str
    url: str
    role: str
    description: str

    @property
    def archive_dir(self) -> Path:
        return ARCHIVE_DIR / self.id

    @property
    def extract_dir(self) -> Path:
        return RAW_DIR / self.id


def load_registry() -> dict[str, Dataset]:
    with REGISTRY_PATH.open(encoding="utf-8") as file:
        registry: dict[str, Any] = json.load(file)

    datasets = {}
    for item in registry["datasets"]:
        dataset = Dataset(
            id=item["id"],
            title=item["title"],
            slug=item["slug"],
            url=item["url"],
            role=item.get("role", "primary"),
            description=item.get("description", ""),
        )
        datasets[dataset.id] = dataset
    return datasets


def selected_dataset(dataset_id: str) -> Dataset:
    registry = load_registry()
    if dataset_id not in registry:
        available = ", ".join(registry)
        raise SystemExit(f"Unknown dataset: {dataset_id}. Available: {available}")
    return registry[dataset_id]


def kaggle_auth_available() -> bool:
    token_file = Path.home() / ".kaggle" / "kaggle.json"
    env_auth = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    return token_file.exists() or bool(env_auth)


def kaggle_command() -> list[str]:
    if shutil.which("kaggle"):
        return ["kaggle"]

    check = subprocess.run(
        [sys.executable, "-m", "kaggle", "--version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check.returncode == 0:
        return [sys.executable, "-m", "kaggle"]

    raise SystemExit("Kaggle CLI is not available. Install dependencies with: uv sync")


def latest_archive(dataset: Dataset) -> Path:
    archives = sorted(dataset.archive_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime)
    if not archives:
        raise SystemExit(
            f"No .zip archive found in {dataset.archive_dir}. "
            f"Run `make download-data DATASET={dataset.id}` first."
        )
    return archives[-1]


def download_dataset(dataset: Dataset, force: bool) -> Path:
    if not kaggle_auth_available():
        raise SystemExit(
            "Kaggle credentials are not configured. Put kaggle.json in ~/.kaggle/ "
            "or set KAGGLE_USERNAME and KAGGLE_KEY."
        )

    dataset.archive_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *kaggle_command(),
        "datasets",
        "download",
        "-d",
        dataset.slug,
        "-p",
        str(dataset.archive_dir),
    ]
    if force:
        cmd.append("--force")

    subprocess.run(cmd, check=True)
    return latest_archive(dataset)


def extract_archive(dataset: Dataset, archive: Path, force: bool) -> None:
    if dataset.extract_dir.exists() and force:
        shutil.rmtree(dataset.extract_dir)
    dataset.extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(dataset.extract_dir)


def list_command(_: argparse.Namespace) -> None:
    for dataset in load_registry().values():
        print(f"{dataset.id}\t{dataset.slug}\t{dataset.role}\t{dataset.title}")


def download_command(args: argparse.Namespace) -> None:
    dataset = selected_dataset(args.dataset)
    archive = download_dataset(dataset=dataset, force=args.force)
    print(f"Downloaded {dataset.id}: {archive}")


def extract_command(args: argparse.Namespace) -> None:
    dataset = selected_dataset(args.dataset)
    archive = Path(args.archive).expanduser().resolve() if args.archive else latest_archive(dataset)
    if not archive.exists():
        raise SystemExit(f"Archive not found: {archive}")

    extract_archive(dataset=dataset, archive=archive, force=args.force)
    print(f"Extracted {dataset.id}: {archive} -> {dataset.extract_dir}")


def add_dataset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "dataset",
        nargs="?",
        default="olist-brazilian-ecommerce",
        help="Dataset id from data/datasets.json. Defaults to olist-brazilian-ecommerce.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or extract the project Kaggle dataset.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List datasets from data/datasets.json.")
    list_parser.set_defaults(func=list_command)

    download_parser = subparsers.add_parser("download", help="Download the Kaggle dataset archive.")
    add_dataset_argument(download_parser)
    download_parser.add_argument("--force", action="store_true", help="Overwrite an existing downloaded archive.")
    download_parser.set_defaults(func=download_command)

    extract_parser = subparsers.add_parser("extract", help="Extract the downloaded archive into data/raw/.")
    add_dataset_argument(extract_parser)
    extract_parser.add_argument("--archive", help="Path to a specific .zip archive.")
    extract_parser.add_argument("--force", action="store_true", help="Remove the existing extracted directory before extracting.")
    extract_parser.set_defaults(func=extract_command)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
