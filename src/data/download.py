"""Acquire the NASA CMAPSS dataset and extract it to the raw data directory

Usage:
    python -m src.data.download                      # download from config URL
    python -m src.data.download --zip CMAPSSData.zip # extract a local copy
"""

import argparse
import io
import logging
import zipfile
from pathlib import Path

import requests

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
EXPECTED_FILES = [f"{split}_{s}.txt" for s in SUBSETS for split in ("train", "test", "RUL")]


def fetch_zip(url: str, timeout: int = 120) -> bytes:
    """Download a zip archive and return its bytes."""
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def extract_cmapss(archive: bytes, out_dir: Path) -> None:
    """Extract CMAPSS text files from an archive into a flat ``out_dir``.

    The NASA repository wraps ``CMAPSSData.zip`` inside another zip, so nested
    archives are opened recursively. Folder structure inside the archive is
    discarded and only ``.txt`` files are written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            if name.lower().endswith(".zip"):
                extract_cmapss(zf.read(name), out_dir)
            elif name.lower().endswith(".txt"):
                (out_dir / Path(name).name).write_bytes(zf.read(name))


def verify(out_dir: Path) -> list[str]:
    """Return the expected CMAPSS files missing from ``out_dir``."""
    return [f for f in EXPECTED_FILES if not (out_dir / f).is_file()]


def main(argv: list[str] | None = None) -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=config["data"]["url"], help="dataset zip URL")
    parser.add_argument("--zip", type=Path, help="extract a local zip instead of downloading")
    parser.add_argument("--out", type=Path, default=resolve_path(config["data"]["raw_dir"]))
    args = parser.parse_args(argv)

    archive = args.zip.read_bytes() if args.zip else fetch_zip(args.url)
    extract_cmapss(archive, args.out)

    missing = verify(args.out)
    if missing:
        raise SystemExit(f"CMAPSS extraction incomplete, missing: {missing}")
    logger.info("CMAPSS data ready in %s", args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
