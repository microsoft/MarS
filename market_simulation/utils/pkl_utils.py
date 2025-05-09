# pyright: strict
import logging
import pickle
import shutil
from pathlib import Path
from time import sleep
from typing import Any

import zstandard as zstd


def _save_pkl_zstd(data: object, filepath: Path, compression_level: int = 8) -> None:
    """Save pickle to zstd file."""
    cctx = zstd.ZstdCompressor(level=compression_level)

    # compress the pickle file
    tmp_path = filepath.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        f.write(cctx.compress(pickle.dumps(data)))

    shutil.move(str(tmp_path), str(filepath))


def _load_pkl_zstd(filepath: Path) -> object:
    """Load pickle from zstd file, it's faster compared with gzip."""
    # create a decompressor object
    dctx = zstd.ZstdDecompressor()

    # decompress the zstd file
    with filepath.open("rb") as f:
        decompressed_data = dctx.decompress(f.read())

    # load the pickle file
    return pickle.loads(decompressed_data)


def unzip_zstd(src_path: Path, dst_path: Path, max_retries: int = 5) -> None:
    """Unzip zstd file to a new file."""
    tmp_path = dst_path.with_suffix(".tmp")
    for retry in range(max_retries + 1):
        try:
            with src_path.open("rb") as compressed_file:
                dctx = zstd.ZstdDecompressor()
                # Decompress the file
                with tmp_path.open("wb") as decompressed_file:
                    dctx.copy_stream(compressed_file, decompressed_file)
                shutil.move(str(tmp_path), str(dst_path))
        except Exception as _:
            sleep(retry)
            logging.exception(f"retry: {retry}, IO exception when unzipping {src_path}")
        else:
            return
    raise OSError(f"failed to unzip {src_path} after {max_retries} retries.")


def save_pkl_zstd(data: object, filepath: Path, compression_level: int = 8, num_retry: int = 10) -> None:
    """Save pickle to zstd file."""
    for retry in range(num_retry + 1):
        try:
            _save_pkl_zstd(data, filepath, compression_level)
            break
        except Exception as _:
            sleep(retry)
            logging.exception(f"retry: {retry}, IO exception when saving pkl {filepath}")


def load_pkl_zstd(filepath: Path, num_retry: int = 10) -> Any:
    """Load pickle from zstd file, it's faster compared with gzip."""
    for retry in range(num_retry + 1):
        try:
            return _load_pkl_zstd(filepath)
        except Exception as _:
            sleep(retry)
            logging.exception(f"retry: {retry}, IO exception when loading pkl {filepath}")
    msg = f"failed to load {filepath} after {num_retry} retries."
    logging.error(msg)
    raise OSError(msg)
