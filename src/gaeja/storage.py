"""Atomic local Parquet storage and immutable, content-addressed run snapshots."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


def utc(value=None) -> pd.Timestamp:
    value = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if value.tzinfo is None:
        raise ValueError("Timestamp must include a timezone")
    return value.tz_convert("UTC")


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".parquet")
    os.close(fd)
    try:
        frame.to_parquet(name, index=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, default=str, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def snapshot(frame: pd.DataFrame, root: Path, metadata: dict) -> Path:
    stamp = utc().strftime("%Y%m%dT%H%M%S%fZ")
    path = root / "versions" / stamp / "features.parquet"
    write_parquet(frame, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json({**metadata, "sha256": digest, "rows": len(frame)}, path.with_suffix(".json"))
    write_parquet(frame, root / "features.parquet")
    return path
