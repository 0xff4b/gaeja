"""Public-source clients; availability is distinct from observation/valid time."""

import io
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gaeja.storage import read_parquet, utc, write_parquet

HYSTREET_URL = (
    "https://data.stadt-zuerich.ch/dataset/hystreet_fussgaengerfrequenzen/"
    "download/hystreet_fussgaengerfrequenzen_seit2021.parquet"
)
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
]


def session() -> requests.Session:
    client = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    client.mount("https://", HTTPAdapter(max_retries=retry))
    client.headers["User-Agent"] = "gaeja-ms2/0.1 public-data research"
    return client


def normalize_counts(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "location_id", "pedestrians_count", "collection_type", "unverified"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing count fields: {required - set(raw.columns)}")
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["count"] = pd.to_numeric(frame["pedestrians_count"], errors="coerce")
    # The export pads sensor 670 with zero rows years before installation. These are not labels.
    valid = (
        frame["collection_type"].eq("measured")
        & frame["unverified"].astype(str).str.lower().eq("false")
        & frame["count"].ge(0)
        & frame["timestamp"].eq(frame["timestamp"].dt.floor("h"))
    )
    frame = frame.loc[valid, ["timestamp", "location_id", "count"]]
    if frame.duplicated(["timestamp", "location_id"]).any():
        raise ValueError("Duplicate sensor intervals in source")
    return frame.sort_values(["timestamp", "location_id"]).reset_index(drop=True)


def fetch_counts(root: Path, retrieved_at=None) -> pd.DataFrame:
    response = session().get(HYSTREET_URL, timeout=(15, 180))
    response.raise_for_status()
    retrieved_at = utc(retrieved_at)
    frame = normalize_counts(pd.read_parquet(io.BytesIO(response.content)))
    frame = frame[frame["timestamp"] + pd.Timedelta(hours=1) <= retrieved_at].copy()
    old = read_parquet(root / "counts.parquet")
    if not old.empty:
        previous = old[["timestamp", "location_id", "first_seen_at", "count"]].rename(
            columns={"count": "previous_count"}
        )
        previous["value_available_at"] = old.get("value_available_at", old["first_seen_at"])
        frame = frame.merge(
            previous, on=["timestamp", "location_id"], how="left", validate="one_to_one"
        )
        changed = frame["count"].ne(frame["previous_count"])
        frame.loc[changed, "value_available_at"] = retrieved_at
        frame = frame.drop(columns="previous_count")
    else:
        frame["first_seen_at"] = pd.NaT
        frame["value_available_at"] = retrieved_at
    frame["first_seen_at"] = pd.to_datetime(frame["first_seen_at"], utc=True).fillna(retrieved_at)
    frame["value_available_at"] = pd.to_datetime(frame["value_available_at"], utc=True).fillna(
        retrieved_at
    )
    frame["retrieved_at"] = retrieved_at
    write_parquet(frame, root / "counts.parquet")
    # Only recently observed intervals inform latency; initial historical backfill is not latency.
    recent = frame[frame["timestamp"] >= retrieved_at - pd.Timedelta(hours=48)].copy()
    recent["observed_latency_minutes"] = (
        recent["first_seen_at"] - recent["timestamp"] - pd.Timedelta(hours=1)
    ).dt.total_seconds() / 60
    write_parquet(recent, root / "latency.parquet")
    return frame


def fetch_weather(root: Path, retrieved_at=None) -> pd.DataFrame:
    response = session().get(
        WEATHER_URL,
        params={
            "latitude": 47.378,
            "longitude": 8.539,
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": "UTC",
            "forecast_days": 2,
        },
        timeout=(15, 60),
    )
    response.raise_for_status()
    retrieved_at = utc(retrieved_at)
    hourly = response.json()["hourly"]
    frame = pd.DataFrame(hourly).rename(columns={"time": "valid_time"})
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    frame["available_at"] = retrieved_at
    old = read_parquet(root / "weather.parquet")
    frame = pd.concat([old, frame], ignore_index=True).drop_duplicates(
        ["valid_time", "available_at"], keep="last"
    )
    write_parquet(frame, root / "weather.parquet")
    return frame


def coverage(counts: pd.DataFrame) -> dict:
    result = {}
    for sensor, group in counts.groupby("location_id"):
        expected = len(pd.date_range(group.timestamp.min(), group.timestamp.max(), freq="h"))
        result[str(sensor)] = {
            "first": group.timestamp.min(),
            "last": group.timestamp.max(),
            "measured_hours": len(group),
            "missing_hours": expected - len(group),
        }
    return result
