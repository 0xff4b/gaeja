"""Aggregate scheduled arrivals from dated GTFS snapshots, without later revisions."""

import hashlib
import io
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import urljoin
from zipfile import ZipFile

import pandas as pd

from gaeja.sources import session
from gaeja.storage import read_parquet, utc, write_json, write_parquet

CATALOG_URL = "https://data.opentransportdata.swiss/api/3/action/package_show"
ARCHIVE_URL = "https://archive.opentransportdata.swiss/timetable_gtfs.php"


def archive_resources(year: int) -> list[dict]:
    response = session().get(ARCHIVE_URL, timeout=(15, 60))
    response.raise_for_status()
    links = re.findall(r'href="([^\"]+\.zip)"', response.text)
    return [
        {
            "url": urljoin(ARCHIVE_URL, link),
            "name": link.rsplit("/", 1)[-1],
            "availability_evidence": "unverified until downloaded",
        }
        for link in links
        if f"timetable-{year}-gtfs2020/" in link
    ]


def download_feed(url: str, root: Path) -> dict:
    # Public official archive only; retain server evidence without treating file names as availability.
    if not url.startswith("https://archive.opentransportdata.swiss/timetable_gtfs/"):
        raise ValueError("Expected an official GTFS archive URL")
    response = session().get(url, timeout=(15, 180), stream=True)
    response.raise_for_status()
    retrieved = utc()
    modified = response.headers.get("Last-Modified")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        path = Path(temporary) / "feed.zip"
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
        # Last-Modified records server file modification, not provable original public availability.
        meta = import_feed(path, root, source=url)
    write_json(
        {
            "url": url,
            "retrieved_at": retrieved,
            "http_last_modified": modified,
            "publication_verified": False,
            "snapshot_id": meta["id"],
        },
        root / "gtfs" / meta["id"] / "download_evidence.json",
    )
    return meta


def discover(year: int) -> list[dict]:
    response = session().get(
        CATALOG_URL, params={"id": f"timetable-{year}-gtfs2020"}, timeout=(15, 60)
    )
    if response.status_code == 403:
        return archive_resources(year)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise ValueError("GTFS catalog request failed")
    return [
        {
            "url": r["url"],
            "name": r.get("name"),
            "created": r.get("created"),
            "last_modified": r.get("last_modified"),
        }
        for r in payload["result"]["resources"]
        if ".zip" in r["url"].lower() or r.get("format", "").lower() == "gtfs"
    ]


def import_feed(archive: Path, root: Path, available_at=None, source: str = "local import") -> dict:
    """Default availability is retrieval time, never a date inferred from the filename."""
    available = utc(available_at)
    content = archive.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    directory = root / "gtfs" / digest
    if (directory / "metadata.json").exists():
        return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with ZipFile(io.BytesIO(content)) as feed:

        def read(name, required=True):
            if name not in feed.namelist():
                if required:
                    raise ValueError(f"GTFS missing {name}")
                return pd.DataFrame()
            return pd.read_csv(feed.open(name), dtype=str)

        stops = read("stops.txt")
        # Exact HB name and its child platforms; do not include every stop containing 'HB'.
        station_ids = set(stops.loc[stops.stop_name.eq("Zürich HB"), "stop_id"])
        if not station_ids:
            raise ValueError("No Zürich HB station in feed")
        selected = station_ids | set(
            stops.loc[
                stops.get("parent_station", pd.Series(index=stops.index, dtype=str)).isin(
                    station_ids
                ),
                "stop_id",
            ]
        )
        parts = []
        for chunk in pd.read_csv(feed.open("stop_times.txt"), dtype=str, chunksize=250_000):
            part = chunk[chunk.stop_id.isin(selected)]
            # A trip can visit multiple platform records; count one arrival per trip and instant.
            parts.append(part[["trip_id", "arrival_time"]])
        times = pd.concat(parts, ignore_index=True).drop_duplicates()
        if times.empty:
            raise ValueError("No Zürich HB arrivals in feed")
        trips = read("trips.txt")
        times = times.merge(trips[["trip_id", "service_id"]], on="trip_id", validate="many_to_one")
        times = times.dropna(subset=["arrival_time"])
        calendar = read("calendar.txt", required=False)
        exceptions = read("calendar_dates.txt", required=False)
        if calendar.empty and exceptions.empty:
            raise ValueError("No GTFS service dates")
        dates = []
        if not calendar.empty:
            dates.extend(calendar.start_date.tolist() + calendar.end_date.tolist())
        if not exceptions.empty:
            dates.extend(exceptions.loc[exceptions.exception_type.eq("1"), "date"].tolist())
        if not dates:
            raise ValueError("No active service coverage")
        write_parquet(times, directory / "times.parquet")
        write_parquet(calendar, directory / "calendar.parquet")
        write_parquet(exceptions, directory / "exceptions.parquet")
    metadata = {
        "id": digest,
        "available_at": available.isoformat(),
        "source": source,
        "service_start": min(dates),
        "service_end": max(dates),
        "calendar_days_back": int(times.arrival_time.str.split(":").str[0].astype(int).max() // 24)
        + 1,
        "stop_ids": sorted(selected),
        "arrivals_in_template": len(times),
        "availability_evidence": "operator supplied"
        if available_at is not None
        else "retrieved now",
    }
    write_json(metadata, directory / "metadata.json")
    return metadata


def active_services(calendar: pd.DataFrame, exceptions: pd.DataFrame, day: pd.Timestamp) -> set:
    key = day.strftime("%Y%m%d")
    weekday = day.day_name().lower()
    active = set()
    if not calendar.empty:
        active = set(
            calendar.loc[
                calendar.start_date.le(key) & calendar.end_date.ge(key) & calendar[weekday].eq("1"),
                "service_id",
            ]
        )
    if not exceptions.empty:
        changes = exceptions[exceptions.date.eq(key)]
        active -= set(changes.loc[changes.exception_type.eq("2"), "service_id"])
        active |= set(changes.loc[changes.exception_type.eq("1"), "service_id"])
    return active


def feed_arrivals(directory: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    times = read_parquet(directory / "times.parquet")
    calendar = read_parquet(directory / "calendar.parquet")
    exceptions = read_parquet(directory / "exceptions.parquet")
    split = times.arrival_time.str.split(":", expand=True).astype(int)
    seconds = split[0] * 3600 + split[1] * 60 + split[2]
    if (seconds < 0).any() or (split[1] >= 60).any() or (split[2] >= 60).any():
        raise ValueError("Invalid GTFS arrival time")
    days_back = int(seconds.max() // 86400) + 1
    days = pd.date_range(
        start.tz_convert("Europe/Zurich").date() - pd.Timedelta(days=days_back),
        end.tz_convert("Europe/Zurich").date(),
        freq="D",
    )
    arrivals = []
    for day in days:
        active = active_services(calendar, exceptions, day)
        # GTFS defines service-day time as local noon minus 12 hours, including DST days.
        anchor = (day + pd.Timedelta(hours=12)).tz_localize("Europe/Zurich")
        anchor = anchor.tz_convert("UTC") - pd.Timedelta(hours=12)
        values = anchor + pd.to_timedelta(seconds[times.service_id.isin(active)], unit="s")
        arrivals.extend(values[(values >= start) & (values < end)].tolist())
    return pd.DatetimeIndex(arrivals, tz="UTC").sort_values()


def aggregate(root: Path, targets: pd.DatetimeIndex, decisions: pd.DatetimeIndex) -> pd.DataFrame:
    result = pd.DataFrame(
        index=targets, data={"gtfs_arrivals": float("nan"), "gtfs_snapshot": None}
    )
    feeds = [
        json.loads(p.read_text(encoding="utf-8")) for p in (root / "gtfs").glob("*/metadata.json")
    ]
    feeds.sort(key=lambda m: utc(m["available_at"]))
    assignments = pd.Series(None, index=targets, dtype=object)
    local_dates = targets.tz_convert("Europe/Zurich").strftime("%Y%m%d")
    end_dates = (targets + pd.Timedelta(minutes=59)).tz_convert("Europe/Zurich").strftime("%Y%m%d")
    for meta in feeds:
        # Require preceding service days too, since trips may arrive after 24:00.
        earliest_service_day = (
            targets.tz_convert("Europe/Zurich")
            - pd.Timedelta(days=meta.get("calendar_days_back", 2))
        ).strftime("%Y%m%d")
        mask = (
            (decisions >= utc(meta["available_at"]))
            & (earliest_service_day >= meta["service_start"])
            & (local_dates >= meta["service_start"])
            & (end_dates <= meta["service_end"])
        )
        assignments.loc[mask] = meta["id"]
    for feed_id in assignments.dropna().unique():
        selected = targets[assignments.eq(feed_id)]
        arrivals = feed_arrivals(
            root / "gtfs" / feed_id, selected.min(), selected.max() + pd.Timedelta(hours=1)
        )
        counts = (
            pd.Series(1, index=arrivals).resample("h").sum()
            if len(arrivals)
            else pd.Series(dtype=int)
        )
        result.loc[selected, "gtfs_arrivals"] = counts.reindex(selected, fill_value=0).values
        result.loc[selected, "gtfs_snapshot"] = feed_id
    return result
