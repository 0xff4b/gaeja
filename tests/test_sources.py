import pandas as pd
import pytest
import requests

from gaeja import sources
from gaeja.features import build_features
from gaeja.storage import read_parquet, write_parquet


def test_first_seen_is_retained_and_recent_latency_logged(tmp_path, monkeypatch):
    old = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-11-04T22:00Z"], utc=True),
            "location_id": [670],
            "count": [12],
            "first_seen_at": pd.to_datetime(["2026-11-05T12:00Z"], utc=True),
        }
    )
    write_parquet(old, tmp_path / "counts.parquet")

    class Response:
        content = b"unused"

        def raise_for_status(self):
            pass

    class Client:
        def get(self, *args, **kwargs):
            return Response()

    raw = pd.DataFrame(
        {
            "timestamp": ["2026-11-04T22:00Z", "2026-11-05T15:00Z"],
            "location_id": [670, 670],
            "pedestrians_count": [13, 14],
            "collection_type": ["measured", "measured"],
            "unverified": [False, False],
        }
    )
    monkeypatch.setattr(sources, "session", lambda: Client())
    monkeypatch.setattr(sources.pd, "read_parquet", lambda *a, **k: raw)
    # Avoid patching the storage reader through the shared pandas module.
    monkeypatch.setattr(sources, "read_parquet", lambda _: old)
    result = sources.fetch_counts(tmp_path, "2026-11-05T15:30Z")
    assert len(result) == 1  # Incomplete target hour excluded.
    assert result.first_seen_at.iloc[0] == old.first_seen_at.iloc[0]
    assert result["count"].iloc[0] == 13
    assert result.value_available_at.iloc[0] == pd.Timestamp("2026-11-05T15:30Z")


def test_http_failure_does_not_overwrite_counts(tmp_path, monkeypatch):
    original = pd.DataFrame({"sentinel": [1]})
    write_parquet(original, tmp_path / "counts.parquet")

    class Client:
        def get(self, *args, **kwargs):
            raise requests.Timeout("test")

    monkeypatch.setattr(sources, "session", lambda: Client())
    with pytest.raises(requests.Timeout):
        sources.fetch_counts(tmp_path)
    pd.testing.assert_frame_equal(read_parquet(tmp_path / "counts.parquet"), original)


def test_default_delay_is_conservative_and_latest_age_is_explicit(tmp_path):
    times = pd.date_range("2026-09-01", periods=300, freq="h", tz="UTC")
    data = pd.DataFrame(
        {
            "timestamp": times,
            "location_id": 670,
            "count": range(300),
            "first_seen_at": times + pd.Timedelta(hours=25),
        }
    )
    result = build_features(data, pd.DatetimeIndex([times[200]]), tmp_path).iloc[0]
    assert result.flow_age_hours == 26
    assert result.flow_latest_available == 174
    assert pd.isna(result.flow_lag_24)
    assert result.flow_mean_3 == 173


def test_missing_gtfs_is_not_zero(tmp_path):
    from gaeja.gtfs import aggregate

    targets = pd.date_range("2026-09-01", periods=2, freq="h", tz="UTC")
    result = aggregate(tmp_path, targets, targets - pd.Timedelta(minutes=10))
    assert result.gtfs_arrivals.isna().all()


def test_archive_discovery_uses_official_links_only(monkeypatch):
    from gaeja import gtfs

    class Response:
        text = (
            '<a href="timetable_gtfs/timetable-2024-gtfs2020/feed.zip">old</a>'
            '<a href="timetable_gtfs/timetable-2026-gtfs2020/feed.zip">current</a>'
        )

        def raise_for_status(self):
            pass

    class Client:
        def get(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(gtfs, "session", lambda: Client())
    result = gtfs.archive_resources(2026)
    assert len(result) == 1
    assert result[0]["url"].startswith("https://archive.opentransportdata.swiss/")
    assert result[0]["availability_evidence"] == "unverified until downloaded"


def test_download_rejects_nonofficial_origin(tmp_path):
    from gaeja.gtfs import download_feed

    with pytest.raises(ValueError, match="official"):
        download_feed("https://example.com/feed.zip", tmp_path)
