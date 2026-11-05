import json
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from gaeja.evaluation import evaluate, monitor, predict_baseline
from gaeja.features import build_features
from gaeja.gtfs import active_services, aggregate, feed_arrivals, import_feed
from gaeja.sources import normalize_counts
from gaeja.storage import read_parquet, snapshot, write_parquet


def counts():
    times = pd.date_range("2026-09-01", periods=1000, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": times,
            "location_id": 670,
            "count": 100 + np.arange(len(times)) % 24,
            "first_seen_at": times + pd.Timedelta(hours=1, minutes=12),
        }
    )


def test_reject_padded_and_unverified_counts():
    raw = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00Z"] * 3,
            "location_id": [670, 329, 330],
            "pedestrians_count": [0, 3, 4],
            "collection_type": [None, "measured", "measured"],
            "unverified": [False, False, True],
        }
    )
    result = normalize_counts(raw)
    assert result.location_id.tolist() == [329]
    assert result["count"].tolist() == [3]
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_counts(pd.concat([raw.iloc[[1]], raw.iloc[[1]]]))


def test_missing_intervals_are_not_positional(tmp_path):
    data = counts()
    target = data.timestamp.iloc[200]
    data = data[
        ~data.timestamp.isin([target - pd.Timedelta(hours=2), target - pd.Timedelta(hours=4)])
    ]
    result = build_features(
        data, pd.DatetimeIndex([target]), tmp_path, assumed_delay_minutes=15
    ).iloc[0]
    assert pd.isna(result.flow_lag_1)
    assert pd.isna(result.flow_lag_2)
    assert pd.isna(result.flow_mean_3)
    assert result.flow_lag_3 == 100 + 197 % 24
    assert pd.isna(result.gtfs_arrivals)


def test_point_in_time_live_and_backfill(tmp_path):
    data = counts()
    target = data.timestamp.iloc[200]
    targets = pd.DatetimeIndex([target])
    historical = build_features(data, targets, tmp_path, assumed_delay_minutes=15)
    live = build_features(
        data, targets, tmp_path, live=True, decision_at=target - pd.Timedelta(minutes=10)
    )
    assert pd.isna(live.flow_lag_1.iloc[0])
    assert live.flow_lag_2.iloc[0] == historical.flow_lag_2.iloc[0]
    # Publishing lag2 after the decision makes it unavailable without affecting older lags.
    data.loc[data.timestamp == target - pd.Timedelta(hours=2), "first_seen_at"] = target
    delayed = build_features(
        data, targets, tmp_path, live=True, decision_at=target - pd.Timedelta(minutes=10)
    )
    assert pd.isna(delayed.flow_lag_2.iloc[0])
    assert delayed.flow_mean_3.iloc[0] == np.mean([100 + i % 24 for i in (195, 196, 197)])
    assert delayed.flow_lag_3.iloc[0] == live.flow_lag_3.iloc[0]


def test_future_weather_revision_is_excluded(tmp_path):
    data = counts()
    target = data.timestamp.iloc[200]
    weather = pd.DataFrame(
        {
            "valid_time": [target, target],
            "available_at": [target - pd.Timedelta(hours=2), target],
            **{
                v: [2.0, 99.0]
                for v in [
                    "temperature_2m",
                    "precipitation",
                    "relative_humidity_2m",
                    "surface_pressure",
                    "wind_speed_10m",
                ]
            },
        }
    )
    write_parquet(weather, tmp_path / "weather.parquet")
    result = build_features(data, pd.DatetimeIndex([target]), tmp_path)
    assert result.forecast_temperature_2m.iloc[0] == 2.0


def make_feed(tmp_path):
    path = tmp_path / "feed.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "stops.txt",
            "stop_id,stop_name,parent_station\nparent,Zürich HB,\nplatform,Zürich HB, parent\nother,Other,\n",
        )
        archive.writestr(
            "trips.txt", "trip_id,service_id,route_id\na,weekday,r\nb,weekday,r\nc,extra,r\n"
        )
        archive.writestr(
            "stop_times.txt",
            "trip_id,stop_id,arrival_time\na,parent,12:15:00\nb,parent,25:10:00\nc,parent,13:00:00\n",
        )
        archive.writestr(
            "calendar.txt",
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\nweekday,1,1,1,1,1,0,0,20260101,20261231\n",
        )
        archive.writestr(
            "calendar_dates.txt",
            "service_id,date,exception_type\nweekday,20260902,2\nextra,20260902,1\n",
        )
    return path


def test_gtfs_publication_cutoff_exceptions_and_overnight(tmp_path):
    feed = make_feed(tmp_path)
    meta = import_feed(feed, tmp_path, "2026-09-01T00:00Z", "test fixture")
    targets = pd.DatetimeIndex(
        ["2026-09-01T10:00Z", "2026-09-01T23:00Z", "2026-09-02T10:00Z", "2026-09-02T11:00Z"]
    )
    result = aggregate(tmp_path, targets, targets - pd.Timedelta(minutes=10))
    assert result.gtfs_arrivals.tolist() == [1, 1, 0, 1]
    before = aggregate(tmp_path, targets[:1], pd.DatetimeIndex(["2026-08-31T23:00Z"]))
    assert pd.isna(before.gtfs_arrivals.iloc[0])
    directory = tmp_path / "gtfs" / meta["id"]
    events = feed_arrivals(
        directory, pd.Timestamp("2026-09-01T23:00Z"), pd.Timestamp("2026-09-02T00:00Z")
    )
    assert events[0] == pd.Timestamp("2026-09-01T23:10Z")


def test_gtfs_service_exceptions():
    calendar = pd.DataFrame(
        {
            "service_id": ["a"],
            "start_date": ["20260101"],
            "end_date": ["20261231"],
            "wednesday": ["1"],
        }
    )
    exceptions = pd.DataFrame(
        {"service_id": ["a", "b"], "date": ["20260902"] * 2, "exception_type": ["2", "1"]}
    )
    assert active_services(calendar, exceptions, pd.Timestamp("2026-09-02")) == {"b"}


def test_gtfs_dst_uses_service_day_anchor(tmp_path):
    meta = import_feed(make_feed(tmp_path), tmp_path, "2026-01-01T00:00Z")
    directory = tmp_path / "gtfs" / meta["id"]
    calendar = read_parquet(directory / "calendar.parquet")
    calendar["sunday"] = "1"
    write_parquet(calendar, directory / "calendar.parquet")
    events = feed_arrivals(
        directory, pd.Timestamp("2026-10-25T00:00Z"), pd.Timestamp("2026-10-26T00:00Z")
    )
    assert pd.Timestamp("2026-10-25T11:15Z") in events


def test_snapshot_and_evaluation(tmp_path):
    data = counts()
    features = build_features(data, pd.DatetimeIndex(data.timestamp), tmp_path)
    path = snapshot(features, tmp_path, {"mode": "test"})
    assert path.exists()
    assert json.loads(path.with_suffix(".json").read_text())["rows"] == len(features)
    split = data.timestamp.iloc[800]
    result = evaluate(features, split, tmp_path)
    assert result["train_end"] + pd.Timedelta(hours=25) <= result["training_cutoff"]
    assert result["training_cutoff"] < split
    assert result["test_start"] >= split
    assert result["test_rows"] == 200
    assert result["success_threshold_mae"] == 0.9 * min(
        result["mae"]["latest_available_persistence"],
        result["mae"]["seasonal_naive_lag168"],
    )
    assert result["lag1_diagnostic"] is None
    assert "Not yet" in result["gtfs_ablation"]["status"]


def test_live_monitoring_preserves_first_prediction(tmp_path):
    data = counts()
    target = data.timestamp.iloc[200]
    features = build_features(
        data,
        pd.DatetimeIndex([target]),
        tmp_path,
        live=True,
        decision_at=target - pd.Timedelta(minutes=10),
    )
    predict_baseline(features, tmp_path)
    original = read_parquet(tmp_path / "predictions.parquet").prediction.iloc[0]
    features["flow_latest_available"] = 9999
    predict_baseline(features, tmp_path)
    assert read_parquet(tmp_path / "predictions.parquet").prediction.iloc[0] == original
    result = monitor(tmp_path, data, target + pd.Timedelta(hours=2))
    assert result["matured_predictions"] == 1
    assert result["paired_rows"] == 1
    assert result["paired_mae"]["prediction"] == abs(original - data["count"].iloc[200])
    assert monitor(tmp_path, data, target)["matured_predictions"] == 0


def test_target_validation(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        build_features(counts(), pd.date_range("2026-09-01", periods=1), tmp_path)
    with pytest.raises(ValueError, match="hourly"):
        build_features(counts(), pd.DatetimeIndex(["2026-09-01T00:10Z"]), tmp_path)


def test_paired_gtfs_ablation_uses_identical_covered_rows(tmp_path):
    data = counts()
    features = build_features(data, pd.DatetimeIndex(data.timestamp), tmp_path)
    features["gtfs_arrivals"] = np.arange(len(features)) % 10
    features.loc[:250, "gtfs_arrivals"] = np.nan
    result = evaluate(features, data.timestamp.iloc[800], tmp_path)
    assert result["gtfs_ablation"]["status"] == "complete"
    assert result["gtfs_ablation"]["test_rows"] == 200
    assert set(result["gtfs_ablation"]["mae"]) == {"with_gtfs", "without_gtfs"}


def test_revised_value_cannot_enter_earlier_live_decision(tmp_path):
    data = counts()
    target = data.timestamp.iloc[200]
    data["value_available_at"] = data.first_seen_at
    data.loc[data.timestamp == target - pd.Timedelta(hours=2), "value_available_at"] = target
    features = build_features(
        data,
        pd.DatetimeIndex([target]),
        tmp_path,
        live=True,
        decision_at=target - pd.Timedelta(minutes=10),
    )
    assert pd.isna(features.flow_lag_2.iloc[0])
    assert features.flow_age_hours.iloc[0] == 3


def test_live_accepts_microsecond_parquet_timestamps(tmp_path):
    data = counts()
    data["first_seen_at"] = data.first_seen_at.astype("datetime64[us, UTC]")
    target = data.timestamp.iloc[200]
    result = build_features(
        data,
        pd.DatetimeIndex([target]),
        tmp_path,
        live=True,
        decision_at=target - pd.Timedelta(minutes=10),
    )
    assert result.flow_age_hours.iloc[0] == 2
