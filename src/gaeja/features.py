"""Shared timestamp-based feature definitions for backfill and live forecasts."""

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from gaeja.gtfs import aggregate
from gaeja.sources import WEATHER_VARIABLES
from gaeja.storage import read_parquet, utc

LAGS = (1, 2, 3, 6, 24, 168)
WINDOWS = (3, 6, 24, 168)


def build_features(
    counts: pd.DataFrame,
    targets: pd.DatetimeIndex,
    root: Path,
    *,
    live: bool = False,
    decision_at=None,
    assumed_delay_minutes: int = 1440,
) -> pd.DataFrame:
    if targets.tz is None or str(targets.tz) != "UTC" or not targets.is_unique:
        raise ValueError("Targets must be unique timezone-aware UTC timestamps")
    if len(targets) == 0 or not targets.equals(targets.floor("h")):
        raise ValueError("Targets must be nonempty exact hourly interval starts")
    if assumed_delay_minutes < 0:
        raise ValueError("Delay cannot be negative")
    decisions = targets - pd.Timedelta(minutes=10)
    if live:
        decision = utc(decision_at)
        if len(targets) != 1 or decision >= targets[0]:
            raise ValueError("Live prediction must precede its one target interval")
        decisions = pd.DatetimeIndex([decision])
    frame = pd.DataFrame(index=targets)
    frame.index.name = "target_time"
    frame["decision_at"] = decisions
    frame["availability_mode"] = "first_seen" if live else "assumed_delay"
    frame["assumed_delay_minutes"] = assumed_delay_minutes if not live else np.nan
    local = targets.tz_convert("Europe/Zurich")
    calendar = holidays.Switzerland(subdiv="ZH", years=sorted(set(local.year)))
    frame["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    frame["weekday"] = local.dayofweek
    frame["month"] = local.month
    frame["is_weekend"] = (local.dayofweek >= 5).astype(int)
    frame["public_holiday"] = [int(day in calendar) for day in local.date]
    school = root.parent / "config" / "school_holidays.csv"
    frame["school_holiday"] = np.nan
    if school.exists():
        periods = pd.read_csv(school)
        known_years = set(pd.to_datetime(periods.start).dt.year) | set(
            pd.to_datetime(periods.end).dt.year
        )
        frame["school_holiday"] = [0.0 if day.year in known_years else np.nan for day in local.date]
        for period in periods.itertuples():
            mask = (local.date >= pd.Timestamp(period.start).date()) & (
                local.date <= pd.Timestamp(period.end).date()
            )
            frame.loc[mask, "school_holiday"] = 1.0
    for sensor, group in counts.groupby("location_id"):
        values = group.set_index("timestamp")["count"].sort_index()
        if live:
            availability_column = (
                "value_available_at" if "value_available_at" in group else "first_seen_at"
            )
            availability = group.set_index("timestamp")[availability_column]
        else:
            availability = pd.Series(
                values.index + pd.Timedelta(hours=1, minutes=assumed_delay_minutes),
                index=values.index,
            )
        prefix = "flow" if int(sensor) == 670 else f"spatial_{sensor}"
        if int(sensor) == 670:
            historical = pd.DataFrame(
                {
                    "observed_time": values.index,
                    "value": values.values,
                    "available_at": availability.reindex(values.index).values,
                }
            )
            historical["available_at"] = pd.to_datetime(historical["available_at"], utc=True)
            historical = historical.dropna(subset=["available_at"]).sort_values("available_at")
            # Cumulative latest event-time prevents late revisions to older intervals regressing state.
            historical["latest_time"] = historical.observed_time.cummax()
            selected = pd.merge_asof(
                pd.DataFrame({"decision_at": decisions}).sort_values("decision_at"),
                historical,
                left_on="decision_at",
                right_on="available_at",
                direction="backward",
            )
            latest_times = pd.DatetimeIndex(selected.latest_time)
            # A decision cannot use an unfinished interval even if upstream metadata is malformed.
            unfinished = latest_times + pd.Timedelta(hours=1) > decisions
            latest_times = pd.DatetimeIndex(latest_times.where(~unfinished, pd.NaT))
            age = (targets - latest_times).total_seconds() / 3600
            latest_values = values.reindex(latest_times).to_numpy(dtype=float)
            latest_values[(age < 1) | (age > 48)] = np.nan
            frame["flow_latest_available"] = latest_values
            frame["flow_age_hours"] = age
        lags = LAGS if int(sensor) == 670 else (2, 24, 168)
        for lag in lags:
            times = targets - pd.Timedelta(hours=lag)
            observed = values.reindex(times).to_numpy(dtype=float)
            available = pd.to_datetime(availability.reindex(times).to_numpy(), utc=True)
            observed[~((available <= decisions) & (times + pd.Timedelta(hours=1) <= decisions))] = (
                np.nan
            )
            frame[f"{prefix}_lag_{lag}"] = observed
        if int(sensor) == 670:
            latest_lag = max(2, int(np.ceil((assumed_delay_minutes + 70) / 60))) if not live else 2
            grid = pd.date_range(
                targets.min() - pd.Timedelta(hours=latest_lag + 168), targets.max(), freq="h"
            )
            complete = values.reindex(grid)
            # Every interval must be present and published; no gap filling.
            for window in WINDOWS:
                rolling = complete.rolling(window, min_periods=window).mean()
                available = availability.reindex(grid)
                latest = (
                    pd.to_datetime(available, utc=True).astype("int64").where(available.notna())
                )
                latest = latest.rolling(window, min_periods=window).max()
                times = latest_times
                result = rolling.reindex(times).to_numpy(dtype=float)
                cutoff = latest.reindex(times).to_numpy()
                result[~(cutoff <= decisions.asi8)] = np.nan
                frame[f"flow_mean_{window}"] = result
            frame["label"] = values.reindex(targets).to_numpy(dtype=float)
    if "label" not in frame:
        raise ValueError("No measured Lintheschergasse data (sensor 670)")
    weather = read_parquet(root / "weather.parquet")
    for variable in WEATHER_VARIABLES:
        frame[f"forecast_{variable}"] = np.nan
    if not weather.empty:
        candidates = (
            frame[["decision_at"]]
            .reset_index()
            .merge(weather, left_on="target_time", right_on="valid_time", how="inner")
        )
        candidates = candidates[candidates.available_at <= candidates.decision_at]
        latest = candidates.sort_values("available_at").drop_duplicates("target_time", keep="last")
        latest = latest.set_index("target_time")
        for variable in WEATHER_VARIABLES:
            frame[f"forecast_{variable}"] = latest[variable].reindex(targets)
    frame = frame.join(aggregate(root, targets, decisions))
    return frame.reset_index()


def model_columns(frame: pd.DataFrame, *, gtfs=False, weather=False) -> list[str]:
    columns = [
        c
        for c in frame
        if c.startswith("flow_")
        or c
        in {
            "hour_sin",
            "hour_cos",
            "weekday",
            "month",
            "is_weekend",
            "public_holiday",
            "school_holiday",
        }
    ]
    if weather:
        columns.extend(c for c in frame if c.startswith("forecast_"))
    if gtfs:
        columns.append("gtfs_arrivals")
    return [c for c in columns if frame[c].notna().any()]
