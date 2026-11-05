"""Fixed chronological holdout and paired GTFS ablation, with no imputation of labels."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from gaeja.features import model_columns
from gaeja.storage import read_parquet, utc, write_json, write_parquet


def evaluate(frame: pd.DataFrame, split, output: Path) -> dict:
    split = utc(split)
    frame = frame.sort_values("target_time").copy()
    # Persistence uses the latest actually/assumed available observation, not an unavailable lag1.
    common = frame.dropna(subset=["label", "flow_latest_available", "flow_lag_168"])
    label_ready = (
        common.target_time
        + pd.Timedelta(hours=1)
        + pd.to_timedelta(common.assumed_delay_minutes, unit="m")
    )
    training_cutoff = split - pd.Timedelta(minutes=10)
    train = common[label_ready <= training_cutoff]
    test = common[common.target_time >= split]
    if len(train) < 168 or len(test) < 24:
        raise ValueError(
            "Need at least 168 training hours and 24 holdout hours with both baselines"
        )
    scores = {
        "latest_available_persistence": float(
            mean_absolute_error(test.label, test.flow_latest_available)
        ),
        "seasonal_naive_lag168": float(mean_absolute_error(test.label, test.flow_lag_168)),
    }
    columns = model_columns(train)
    model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15, random_state=42)
    model.fit(train[columns], train.label)
    prediction = np.maximum(0, model.predict(test[columns]))
    scores["lag_calendar_model"] = float(mean_absolute_error(test.label, prediction))
    available_lag1 = test.dropna(subset=["flow_lag_1"])
    lag1 = (
        None
        if available_lag1.empty
        else {
            "rows": len(available_lag1),
            "mae": float(mean_absolute_error(available_lag1.label, available_lag1.flow_lag_1)),
        }
    )
    better = min(scores["latest_available_persistence"], scores["seasonal_naive_lag168"])
    ablation = {"status": "Not yet: insufficient point-in-time GTFS coverage"}
    paired = common.dropna(subset=["gtfs_arrivals"])
    pair_train = paired[
        paired.target_time
        + pd.Timedelta(hours=1)
        + pd.to_timedelta(paired.assumed_delay_minutes, unit="m")
        <= training_cutoff
    ]
    pair_test = paired[paired.target_time >= split]
    if len(pair_train) >= 168 and len(pair_test) >= 24:
        paired_scores = {}
        for use_gtfs in (False, True):
            selected = model_columns(pair_train, gtfs=use_gtfs)
            estimator = HistGradientBoostingRegressor(
                max_iter=100, max_leaf_nodes=15, random_state=42
            )
            estimator.fit(pair_train[selected], pair_train.label)
            predicted = np.maximum(0, estimator.predict(pair_test[selected]))
            paired_scores["with_gtfs" if use_gtfs else "without_gtfs"] = float(
                mean_absolute_error(pair_test.label, predicted)
            )
        ablation = {
            "status": "complete",
            "training_rows": len(pair_train),
            "test_rows": len(pair_test),
            "mae": paired_scores,
        }
    counts = read_parquet(output / "counts.parquet")
    oracle_lag1 = None
    if not counts.empty:
        target_counts = counts[counts.location_id.eq(670)].set_index("timestamp")["count"]
        values = target_counts.reindex(test.target_time - pd.Timedelta(hours=1)).to_numpy()
        valid = ~pd.isna(values)
        if valid.any():
            oracle_lag1 = {
                "mae": float(mean_absolute_error(test.label.to_numpy()[valid], values[valid])),
                "rows": int(valid.sum()),
                "deployable": False,
                "reason": "Hour not complete at pre-hour cutoff",
            }
    report = {
        "oracle_lag1_persistence": oracle_lag1,
        "split": split,
        "training_cutoff": training_cutoff,
        "train_end": train.target_time.max(),
        "test_start": test.target_time.min(),
        "test_end": test.target_time.max(),
        "training_rows": len(train),
        "test_rows": len(test),
        "mae": scores,
        "lag1_diagnostic": lag1,
        "success_threshold_mae": 0.9 * better,
        "success": scores["lag_calendar_model"] <= 0.9 * better,
        "gtfs_ablation": ablation,
        "feature_columns": columns,
        "availability_caveat": "Historical publication times are assumed (default 24h), not proven; training labels obey the same delay. See latency log.",
        "policy": "Forecast target hour before interval starts; persistence uses latest available count, max age 48h.",
    }
    write_json(report, output / "evaluation.json")
    write_parquet(
        pd.DataFrame(
            {"target_time": test.target_time, "label": test.label, "prediction": prediction}
        ),
        output / "holdout_predictions.parquet",
    )
    return report


def monitor(root: Path, counts: pd.DataFrame, now=None) -> dict:
    now = utc(now)
    predictions = read_parquet(root / "predictions.parquet")
    if predictions.empty:
        result = {"matured_predictions": 0, "mae": {}, "status": "No live predictions yet"}
    else:
        labels = counts[counts.location_id.eq(670)][["timestamp", "count"]]
        matured = predictions[predictions.target_time + pd.Timedelta(hours=1) <= now]
        matured = matured.merge(labels, left_on="target_time", right_on="timestamp", how="inner")
        scores = {}
        for column in ("prediction", "persistence", "seasonal"):
            valid = matured.dropna(subset=[column, "count"])
            if not valid.empty:
                scores[column] = {
                    "mae": float(mean_absolute_error(valid["count"], valid[column])),
                    "rows": len(valid),
                }
        # Also compare all three on identical rows whenever the seasonal baseline exists.
        paired = matured.dropna(subset=["prediction", "persistence", "seasonal", "count"])
        paired_scores = (
            {
                column: float(mean_absolute_error(paired["count"], paired[column]))
                for column in ("prediction", "persistence", "seasonal")
            }
            if len(paired)
            else {}
        )
        result = {
            "matured_predictions": len(matured),
            "mae": scores,
            "paired_rows": len(paired),
            "paired_mae": paired_scores,
            "status": "complete" if len(matured) else "Waiting for labels",
        }
        write_parquet(matured, root / "monitoring.parquet")
    write_json({**result, "computed_at": now}, root / "monitoring.json")
    return result


def predict_baseline(features: pd.DataFrame, root: Path) -> None:
    row = features.iloc[0]
    if pd.isna(row.flow_latest_available):
        raise ValueError("Latest available persistence missing; refuse stale live forecast")
    prediction = pd.DataFrame(
        [
            {
                "target_time": row.target_time,
                "decision_at": row.decision_at,
                "prediction": row.flow_latest_available,
                "persistence": row.flow_latest_available,
                "seasonal": row.flow_lag_168,
                "model_version": "baseline-latest-v1",
            }
        ]
    )
    old = read_parquet(root / "predictions.parquet")
    # Never replace a forecast after seeing its outcome or revising its features.
    combined = pd.concat([old, prediction], ignore_index=True).drop_duplicates(
        "target_time", keep="first"
    )
    write_parquet(combined, root / "predictions.parquet")
