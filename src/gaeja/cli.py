"""Command-line entry points used identically in Docker and GitHub Actions."""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
import requests

from gaeja.evaluation import evaluate, monitor, predict_baseline
from gaeja.features import build_features
from gaeja.gtfs import discover, download_feed, import_feed
from gaeja.sources import coverage, fetch_counts, fetch_weather
from gaeja.storage import read_parquet, snapshot, utc, write_json

LOG = logging.getLogger("gaeja")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(os.getenv("GAEJA_DATA_DIR", "data")))
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser(
        "ingest", help="Fetch measured counts and archive current forecasts"
    )
    ingest.add_argument("--skip-weather", action="store_true")
    backfill = commands.add_parser("backfill", help="Build historical feature snapshot")
    backfill.add_argument("--start", required=True)
    backfill.add_argument("--end", required=True, help="Exclusive UTC bound")
    backfill.add_argument("--delay-minutes", type=int, default=1440)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--split", required=True, help="Frozen chronological split, UTC")
    commands.add_parser("live", help="Archive next whole-hour features and baseline prediction")
    commands.add_parser("monitor")
    gtfs = commands.add_parser("gtfs-import")
    gtfs.add_argument("archive", type=Path)
    gtfs.add_argument("--available-at", help="Only with independent publication-time evidence")
    gtfs.add_argument("--source", default="local import")
    download = commands.add_parser("gtfs-download")
    download.add_argument(
        "url", help="Official archive ZIP URL; availability remains retrieval time"
    )
    catalog = commands.add_parser("gtfs-discover")
    catalog.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    root = args.data
    root.mkdir(parents=True, exist_ok=True)
    if args.command == "ingest":
        counts = fetch_counts(root)
        result = {"counts": coverage(counts), "weather": "skipped"}
        if not args.skip_weather:
            try:
                weather = fetch_weather(root)
                result["weather"] = f"archived {len(weather)} forecast rows"
            except (requests.RequestException, KeyError, ValueError) as exc:
                result["weather"] = f"unavailable: {exc}"
                LOG.warning("Weather unavailable; counts remain usable: %s", exc)
        write_json(result, root / "coverage.json")
        LOG.info("%s", json.dumps(result, default=str))
    elif args.command == "backfill":
        start, end = utc(args.start), utc(args.end)
        if start >= end:
            parser.error("--start must precede --end")
        counts = read_parquet(root / "counts.parquet")
        if counts.empty:
            parser.error("Run ingest before backfill")
        features = build_features(
            counts,
            pd.date_range(start, end, freq="h", inclusive="left"),
            root,
            assumed_delay_minutes=args.delay_minutes,
        )
        path = snapshot(
            features,
            root,
            {
                "mode": "backfill",
                "start": start,
                "end": end,
                "assumed_delay_minutes": args.delay_minutes,
            },
        )
        LOG.info("Wrote %s rows: %s", len(features), path)
    elif args.command == "evaluate":
        report = evaluate(read_parquet(root / "features.parquet"), args.split, root)
        LOG.info("%s", json.dumps(report, default=str))
    elif args.command == "live":
        now = utc()
        target = now.ceil("h")
        counts = read_parquet(root / "counts.parquet")
        features = build_features(
            counts, pd.DatetimeIndex([target]), root, live=True, decision_at=now
        )
        path = root / "live" / f"{target.strftime('%Y%m%dT%H%MZ')}.parquet"
        if path.exists():
            LOG.info("Prediction already archived; keeping original features")
            return
        from gaeja.storage import write_parquet

        predict_baseline(features, root)
        write_parquet(features, path)
        monitor(root, counts, now)
        LOG.info("Archived baseline forecast for %s", target)
    elif args.command == "monitor":
        LOG.info("%s", monitor(root, read_parquet(root / "counts.parquet")))
    elif args.command == "gtfs-download":
        LOG.info("%s", download_feed(args.url, root))
    elif args.command == "gtfs-import":
        LOG.info("%s", import_feed(args.archive, root, args.available_at, args.source))
    elif args.command == "gtfs-discover":
        try:
            result = discover(args.year)
        except (requests.RequestException, ValueError) as exc:
            write_json(
                {
                    "year": args.year,
                    "status": "unavailable",
                    "error": str(exc),
                    "checked_at": utc(),
                },
                root / "gtfs_catalog.json",
            )
            LOG.error("GTFS history remains unsecured: %s", exc)
            raise SystemExit(1) from exc
        write_json(
            {"year": args.year, "status": "discovered", "resources": result},
            root / "gtfs_catalog.json",
        )
        LOG.info("%s", result)


if __name__ == "__main__":
    main()
