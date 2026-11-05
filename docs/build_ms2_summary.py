"""Rebuild the two-page MS2 summary from the locally verified evaluation report."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "data" / "evaluation.json").read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle("Compact", parent=styles["BodyText"], fontSize=9, leading=11, spaceAfter=5)
    )
    styles.add(
        ParagraphStyle("SmallNote", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=4)
    )
    story = []

    def text(value, style="Compact"):
        story.append(Paragraph(value, styles[style]))

    def response(label, status, value):
        text(f"<b>{label} - {status}.</b> {value}")

    text("MS2 Summary - Predicting Pedestrian Flow", "Title")
    text("Timo Schlumpf | 5 November 2026 | gaeja / P15", "SmallNote")
    text("1. What I achieved in this milestone", "Heading2")
    text(
        "Implemented a Python CLI for live public-data ingestion, timestamp-based historical backfill, "
        "versioned Parquet features, chronological evaluation, baseline prediction archiving and "
        "matured-label MAE monitoring. Shared feature definitions serve historical and live paths. "
        "Pinned dependencies, Docker, CI and an hourly GitHub Actions workflow are included. "
        "First files to inspect: README.md, src/gaeja/features.py, tests/, .github/workflows/features.yml."
    )
    text(
        "<b>Important correction:</b> the export pads Lintheschergasse with zero placeholders before "
        "installation. Filtering verified measured records gives 23,184 real hourly labels from "
        "2024-03-13 23:00 through 2026-11-04 22:00 UTC, not five years. Other sensors stay spatial context."
    )
    text(
        "<b>Availability:</b> the public export was about 16 hours behind on the first fetch. "
        "Historical backfill therefore defaults to an explicit, unverified 24-hour publication delay; "
        "live features use observed first-seen times. At a decision 10 minutes before the next clock "
        "hour, lag1 is not yet complete. Persistence uses the latest available count with a 48-hour "
        "staleness cap, rather than pretending lag1 is available."
    )
    text(
        "<b>Evaluation:</b> frozen split 2026-10-01 00:00 UTC; training labels must be published "
        "before the first holdout decision. Fixed model parameters; 839 identical test hours, through "
        "2026-11-04 22:00 UTC. Historical availability remains an assumption, not a live accuracy claim."
    )
    rows = [["Forecast", "Holdout MAE"]]
    for label, key in [
        ("Latest available persistence", "latest_available_persistence"),
        ("Seasonal naive (lag168)", "seasonal_naive_lag168"),
        ("Target lag/calendar model", "lag_calendar_model"),
    ]:
        rows.append([label, f"{report['mae'][key]:.2f}"])
    oracle = report.get("oracle_lag1_persistence")
    if oracle:
        rows.append(["Lag1 oracle (not deployable at cutoff)", f"{oracle['mae']:.2f}"])
    table = Table(rows, colWidths=[125 * mm, 35 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 4 * mm))
    improvement = 100 * (
        1
        - report["mae"]["lag_calendar_model"]
        / min(report["mae"]["latest_available_persistence"], report["mae"]["seasonal_naive_lag168"])
    )
    text(
        f"Success threshold: MAE &lt;= {report['success_threshold_mae']:.2f} (10% below the better "
        f"deployable baseline). Initial model improvement: {improvement:.1f}%. "
        "GTFS ablation: not yet possible on real point-in-time history."
    )
    text(
        "<b>Verified locally:</b> 18 tests pass with warnings treated as errors; Ruff passes; live "
        "Hystreet ingestion, 48 Open-Meteo forecast rows, backfill, evaluation and a versioned baseline "
        "forecast ran. No live labels have matured yet. Data/reports stay gitignored; commands reproduce them."
    )
    text(
        "<b>Not claimed complete:</b> official GTFS catalog returns HTTP 403; historical coverage is "
        "unsecured. Docker build and container CLI smoke test now pass. GitHub workflows must "
        "be pushed and observed before calling scheduling operational. Artifacts have 30-day retention. "
        "No school-holiday dates or weather-prototype provenance were invented."
    )

    story.append(PageBreak())
    text("2. Response to MS1 reviews", "Heading2")
    text(
        "Every distinct editorial/reviewer concern is addressed below; shared labels are combined. "
        "References are repository files, not claims of uncreated commits.",
        "SmallNote",
    )
    response(
        "E1",
        "Partly",
        "Named the official Swiss GTFS catalog/cookbook in README.md. "
        "gtfs-discover and gtfs-import provide resource discovery, hashed provenance and "
        "availability-aware import (src/gaeja/gtfs.py). Catalog access returns 403 here; earliest "
        "archive fallback now finds 2024/2025/2026 snapshots and downloads a May 2026 ZIP. "
        "Original publication times remain unverified. A user-downloaded 2026-11-04 feed was imported at "
        "2026-11-05 15:34 UTC; 113 next-hour HB arrivals verified. No historical backfill gain claimed.",
    )
    response(
        "A1 / B5",
        "Changed",
        "src/gaeja/evaluation.py compares persistence, weekly baseline and "
        "a target-lag/calendar model on the same holdout. Success is 10% below the better deployable "
        "baseline. Lag1 is separately shown as an unavailable oracle; paired GTFS ablation is "
        "implemented but awaits real schedule history. Results and cutoff rationale are on page 1.",
    )
    response(
        "B2",
        "Partly",
        "Static schedules may only duplicate weekly calendar structure. "
        "The paired with/without-GTFS experiment is implemented on identical covered rows, "
        "with no real result claimed yet. Keep this as the MS3 headline, including a null result; "
        "consider delays/occupancy only if justified (README.md, evaluation.py).",
    )
    response(
        "E2",
        "Partly",
        "Replaced per-event scope with next-hour Zurich HB arrival counts. "
        "Importer handles service exceptions, overnight times and DST; synthetic tests exercise "
        "these rules. The user-supplied national feed now imports and yields real hourly arrivals "
        "(src/gaeja/gtfs.py, tests/test_pipeline.py).",
    )
    response(
        "A3 / B6",
        "Partly",
        "Named Open-Meteo; ingest archives five next-hour forecast variables "
        "with retrieval-time availability. Historical stitched forecasts are not used as "
        "publication vintages. Verified forecast backfill and a separate observation source "
        "remain outstanding (sources.py, features.py, README.md).",
    )
    response(
        "A2",
        "Partly",
        "First-seen timestamps and recent latency logs are implemented; first "
        "fetch found the export around 16 hours stale. Live persistence selects the actual latest "
        "available interval, and short lags stay missing. Historical default is a labeled 24-hour "
        "delay assumption, not validated historical latency. More scheduled samples are needed "
        "(sources.py, features.py, data/latency.parquet after ingestion).",
    )
    response(
        "B3",
        "Changed",
        "Core monitor scores immutable archived forecasts against matured "
        "labels and both baselines, logging MAE, sample counts and paired-row comparisons. "
        "No-matured-label state is explicit; no zero-error claim (evaluation.py).",
    )
    response(
        "B4",
        "Changed",
        "README.md FTI diagram connects the feature store to inference and "
        "shows future-known GTFS/weather entering backfill and inference. Model registry is "
        "clearly marked MS3/MS4 work, not a completed component.",
    )
    response(
        "E3",
        "Not yet",
        "README.md explicitly says no weather prototype existed in this "
        "repository and its provenance was not provided. Current ingestion is new code; original "
        "prototype provenance must still be supplied by the author.",
    )
    response(
        "E4",
        "Changed",
        "Sensor 670 remains the only label; other sensors are lagged spatial "
        "features, never pooled targets. The comparison model is target-lags/calendar only "
        "(sources.py, features.py, evaluation.py).",
    )
    response(
        "B1",
        "Rejected",
        "The target remains counted passers-by, explicitly not customers. "
        "README.md narrows the one-hour business rationale to immediate preparation/queue "
        "management/promotions; customer conversion calibration remains a future extension.",
    )
    text(
        "<b>Submission checklist:</b> commit source and this PDF, push, observe CI and the scheduled "
        "pipeline, then tag/push ms2. Resolve GTFS history access and verify publication cadence. "
        "No commit, push, milestone tag or cloud deployment is claimed by this local implementation.",
        "SmallNote",
    )
    output = ROOT / "docs" / "ms2_summary.pdf"
    SimpleDocTemplate(
        str(output),
        pagesize=(210 * mm, 297 * mm),
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=13 * mm,
        bottomMargin=13 * mm,
        title="MS2 Summary - Predicting Pedestrian Flow",
        author="Timo Schlumpf",
    ).build(story)
    from pypdf import PdfReader

    pages = len(PdfReader(output).pages)
    if pages != 2:
        raise RuntimeError(f"Summary must be two pages, got {pages}")
    print(f"Built {output}: {pages} pages; evaluation cutoff {escape(str(report['split']))}")


if __name__ == "__main__":
    main()
