#!/usr/bin/env python3
"""Summarise a worker log into the latency tables used by implementation_plan.md.

Reproduces the analysis that produced the 2026-08-24 baseline so before/after
comparisons are a single command instead of ad-hoc shell pipelines.

Usage:
    uv run python scripts/analyze_job_log.py .local-tmp/worker_server.log
    uv run python scripts/analyze_job_log.py <log> --json      # machine-readable

Reads the structured JSON lines emitted by src/config/logging_config.py and
reports per-job wall clock, stage durations, LLM call stats (including
thinking tokens), effective concurrency, batch-plan pool utilisation, and
wasted-work counters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

_LLM_DONE = re.compile(
    r"(?P<op>do_llm_translate|do_translate) done:.*?"
    r"model=(?P<model>\S+).*?"
    r"latency_s=(?P<latency>[\d.]+)\s+"
    r"input_chars=(?P<input_chars>\d+)\s+"
    r"prompt_chars=(?P<prompt_chars>\d+)\s+"
    r"out_chars=(?P<out_chars>\d+)\s+"
    r"input_tokens=(?P<input_tokens>\d+)\s+"
    r"output_tokens=(?P<output_tokens>\d+)"
    r"(?:\s+thinking_tokens=(?P<thinking_tokens>\d+))?"
    r"(?:\s+billable_output_tokens=(?P<billable>\d+))?"
    r"(?:\s+batch_items=(?P<batch_items>\d+))?"
)
_STAGE_DONE = re.compile(
    r"Pipeline stage finished: stage=(?P<stage>.+?) part=.*?"
    r"duration_s=(?P<duration>[\d.]+)\s+units=(?P<units>\d+)"
)
_TASK_RECV = re.compile(r"Received translate task request job_id=(?P<job>[0-9a-f-]+)")
_PIPELINE_END = re.compile(r"Translation pipeline finished for job (?P<job>[0-9a-f-]+)")
_JUDGE = re.compile(
    r"Judge evaluate done: model=(?P<model>\S+) latency_s=(?P<latency>[\d.]+)"
    r".*?final=(?P<final>[\d.]+)"
)
_BATCH_PLAN = re.compile(
    r"batch_plan stage=(?P<stage>\S+) batch_count=(?P<count>\d+)"
    r".*?pool_utilisation=(?P<util>[\d.]+)"
)
_VALIDATION_FAIL = re.compile(r"validation failed \((?P<reason>\w+)\)")

# Boilerplate prompt overhead, subtracted to get the real translatable payload.
_BOILERPLATE_TOKENS = 760


def _parse(path: Path) -> list[tuple[dt.datetime, str, str]]:
    """Load structured log lines as (timestamp, logger, message), time-ordered."""
    rows: list[tuple[dt.datetime, str, str]] = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue  # nohup banners, uvicorn startup text, etc.
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        stamp = entry.get("time")
        if not stamp:
            continue
        try:
            parsed = dt.datetime.fromisoformat(stamp)
        except ValueError:
            continue
        rows.append((parsed, entry.get("logger", ""), entry.get("message", "")))
    rows.sort(key=lambda row: row[0])
    return rows


def _pct(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def analyse(rows: list[tuple[dt.datetime, str, str]]) -> dict[str, Any]:
    """Extract every metric this report needs in a single pass."""
    calls: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    jobs: dict[str, dict[str, Any]] = defaultdict(dict)
    judges: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    validation: Counter = Counter()
    fallbacks = 0

    for ts, _logger, message in rows:
        if match := _LLM_DONE.search(message):
            latency = float(match["latency"])
            calls.append(
                {
                    "ts": ts,
                    "start": ts - dt.timedelta(seconds=latency),
                    "op": match["op"],
                    "model": match["model"],
                    "latency": latency,
                    "input_tokens": int(match["input_tokens"]),
                    "output_tokens": int(match["output_tokens"]),
                    "thinking_tokens": int(match["thinking_tokens"] or 0),
                    "batch_items": (
                        int(match["batch_items"]) if match["batch_items"] else None
                    ),
                }
            )
        elif match := _STAGE_DONE.search(message):
            stages.append(
                {
                    "stage": match["stage"],
                    "duration": float(match["duration"]),
                    "units": int(match["units"]),
                }
            )
        elif match := _TASK_RECV.search(message):
            jobs[match["job"]]["start"] = ts
        elif match := _PIPELINE_END.search(message):
            jobs[match["job"]]["end"] = ts
        elif match := _JUDGE.search(message):
            judges.append(
                {
                    "model": match["model"],
                    "latency": float(match["latency"]),
                    "final": float(match["final"]),
                }
            )
        elif match := _BATCH_PLAN.search(message):
            plans.append(
                {
                    "stage": match["stage"],
                    "count": int(match["count"]),
                    "util": float(match["util"]),
                }
            )
        elif match := _VALIDATION_FAIL.search(message):
            validation[match["reason"]] += 1
        elif (
            "Fallback to simple translation" in message
            or "fallback re-batching" in message
        ):
            fallbacks += 1

    return {
        "calls": calls,
        "stages": stages,
        "jobs": jobs,
        "judges": judges,
        "plans": plans,
        "validation": validation,
        "fallbacks": fallbacks,
    }


def _report_jobs_and_stages(data: dict[str, Any]) -> None:
    print("=" * 72)
    print("JOB WALL CLOCK (task receipt -> pipeline finished)")
    print("=" * 72)
    for job, span in data["jobs"].items():
        if "start" in span and "end" in span:
            secs = (span["end"] - span["start"]).total_seconds()
            print(f"  {job[:8]}  {secs:7.0f}s  ({secs / 60:.1f} min)")
        else:
            print(f"  {job[:8]}  (incomplete - still running or log truncated)")

    if not data["stages"]:
        return
    print()
    print("=" * 72)
    print("PIPELINE STAGES")
    print("=" * 72)
    total = sum(s["duration"] for s in data["stages"])
    for stage in sorted(data["stages"], key=lambda s: -s["duration"]):
        share = stage["duration"] / total * 100 if total else 0
        print(
            f"  {stage['stage']:44s} {stage['duration']:7.1f}s  "
            f"{share:4.1f}%  units={stage['units']}"
        )
    print(f"  {'TOTAL':44s} {total:7.1f}s")


def _report_llm(calls: list[dict[str, Any]]) -> None:
    print()
    print("=" * 72)
    print("LLM CALLS")
    print("=" * 72)
    if not calls:
        print("  (none found)")
        return

    latencies = [c["latency"] for c in calls]
    out_tokens = sum(c["output_tokens"] for c in calls)
    thinking = sum(c["thinking_tokens"] for c in calls)
    total_latency = sum(latencies) or 1.0
    wall = (
        max(c["ts"] for c in calls) - min(c["start"] for c in calls)
    ).total_seconds() or 1.0

    print(f"  calls               : {len(calls)}")
    print(f"  models              : {dict(Counter(c['model'] for c in calls))}")
    print(
        f"  latency med/p90/max : {_pct(latencies, 0.5):.1f}s / "
        f"{_pct(latencies, 0.9):.1f}s / {max(latencies):.1f}s"
    )
    print(f"  output tokens       : {out_tokens}")
    note = " <- billed as output" if thinking else " (not reported by this build)"
    print(f"  thinking tokens     : {thinking}{note}")
    print(f"  output tok/s        : {out_tokens / total_latency:.1f}")
    print(f"  sum latency         : {total_latency:.0f}s over {wall:.0f}s wall")
    print(f"  eff. concurrency    : {total_latency / wall:.2f}")

    sized = [c["batch_items"] for c in calls if c["batch_items"]]
    if sized:
        print(
            f"  batch_items med/max : "
            f"{_pct([float(i) for i in sized], 0.5):.0f} / {max(sized)}"
        )

    print()
    print("  payload tokens (input - ~760 boilerplate) vs efficiency:")
    print(f"    {'range':>14} {'n':>4} {'med_lat':>9} {'out_tok/s':>10}")
    for low, high in (
        (0, 100),
        (100, 500),
        (500, 1000),
        (1000, 2000),
        (2000, 5000),
        (5000, 10**9),
    ):
        chosen = [
            c
            for c in calls
            if low <= max(c["input_tokens"] - _BOILERPLATE_TOKENS, 0) < high
        ]
        if not chosen:
            continue
        lat = sum(c["latency"] for c in chosen) or 1.0
        tok = sum(c["output_tokens"] for c in chosen)
        label = f"{low}-{high if high < 10**9 else '+'}"
        med = _pct([c["latency"] for c in chosen], 0.5)
        print(f"    {label:>14} {len(chosen):4d} {med:8.1f}s {tok / lat:10.1f}")


def _report_extras(data: dict[str, Any]) -> None:
    if data["plans"]:
        print()
        print("=" * 72)
        print("BATCH PLANS (pool utilisation)")
        print("=" * 72)
        for plan in data["plans"]:
            flag = "" if plan["util"] >= 0.8 else "   <- pool under-filled"
            print(
                f"  {plan['stage']:26s} batches={plan['count']:4d}  "
                f"utilisation={plan['util']:.2f}{flag}"
            )

    if data["judges"]:
        print()
        print("=" * 72)
        print("QUALITY JUDGE")
        print("=" * 72)
        for judge in data["judges"]:
            print(
                f"  {judge['model']:22s} latency={judge['latency']:6.1f}s  "
                f"final={judge['final']:.3f}"
            )

    if data["validation"] or data["fallbacks"]:
        print()
        print("=" * 72)
        print("WASTED WORK")
        print("=" * 72)
        for reason, count in data["validation"].most_common():
            print(f"  validation failed ({reason}): {count}")
        print(f"  fallback events            : {data['fallbacks']}")


def report(data: dict[str, Any]) -> None:
    _report_jobs_and_stages(data)
    _report_llm(data["calls"])
    _report_extras(data)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise a worker log into latency tables.",
    )
    parser.add_argument("log", type=Path, help="worker_server.log to analyse")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    if not args.log.is_file():
        print(f"error: {args.log} not found", file=sys.stderr)
        return 1

    data = analyse(_parse(args.log))

    if args.json:
        calls = data["calls"]
        latency = sum(c["latency"] for c in calls)
        out_tokens = sum(c["output_tokens"] for c in calls)
        print(
            json.dumps(
                {
                    "llm_calls": len(calls),
                    "sum_latency_s": round(latency, 1),
                    "output_tokens": out_tokens,
                    "thinking_tokens": sum(c["thinking_tokens"] for c in calls),
                    "output_tokens_per_s": round(out_tokens / latency, 2)
                    if latency
                    else 0,
                    "stages": data["stages"],
                    "batch_plans": data["plans"],
                    "validation_failures": dict(data["validation"]),
                    "fallback_events": data["fallbacks"],
                },
                indent=2,
            )
        )
    else:
        report(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
