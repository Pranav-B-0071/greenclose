#!/usr/bin/env python
"""Nightly forward-prediction runner. Predictions only: no broker, no orders, no money.

Run any evening after the market closes:

    uv run python scripts/daily_forward.py            # use the latest snapshot already on disk
    uv run python scripts/daily_forward.py --fetch    # take a fresh snapshot and rebuild features

It writes one row per symbol to data/forward_log.parquet, back-fills the outcomes of every earlier
date that is now observable, and appends one line to research/forward_log_summary.md.

Exit codes: 0 success, 1 the run failed (the message says why). Nothing here can place an order.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from greenclose.config import Paths, load_config
from greenclose.evaluation.forward import forward_metrics, run_forward
from greenclose.features.build import build
from greenclose.ingest.pipeline import take_snapshot
from greenclose.ops.logging import configure, get_logger
from greenclose.pipeline_io import features_path, load_features, open_store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true", help="take a fresh snapshot and rebuild features first")
    ap.add_argument("--snapshot", default=None, help="snapshot id (default: latest)")
    ap.add_argument("--asof", default="", help="trading date to score (default: the last date in the panel)")
    args = ap.parse_args(argv)

    configure("INFO")
    log = get_logger("daily_forward")
    cfg, paths = load_config(), Paths()
    try:
        if args.fetch:
            target, drift = take_snapshot(cfg, paths, notes="nightly forward runner")
            if drift is not None and drift.changed:
                log.warning("snapshot_drift", lines=len(drift.lines()))
                print("\n".join(drift.lines()))
            store = open_store(paths, target.name)
            last = pd.Timestamp(store.trading_dates(cutoff=pd.Timestamp("2100-01-01"))[-1])
            panel = build(last, store, cfg.universe, cfg)
            out = features_path(paths, store.snapshot_id)
            out.parent.mkdir(parents=True, exist_ok=True)
            panel.frame.to_parquet(out)
            feats = panel.frame
        else:
            store = open_store(paths, args.snapshot)
            feats = load_features(paths, store.snapshot_id)

        res = run_forward(
            cfg, paths, store=store, features=feats, asof=pd.Timestamp(args.asof) if args.asof else None
        )
    except Exception as exc:  # a failed run must be loud and must not write a partial log
        traceback.print_exc()
        print(f"FORWARD RUN FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"scored {res.n_rows} symbols at cut-off {res.asof.date()}; {res.n_selected} selected")
    if len(res.top_picks):
        print(res.top_picks.to_string(index=False))
    else:
        print("no picks: every candidate abstained on the cost threshold")
    print(f"back-filled outcomes for {len(res.backfilled_dates)} earlier date(s)")
    print(forward_metrics(paths))
    print(res.summary_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
