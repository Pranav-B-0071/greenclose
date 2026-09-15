"""The locked holdout, evaluated exactly once (brief invariant 10, phase P10).

The last 24 months are never touched by any other command: every other run stops at
`holdout.start`. This module evaluates them once, writes an indelible record of the touch, and
refuses to run a second time.

If the holdout disagrees with the walk-forward, believe the holdout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from greenclose.backtest.engine import DayPanel
from greenclose.config import AppConfig, Paths
from greenclose.evaluation.ladder import build_head, evaluate_predictions, model_columns_of
from greenclose.models.train import walk_forward_predict
from greenclose.ops.provenance import Provenance, git_sha, utc_now_iso
from greenclose.ops.research_log import ResearchLog, Trial
from greenclose.pipeline_io import load_features, load_labels, open_store

TOUCH_FILE = "holdout_touch.log"


class HoldoutAlreadyTouched(RuntimeError):
    pass


def touch_path(paths: Paths) -> Path:
    return paths.research_dir / TOUCH_FILE


def already_touched(paths: Paths) -> bool:
    p = touch_path(paths)
    return p.exists() and bool(p.read_text(encoding="utf-8").strip())


def record_touch(paths: Paths, payload: dict[str, object]) -> Path:
    p = touch_path(paths)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return p


def run_holdout(
    cfg: AppConfig,
    paths: Paths,
    *,
    snapshot: str | None = None,
    head_kind: str = "lgbm_huber",
    seed: int = 0,
    n_boot: int = 10_000,
) -> list[str]:
    if already_touched(paths):
        prior = touch_path(paths).read_text(encoding="utf-8").strip().splitlines()[0]
        raise HoldoutAlreadyTouched(
            "The locked holdout has already been evaluated. It may be touched exactly once.\n"
            f"First touch: {prior}\n"
            "If you genuinely need to re-run it, that is a new experiment: say so in writing, move the "
            f"existing {TOUCH_FILE} aside deliberately, and report both results."
        )

    store = open_store(paths, snapshot)
    feats = load_features(paths, store.snapshot_id)
    labels = load_labels(paths, store.snapshot_id, cfg.base.labels.horizon_days, cfg.base.labels.horizon)
    panel = DayPanel.from_features(feats)
    holdout_start = pd.Timestamp(cfg.base.holdout.start)
    dates = panel.dates[panel.dates >= holdout_start]
    if not len(dates):
        raise ValueError(f"no dates at or after {holdout_start.date()} in this snapshot")

    idx = store.get_index(cutoff=feats.index.get_level_values("date").max())
    nifty = idx.loc[idx["symbol"] == cfg.base.data.benchmark].set_index("date")["close"]
    nifty_ret = nifty.reindex(nifty.index.union(dates)).ffill().reindex(dates).pct_change().fillna(0.0)

    # the same monthly walk-forward as everywhere else, now stepping through the holdout window
    run = walk_forward_predict(
        feats,
        labels,
        model_columns=model_columns_of(feats),
        head=build_head(head_kind, cfg),
        cfg=cfg,
        oos_dates=dates,
        seed=seed,
    )
    ev = evaluate_predictions(
        f"HOLDOUT_{head_kind}_s{seed}",
        run.predictions,
        cfg=cfg,
        paths=paths,
        panel=panel,
        labels=labels,
        dates=dates,
        nifty_ret=nifty_ret,
        snapshot_id=store.snapshot_id,
        seed=seed,
        description=f"LOCKED HOLDOUT, evaluated once: {head_kind}",
        n_boot=n_boot,
        log_to=None,
    )
    prov = Provenance.create(
        config_hash=cfg.hash(), snapshot_id=store.snapshot_id, seed=seed, label="holdout"
    )
    log = ResearchLog(paths.research_dir / "research_log.jsonl")
    log.append(
        Trial(
            description=f"LOCKED HOLDOUT once: {head_kind} {dates[0].date()}..{dates[-1].date()}",
            family="holdout",
            config_hash=cfg.hash(),
            snapshot_id=store.snapshot_id,
            git_sha=git_sha(paths.root),
            seed=seed,
            oos_ic=ev.family_a.get("ic_mean"),
            oos_sharpe=ev.family_b.get("sharpe_per_period"),
            oos_net_return_pa=ev.family_b.get("net_return_pa"),
            n_days=len(dates),
            touched_holdout=True,
            counts_for_dsr=False,
            extra={"head": head_kind},
        )
    )
    payload = {
        "touched_at": utc_now_iso(),
        "git_sha": git_sha(paths.root),
        "config_hash": cfg.hash(),
        "snapshot_id": store.snapshot_id,
        "run_id": prov.run_id,
        "head": head_kind,
        "seed": seed,
        "holdout_start": str(dates[0].date()),
        "holdout_end": str(dates[-1].date()),
        "n_days": len(dates),
        "ic_mean": ev.family_a.get("ic_mean"),
        "ic_t_nw": ev.family_a.get("ic_t_nw"),
        "net_return_pa": ev.family_b.get("net_return_pa"),
        "gross_return_pa": ev.family_b.get("gross_return_pa"),
        "sharpe": ev.family_b.get("sharpe"),
        "costs_over_gross_pnl": ev.family_b.get("costs_over_gross_pnl"),
    }
    record_touch(paths, payload)
    (paths.research_dir / "holdout_result.json").write_text(
        json.dumps(
            {"summary": payload, "family_a": ev.family_a, "family_b": ev.family_b}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    return [
        f"LOCKED HOLDOUT evaluated once, {dates[0].date()}..{dates[-1].date()} ({len(dates)} days)",
        f"  head {head_kind}  seed {seed}  git {payload['git_sha']}  config {payload['config_hash']}",
        f"  rank IC        {ev.family_a.get('ic_mean', float('nan')):+.4f} "
        f"(NW t {ev.family_a.get('ic_t_nw', float('nan')):+.2f})",
        f"  gross p.a.     {ev.family_b.get('gross_return_pa', float('nan')):+.1%}",
        f"  net p.a.       {ev.family_b.get('net_return_pa', float('nan')):+.1%}",
        f"  Sharpe         {ev.family_b.get('sharpe', float('nan')):+.2f}",
        f"  touch recorded in {touch_path(paths)}",
        "  This result stands whether or not it agrees with the walk-forward. If it disagrees, "
        "believe the holdout.",
    ]
