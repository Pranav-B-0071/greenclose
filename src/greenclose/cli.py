"""GREENCLOSE command line. Paper only: nothing here can place an order."""

from __future__ import annotations

from pathlib import Path

import typer

from greenclose import __version__
from greenclose.config import Paths, load_config
from greenclose.ops.logging import configure

app = typer.Typer(
    name="greenclose",
    help="GREENCLOSE research instrument. PAPER ONLY. No broker, no orders, no money.",
    no_args_is_help=True,
)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging")) -> None:
    configure("DEBUG" if verbose else "INFO")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def fetch(
    from_csv: Path | None = typer.Option(None, "--from-csv", help="Directory of per-symbol CSVs (fallback)"),
    start: str | None = typer.Option(None, help="First date (default: config data.start)"),
    end: str | None = typer.Option(None, help="Exclusive end date (default: tomorrow)"),
    notes: str = typer.Option("", help="Free text stored in the manifest"),
) -> None:
    """Acquire bars into a new immutable, content-hashed snapshot and diff it against the previous one."""
    from greenclose.ingest.pipeline import coverage_table, take_snapshot
    from greenclose.ingest.snapshots import read_bars, read_manifest

    cfg, paths = load_config(), Paths()
    target, drift = take_snapshot(cfg, paths, from_csv=from_csv, start=start, end=end, notes=notes)
    m = read_manifest(target)
    cov = coverage_table(read_bars(target))
    ten_y = int((cov["years"] >= 9.5).sum())
    typer.echo(f"snapshot      {m.snapshot_id}")
    typer.echo(f"content_sha   {m.content_sha256}")
    typer.echo(f"rows          {m.row_count}  index_rows {m.index_row_count}")
    typer.echo(f"symbols       {len(m.symbols)} fetched, {len(m.failed_symbols)} failed {m.failed_symbols}")
    typer.echo(f"range         {m.start} .. {m.end}")
    typer.echo(f">=9.5y bars   {ten_y} of {len(cfg.universe.all_symbols)} symbols")
    typer.echo(cov.to_string())
    if drift is None:
        typer.echo("drift         (first snapshot, nothing to compare)")
    else:
        typer.echo("\n".join(drift.lines()))


@app.command()
def validate(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    apply_drops: bool = typer.Option(
        False, "--apply-drops", help="Record failing symbols in config/universe.yaml"
    ),
) -> None:
    """Data-quality report (reports/data_quality.html) with a mechanical verdict line."""
    from greenclose.ingest.snapshots import read_bars, read_index, read_manifest, resolve_snapshot
    from greenclose.ingest.validate import apply_drops as _apply
    from greenclose.ingest.validate import render_html
    from greenclose.ingest.validate import validate as _validate

    cfg, paths = load_config(), Paths()
    d = resolve_snapshot(paths.data_dir / "snapshots", snapshot)
    m = read_manifest(d)
    res = _validate(
        read_bars(d),
        read_index(d),
        snapshot_id=m.snapshot_id,
        vix_symbol=cfg.base.data.vix,
        vix_min_coverage=cfg.base.data.vix_min_coverage,
        expected_symbols=cfg.universe.all_symbols,
    )
    out = paths.reports_dir / "data_quality.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(res), encoding="utf-8")
    typer.echo(res.calendar.to_string(index=False))
    typer.echo(res.coverage.to_string(index=False))
    typer.echo(f"suspicious prints: {len(res.suspicious)}   split/adjustment artefacts: {len(res.splits)}")
    typer.echo(f"India VIX usable: {res.vix_usable}")
    typer.echo(f"phantom days removed: {res.phantom_days}")
    for s, r in res.flagged.items():
        typer.echo(f"FLAG {s}: {'; '.join(r)}")
    for s, r in res.drop.items():
        typer.echo(f"DROP {s}: {r}")
    typer.echo(f"report: {out}")
    typer.echo(f"VERDICT: {res.verdict_line}")
    if apply_drops:
        _apply(paths.config_dir / "universe.yaml", res.drop)
        typer.echo(f"universe.yaml updated: dropped={sorted(res.drop)}")


@app.command()
def features(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    check_truncation: int = typer.Option(0, help="Also verify truncation invariance at N random cut-offs"),
) -> None:
    """Build the point-in-time feature panel at the last available date and cache it under data/features."""
    import time

    import numpy as np
    import pandas as pd

    from greenclose.features.build import build, nan_report
    from greenclose.ingest.store import ParquetBarStore

    cfg, paths = load_config(), Paths()
    store = ParquetBarStore.open(paths.data_dir / "snapshots", snapshot)
    last = pd.Timestamp(store.trading_dates(cutoff=pd.Timestamp.max.normalize() - pd.Timedelta(days=1))[-1])
    t0 = time.perf_counter()
    panel = build(last, store, cfg.universe, cfg)
    dt = time.perf_counter() - t0
    out_dir = paths.data_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{store.snapshot_id}__{panel.registry_version}.parquet"
    panel.frame.to_parquet(out)
    typer.echo(f"snapshot {store.snapshot_id}  registry {panel.registry_version}  built in {dt:.1f}s")
    typer.echo(
        f"rows {len(panel.frame)}  dates {panel.frame.index.get_level_values(0).nunique()}  "
        f"symbols {panel.frame.index.get_level_values(1).nunique()}"
    )
    typer.echo(
        f"model features {panel.n_model_features} (tier1 {len(panel.tier1)}, tier2 {len(panel.tier2)})"
        f"  omitted {panel.omitted}  skipped days {len(panel.skipped_days)}"
    )
    typer.echo(nan_report(panel).to_string())
    typer.echo(f"saved {out}")
    if check_truncation > 0:
        dates = panel.frame.index.get_level_values(0).unique()
        rng = np.random.default_rng(13)
        picks = sorted(rng.choice(np.arange(300, len(dates) - 1), size=check_truncation, replace=False))
        for i in picks:
            t = pd.Timestamp(dates[i])
            part = build(t, store, cfg.universe, cfg).frame
            expected = panel.frame.loc[panel.frame.index.get_level_values(0) <= t]
            pd.testing.assert_frame_equal(part, expected, check_exact=True, obj=f"cutoff={t.date()}")
            typer.echo(f"truncation invariance at {t.date()}: identical ({len(part)} rows)")
        typer.echo(f"TRUNCATION INVARIANCE: PASS for {check_truncation} random cut-offs")


@app.command()
def labels(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    horizon_days: int | None = typer.Option(None, help="Forward horizon in trading days (default: config)"),
) -> None:
    """Compute the six label variants aligned to the cached feature panel and save them under data/labels."""
    from greenclose.labels.targets import build_labels, summarise
    from greenclose.pipeline_io import load_features, open_store

    cfg, paths = load_config(), Paths()
    store = open_store(paths, snapshot)
    feats = load_features(paths, store.snapshot_id)
    lp = build_labels(feats, store, cfg, horizon_days=horizon_days)
    out_dir = paths.data_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{store.snapshot_id}__h{lp.horizon_days}_{lp.base}.parquet"
    lp.frame.to_parquet(out)
    typer.echo(f"snapshot {store.snapshot_id}  horizon {lp.horizon_days}d  base {lp.base}")
    typer.echo(summarise(lp).to_string())
    typer.echo(f"saved {out}")


@app.command()
def baselines(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    b0_sims: int = typer.Option(1000, help="Number of uniform-random simulations for B0"),
    n_boot: int = typer.Option(10_000, help="Stationary-bootstrap resamples for confidence intervals"),
    jobs: int = typer.Option(1, help="Worker processes for the B0 null distribution"),
) -> None:
    """Run baselines B0-B5 through the shared decision layer on the common out-of-sample window."""
    import time

    from greenclose.backtest.engine import DayPanel
    from greenclose.evaluation.baselines import format_table, run_baselines
    from greenclose.ops.research_log import ResearchLog
    from greenclose.pipeline_io import load_features, load_labels, open_store

    cfg, paths = load_config(), Paths()
    store = open_store(paths, snapshot)
    feats = load_features(paths, store.snapshot_id)
    labels_df = load_labels(paths, store.snapshot_id, cfg.base.labels.horizon_days, cfg.base.labels.horizon)
    panel = DayPanel.from_features(feats)
    idx = store.get_index(cutoff=feats.index.get_level_values("date").max())
    nifty = idx.loc[idx["symbol"] == cfg.base.data.benchmark].set_index("date")["close"]
    log = ResearchLog(paths.research_dir / "research_log.jsonl")
    t0 = time.perf_counter()
    table, b0 = run_baselines(
        cfg,
        paths,
        panel=panel,
        labels=labels_df,
        nifty_close=nifty,
        snapshot_id=store.snapshot_id,
        b0_sims=b0_sims,
        n_boot=n_boot,
        log=log,
        n_universe=len(cfg.universe.symbols),
        jobs=jobs,
    )
    oos = panel.oos_dates(
        min_train_years=cfg.base.walk_forward.min_train_years,
        start=cfg.base.data.start,
        holdout_start=cfg.base.holdout.start,
    )
    typer.echo(f"OOS window: {oos[0].date()} .. {oos[-1].date()} ({len(oos)} days)")
    typer.echo(format_table(table))
    typer.echo(f"research_log rows: {log.count()}  (counting for DSR: {log.count(dsr_only=True)})")
    typer.echo(f"elapsed {time.perf_counter() - t0:.0f}s")


@app.command()
def leakage(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    n_truncation: int = typer.Option(10, help="Random cut-offs for the truncation-invariance check"),
) -> None:
    """Run the six leakage checks against the REAL snapshot. Phase 6 gates Phase 7."""
    import json
    import time

    from greenclose.evaluation.leakage import run_suite
    from greenclose.pipeline_io import load_features, load_labels, open_store

    cfg, paths = load_config(), Paths()
    store = open_store(paths, snapshot)
    feats = load_features(paths, store.snapshot_id)
    labels_df = load_labels(paths, store.snapshot_id, cfg.base.labels.horizon_days, cfg.base.labels.horizon)
    model_columns = [c for c in feats.columns if c.endswith("_rk")] + [
        c for c in feats.columns if c.startswith(("nifty_", "breadth", "xs_", "india_", "avg_pairwise"))
    ]
    t0 = time.perf_counter()
    report = run_suite(
        cfg,
        paths,
        store=store,
        features=feats,
        labels=labels_df,
        model_columns=model_columns,
        n_truncation=n_truncation,
    )
    typer.echo(report.summary())
    typer.echo(f"elapsed {time.perf_counter() - t0:.0f}s")
    out = paths.research_dir / "leakage_suite.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "snapshot_id": report.snapshot_id,
                "ran_at": report.ran_at,
                "n_passed": report.n_passed,
                "n_checks": len(report.checks),
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail, **c.numbers}
                    for c in report.checks
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"saved {out}")
    if not report.all_passed:
        raise typer.Exit(code=1)


@app.command()
def ladder(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    rungs: str = typer.Option("3-8", help="Rung range, e.g. 3-5 or 3-8"),
    seeds: str = typer.Option("", help="Comma-separated seeds (default: config seeds)"),
    jobs: int = typer.Option(1, help="Worker processes for fitting (rung, seed) pairs"),
    n_boot: int = typer.Option(10_000, help="Stationary-bootstrap resamples"),
    h2: bool = typer.Option(True, help="Also run the H2 ranking-vs-classification experiment"),
) -> None:
    """Climb the model ladder: identical folds per rung, paired-bootstrap comparisons, seeds reported."""
    from greenclose.evaluation.ladder_run import run_ladder

    out = run_ladder(snapshot=snapshot, rungs=rungs, seeds=seeds, jobs=jobs, n_boot=n_boot, with_h2=h2)
    for line in out.lines:
        typer.echo(line)


@app.command()
def sweep(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    group: str = typer.Option("decision", help="decision | retrain | all"),
    model: str = typer.Option("", help="Cached rung whose predictions to replay (default: best by IC)"),
    seed: int = typer.Option(0, help="Seed of the cached prediction frame to replay"),
    n_boot: int = typer.Option(2000, help="Stationary-bootstrap resamples per cell"),
) -> None:
    """Run the required sweeps. Decision-layer cells replay cached predictions; retrain cells refit."""
    import time

    import pandas as pd

    from greenclose.backtest.engine import DayPanel
    from greenclose.evaluation.ladder import ladder_rungs, model_columns_of, predictions_path
    from greenclose.evaluation.sweeps import format_sweep, run_decision_sweep, run_retrain_sweep
    from greenclose.ops.research_log import ResearchLog
    from greenclose.pipeline_io import load_features, load_labels, open_store

    cfg, paths = load_config(), Paths()
    store = open_store(paths, snapshot)
    feats = load_features(paths, store.snapshot_id)
    labels_df = load_labels(paths, store.snapshot_id, cfg.base.labels.horizon_days, cfg.base.labels.horizon)
    panel = DayPanel.from_features(feats)
    dates = panel.oos_dates(
        min_train_years=cfg.base.walk_forward.min_train_years,
        start=cfg.base.data.start,
        holdout_start=cfg.base.holdout.start,
    )
    idx = store.get_index(cutoff=feats.index.get_level_values("date").max())
    nifty = idx.loc[idx["symbol"] == cfg.base.data.benchmark].set_index("date")["close"]
    nifty_ret = nifty.reindex(nifty.index.union(dates)).ffill().reindex(dates).pct_change().fillna(0.0)
    log = ResearchLog(paths.research_dir / "research_log.jsonl")
    out_dir = paths.data_dir / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    frames = []

    if group in ("decision", "all"):
        candidates = [
            r for r in ladder_rungs() if predictions_path(paths, r, cfg, store.snapshot_id, seed).exists()
        ]
        if model:
            candidates = [r for r in candidates if r.name == model]
        if not candidates:
            raise typer.BadParameter("no cached predictions; run `greenclose ladder` first")
        rung = candidates[-1]
        preds = pd.read_parquet(predictions_path(paths, rung, cfg, store.snapshot_id, seed))
        typer.echo(f"decision sweep replaying {rung.name} seed {seed} ({len(preds)} rows)")
        df = run_decision_sweep(
            preds,
            cfg=cfg,
            paths=paths,
            panel=panel,
            labels=labels_df,
            dates=dates,
            nifty_ret=nifty_ret,
            snapshot_id=store.snapshot_id,
            model_name=rung.name,
            seed=seed,
            log=log,
            n_boot=n_boot,
        )
        df.to_parquet(out_dir / f"decision__{rung.name}__s{seed}.parquet", index=False)
        frames.append(df)
        typer.echo(format_sweep(df))

    if group in ("retrain", "all"):
        typer.echo("retrain sweep: refitting per horizon and training window")
        df = run_retrain_sweep(
            cfg=cfg,
            paths=paths,
            store=store,
            features=feats,
            panel=panel,
            dates=dates,
            nifty_ret=nifty_ret,
            snapshot_id=store.snapshot_id,
            head_kind=model or "lgbm_huber",
            model_columns=model_columns_of(feats),
            seed=seed,
            log=log,
            n_boot=n_boot,
        )
        df.to_parquet(out_dir / f"retrain__s{seed}.parquet", index=False)
        frames.append(df)
        typer.echo(format_sweep(df))

    typer.echo(f"research_log rows: {log.count()} (counting for DSR: {log.count(dsr_only=True)})")
    typer.echo(f"elapsed {time.perf_counter() - t0:.0f}s")


@app.command()
def report(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    run_id: str = typer.Option("", help="Report file name (default: a timestamped id)"),
) -> None:
    """Render the single self-contained HTML report with a mechanically written verdict block."""
    from greenclose.evaluation.report import gather, verdict_lines, write_report
    from greenclose.ops.provenance import Provenance
    from greenclose.pipeline_io import open_store

    cfg, paths = load_config(), Paths()
    store = open_store(paths, snapshot)
    inp = gather(cfg, paths, store.snapshot_id)
    rid = (
        run_id
        or Provenance.create(
            config_hash=cfg.hash(), snapshot_id=store.snapshot_id, seed=0, root=paths.root
        ).run_id
    )
    out = write_report(inp, run_id=rid)
    for line in verdict_lines(inp):
        typer.echo(line)
    typer.echo(f"saved {out}")


@app.command("pin-model")
def pin_model(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    head: str = typer.Option("lgbm_huber", help="Head kind to freeze for the forward runner"),
    seed: int = typer.Option(0, help="Seed"),
    reason: str = typer.Option("", help="Why this artefact was pinned"),
) -> None:
    """Fit the final artefact on every labelled row and pin it for the forward runner."""
    import json

    from greenclose.evaluation.ladder import build_head, ladder_rungs, model_columns_of, schedule_path
    from greenclose.models.registry import ModelCard, ModelRegistry, make_version
    from greenclose.models.train import train_final
    from greenclose.ops.provenance import git_sha
    from greenclose.pipeline_io import load_features, load_labels, open_store

    cfg, paths = load_config(), Paths()
    store = open_store(paths, snapshot)
    feats = load_features(paths, store.snapshot_id)
    labels_df = load_labels(paths, store.snapshot_id, cfg.base.labels.horizon_days, cfg.base.labels.horizon)
    cols = model_columns_of(feats)
    rung = next((r for r in ladder_rungs() if r.head_kind == head), None)
    sched = None
    if rung is not None:
        f = schedule_path(paths, rung, cfg, store.snapshot_id)
        if f.exists():
            sched = json.loads(f.read_text(encoding="utf-8"))
            typer.echo(f"reusing the walk-forward hyperparameter choice from {f.name}")
    model, pre, meta = train_final(
        feats,
        labels_df,
        model_columns=cols,
        head=build_head(head, cfg),
        cfg=cfg,
        seed=seed,
        hparam_schedule=sched,
    )
    last = feats.index.get_level_values("date").max()
    registry = ModelRegistry(paths.data_dir / "models")
    card = ModelCard(
        model_version=make_version(head, store.snapshot_id, seed, last.date()),
        head_kind=head,
        target=build_head(head, cfg).target,
        seed=seed,
        git_sha=git_sha(paths.root),
        config_hash=cfg.hash(),
        snapshot_id=store.snapshot_id,
        feature_registry=str(feats.attrs.get("registry_version", "")),
        model_columns=cols,
        train_start=meta["train_start"],
        train_end=meta["train_end"],
        hyperparameters=meta["hyperparameters"],
        notes=reason or "pinned for the forward prediction log",
    )
    d = registry.save(card, model, pre)
    registry.pin(card.model_version, reason=reason)
    typer.echo(f"trained on {meta['n_train_rows']:,} rows, {meta['train_start']}..{meta['train_end']}")
    typer.echo(f"pinned {card.model_version}")
    typer.echo(f"saved {d}")


@app.command()
def forward(
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    asof: str = typer.Option("", help="Trading date to score (default: the last date in the panel)"),
    fetch_first: bool = typer.Option(
        False, "--fetch", help="Take a fresh snapshot and rebuild features first"
    ),
) -> None:
    """Append tonight's predictions for every symbol and back-fill yesterday's outcomes. No orders."""
    import pandas as pd

    from greenclose.evaluation.forward import forward_metrics, run_forward
    from greenclose.pipeline_io import features_path, load_features, open_store

    cfg, paths = load_config(), Paths()
    if fetch_first:
        from greenclose.features.build import build
        from greenclose.ingest.pipeline import take_snapshot

        target, drift = take_snapshot(cfg, paths, notes="forward runner")
        if drift is not None and drift.changed:
            typer.echo("\n".join(drift.lines()))
        store = open_store(paths, target.name)
        last = pd.Timestamp(store.trading_dates(cutoff=pd.Timestamp("2100-01-01"))[-1])
        panel = build(last, store, cfg.universe, cfg)
        out = features_path(paths, store.snapshot_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        panel.frame.to_parquet(out)
        feats = panel.frame
    else:
        store = open_store(paths, snapshot)
        feats = load_features(paths, store.snapshot_id)

    res = run_forward(cfg, paths, store=store, features=feats, asof=pd.Timestamp(asof) if asof else None)
    typer.echo(f"scored {res.n_rows} symbols at cut-off {res.asof.date()}; {res.n_selected} selected")
    if len(res.top_picks):
        typer.echo(res.top_picks.to_string(index=False))
    else:
        typer.echo("no picks: every candidate abstained on the cost threshold")
    typer.echo(f"back-filled outcomes for {len(res.backfilled_dates)} earlier date(s)")
    m = forward_metrics(paths)
    typer.echo(f"forward log to date: {m}")
    typer.echo(res.summary_line)
    typer.echo(f"log {res.log_path}")


@app.command()
def holdout(
    confirm: bool = typer.Option(False, "--confirm", help="Required. The holdout may be touched once."),
    snapshot: str | None = typer.Option(None, help="Snapshot id (default: latest)"),
    head: str = typer.Option("lgbm_huber", help="Head to evaluate on the holdout"),
    seed: int = typer.Option(0, help="Seed"),
) -> None:
    """Evaluate the locked holdout. Once, ever. Refuses to run twice and records the touch."""
    from greenclose.evaluation.holdout import run_holdout

    if not confirm:
        raise typer.BadParameter("pass --confirm: the locked holdout may be evaluated exactly once")
    cfg, paths = load_config(), Paths()
    for line in run_holdout(cfg, paths, snapshot=snapshot, head_kind=head, seed=seed):
        typer.echo(line)


@app.command()
def snapshots() -> None:
    """List snapshots with their content hashes."""
    from greenclose.ingest.snapshots import list_snapshots, read_manifest

    for d in list_snapshots(Paths().data_dir / "snapshots"):
        m = read_manifest(d)
        typer.echo(f"{m.snapshot_id}  {m.content_sha256[:16]}  rows={m.row_count}  {m.start}..{m.end}")


if __name__ == "__main__":
    app()
