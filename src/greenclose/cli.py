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
def snapshots() -> None:
    """List snapshots with their content hashes."""
    from greenclose.ingest.snapshots import list_snapshots, read_manifest

    for d in list_snapshots(Paths().data_dir / "snapshots"):
        m = read_manifest(d)
        typer.echo(f"{m.snapshot_id}  {m.content_sha256[:16]}  rows={m.row_count}  {m.start}..{m.end}")


if __name__ == "__main__":
    app()
