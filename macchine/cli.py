"""Click-based CLI entry point for the macchine platform."""

from __future__ import annotations

from pathlib import Path

import click

from macchine.config import DEFAULT_RAW_DIR


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Bauer fleet data analysis platform."""


@cli.command()
@click.argument("raw_dir", type=click.Path(exists=True, path_type=Path), default=str(DEFAULT_RAW_DIR))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option("--skip-existing", is_flag=True, default=True, help="Skip already-converted files.")
@click.option("--format", "fmt", type=click.Choice(["json", "dat", "all"]), default="all", help="File format to convert.")
def convert(raw_dir: Path, output_dir: Path, skip_existing: bool, fmt: str):
    """Convert raw JSON/DAT files to Parquet."""
    from macchine.storage.writer import run_conversion

    run_conversion(raw_dir, output_dir, skip_existing=skip_existing, fmt=fmt)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
def fleet(output_dir: Path):
    """Show fleet registry (machines, sites, assignments)."""
    from macchine.storage.catalog import load_fleet_registry

    registry = load_fleet_registry(output_dir)
    click.echo(registry.summary())


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
def info(output_dir: Path):
    """Show dataset summary statistics."""
    from macchine.storage.catalog import dataset_info

    click.echo(dataset_info(output_dir))


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--site", help="Generate report for specific site only.")
def quality(output_dir: Path, site: str | None):
    """Generate data quality reports."""
    from macchine.quality.site_report import generate_reports

    generate_reports(output_dir, site_filter=site)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--machine", help="Filter by machine slug.")
@click.option("--site", help="Filter by site ID.")
def utilization(output_dir: Path, machine: str | None, site: str | None):
    """Compute machine utilization statistics."""
    from macchine.analysis.utilization import compute_utilization

    compute_utilization(output_dir, machine=machine, site=site)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--machine", help="Filter by machine slug.")
@click.option("--site", help="Filter by site ID.")
def operators(output_dir: Path, machine: str | None, site: str | None):
    """Compare operator performance."""
    from macchine.analysis.operator_comparison import compare_operators

    compare_operators(output_dir, machine=machine, site=site)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--machine", help="Filter by machine slug.")
@click.option("--window", default=30, help="Rolling window in days.")
def degradation(output_dir: Path, machine: str | None, window: int):
    """Analyze machine performance degradation."""
    from macchine.analysis.degradation import analyze_degradation

    analyze_degradation(output_dir, machine=machine, window_days=window)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--site", help="Filter by site ID.")
@click.option("--technique", help="Filter by technique code.")
def piles(output_dir: Path, site: str | None, technique: str | None):
    """Analyze pile/element quality metrics."""
    from macchine.analysis.pile_quality import analyze_pile_quality

    analyze_pile_quality(output_dir, site=site, technique=technique)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--machine", help="Filter by machine slug.")
def anomalies(output_dir: Path, machine: str | None):
    """Detect anomalous traces."""
    from macchine.analysis.predictive import detect_anomalies

    detect_anomalies(output_dir, machine=machine)


@cli.command()
@click.argument("output_dir", type=click.Path(exists=True, path_type=Path))
def trends(output_dir: Path):
    """Show fleet-wide performance trends."""
    from macchine.analysis.predictive import trend_summary

    trend_summary(output_dir)


if __name__ == "__main__":
    cli()
