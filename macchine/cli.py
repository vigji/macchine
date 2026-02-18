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


if __name__ == "__main__":
    cli()
