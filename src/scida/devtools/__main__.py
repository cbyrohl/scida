"""
CLI for building and deploying scida pre-cache files.

Usage::

    uv run python -m scida.devtools cache build /virgotng/universe/IllustrisTNG/L75n1820TNG
    uv run python -m scida.devtools cache deploy /virgotng/universe/IllustrisTNG/L75n1820TNG
    uv run python -m scida.devtools cache build-deploy-all
"""

from __future__ import annotations

import os

try:
    import typer
except ImportError:
    raise SystemExit(
        "typer is required for the devtools CLI. "
        "Install it with: uv sync --group dev"
    )

import scida
from scida.devtools import (
    MPCDF_TARGETS,
    deploy_precache,
    deploy_series_precache,
)

app = typer.Typer(help="scida developer tools.")
cache_app = typer.Typer(help="Build and deploy dataset caches.")
app.add_typer(cache_app, name="cache")


@cache_app.command()
def build(
    path: str = typer.Argument(..., help="Path to a simulation directory."),
    overwrite: bool = typer.Option(False, help="Overwrite existing caches."),
):
    """Build local caches by loading the dataset series at PATH."""
    typer.echo(f"Building caches for {path} ...")
    scida.load(path, overwrite=overwrite, lazy=False)
    typer.echo("Done.")


@cache_app.command()
def deploy(
    path: str = typer.Argument(..., help="Path to a simulation directory."),
    basefolder: str = typer.Option(None, help="Target basefolder for deployment."),
    overwrite: bool = typer.Option(False, help="Overwrite existing deployed files."),
    depth: int = typer.Option(2, help="Parent levels for dataset cache ancestor."),
    series_depth: int = typer.Option(
        3, help="Parent levels for series cache ancestor."
    ),
):
    """Deploy local caches to a shared basefolder for PATH."""
    typer.echo(f"Loading series for {path} ...")
    series = scida.load(path)

    typer.echo(f"Deploying dataset caches ({len(series.paths)} paths) ...")
    deploy_precache(
        series.paths, basefolder=basefolder, overwrite=overwrite, depth=depth
    )

    typer.echo("Deploying series cache ...")
    deploy_series_precache(
        series.paths, basefolder=basefolder, overwrite=overwrite, depth=series_depth
    )
    typer.echo("Done.")


@cache_app.command()
def build_deploy(
    path: str = typer.Argument(..., help="Path to a simulation directory."),
    basefolder: str = typer.Option(None, help="Target basefolder for deployment."),
    overwrite: bool = typer.Option(False, help="Overwrite existing caches."),
    depth: int = typer.Option(2, help="Parent levels for dataset cache ancestor."),
    series_depth: int = typer.Option(
        3, help="Parent levels for series cache ancestor."
    ),
):
    """Build caches and deploy them in one step for PATH."""
    typer.echo(f"Building caches for {path} ...")
    series = scida.load(path, overwrite=overwrite, lazy=False)

    typer.echo(f"Deploying dataset caches ({len(series.paths)} paths) ...")
    deploy_precache(
        series.paths, basefolder=basefolder, overwrite=overwrite, depth=depth
    )

    typer.echo("Deploying series cache ...")
    deploy_series_precache(
        series.paths, basefolder=basefolder, overwrite=overwrite, depth=series_depth
    )
    typer.echo("Done.")


def _run_for_all_targets(action, **kwargs):
    """Run *action* for each MPCDF_TARGETS path that exists on this machine."""
    failures = []
    skipped = 0
    for target in MPCDF_TARGETS:
        if not os.path.isdir(target):
            skipped += 1
            continue
        typer.echo(f"\n{'='*60}\n{target}\n{'='*60}")
        try:
            action(target, **kwargs)
        except Exception as exc:
            typer.echo(f"  FAILED: {exc}", err=True)
            failures.append((target, str(exc)))

    typer.echo(f"\nFinished. Skipped {skipped} missing paths.")
    if failures:
        typer.echo(f"{len(failures)} failure(s):")
        for t, msg in failures:
            typer.echo(f"  {t}: {msg}")
        raise typer.Exit(code=1)


@cache_app.command()
def build_all(
    overwrite: bool = typer.Option(False, help="Overwrite existing caches."),
):
    """Build caches for all MPCDF_TARGETS that exist on this machine."""

    def _build(target, *, overwrite):
        scida.load(target, overwrite=overwrite, lazy=False)

    _run_for_all_targets(_build, overwrite=overwrite)


@cache_app.command()
def deploy_all(
    basefolder: str = typer.Option(None, help="Target basefolder for deployment."),
    overwrite: bool = typer.Option(False, help="Overwrite existing deployed files."),
    depth: int = typer.Option(2, help="Parent levels for dataset cache ancestor."),
    series_depth: int = typer.Option(
        3, help="Parent levels for series cache ancestor."
    ),
):
    """Deploy caches for all MPCDF_TARGETS that exist on this machine."""

    def _deploy(target, *, basefolder, overwrite, depth, series_depth):
        series = scida.load(target)
        deploy_precache(
            series.paths, basefolder=basefolder, overwrite=overwrite, depth=depth
        )
        deploy_series_precache(
            series.paths,
            basefolder=basefolder,
            overwrite=overwrite,
            depth=series_depth,
        )

    _run_for_all_targets(
        _deploy,
        basefolder=basefolder,
        overwrite=overwrite,
        depth=depth,
        series_depth=series_depth,
    )


@cache_app.command()
def build_deploy_all(
    basefolder: str = typer.Option(None, help="Target basefolder for deployment."),
    overwrite: bool = typer.Option(False, help="Overwrite existing caches."),
    depth: int = typer.Option(2, help="Parent levels for dataset cache ancestor."),
    series_depth: int = typer.Option(
        3, help="Parent levels for series cache ancestor."
    ),
):
    """Build and deploy caches for all MPCDF_TARGETS that exist."""

    def _build_deploy(target, *, basefolder, overwrite, depth, series_depth):
        series = scida.load(target, overwrite=overwrite, lazy=False)
        deploy_precache(
            series.paths, basefolder=basefolder, overwrite=overwrite, depth=depth
        )
        deploy_series_precache(
            series.paths,
            basefolder=basefolder,
            overwrite=overwrite,
            depth=series_depth,
        )

    _run_for_all_targets(
        _build_deploy,
        basefolder=basefolder,
        overwrite=overwrite,
        depth=depth,
        series_depth=series_depth,
    )


if __name__ == "__main__":
    app()
