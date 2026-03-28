from __future__ import annotations

from pathlib import Path

from DREAM import DREAMSettings
import DREAM.Settings.RadialGrid as RadialGrid


DEFAULT_MINOR_RADIUS = 0.18


def configure_stellarator_geometry(
    ds: DREAMSettings,
    *,
    source: Path,
    provider: str | None,
    radial_nr: int,
    nr_equil: int,
    ntheta_equil: int,
    nphi_equil: int,
    with_boozer: bool = False,
    cache_filename: Path | None = None,
    write_cache: bool = False,
    minor_radius: float = DEFAULT_MINOR_RADIUS,
) -> None:
    ds.radialgrid.setType(RadialGrid.TYPE_STELLARATOR)
    ds.radialgrid.setNr(radial_nr)
    ds.radialgrid.setMinorRadius(minor_radius)
    ds.radialgrid.setWallRadius(minor_radius)

    kwargs = {
        "nr_equil": nr_equil,
        "ntheta_equil": ntheta_equil,
        "nphi_equil": nphi_equil,
        "with_boozer": with_boozer,
    }
    if cache_filename is not None:
        kwargs["cache_filename"] = str(cache_filename)
        kwargs["write_cache"] = write_cache
    if provider is not None:
        kwargs["provider"] = provider

    ds.radialgrid.setStellarator(str(source), **kwargs)
