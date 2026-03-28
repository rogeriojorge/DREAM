#!/usr/bin/env python3

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from DREAM import DREAMSettings

from common import configure_stellarator_geometry
from package_no_bootstrap_smoke import build_settings, resolve_dreami, run_kernel


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = ROOT / "data" / "stellarator_geometry_v2.h5"
DEFAULT_WOUT = ROOT / "data" / "wout_LandremanPaul2021_QA_lowres_reference.nc"
def _default_source(provider: str) -> Path:
    if provider in ("desc", "vmec_jax"):
        return DEFAULT_WOUT
    return DEFAULT_PACKAGE


def main():
    parser = argparse.ArgumentParser(
        description="Recommended stellarator geometry workflow: load or build a geometry package, inspect it, evaluate a flux tube, and optionally run a no-bootstrap DREAM smoke case."
    )
    parser.add_argument("--source", type=Path, default=None, help="Geometry source. Defaults to a packaged geometry for provider=auto/package, or the QA VMEC wout for provider=desc/vmec_jax.")
    parser.add_argument("--provider", choices=("auto", "package", "desc", "vmec_jax"), default="auto")
    parser.add_argument("--cache", type=Path, default=None, help="Optional package filename to write or reuse.")
    parser.add_argument("--nr", type=int, default=4)
    parser.add_argument("--ntheta-equil", type=int, default=17)
    parser.add_argument("--nphi-equil", type=int, default=17)
    parser.add_argument("--with-boozer", action="store_true", help="Request a Boozer-capable package when the provider supports it.")
    parser.add_argument("--run-smoke", action="store_true", help="Also run the minimal no-bootstrap DREAM smoke case.")
    parser.add_argument("--dreami", type=Path, default=None)
    args = parser.parse_args()

    source = args.source or _default_source(args.provider)
    provider = None if args.provider == "auto" else args.provider
    dreami = resolve_dreami(args.dreami)
    cache_existed = args.cache is not None and args.cache.exists()

    ds = DREAMSettings()
    configure_stellarator_geometry(
        ds,
        source=source,
        provider=provider,
        radial_nr=args.nr - 1,
        nr_equil=args.nr,
        ntheta_equil=args.ntheta_equil,
        nphi_equil=args.nphi_equil,
        with_boozer=args.with_boozer,
        cache_filename=args.cache,
        write_cache=args.cache is not None,
    )
    package = ds.radialgrid.num_stellarator.package
    metadata = package.metadata

    print("Step 1: Geometry package")
    print(f"  Source: {source}")
    print(f"  Resolved provider: {ds.radialgrid.stellarator_provider}")
    print(f"  Source kind: {metadata.get('source_kind', 'unknown')}")
    print(f"  Schema version: v{package.schema_version}")
    print(f"  Boozer block: {'present' if package.boozer is not None else 'absent'}")
    if args.cache is not None:
        print(f"  Cache/package path: {args.cache}")
        print(f"  Cache status: {'reused existing package' if cache_existed else 'rebuilt geometry package'}")

    print("Step 2: Geometry summary")
    print(f"  Major radius: {ds.radialgrid.R0:.8f} m")
    print(f"  Minor radius: {ds.radialgrid.a:.8f} m")
    print(f"  Sample grid: nrho={ds.radialgrid.rho.size}, ntheta={ds.radialgrid.theta.size}, nphi={ds.radialgrid.phi.size}")
    requested_ntheta = metadata.get("requested_ntheta_equil")
    requested_nphi = metadata.get("requested_nphi_equil")
    resolved_ntheta = metadata.get("resolved_ntheta_equil")
    resolved_nphi = metadata.get("resolved_nphi_equil")
    if requested_ntheta is not None and requested_nphi is not None:
        print(f"  Requested equilibrium grid: ntheta={requested_ntheta}, nphi={requested_nphi}")
    if resolved_ntheta is not None and resolved_nphi is not None:
        print(f"  Resolved equilibrium grid: ntheta={resolved_ntheta}, nphi={resolved_nphi}")

    if package.boozer is not None:
        trace = ds.radialgrid.getStellaratorFluxTubeEvaluator().evaluate(s=0.5, alpha=0.0, nturns=1, npoints=64)
        print("Step 3: Flux-tube evaluation")
        print(f"  Flux-tube sample points: {trace['R'].size}")
        print(f"  Mid-radius iota: {trace['iota']:.8f}")
        print(f"  |B| range: [{trace['|B|'].min():.8f}, {trace['|B|'].max():.8f}] T")
    else:
        print("Step 3: Flux-tube evaluation")
        print("  Skipped because this package does not contain a Boozer-capable block.")

    if args.run_smoke:
        with tempfile.TemporaryDirectory(prefix="dream-stellarator-workflow-") as td:
            td = Path(td)
            settings_path = td / "settings.h5"
            output_path = td / "output.h5"
            smoke_ds = build_settings(
                source,
                settings_path,
                output_path,
                nr=max(args.nr + 2, 6),
                provider=ds.radialgrid.stellarator_provider,
                cache_filename=args.cache,
            )
            runtime = run_kernel(dreami, settings_path)
            print("Step 4: No-bootstrap smoke run")
            print(f"  Smoke provider: {smoke_ds.radialgrid.stellarator_provider}")
            print(f"  Settings file: {settings_path}")
            print(f"  Output file: {output_path}")
            print(f"  DREAMi executable: {dreami}")
            print(f"  Kernel runtime: {runtime:.3f} s")


if __name__ == "__main__":
    main()
