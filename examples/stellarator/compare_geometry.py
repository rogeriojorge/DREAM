#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from DREAM import DREAMSettings
import DREAM.Settings.RadialGrid as RadialGrid


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = ROOT / "data" / "stellarator_geometry_v2.h5"
DEFAULT_WOUT = ROOT / "data" / "wout_LandremanPaul2021_QA_lowres_reference.nc"


def _load_radialgrid(source, *, provider, cache_filename, nr, ntheta_equil, nphi_equil, with_boozer):
    ds = DREAMSettings()
    ds.radialgrid.setType(RadialGrid.TYPE_STELLARATOR)
    ds.radialgrid.setNr(nr - 1)
    ds.radialgrid.setMinorRadius(0.18)
    ds.radialgrid.setWallRadius(0.18)

    kwargs = {
        "nr_equil": nr,
        "ntheta_equil": ntheta_equil,
        "nphi_equil": nphi_equil,
        "with_boozer": with_boozer,
    }
    if cache_filename is not None:
        kwargs["cache_filename"] = str(cache_filename)
        kwargs["write_cache"] = True
    if provider != "auto":
        kwargs["provider"] = provider

    ds.radialgrid.setStellarator(str(source), **kwargs)
    return ds.radialgrid


def _max_abs_rel(a, b):
    diff = abs(a - b)
    scale = max(abs(a), 1e-30)
    return diff, diff / scale


def _array_metrics(a, b):
    if a.shape != b.shape:
        return {"shape_match": False, "shape_a": a.shape, "shape_b": b.shape}

    import numpy as np

    diff = np.abs(a - b)
    rel = diff / np.maximum(np.abs(a), 1e-30)
    return {
        "shape_match": True,
        "shape": a.shape,
        "max_abs": float(np.max(diff)) if diff.size > 0 else 0.0,
        "max_rel": float(np.max(rel)) if rel.size > 0 else 0.0,
    }


def _boozer_summary(package):
    if package.boozer is None:
        return "absent"
    return package.boozer.get("representation", "present")


def _compare_metadata(pkg_a, pkg_b, *, atol, rtol):
    checks = []
    numeric_keys = ("major_radius", "minor_radius")
    scalar_keys = (
        "provider",
        "source_kind",
        "nfp",
        "schema_version",
        "requested_ntheta_equil",
        "requested_nphi_equil",
        "resolved_ntheta_equil",
        "resolved_nphi_equil",
    )

    for key in scalar_keys:
        a = pkg_a.metadata.get(key)
        b = pkg_b.metadata.get(key)
        ok = a == b
        checks.append((key, ok, a, b))

    for key in numeric_keys:
        a = float(pkg_a.metadata.get(key))
        b = float(pkg_b.metadata.get(key))
        max_abs, max_rel = _max_abs_rel(a, b)
        ok = max_abs <= atol or max_rel <= rtol
        checks.append((key, ok, a, b, max_abs, max_rel))

    return checks


def _print_metadata_report(pkg_a, pkg_b, *, atol, rtol):
    print("Metadata")
    status = True
    for item in _compare_metadata(pkg_a, pkg_b, atol=atol, rtol=rtol):
        key = item[0]
        ok = item[1]
        if len(item) == 4:
            _, _, a, b = item
            print(f"  {key}: {'PASS' if ok else 'DRIFT'} ({a!r} vs {b!r})")
        else:
            _, _, a, b, max_abs, max_rel = item
            print(f"  {key}: {'PASS' if ok else 'DRIFT'} (A={a:.16e}, B={b:.16e}, max_abs={max_abs:.3e}, max_rel={max_rel:.3e})")
        status = status and ok

    boozer_match = _boozer_summary(pkg_a) == _boozer_summary(pkg_b)
    print(f"  boozer_representation: {'PASS' if boozer_match else 'DRIFT'} ({_boozer_summary(pkg_a)} vs {_boozer_summary(pkg_b)})")
    return status and boozer_match


def _print_array_report(title, names, get_array, *, rg_a, rg_b, atol, rtol):
    print(title)
    status = True
    for name in names:
        metrics = _array_metrics(get_array(rg_a, name), get_array(rg_b, name))
        if not metrics["shape_match"]:
            print(f"  {name}: DRIFT (shape {metrics['shape_a']} vs {metrics['shape_b']})")
            status = False
            continue
        ok = metrics["max_abs"] <= atol or metrics["max_rel"] <= rtol
        print(
            f"  {name}: {'PASS' if ok else 'DRIFT'} "
            f"(shape={metrics['shape']}, max_abs={metrics['max_abs']:.3e}, max_rel={metrics['max_rel']:.3e})"
        )
        status = status and ok
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Compare two stellarator geometry sources or packages and print a compact parity report."
    )
    parser.add_argument("--source-a", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--provider-a", choices=("auto", "package", "desc", "vmec_jax"), default="auto")
    parser.add_argument("--cache-a", type=Path, default=None)
    parser.add_argument("--source-b", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--provider-b", choices=("auto", "package", "desc", "vmec_jax"), default="auto")
    parser.add_argument("--cache-b", type=Path, default=None)
    parser.add_argument("--nr", type=int, default=4)
    parser.add_argument("--ntheta-equil", type=int, default=17)
    parser.add_argument("--nphi-equil", type=int, default=17)
    parser.add_argument("--with-boozer", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--rtol", type=float, default=1e-13)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    rg_a = _load_radialgrid(
        args.source_a,
        provider=args.provider_a,
        cache_filename=args.cache_a,
        nr=args.nr,
        ntheta_equil=args.ntheta_equil,
        nphi_equil=args.nphi_equil,
        with_boozer=args.with_boozer,
    )
    rg_b = _load_radialgrid(
        args.source_b,
        provider=args.provider_b,
        cache_filename=args.cache_b,
        nr=args.nr,
        ntheta_equil=args.ntheta_equil,
        nphi_equil=args.nphi_equil,
        with_boozer=args.with_boozer,
    )

    pkg_a = rg_a.num_stellarator.package
    pkg_b = rg_b.num_stellarator.package

    print("Geometry parity report")
    print(f"  Source A: {args.source_a}")
    print(f"  Resolved provider A: {rg_a.stellarator_provider}")
    print(f"  Source B: {args.source_b}")
    print(f"  Resolved provider B: {rg_b.stellarator_provider}")

    metadata_ok = _print_metadata_report(pkg_a, pkg_b, atol=args.atol, rtol=args.rtol)
    grid_ok = _print_array_report(
        "Grid arrays",
        ("rho", "theta", "phi"),
        lambda rg, name: getattr(rg, name),
        rg_a=rg_a,
        rg_b=rg_b,
        atol=args.atol,
        rtol=args.rtol,
    )
    profiles_ok = _print_array_report(
        "Profiles",
        ("f_passing", "B_min", "B_max", "G", "I", "iota", "psi_T"),
        lambda rg, name: getattr(rg, name),
        rg_a=rg_a,
        rg_b=rg_b,
        atol=args.atol,
        rtol=args.rtol,
    )
    sampled_ok = _print_array_report(
        "Sampled geometry",
        ("R", "Z", "B", "BdotGradPhi", "Jacobian", "g_tt", "g_tp", "lambda_t", "lambda_p"),
        lambda rg, name: getattr(rg, name),
        rg_a=rg_a,
        rg_b=rg_b,
        atol=args.atol,
        rtol=args.rtol,
    )

    overall_ok = metadata_ok and grid_ok and profiles_ok and sampled_ok
    print(f"Overall status: {'PASS' if overall_ok else 'DRIFT'}")

    if args.fail_on_drift and not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
