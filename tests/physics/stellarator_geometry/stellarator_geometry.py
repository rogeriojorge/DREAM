#!/usr/bin/env python3

import os
import pathlib
import subprocess
import sys
import tempfile
import importlib
import shutil
import warnings

import h5py
import numpy as np

import dreamtests

import DREAM.Settings.RadialGrid as RadialGrid
from DREAM.Settings.StellaratorFluxTube import FluxTubeEvaluator
from DREAM.Settings.StellaratorGeometry import StellaratorGeometryPackage
from DREAM.Settings.StellaratorGeometryProviders import DescProvider, VmecJaxProvider


ROOT = pathlib.Path(__file__).resolve().parents[3] / "examples" / "stellarator" / "data"
EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[3] / "examples" / "stellarator"
PACKAGE_V1 = ROOT / "stellarator_geometry_v1.h5"
PACKAGE_V2 = ROOT / "stellarator_geometry_v2.h5"
LEGACY_CACHE = ROOT / "legacy_numeric_stellarator_cache.h5"
WOUT = ROOT / "wout_LandremanPaul2021_QA_lowres_reference.nc"
KERNEL_SMOKE = pathlib.Path(__file__).resolve().parents[3] / "examples" / "stellarator" / "package_no_bootstrap_smoke.py"
RECOMMENDED_WORKFLOW = pathlib.Path(__file__).resolve().parents[3] / "examples" / "stellarator" / "recommended_workflow.py"
TEST_MINOR_RADIUS = 0.18


def _resolve_dreami():
    candidates = []
    dreampath = os.environ.get("DREAMPATH")
    if dreampath:
        candidates.append(pathlib.Path(dreampath) / "iface" / "dreami")
    candidates.append(pathlib.Path(__file__).resolve().parents[3] / "build-brewpetsc" / "iface" / "dreami")
    discovered = shutil.which("dreami")
    if discovered:
        candidates.append(pathlib.Path(discovered))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


DREAMI = _resolve_dreami()


def _desc_available():
    try:
        DescProvider(str(WOUT)).build_package(nr=4, ntheta=17, nphi=17, with_boozer=False)
        return True
    except Exception:
        return False


def _vmec_jax_available():
    try:
        importlib.import_module("vmec_jax")
        importlib.import_module("vmec_jax.wout")
        VmecJaxProvider(str(WOUT)).build_package(nr=4, ntheta=17, nphi=17, with_boozer=False)
        return True
    except Exception:
        return False


def _booz_xform_available():
    try:
        importlib.import_module("booz_xform_jax")
        return True
    except Exception:
        return False


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _run_smoke(source, *, provider="package", cache_filename=None):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        settings = td / "settings.h5"
        output = td / "output.h5"
        command = [
            sys.executable,
            str(KERNEL_SMOKE),
            "--source",
            str(source),
            "--provider",
            provider,
            "--dreami",
            str(DREAMI),
            "--settings",
            str(settings),
            "--output",
            str(output),
            "--run",
        ]
        if cache_filename is not None:
            command.extend(["--cache-filename", str(cache_filename)])

        subprocess.run(
            command,
            check=True,
            env=dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).resolve().parents[3] / "py")),
        )
        _assert(output.is_file(), f"Kernel smoke test did not produce an output file for provider '{provider}'.")
        with h5py.File(output, "r") as hf:
            return {
                "Bmax": np.asarray(hf["grid/geometry/Bmax"][:], dtype=np.float64),
                "Bmin": np.asarray(hf["grid/geometry/Bmin"][:], dtype=np.float64),
                "GR0": np.asarray(hf["grid/geometry/GR0"][:], dtype=np.float64),
                "IR0": np.asarray(hf["grid/geometry/IR0"][:], dtype=np.float64),
                "toroidalFlux": np.asarray(hf["grid/geometry/toroidalFlux"][:], dtype=np.float64),
                "E_field": np.asarray(hf["eqsys/E_field"][:], dtype=np.float64),
                "j_ohm": np.asarray(hf["eqsys/j_ohm"][:], dtype=np.float64),
                "psi_p": np.asarray(hf["eqsys/psi_p"][:], dtype=np.float64),
                "I_p": np.asarray(hf["eqsys/I_p"][:], dtype=np.float64),
            }

def _load_radialgrid_from_source(source, *, provider=None, cache_filename=None, with_boozer=False):
    rg = RadialGrid.RadialGrid(ttype=RadialGrid.TYPE_STELLARATOR)
    rg.setNr(3)
    rg.setMinorRadius(0.18)
    rg.setWallRadius(0.18)

    kwargs = {
        "nr_equil": 4,
        "ntheta_equil": 17,
        "nphi_equil": 17,
        "with_boozer": with_boozer,
    }
    if provider is not None:
        kwargs["provider"] = provider
    if cache_filename is not None:
        kwargs["cache_filename"] = str(cache_filename)
        kwargs["write_cache"] = True

    rg.setStellarator(str(source), **kwargs)
    return rg


def _assert_geometry_parity(reference, candidate, *, context):
    _assert(reference.num_stellarator.package.metadata["source_kind"] == candidate.num_stellarator.package.metadata["source_kind"], f"{context}: source_kind metadata drifted.")
    _assert(reference.num_stellarator.package.metadata["provider"] == candidate.num_stellarator.package.metadata["provider"], f"{context}: provider metadata drifted.")
    _assert(reference.num_stellarator.package.metadata["nfp"] == candidate.num_stellarator.package.metadata["nfp"], f"{context}: nfp metadata drifted.")
    _assert(np.isclose(reference.num_stellarator.package.metadata["major_radius"], candidate.num_stellarator.package.metadata["major_radius"]), f"{context}: major radius metadata drifted.")
    _assert(np.isclose(reference.num_stellarator.package.metadata["minor_radius"], candidate.num_stellarator.package.metadata["minor_radius"]), f"{context}: minor radius metadata drifted.")

    for key in ("requested_ntheta_equil", "requested_nphi_equil", "resolved_ntheta_equil", "resolved_nphi_equil"):
        if key in reference.num_stellarator.package.metadata or key in candidate.num_stellarator.package.metadata:
            _assert(reference.num_stellarator.package.metadata.get(key) == candidate.num_stellarator.package.metadata.get(key), f"{context}: metadata field '{key}' drifted.")

    for key in ("rho", "theta", "phi", "f_passing", "B_min", "B_max", "G", "I", "iota", "psi_T", "R", "Z", "B", "BdotGradPhi", "Jacobian", "g_tt", "g_tp", "lambda_t", "lambda_p"):
        _assert(
            np.allclose(getattr(reference, key), getattr(candidate, key), rtol=1e-13, atol=1e-13),
            f"{context}: field '{key}' changed between provider-backed and package-backed loads.",
        )

    ref_boozer = reference.num_stellarator.package.boozer
    cand_boozer = candidate.num_stellarator.package.boozer
    if ref_boozer is None or cand_boozer is None:
        _assert(ref_boozer is cand_boozer, f"{context}: Boozer block presence changed.")
        return

    _assert(ref_boozer.get("representation") == cand_boozer.get("representation"), f"{context}: Boozer representation drifted.")

    if "booz_xform" in ref_boozer and "booz_xform" in cand_boozer:
        for key in ("xm_b", "xn_b", "bmnc_b", "rmnc_b", "zmns_b", "iota", "s_b"):
            _assert(
                np.allclose(ref_boozer["booz_xform"][key], cand_boozer["booz_xform"][key], rtol=1e-13, atol=1e-13),
                f"{context}: Boozer field '{key}' drifted.",
            )
    elif "vmec" in ref_boozer and "vmec" in cand_boozer:
        for key in ("xm", "xn", "rmnc", "zmns", "bmnc"):
            _assert(
                np.allclose(ref_boozer["vmec"][key], cand_boozer["vmec"][key], rtol=1e-13, atol=1e-13),
                f"{context}: VMEC spectral field '{key}' drifted.",
            )
        for key in ("s", "iota"):
            _assert(
                np.allclose(ref_boozer[key], cand_boozer[key], rtol=1e-13, atol=1e-13),
                f"{context}: VMEC spectral profile '{key}' drifted.",
            )
    elif "desc" in ref_boozer and "desc" in cand_boozer:
        for key in ("xm_b", "xn_b", "bmnc_b", "rmnc_b", "gmn_b", "iota", "s"):
            _assert(
                np.allclose(ref_boozer["desc"][key] if key in ref_boozer["desc"] else ref_boozer[key], cand_boozer["desc"][key] if key in cand_boozer["desc"] else cand_boozer[key], rtol=1e-13, atol=1e-13),
                f"{context}: DESC Boozer field '{key}' drifted.",
            )


def _new_stellarator_radialgrid():
    rg = RadialGrid.RadialGrid(ttype=RadialGrid.TYPE_STELLARATOR)
    rg.setNr(3)
    rg.setMinorRadius(TEST_MINOR_RADIUS)
    rg.setWallRadius(TEST_MINOR_RADIUS)
    return rg


def test_package_smoke():
    rg = _new_stellarator_radialgrid()
    rg.setStellarator(str(PACKAGE_V1), provider="package")

    _assert(rg.B.size == rg.phi.size * rg.rho.size * rg.theta.size, "Unexpected sampled array size.")
    _assert(rg.f_passing.shape == (rg.rho.size,), "Unexpected passing-fraction profile shape.")
    rg.verifySettings()


def test_provider_autodetect_package():
    rg = RadialGrid.RadialGrid(ttype=RadialGrid.TYPE_STELLARATOR)
    rg.setNr(3)
    rg.setMinorRadius(TEST_MINOR_RADIUS)
    rg.setWallRadius(TEST_MINOR_RADIUS)
    rg.setStellarator(str(PACKAGE_V1))

    _assert(rg.stellarator_provider == "package", "Geometry-package sources should auto-select provider='package'.")


def test_legacy_argument_warnings():
    rg = RadialGrid.RadialGrid(ttype=RadialGrid.TYPE_STELLARATOR)
    rg.setNr(3)
    rg.setMinorRadius(TEST_MINOR_RADIUS)
    rg.setWallRadius(TEST_MINOR_RADIUS)

    with tempfile.TemporaryDirectory() as td:
        legacy_cache = pathlib.Path(td) / "legacy-cache.h5"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rg.setStellarator(
                str(PACKAGE_V1),
                provider="package",
                datafilename=str(legacy_cache),
            )

    messages = [str(w.message) for w in caught]
    _assert(any("datafilename" in message for message in messages), "Using 'datafilename' should emit a deprecation warning.")
    _assert(rg.stellarator_cache_filename == str(legacy_cache), "Deprecated 'datafilename' did not map to 'cache_filename'.")


def test_legacy_format_conflict():
    rg = RadialGrid.RadialGrid(ttype=RadialGrid.TYPE_STELLARATOR)
    rg.setNr(3)
    rg.setMinorRadius(TEST_MINOR_RADIUS)
    rg.setWallRadius(TEST_MINOR_RADIUS)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            rg.setStellarator(
                str(PACKAGE_V1),
                provider="package",
                format=RadialGrid.FILE_FORMAT_DESC,
            )
    except Exception as ex:
        _assert("Conflicting stellarator source selection" in str(ex), "Unexpected error message for legacy format/provider conflict.")
        return

    raise AssertionError("Conflicting 'format' and 'provider' arguments should have failed.")


def test_settings_roundtrip():
    rg = _new_stellarator_radialgrid()
    rg.setStellarator(str(PACKAGE_V1), provider="package")

    data = rg.todict(verify=False)
    rg2 = RadialGrid.RadialGrid(ttype=RadialGrid.TYPE_STELLARATOR)
    rg2.fromdict(data)
    rg2.verifySettings()

    _assert(np.allclose(rg.B_min, rg2.B_min), "B_min profile changed after settings roundtrip.")
    _assert(np.allclose(rg.B, rg2.B), "3D B array changed after settings roundtrip.")


def test_cache_readback():
    rg = _new_stellarator_radialgrid()
    rg.setStellarator(str(WOUT), provider="package", cache_filename=str(LEGACY_CACHE))
    rg.verifySettings()
    _assert(np.all(np.isfinite(rg.R)), "Legacy cache load produced non-finite R data.")


def test_package_roundtrip():
    pkg = StellaratorGeometryPackage.read(PACKAGE_V1)
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "roundtrip_package.h5"
        pkg.write(out)
        reloaded = StellaratorGeometryPackage.read(out)

    for key in pkg.sampled:
        _assert(np.allclose(pkg.sampled[key], reloaded.sampled[key]), f"Sampled array '{key}' changed after package roundtrip.")


def test_flux_tube_evaluator():
    ft = FluxTubeEvaluator(PACKAGE_V2)
    out = ft.evaluate(s=0.5, alpha=0.1, nturns=1, npoints=64)
    _assert(out["R"].shape == (64,), "Flux-tube R trace has the wrong shape.")
    _assert(np.all(np.isfinite(out["|B|"])), "Flux-tube trace contains non-finite B values.")


def test_recommended_workflow_example():
    command = [
        sys.executable,
        str(RECOMMENDED_WORKFLOW),
        "--provider",
        "package",
        "--source",
        str(PACKAGE_V2),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(pathlib.Path(__file__).resolve().parents[3] / "py")),
    )
    _assert("Step 1: Geometry package" in result.stdout, "Recommended workflow example did not report the geometry-package step.")
    _assert("Step 3: Flux-tube evaluation" in result.stdout, "Recommended workflow example did not reach the flux-tube step.")


def test_examples_directory_readme():
    readme = (EXAMPLES_DIR / "README.md").read_text()
    _assert("Maintained provider/package workflow examples" in readme, "Examples README does not document the maintained workflow section.")
    _assert("Legacy exploratory scripts" in readme, "Examples README does not document the legacy examples section.")
    _assert("recommended_workflow.py" in readme, "Examples README does not point users to the maintained workflow entry point.")


def test_legacy_scripts_guarded():
    for name in ("runStellarator.py", "runSPARC_old.py"):
        src = (EXAMPLES_DIR / name).read_text()
        _assert('if __name__ == "__main__":' in src, f"{name} should guard its legacy execution path behind __main__.")
        _assert("legacy" in src.lower(), f"{name} should clearly mark itself as legacy.")


def test_kernel_smoke():
    if DREAMI is None:
        return "skip"
    _run_smoke(PACKAGE_V1, provider="package")


def test_kernel_package_legacy_parity():
    if DREAMI is None:
        return "skip"

    pkg = _run_smoke(PACKAGE_V1, provider="package")
    legacy = _run_smoke(WOUT, provider="package", cache_filename=LEGACY_CACHE)

    for key in pkg:
        _assert(
            np.allclose(pkg[key], legacy[key], rtol=1e-11, atol=1e-12),
            f"Package-backed and legacy-cache-backed kernel outputs differ for '{key}'.",
        )


def test_desc_optional():
    if not _desc_available():
        return "skip"

    pkg = DescProvider(str(WOUT)).build_package(nr=4, ntheta=17, nphi=17, with_boozer=False)
    ref = StellaratorGeometryPackage.read(PACKAGE_V1)
    _assert(np.allclose(pkg.profiles["iota"], ref.profiles["iota"], rtol=1e-4, atol=1e-6), "DESC iota profile drifted from the reference package.")


def test_desc_generated_package_frontend_parity_optional():
    if not _desc_available():
        return "skip"

    with tempfile.TemporaryDirectory() as td:
        package_path = pathlib.Path(td) / "desc-generated-package.h5"
        provider_rg = _load_radialgrid_from_source(
            WOUT,
            provider="desc",
            cache_filename=package_path,
            with_boozer=True,
        )
        package_rg = _load_radialgrid_from_source(package_path, provider="package", with_boozer=True)

    _assert_geometry_parity(provider_rg, package_rg, context="DESC package parity")


def test_desc_flux_tube_optional():
    if not _desc_available():
        return "skip"

    desc_pkg = DescProvider(str(WOUT)).build_package(nr=4, ntheta=17, nphi=17, with_boozer=True)
    vmec_pkg = VmecJaxProvider(str(WOUT)).build_package(nr=4, ntheta=17, nphi=17, with_boozer=True)

    _assert(desc_pkg.boozer is not None and desc_pkg.boozer.get("representation") == "desc_boozer", "DESC provider did not attach a DESC Boozer block.")

    desc_trace = FluxTubeEvaluator(desc_pkg).evaluate(s=0.5, alpha=0.0, nturns=1, npoints=64)
    vmec_trace = FluxTubeEvaluator(vmec_pkg).evaluate(s=0.5, alpha=0.0, nturns=1, npoints=64)

    for key in ("R", "Z", "|B|", "sqrt(g)"):
        _assert(
            np.allclose(desc_trace[key], vmec_trace[key], rtol=2e-2, atol=2e-3),
            f"DESC and vmec_jax flux-tube traces drifted for '{key}'.",
        )


def test_vmec_optional():
    if not _vmec_jax_available():
        return "skip"

    pkg = VmecJaxProvider(str(WOUT)).build_package(nr=4, ntheta=17, nphi=17, with_boozer=False)
    ref = StellaratorGeometryPackage.read(PACKAGE_V1)
    iota_on_ref = np.interp(ref.profiles["s"], pkg.profiles["s"], pkg.profiles["iota"])
    _assert(np.allclose(iota_on_ref, ref.profiles["iota"], rtol=3e-4, atol=1e-6), "vmec_jax iota profile drifted from the reference package.")
    _assert(pkg.boozer is not None, "vmec_jax provider did not attach a spectral block.")


def test_vmec_generated_package_frontend_parity_optional():
    if not _vmec_jax_available():
        return "skip"

    with tempfile.TemporaryDirectory() as td:
        package_path = pathlib.Path(td) / "vmec-generated-package.h5"
        provider_rg = _load_radialgrid_from_source(
            WOUT,
            provider="vmec_jax",
            cache_filename=package_path,
            with_boozer=_booz_xform_available(),
        )
        package_rg = _load_radialgrid_from_source(package_path, provider="package", with_boozer=_booz_xform_available())

    _assert_geometry_parity(provider_rg, package_rg, context="VMEC package parity")


def test_vmec_boozer_optional():
    if not _vmec_jax_available() or not _booz_xform_available():
        return "skip"

    pkg = VmecJaxProvider(str(WOUT)).build_package(nr=4, ntheta=17, nphi=17, with_boozer=True)
    _assert(pkg.boozer is not None, "vmec_jax Boozer package did not attach a spectral block.")
    _assert(pkg.boozer.get("representation") == "vmec_wout+booz_xform", "vmec_jax Boozer package did not record the Boozer representation.")
    _assert("booz_xform" in pkg.boozer, "vmec_jax Boozer package did not include the booz_xform block.")


def run(args):
    tests = [
        ("package_smoke", test_package_smoke),
        ("provider_autodetect_package", test_provider_autodetect_package),
        ("legacy_argument_warnings", test_legacy_argument_warnings),
        ("legacy_format_conflict", test_legacy_format_conflict),
        ("settings_roundtrip", test_settings_roundtrip),
        ("cache_readback", test_cache_readback),
        ("package_roundtrip", test_package_roundtrip),
        ("flux_tube_evaluator", test_flux_tube_evaluator),
        ("recommended_workflow_example", test_recommended_workflow_example),
        ("examples_directory_readme", test_examples_directory_readme),
        ("legacy_scripts_guarded", test_legacy_scripts_guarded),
        ("kernel_smoke", test_kernel_smoke),
        ("kernel_package_legacy_parity", test_kernel_package_legacy_parity),
        ("desc_optional", test_desc_optional),
        ("desc_generated_package_frontend_parity_optional", test_desc_generated_package_frontend_parity_optional),
        ("desc_flux_tube_optional", test_desc_flux_tube_optional),
        ("vmec_optional", test_vmec_optional),
        ("vmec_generated_package_frontend_parity_optional", test_vmec_generated_package_frontend_parity_optional),
        ("vmec_boozer_optional", test_vmec_boozer_optional),
    ]

    success = True
    skipped = []

    for name, test in tests:
        try:
            result = test()
            if result == "skip":
                skipped.append(name)
                continue
            dreamtests.print_ok(name)
        except Exception as ex:
            dreamtests.print_error(f"{name}: {ex}")
            success = False

    if skipped and args.get("verbose", False):
        print("Skipped optional tests: {}".format(", ".join(skipped)))

    return success
