import pathlib
import warnings
from importlib import metadata as importlib_metadata

import numpy as np

from DREAM.DREAMException import DREAMException

from .StellaratorGeometry import (
    GEOMETRY_PACKAGE_V1,
    GEOMETRY_PACKAGE_V2,
    GeometryProvider,
    StellaratorGeometryPackage,
    import_optional,
)


def _dependency_versions(*packages):
    versions = {}
    for package in packages:
        if package is None:
            continue
        name = getattr(package, "__name__", package.__class__.__name__)
        versions[name] = getattr(package, "__version__", "unknown")
    return versions


def _installed_version(*names):
    for name in names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


def _reshape_desc_samples(eq, grid, name):
    return np.asarray(eq.compute(name, grid=grid)[name], dtype=np.float64)


def _resolve_desc_equilibrium(source):
    try:
        source = str(source)
        vmec = import_optional("desc.vmec")
        io = import_optional("desc.io")
        grid_mod = import_optional("desc.grid")
        if source.endswith(".nc"):
            eq = vmec.VMECIO.load(source)
            source_kind = "vmec_wout"
        else:
            eq = io.load(source)
            source_kind = "desc_equilibrium"
        LinearGrid = grid_mod.LinearGrid
    except Exception as ex:
        raise DREAMException(
            "DESC-based stellarator loading is unavailable. Install DESC in a Python environment with its optional GUI/backend "
            "requirements satisfied, or load a precomputed DREAM geometry package instead."
        ) from ex

    dependency_versions = {"desc-opt": _installed_version("desc-opt", "desc")}
    return eq, LinearGrid, source_kind, dependency_versions


def _desc_sample_package(source, nr, ntheta, nphi, provider_name="desc"):
    eq, LinearGrid, source_kind, dependency_versions = _resolve_desc_equilibrium(source)

    grid = LinearGrid(
        L=int(nr - 1),
        M=int((ntheta - 1) / 2),
        N=int((nphi - 1) / 2),
        endpoint=True,
        NFP=eq.NFP,
    )

    R = _reshape_desc_samples(eq, grid, "R")
    Z = _reshape_desc_samples(eq, grid, "Z")
    B = _reshape_desc_samples(eq, grid, "|B|")
    BdotGradPhi = _reshape_desc_samples(eq, grid, "B^zeta")
    Jacobian = _reshape_desc_samples(eq, grid, "sqrt(g)") / float(eq.compute("a", grid=grid)["a"])
    g_tt = _reshape_desc_samples(eq, grid, "g_tt")
    g_tp = _reshape_desc_samples(eq, grid, "g_tz")

    iota_full = _reshape_desc_samples(eq, grid, "iota")
    lambda_t = _reshape_desc_samples(eq, grid, "lambda_t") + iota_full * _reshape_desc_samples(eq, grid, "nu_t")
    lambda_p = _reshape_desc_samples(eq, grid, "lambda_z") + iota_full * _reshape_desc_samples(eq, grid, "nu_z")

    with np.errstate(divide="ignore", invalid="ignore"):
        g_tt = np.divide(g_tt, Jacobian**2, out=np.zeros_like(g_tt), where=np.abs(Jacobian) > 0)
        g_tp = np.divide(g_tp, Jacobian**2, out=np.zeros_like(g_tp), where=np.abs(Jacobian) > 0)

    is_axis = np.where(grid.nodes[:, 0] == 0)[0]
    if is_axis.size > 0 and is_axis[-1] + ntheta < g_tt.size:
        g_tt[is_axis] = g_tt[is_axis + ntheta]
        g_tp[is_axis] = g_tp[is_axis + ntheta]

    a = float(eq.compute("a", grid=grid)["a"])
    rho = np.asarray(grid.nodes[grid.unique_rho_idx, 0], dtype=np.float64) * a
    theta = np.asarray(grid.nodes[grid.unique_theta_idx, 1], dtype=np.float64)
    phi = np.asarray(grid.nodes[grid.unique_zeta_idx, 2], dtype=np.float64)

    profiles = {
        "s": np.square(rho / a) if a > 0 else np.zeros_like(rho),
        "f_passing": np.asarray(
            1 - eq.compute("trapped fraction", grid=grid)["trapped fraction"][grid.unique_rho_idx], dtype=np.float64
        ),
        "B_min": np.asarray(eq.compute("min_tz |B|", grid=grid)["min_tz |B|"][grid.unique_rho_idx], dtype=np.float64),
        "B_max": np.asarray(eq.compute("max_tz |B|", grid=grid)["max_tz |B|"][grid.unique_rho_idx], dtype=np.float64),
        "G": np.asarray(eq.compute("G", grid=grid)["G"][grid.unique_rho_idx], dtype=np.float64),
        "I": np.asarray(eq.compute("I", grid=grid)["I"][grid.unique_rho_idx], dtype=np.float64),
        "iota": np.asarray(iota_full[grid.unique_rho_idx], dtype=np.float64),
        "psi_T": np.asarray(eq.compute("Psi", grid=grid)["Psi"][grid.unique_rho_idx], dtype=np.float64),
    }
    sampled = {
        "R": R,
        "Z": Z,
        "B": B,
        "BdotGradPhi": BdotGradPhi,
        "Jacobian": Jacobian,
        "g_tt": g_tt,
        "g_tp": g_tp,
        "lambda_t": lambda_t,
        "lambda_p": lambda_p,
    }
    metadata = {
        "schema_version": GEOMETRY_PACKAGE_V1,
        "provider": provider_name,
        "source_kind": source_kind,
        "source_path": str(source),
        "major_radius": float(eq.axis.R_n[eq.axis.R_basis.get_idx(0)]),
        "minor_radius": a,
        "nfp": int(eq.NFP),
        "dependency_versions": dependency_versions,
    }

    return StellaratorGeometryPackage(metadata=metadata, grid={"rho": rho, "theta": theta, "phi": phi}, profiles=profiles, sampled=sampled)


def _import_vmec_jax_stack():
    vmec_jax = import_optional("vmec_jax", package_root_env="VMEC_JAX_ROOT")
    wout = import_optional("vmec_jax.wout", package_root_env="VMEC_JAX_ROOT")
    return vmec_jax, wout


def _build_vmec_spectral_block(source, with_boozer=False):
    vmec_jax, wout_mod = _import_vmec_jax_stack()
    wout = wout_mod.read_wout(source)

    iota_profile = np.asarray(int(getattr(wout, "signgs", 1)) * getattr(wout, "iotas", getattr(wout, "iotaf")), dtype=np.float64)
    if iota_profile.size > 1 and abs(iota_profile[0]) < 1e-14:
        iota_profile[0] = iota_profile[1]

    block = {
        "representation": "vmec_wout",
        "spectral_source": str(source),
        "nfp": int(wout.nfp),
        "signgs": int(getattr(wout, "signgs", 1)),
        "s": np.linspace(0.0, 1.0, int(wout.ns)),
        "iota": iota_profile,
        "vmec": {
            "xm": np.asarray(wout.xm, dtype=np.int64),
            "xn": np.asarray(wout.xn, dtype=np.int64),
            "xm_nyq": np.asarray(wout.xm_nyq, dtype=np.int64),
            "xn_nyq": np.asarray(wout.xn_nyq, dtype=np.int64),
            "rmnc": np.asarray(wout.rmnc, dtype=np.float64),
            "rmns": np.asarray(getattr(wout, "rmns", np.zeros_like(wout.rmnc)), dtype=np.float64),
            "zmns": np.asarray(wout.zmns, dtype=np.float64),
            "zmnc": np.asarray(getattr(wout, "zmnc", np.zeros_like(wout.zmns)), dtype=np.float64),
            "lmns": np.asarray(getattr(wout, "lmns", np.zeros_like(wout.rmnc)), dtype=np.float64),
            "lmnc": np.asarray(getattr(wout, "lmnc", np.zeros_like(getattr(wout, "lmns", np.zeros_like(wout.rmnc)))), dtype=np.float64),
            "bmnc": np.asarray(wout.bmnc, dtype=np.float64),
            "bmns": np.asarray(getattr(wout, "bmns", np.zeros_like(wout.bmnc)), dtype=np.float64),
            "gmnc": np.asarray(getattr(wout, "gmnc", np.zeros_like(wout.bmnc)), dtype=np.float64),
            "gmns": np.asarray(getattr(wout, "gmns", np.zeros_like(getattr(wout, "gmnc", np.zeros_like(wout.bmnc)))), dtype=np.float64),
            "bsupumnc": np.asarray(getattr(wout, "bsupumnc", np.zeros_like(wout.bmnc)), dtype=np.float64),
            "bsupumns": np.asarray(getattr(wout, "bsupumns", np.zeros_like(getattr(wout, "bsupumnc", np.zeros_like(wout.bmnc)))), dtype=np.float64),
            "bsupvmnc": np.asarray(getattr(wout, "bsupvmnc", np.zeros_like(wout.bmnc)), dtype=np.float64),
            "bsupvmns": np.asarray(getattr(wout, "bsupvmns", np.zeros_like(getattr(wout, "bsupvmnc", np.zeros_like(wout.bmnc)))), dtype=np.float64),
        },
        "dependency_versions": _dependency_versions(vmec_jax),
    }

    if with_boozer:
        try:
            booz_xform_mod = import_optional("booz_xform_jax", package_root_env="BOOZ_XFORM_JAX_ROOT")
            bx = booz_xform_mod.Booz_xform()
            if hasattr(bx, "verbose"):
                bx.verbose = 0
            if hasattr(bx, "read_wout_data"):
                bx.read_wout_data(wout, flux=True)
                reader = "read_wout_data"
            else:
                bx.read_wout(str(source), flux=True)
                reader = "read_wout"
            bx.run()

            block["representation"] = "vmec_wout+booz_xform"
            block["booz_xform"] = {
                "xm_b": np.asarray(bx.xm_b, dtype=np.int64),
                "xn_b": np.asarray(bx.xn_b, dtype=np.int64),
                "bmnc_b": np.asarray(bx.bmnc_b, dtype=np.float64),
                "rmnc_b": np.asarray(bx.rmnc_b, dtype=np.float64),
                "zmns_b": np.asarray(bx.zmns_b, dtype=np.float64),
                "gmnc_b": np.asarray(getattr(bx, "gmnc_b", np.zeros_like(bx.bmnc_b)), dtype=np.float64),
                "iota": np.asarray(getattr(bx, "iota", iota_profile), dtype=np.float64),
                "s_b": np.asarray(getattr(bx, "s_b", block["s"][1:]), dtype=np.float64),
                "compute_surfs": np.asarray(getattr(bx, "compute_surfs", np.arange(len(block["s"]) - 1)), dtype=np.int64),
                "mboz": int(getattr(bx, "mboz", 0) or 0),
                "nboz": int(getattr(bx, "nboz", 0) or 0),
                "transform_provenance": {
                    "reader": reader,
                    "source_path": str(source),
                    "flux_profiles": True,
                },
            }
            block["dependency_versions"]["booz_xform_jax"] = _installed_version("booz_xform_jax")
        except ImportError as ex:
            warnings.warn(
                f"VmecJaxProvider: Unable to populate Boozer block because an optional dependency is missing ({ex}).",
                RuntimeWarning,
            )
        except Exception as ex:
            warnings.warn(
                f"VmecJaxProvider: Boozer transform failed for '{source}' ({ex}).",
                RuntimeWarning,
            )

    return block


def _build_desc_boozer_block(source, surfs, M_booz=None, N_booz=None):
    eq, LinearGrid, source_kind, dependency_versions = _resolve_desc_equilibrium(source)
    vmec_utils = import_optional("desc.vmec_utils")

    if M_booz is None:
        M_booz = max(2, 2 * int(eq.M))
    if N_booz is None:
        N_booz = max(2, 2 * int(eq.N))
    surfs = max(int(surfs), 3)

    s_full = np.linspace(0.0, 1.0, surfs)
    hs = 1.0 / float(surfs - 1)
    s_half = s_full[:-1] + 0.5 * hs
    r_half = np.sqrt(s_half)

    grid = LinearGrid(M=2 * int(M_booz), N=2 * int(N_booz), NFP=eq.NFP, rho=r_half, sym=False)

    transforms = vmec_utils.get_transforms(
        "|B|_mn_B",
        obj=eq,
        grid=grid,
        M_booz=int(M_booz) - 1,
        N_booz=int(N_booz),
    )
    basis = transforms["B"].basis
    matrix, modes = vmec_utils.ptolemy_linear_transform(basis.modes)
    num_modes = modes.shape[0] if eq.sym else int((modes.shape[0] + 1) / 2)

    if eq.sym:
        transforms_sin = vmec_utils.get_transforms(
            "|B|_mn_B",
            obj=eq,
            grid=grid,
            M_booz=int(M_booz) - 1,
            N_booz=int(N_booz),
            sym="sin",
        )
        basis_sin = transforms_sin["B"].basis
        matrix_sin, modes_sin = vmec_utils.ptolemy_linear_transform(basis_sin.modes)
    else:
        transforms_sin = transforms
        matrix_sin = matrix
        modes_sin = modes

    keys = [
        "|B|",
        "<|B|^2>",
        "R",
        "Z",
        "sqrt(g)",
        "rho",
        "psi_r",
        "G",
        "I",
        "iota",
        "w_Boozer",
        "nu",
        "sqrt(g)_Boozer_DESC",
    ]
    data_keys = ["|B|_mn_B", "R_mn_B", "sqrt(g)_Boozer_mn"] + keys
    data_keys = data_keys + ["Z_mn_B", "nu_B_mn"] if not eq.sym else data_keys
    data_keys_sin = ["Z_mn_B", "nu_B_mn"]

    data = eq.compute(data_keys, grid=grid, transforms=transforms)
    if eq.sym:
        data.pop("Boozer transform modes norm", None)
        data_sin = eq.compute(data_keys_sin, grid=grid, transforms=transforms_sin, data=data)
    else:
        data_sin = data

    m_neg_inds = np.where(transforms["B"].basis.modes[:, 1] < 0)

    b_mn = np.asarray(data["|B|_mn_B"]).reshape((grid.num_rho, -1))
    mask = np.zeros_like(b_mn, dtype=bool)
    mask[:, m_neg_inds[0]] = True
    b_mn = np.where(mask, -b_mn, b_mn)
    B_mn = np.atleast_2d(matrix @ b_mn.T).T

    r_mn = np.asarray(data["R_mn_B"]).reshape((grid.num_rho, -1))
    r_mn = np.where(mask, -r_mn, r_mn)
    R_mn = np.atleast_2d(matrix @ r_mn.T).T

    sqrt_g_b_mn = (
        np.asarray(data["sqrt(g)_Boozer_mn"]).reshape((grid.num_rho, -1))
        / grid.compress(data["psi_r"])[:, None]
    )
    sqrt_g_b_mn = np.where(mask, -sqrt_g_b_mn, sqrt_g_b_mn)
    Sqrt_g_B_mn = np.atleast_2d(matrix @ sqrt_g_b_mn.T).T

    if eq.sym:
        z_mn = np.asarray(data_sin["Z_mn_B"]).reshape((grid.num_rho, -1))
        mask_sin = np.zeros_like(z_mn, dtype=bool)
        m_neg_inds_sin = np.where(transforms_sin["B"].basis.modes[:, 1] < 0)
        mask_sin[:, m_neg_inds_sin[0]] = True
        z_mn = np.where(mask_sin, -z_mn, z_mn)
        Z_mn = np.atleast_2d(matrix_sin @ z_mn.T).T

        nu_b_mn = np.asarray(data_sin["nu_B_mn"]).reshape((grid.num_rho, -1))
        nu_b_mn = np.where(mask_sin, -nu_b_mn, nu_b_mn)
        nu_B_mn = np.atleast_2d(matrix_sin @ nu_b_mn.T).T
    else:
        z_mn = np.asarray(data["Z_mn_B"]).reshape((grid.num_rho, -1))
        z_mn = np.where(mask, -z_mn, z_mn)
        Z_mn = np.atleast_2d(matrix @ z_mn.T).T

        nu_b_mn = np.asarray(data["nu_B_mn"]).reshape((grid.num_rho, -1))
        nu_b_mn = np.where(mask, -nu_b_mn, nu_b_mn)
        nu_B_mn = np.atleast_2d(matrix @ nu_b_mn.T).T

    inds_cos = np.where(modes[:, 0] == 1)[0]
    inds_sin = np.where(modes_sin[:, 0] == -1)[0] if eq.sym else np.where(modes[:, 0] == -1)[0]
    if eq.sym:
        modes_sin = np.insert(modes_sin, 0, [-1, 0, 0], axis=0)

    mode_m = np.asarray(modes[:, 1] if eq.sym else modes[inds_cos, 1], dtype=np.int64)
    mode_n = np.asarray((modes_sin[:, 2] if eq.sym else modes[inds_cos, 2]) * eq.NFP, dtype=np.int64)

    rmnc = np.asarray(R_mn[:, inds_cos], dtype=np.float64)
    bmnc = np.asarray(B_mn[:, inds_cos], dtype=np.float64)
    gmnc = np.asarray(Sqrt_g_B_mn[:, inds_cos], dtype=np.float64)

    if eq.sym:
        zeros = np.zeros((rmnc.shape[0], mode_m.size), dtype=np.float64)
        rmns = zeros.copy()
        bmns = zeros.copy()
        gmns = zeros.copy()
        zmnc = zeros.copy()
        zmns = np.insert(np.asarray(Z_mn[:, inds_sin], dtype=np.float64), 0, 0.0, axis=1)
        pmnc = zeros.copy()
        pmns = -np.insert(np.asarray(nu_B_mn[:, inds_sin], dtype=np.float64), 0, 0.0, axis=1)
    else:
        rmns = np.insert(np.asarray(R_mn[:, inds_sin], dtype=np.float64), 0, 0.0, axis=1)
        bmns = np.insert(np.asarray(B_mn[:, inds_sin], dtype=np.float64), 0, 0.0, axis=1)
        gmns = np.insert(np.asarray(Sqrt_g_B_mn[:, inds_sin], dtype=np.float64), 0, 0.0, axis=1)
        zmnc = np.asarray(Z_mn[:, inds_cos], dtype=np.float64)
        zmns = np.insert(np.asarray(Z_mn[:, inds_sin], dtype=np.float64), 0, 0.0, axis=1)
        pmnc = -np.asarray(nu_B_mn[:, inds_cos], dtype=np.float64)
        pmns = -np.insert(np.asarray(nu_B_mn[:, inds_sin], dtype=np.float64), 0, 0.0, axis=1)

    return {
        "representation": "desc_boozer",
        "spectral_source": str(source),
        "source_kind": source_kind,
        "nfp": int(eq.NFP),
        "s": np.asarray(s_half, dtype=np.float64),
        "iota": -np.asarray(grid.compress(data["iota"]), dtype=np.float64),
        "I": -np.asarray(grid.compress(data["I"]), dtype=np.float64),
        "G": np.asarray(grid.compress(data["G"]), dtype=np.float64),
        "dependency_versions": dependency_versions,
        "desc": {
            "sym": bool(eq.sym),
            "xm_b": mode_m,
            "xn_b": mode_n,
            "bmnc_b": bmnc,
            "bmns_b": bmns,
            "rmnc_b": rmnc,
            "rmns_b": rmns,
            "zmns_b": zmns,
            "zmnc_b": zmnc,
            "gmn_b": gmnc,
            "gmns_b": gmns,
            "pmns_b": pmns,
            "pmnc_b": pmnc,
            "mboz": int(M_booz),
            "nboz": int(N_booz),
            "mnboz": int(num_modes),
            "compute_surfs": np.arange(2, 2 + s_half.size, dtype=np.int64),
            "transform_provenance": {
                "source_path": str(source),
                "surfs": int(surfs),
                "M_booz": int(M_booz),
                "N_booz": int(N_booz),
            },
        },
    }


def _fourier_eval(coeff_cos, coeff_sin, m, xn, theta, phi):
    phase = np.outer(m, theta) - np.outer(xn, phi)
    return np.sum(coeff_cos[:, None] * np.cos(phase) + coeff_sin[:, None] * np.sin(phase), axis=0)


def _fourier_eval_dtheta(coeff_cos, coeff_sin, m, xn, theta, phi):
    phase = np.outer(m, theta) - np.outer(xn, phi)
    return np.sum(-m[:, None] * coeff_cos[:, None] * np.sin(phase) + m[:, None] * coeff_sin[:, None] * np.cos(phase), axis=0)


def _fourier_eval_dphi(coeff_cos, coeff_sin, m, xn, theta, phi):
    phase = np.outer(m, theta) - np.outer(xn, phi)
    return np.sum(xn[:, None] * coeff_cos[:, None] * np.sin(phase) - xn[:, None] * coeff_sin[:, None] * np.cos(phase), axis=0)


def _vmec_sample_package(source, ntheta, nphi, provider_name="vmec_jax"):
    _, wout_mod = _import_vmec_jax_stack()
    wout = wout_mod.read_wout(source)

    s = np.linspace(0.0, 1.0, int(wout.ns))
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=True)
    phi = np.linspace(0.0, 2.0 * np.pi / float(wout.nfp), int(nphi), endpoint=True)

    m = np.asarray(wout.xm, dtype=np.float64)
    xn = np.asarray(wout.xn, dtype=np.float64)
    m_nyq = np.asarray(wout.xm_nyq, dtype=np.float64)
    xn_nyq = np.asarray(wout.xn_nyq, dtype=np.float64)

    raxis_cc = np.asarray(getattr(wout, "raxis_cc", np.array([0.0])), dtype=np.float64)
    zaxis_cs = np.asarray(getattr(wout, "zaxis_cs", np.array([0.0])), dtype=np.float64)
    axis_phi = np.linspace(0.0, 2.0 * np.pi / float(wout.nfp), 257, endpoint=True)
    axis_phase = np.outer(np.arange(raxis_cc.size, dtype=np.float64), axis_phi * float(wout.nfp))
    axis_r = np.sum(raxis_cc[:, None] * np.cos(axis_phase), axis=0)
    major_radius = float(np.mean(axis_r))

    rmnc_edge = np.asarray(wout.rmnc[-1], dtype=np.float64)
    rmns_edge = np.asarray(getattr(wout, "rmns", np.zeros_like(wout.rmnc))[-1], dtype=np.float64)
    zmns_edge = np.asarray(wout.zmns[-1], dtype=np.float64)
    zmnc_edge = np.asarray(getattr(wout, "zmnc", np.zeros_like(wout.zmns))[-1], dtype=np.float64)
    edge_theta = np.linspace(0.0, 2.0 * np.pi, 513, endpoint=True)
    edge_phi = np.zeros_like(edge_theta)
    edge_R = _fourier_eval(rmnc_edge, rmns_edge, m, xn, edge_theta, edge_phi)
    edge_Z = _fourier_eval(zmnc_edge, zmns_edge, m, xn, edge_theta, edge_phi)
    axis_z = np.sum(zaxis_cs[:, None] * np.sin(axis_phase), axis=0)
    minor_radius = float(max(0.5 * (np.max(edge_R) - np.min(edge_R)), np.max(np.abs(edge_Z - np.mean(axis_z)))))

    shape = (len(phi), len(s), len(theta))
    R = np.empty(shape, dtype=np.float64)
    Z = np.empty(shape, dtype=np.float64)
    B = np.empty(shape, dtype=np.float64)
    Jacobian = np.empty(shape, dtype=np.float64)
    g_tt = np.empty(shape, dtype=np.float64)
    g_tp = np.empty(shape, dtype=np.float64)
    BdotGradPhi = np.empty(shape, dtype=np.float64)
    lambda_t = np.empty(shape, dtype=np.float64)
    lambda_p = np.empty(shape, dtype=np.float64)

    rmns = np.asarray(getattr(wout, "rmns", np.zeros_like(wout.rmnc)), dtype=np.float64)
    zmnc = np.asarray(getattr(wout, "zmnc", np.zeros_like(wout.zmns)), dtype=np.float64)
    lmns = np.asarray(getattr(wout, "lmns", np.zeros_like(wout.rmnc)), dtype=np.float64)
    lmnc = np.asarray(getattr(wout, "lmnc", np.zeros_like(lmns)), dtype=np.float64)
    bmns = np.asarray(getattr(wout, "bmns", np.zeros_like(wout.bmnc)), dtype=np.float64)
    gmns = np.asarray(getattr(wout, "gmns", np.zeros_like(wout.gmnc)), dtype=np.float64)
    bsupumnc = np.asarray(getattr(wout, "bsupumnc", np.zeros_like(wout.bmnc)), dtype=np.float64)
    bsupumns = np.asarray(getattr(wout, "bsupumns", np.zeros_like(bsupumnc)), dtype=np.float64)
    bsupvmnc = np.asarray(getattr(wout, "bsupvmnc", np.zeros_like(wout.bmnc)), dtype=np.float64)
    bsupvmns = np.asarray(getattr(wout, "bsupvmns", np.zeros_like(bsupvmnc)), dtype=np.float64)

    for iphi, phi_value in enumerate(phi):
        phi_line = np.full(theta.shape, phi_value)
        for ir in range(len(s)):
            R_it = _fourier_eval(np.asarray(wout.rmnc[ir], dtype=np.float64), rmns[ir], m, xn, theta, phi_line)
            Z_it = _fourier_eval(zmnc[ir], np.asarray(wout.zmns[ir], dtype=np.float64), m, xn, theta, phi_line)
            B_it = _fourier_eval(np.asarray(wout.bmnc[ir], dtype=np.float64), bmns[ir], m_nyq, xn_nyq, theta, phi_line)
            sqrtg_it = _fourier_eval(np.asarray(wout.gmnc[ir], dtype=np.float64), gmns[ir], m_nyq, xn_nyq, theta, phi_line)
            bsupu_it = _fourier_eval(bsupumnc[ir], bsupumns[ir], m_nyq, xn_nyq, theta, phi_line)
            bsupv_it = _fourier_eval(bsupvmnc[ir], bsupvmns[ir], m_nyq, xn_nyq, theta, phi_line)

            R_theta = _fourier_eval_dtheta(np.asarray(wout.rmnc[ir], dtype=np.float64), rmns[ir], m, xn, theta, phi_line)
            Z_theta = _fourier_eval_dtheta(zmnc[ir], np.asarray(wout.zmns[ir], dtype=np.float64), m, xn, theta, phi_line)
            R_phi = _fourier_eval_dphi(np.asarray(wout.rmnc[ir], dtype=np.float64), rmns[ir], m, xn, theta, phi_line)
            Z_phi = _fourier_eval_dphi(zmnc[ir], np.asarray(wout.zmns[ir], dtype=np.float64), m, xn, theta, phi_line)

            with np.errstate(divide="ignore", invalid="ignore"):
                jac_use = sqrtg_it / minor_radius if minor_radius > 0 else sqrtg_it
                g_tt_it = np.divide(R_theta**2 + Z_theta**2, jac_use**2, out=np.zeros_like(R_theta), where=np.abs(jac_use) > 0)
                g_tp_it = np.divide(R_theta * R_phi + Z_theta * Z_phi, jac_use**2, out=np.zeros_like(R_theta), where=np.abs(jac_use) > 0)

            lambda_t_it = _fourier_eval_dtheta(lmnc[ir], lmns[ir], m, xn, theta, phi_line)
            lambda_p_it = _fourier_eval_dphi(lmnc[ir], lmns[ir], m, xn, theta, phi_line)

            R[iphi, ir, :] = R_it
            Z[iphi, ir, :] = Z_it
            B[iphi, ir, :] = B_it
            Jacobian[iphi, ir, :] = jac_use
            g_tt[iphi, ir, :] = g_tt_it
            g_tp[iphi, ir, :] = g_tp_it
            BdotGradPhi[iphi, ir, :] = bsupv_it
            lambda_t[iphi, ir, :] = lambda_t_it
            lambda_p[iphi, ir, :] = lambda_p_it

    B_min = np.min(B, axis=(0, 2))
    B_max = np.max(B, axis=(0, 2))
    mirror = np.maximum(1.0 - np.divide(B_min, B_max, out=np.ones_like(B_min), where=B_max > 0), 0.0)
    f_passing = 1.0 - np.sqrt(mirror)

    metadata = {
        "schema_version": GEOMETRY_PACKAGE_V1,
        "provider": provider_name,
        "source_kind": "vmec_wout",
        "source_path": str(source),
        "major_radius": major_radius,
        "minor_radius": minor_radius,
        "nfp": int(wout.nfp),
        "kernel_sample_source": "vmec_jax_wout_fourier",
        "dependency_versions": {"vmec_jax": "local"},
        "notes": {
            "f_passing": "Approximated from the surface mirror ratio because a DESC trapped-fraction operator was unavailable.",
        },
    }
    iota_profile = np.asarray(getattr(wout, "signgs", 1) * getattr(wout, "iotas", getattr(wout, "iotaf")), dtype=np.float64)
    if iota_profile.size > 1 and abs(iota_profile[0]) < 1e-14:
        iota_profile[0] = iota_profile[1]

    profiles = {
        "s": s,
        "f_passing": np.asarray(f_passing, dtype=np.float64),
        "B_min": np.asarray(B_min, dtype=np.float64),
        "B_max": np.asarray(B_max, dtype=np.float64),
        "G": np.asarray(getattr(wout, "bvco", np.zeros_like(s)), dtype=np.float64),
        "I": np.asarray(getattr(wout, "buco", np.zeros_like(s)), dtype=np.float64),
        "iota": iota_profile,
        "psi_T": np.asarray(getattr(wout, "phi", np.zeros_like(s)), dtype=np.float64),
    }
    grid = {"rho": np.sqrt(s) * minor_radius, "theta": theta, "phi": phi}
    sampled = {
        "R": R.reshape(-1),
        "Z": Z.reshape(-1),
        "B": B.reshape(-1),
        "BdotGradPhi": BdotGradPhi.reshape(-1),
        "Jacobian": Jacobian.reshape(-1),
        "g_tt": g_tt.reshape(-1),
        "g_tp": g_tp.reshape(-1),
        "lambda_t": lambda_t.reshape(-1),
        "lambda_p": lambda_p.reshape(-1),
    }
    return StellaratorGeometryPackage(metadata=metadata, grid=grid, profiles=profiles, sampled=sampled)


class DescProvider(GeometryProvider):
    name = "desc"

    def build_package(self, nr, ntheta, nphi, with_boozer=False):
        package = _desc_sample_package(self.source, nr, ntheta, nphi, provider_name=self.name)
        if with_boozer:
            try:
                spectral_block = _build_desc_boozer_block(
                    self.source,
                    surfs=nr,
                    M_booz=max(4, int((ntheta - 1) / 2)),
                    N_booz=max(4, int((nphi - 1) / 2)),
                )
                package.metadata["schema_version"] = GEOMETRY_PACKAGE_V2
                package.metadata["dependency_versions"].update(spectral_block["dependency_versions"])
                package.boozer = spectral_block
                package.validate()
            except Exception as ex:
                warnings.warn(
                    f"DescProvider: Unable to attach a DESC Boozer block for '{self.source}' ({ex}).",
                    RuntimeWarning,
                )
        return package


class VmecJaxProvider(GeometryProvider):
    name = "vmec_jax"

    def build_package(self, nr, ntheta, nphi, with_boozer=False):
        if not str(self.source).endswith(".nc"):
            raise DREAMException("VmecJaxProvider: Only VMEC 'wout_*.nc' files are supported.")

        try:
            package = _desc_sample_package(self.source, nr, ntheta, nphi, provider_name=self.name)
            package.metadata["kernel_sample_source"] = "desc.vmec"
        except DREAMException:
            warnings.warn(
                "VmecJaxProvider: Falling back to direct VMEC wout Fourier sampling because DESC is unavailable. "
                "This path uses an approximate passing-fraction profile.",
                RuntimeWarning,
            )
            package = _vmec_sample_package(self.source, ntheta, nphi, provider_name=self.name)

        package.metadata["source_kind"] = "vmec_wout"
        package.metadata["spectral_source"] = "vmec_jax"
        package.metadata["schema_version"] = GEOMETRY_PACKAGE_V2
        spectral_block = _build_vmec_spectral_block(self.source, with_boozer=with_boozer)
        package.metadata["dependency_versions"].update(spectral_block["dependency_versions"])
        package.boozer = spectral_block
        package.validate()
        return package


class PackageProvider(GeometryProvider):
    name = "package"

    def build_package(self, nr, ntheta, nphi, with_boozer=False):
        return StellaratorGeometryPackage.read(self.source)


def load_geometry_package(source):
    if StellaratorGeometryPackage.is_geometry_package(source):
        return StellaratorGeometryPackage.read(source)
    return None
