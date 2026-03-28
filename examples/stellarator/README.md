# Stellarator Examples

This directory currently contains two kinds of material:

## Maintained provider/package workflow examples
These are the supported entry points for the current stellarator geometry stack.

- `recommended_workflow.py`: end-to-end workflow that loads or builds a geometry package, inspects it, evaluates a flux tube, and can run the no-bootstrap smoke case.
- `geometry_from_package.py`: load a precomputed DREAM geometry package and inspect it.
- `geometry_from_vmec.py`: build a geometry package from a VMEC `wout` file through `provider="vmec_jax"`.
- `geometry_from_desc.py`: build a geometry package through `provider="desc"`.
- `package_no_bootstrap_smoke.py`: run a minimal no-bootstrap DREAM case from a package or provider-backed source.

## Legacy exploratory scripts
These files are kept for historical reference, but they are not the recommended
interface for new work and they do not define the maintained public workflow.

- `runStellarator.py`
- `QAS.py`
- `runSPARC_old.py`
- `SPARC.py`
- `Exceptions.py`

If you are evaluating or extending the current stellarator branch, start from
the maintained examples above and the provider-based API:

```python
ds.radialgrid.setStellarator(
    source,
    provider="package",   # or "desc", "vmec_jax"
    cache_filename="stellarator_geometry.h5",
    write_cache=True,
)
```
