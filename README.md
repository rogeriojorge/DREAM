# DREAM

![The DREAM logo](https://raw.githubusercontent.com/chalmersplasmatheory/DREAM/refs/heads/master/media/logo1.png "The DREAM logo")

This directory contains the Disruption Runaway Electron Analysis Model (DREAM)
code. The **online documentation** is available at https://ft.nephy.chalmers.se/dream.

DREAM is a physics simulation framework developed for studying relativistic
runaway electrons in [tokamak](https://en.wikipedia.org/wiki/Tokamak) fusion
devices. Specifically, DREAM solves a system of non-linear partial differential
equations which describe the time evolution of a tokamak plasma. What sets DREAM
apart from other tokamak transport codes is its wide range of models for
studying runaway electron generation and dynamics. In particular, the fluid
equations solved by DREAM can be coupled to a set of kinetic equations for
electrons in order to more accurately describe the runaway electrons.

The official DREAM paper is
[doi:10.1016/j.cpc.2021.108098](https://doi.org/10.1016/j.cpc.2021.108098)
(it is also on arXiv: [2103.16457](https://arxiv.org/abs/2103.16457)).

## Requirements
To compile DREAM, you need to have the following software installed:

- [CMake](https://cmake.org/) >= 3.12
- A C++17 compatible compiler (such as gcc >= 7.0)
- [GNU Scientific Library](https://www.gnu.org/software/gsl/) >= 2.4
- [HDF5](https://www.hdfgroup.org/)
- [PETSc](https://www.mcs.anl.gov/petsc)
- Python 3 (required for generating ADAS data and using the Python interface)

Additionally, to use the DREAM Python interface, you need the following
Python packages:

- h5py
- matplotlib
- numpy
- packaging
- scipy

### Notes on PETSc
While most of the software required by DREAM can be installed directly from
your Linux distribution's package repository, PETSc usually requires a manual
setup. To install PETSc, grab its sources from the PETSc website or clone the
PETSc git repository:
```bash
$ git clone -b release https://gitlab.com/petsc/petsc.git petsc
```
After this, compiling PETSc should be a matter of running the following
commands:
```bash
$ ./configure PETSC_ARCH=linux-c-opt --with-mpi=0
...
$ make PETSC_DIR=/path/to/petsc PETSC_ARCH=linux-c-opt all
...
```
Optionally, you can also run ``make check`` after ``make all``.

Once PETSc has been compiled with the above commands, you only need to make sure
that DREAM will be able to find your PETSc installation. The easiest way to
achieve this is to add the ``PETSC_DIR`` and ``PETSC_ARCH`` environment
variables used above to your ``~/.bashrc`` file (if you use bash; if you're
unsure, you probably do):
```bash
...
export PETSC_DIR="/path/to/petsc"
export PETSC_ARCH=linux-c-opt
```
The value for ``PETSC_DIR`` should be modified according to where you installed
PETSc. An alternative to modifying your ``~/.bashrc`` file is to just give these
variables directly to CMake every time you reconfigure DREAM (which is usually
not very often, unless you're a DREAM developer).

## Compilation
*If you're trying to install DREAM on a cluster which has previously been used
to run DREAM, have a look in the
[setup](https://github.com/chalmersplasmatheory/DREAM/tree/master/setup)
directory to see if a build script is already available.*

To compile DREAM, go to the root DREAM directory and run the following commands:
```bash
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j NTHREADS
```
where ``NTHREADS`` is the number of CPU threads on your computer. If CMake can't
find PETSc, you can change the ``cmake`` command above to read
```bash
$ cmake .. -DPETSC_DIR=/path/to/petsc -DPETSC_ARCH=linux-c-opt
```
where ``/path/to/petsc`` is the path to the directory containing your PETSc
installation.

## Documentation
Online documentation for how to run and extend the code is available at
https://ft.nephy.chalmers.se/dream. LaTeX sources for documentation of the
physics model and various mathematical details can be found under
[doc/notes/](https://github.com/chalmersplasmatheory/DREAM/tree/master/doc/notes).

## Stellarator Geometry Workflow
The `stellarator` branch includes an experimental stellarator geometry workflow
built around a provider-based geometry API. It is intended to make VMEC/DESC
geometry loading, caching, and inspection reproducible while preserving the
existing reduced DREAM kernel physics. The supported public interface is:

```python
ds.radialgrid.setStellarator(
    source,
    provider="package",   # or "desc", "vmec_jax"
    cache_filename="stellarator_geometry.h5",
    write_cache=True,
    nr_equil=4,
    ntheta_equil=17,
    nphi_equil=17,
    with_boozer=True,
)
```

The current recommended workflow is:

1. Build or load a `StellaratorGeometryPackage`.
2. Inspect the packaged geometry and optional Boozer block in Python.
3. Run no-bootstrap DREAM smoke cases from the packaged geometry.

This workflow does not introduce a new native 3D stellarator kernel. Its role
is to provide a clean geometry contract for the existing stellarator-capable
frontend and to make that geometry easier to validate, reload, and compare.

Available example entry points:

- `examples/stellarator/geometry_from_package.py`: load a precomputed DREAM geometry package.
- `examples/stellarator/geometry_from_vmec.py`: build a package from a VMEC `wout` file using `vmec_jax`.
- `examples/stellarator/geometry_from_desc.py`: build a package through the DESC-backed frontend.
- `examples/stellarator/recommended_workflow.py`: recommended end-to-end workflow that loads or builds a package, inspects it, evaluates a flux tube, and can run the no-bootstrap smoke case.
- `examples/stellarator/compare_geometry.py`: compare two geometry sources or packages and print a compact parity report.
- `examples/stellarator/package_no_bootstrap_smoke.py`: run a minimal no-bootstrap DREAM case from a geometry source or package.

The example directory also contains older exploratory scripts. Those are now
documented separately in `examples/stellarator/README.md` and should be treated
as legacy reference material rather than the maintained interface.

Typical workflows:

```bash
python examples/stellarator/recommended_workflow.py
python examples/stellarator/recommended_workflow.py --provider vmec_jax --with-boozer --cache /tmp/stellarator_geometry_vmec.h5
python examples/stellarator/recommended_workflow.py --provider package --source examples/stellarator/data/stellarator_geometry_v2.h5 --run-smoke
python examples/stellarator/compare_geometry.py --source-a examples/stellarator/data/stellarator_geometry_v2.h5 --provider-a package --source-b examples/stellarator/data/stellarator_geometry_v2.h5 --provider-b package
```

Legacy arguments are still accepted for compatibility:

- `format=FILE_FORMAT_DESC`
- `datafilename=...`

but they are deprecated and will emit warnings. New code should use
`provider=` and `cache_filename=` instead.

## Citing DREAM
If you use DREAM in your scientific publications, please cite the
[DREAM paper](https://doi.org/10.1016/j.cpc.2021.108098):
```
@article {DREAM,
    title = {DREAM: A fluid-kinetic framework for tokamak disruption runaway electron simulations},
    journal = {Computer Physics Communications},
    volume = {268},
    pages = {108098},
    year = {2021},
    issn = {0010-4655},
    doi = {10.1016/j.cpc.2021.108098},
    url = {https://doi.org/10.1016/j.cpc.2021.108098},
    author = {Mathias Hoppe and Ola Embreus and Tünde Fülöp}
}
```
If you use certain functionality, you should also cite the relevant reference
for that functionality:

- Shattered Pellet Injection: [O. Vallhagen, MSc thesis, Chalmers University of Technology (2021)](https://hdl.handle.net/20.500.12380/302296)
- Plasmoid drift of SPI shards: [O. Vallhagen *et al*, JPP **89** 905890306 (2023)](https://doi.org/10.1017/S0022377823000466)
- Fluid runaway ionization: [M. Hoppe *et al*, PPCF **67** 045015 (2025)](https://doi.org/10.1088/1361-6587/adbcd5)
- Runaway scrape-off: [O. Vallhagen *et al*, JPP **91** E78 (2025)](https://doi.org/10.1017/s0022377825000327)

## Development
DREAM development is overseen by the DREAM Developer Council which consists of

![The DREAM Developer Council 2025](https://raw.githubusercontent.com/chalmersplasmatheory/DREAM/refs/heads/master/media/DDC/Council-2025.jpg "The DREAM Developer Council in 2025")
*The DREAM Developer Council in 2025. From left: Linn Ekman, Ida Ekmark, Mathias Hoppe, Oskar Vallhagen, Lorenzo Votta, Peter Halldestam.*

- [Mathias Hoppe](https://www.kth.se/profile/mhop?l=en), KTH Royal Institute of Technology, Stockholm, Sweden
- [Ida Ekmark](https://ft.nephy.chalmers.se/?p=people&id=61), Chalmers University of Technology, Gothenburg, Sweden
- [Lorenzo Votta](https://www.kth.se/profile/votta?l=en), KTH Royal Institute of Technology, Stockholm, Sweden
- [Linn Ekman](https://www.kth.se/profile/linnekm?l=en), KTH Royal Institute of Technology, Stockholm, Sweden
- [Oskar Vallhagen](https://ft.nephy.chalmers.se/?p=people&id=16), Chalmers University of Technology, Gothenburg, Sweden
- [Peter Halldestam](https://www.ipp.mpg.de/person/140267/5497846), Max Planck Institute for Plasma Physics, Garching-bei-München, Germany

The code was originally developed within the
[Plasma Theory group at Chalmers](https://ft.nephy.chalmers.se/) but is now
coordinated from KTH Royal Institute of Technology, still in close collaboration
with Chalmers. Over the lifetime of the code, a large number of people from
across the world have contributed to its development. A regularly updated list
of contributors can be found in
[CONTRIBUTORS.md](https://github.com/chalmersplasmatheory/DREAM/blob/master/CONTRIBUTORS.md).

The archive of DREAM newsletters, where changes to DREAM are recorded, can be
found at [https://ft.nephy.chalmers.se/dreamnews/](https://ft.nephy.chalmers.se/dreamnews/).

The original idea for DREAM was proposed by
[Ola Embreus](https://github.com/Embreus) who, together with Mathias Hoppe,
developed the first version.
