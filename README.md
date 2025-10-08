# openmc2dolfinx

[![Conda CI](https://github.com/festim-dev/openmc2dolfinx/actions/workflows/ci_conda.yml/badge.svg)](https://github.com/festim-dev/openmc2dolfinx/actions/workflows/ci_conda.yml)
[![Docker CI](https://github.com/festim-dev/openmc2dolfinx/actions/workflows/ci_docker.yml/badge.svg)](https://github.com/festim-dev/openmc2dolfinx/actions/workflows/ci_docker.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`openmc2dolfinx` is a lightweight tool for converting [OpenMC](https://github.com/openmc-dev/openmc) output data (in `.vtk` format) into [dolfinx](https://github.com/FEniCS/dolfinx)-compatible `fem.Function` objects.
It is primarily designed to facilitate multiphysics coupling between OpenMC and finite element simulations (e.g. thermal, diffusion, or tritium transport analyses).

## Key features

 - Convert structured and unstructured VTK meshes to dolfinx meshes.
 - Interpolate OpenMC tally results directly into dolfinx Function spaces.
 - Integrated with PyVista for mesh and data inspection.

## Installation

Using **conda**:

```bash
conda create -n openmc2dolfinx-env
conda activate openmc2dolfinx-env
conda install -c conda-forge fenics-dolfinx=0.9.0 mpich pyvista
```
Once in the created in environment:
```bash
python -m pip install openmc2dolfinx
```

## Example usage

```python
from openmc2dolfinx import StructuredGridReader, UnstructuredMeshReader
import pyvista as pv
import numpy as np
import dolfinx
from mpi4py import MPI

# download an example tetmesh
filename = pv.examples.download_tetrahedron(load=False)

grid = pv.read(filename)

# assign random cell data
grid.cell_data["mean"] = np.arange(grid.n_cells)
grid.save("out.vtk")

# read the vtk file
reader = UnstructuredMeshReader("out.vtk")

# make a dolfinx function
u = reader.create_dolfinx_function("mean")

# export to vtk for visualisation
writer = dolfinx.io.VTXWriter(MPI.COMM_WORLD, "out.bp", u, "BP5")
writer.write(t=0)
```
