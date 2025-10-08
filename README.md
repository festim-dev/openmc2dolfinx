# openmc2dolfinx

[![Conda CI](https://github.com/festim-dev/openmc2dolfinx/actions/workflows/ci_conda.yml/badge.svg)](https://github.com/festim-dev/openmc2dolfinx/actions/workflows/ci_conda.yml)
[![Docker CI](https://github.com/festim-dev/openmc2dolfinx/actions/workflows/ci_docker.yml/badge.svg)](https://github.com/festim-dev/openmc2dolfinx/actions/workflows/ci_docker.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`openmc2dolfinx` is a tool for converting OpenMC output `.vtk` files to functions that can be used within [dolfinx](https://github.com/FEniCS/dolfinx).

## Installation

```bash
conda create -n openmc2dolfinx-env
conda activate openmc2dolfinx-env
conda install -c conda-forge fenics-dolfinx=0.9.0 mpich pyvista
```
Once in the created in environment:
```bash
python -m pip install openmc2dolfinx
```

## Exmaple usage
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
