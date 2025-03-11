# openmc2dolfinx
A repository to handle the conversion of results in OpenMC vtk files to dolfinx functions


## Usage
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
