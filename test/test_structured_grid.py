import numpy as np
import pyvista as pv
from dolfinx import fem

from openmc2dolfinx import StructuredGridReader


def test_read_and_generation_of_dolfinx_function_from_structured_grid(tmpdir):
    """Test StructuredGridReader"""
    xrng = np.arange(0, 20, 5, dtype=np.float32)
    yrng = np.arange(0, 20, 5, dtype=np.float32)
    zrng = np.arange(0, 20, 5, dtype=np.float32)
    x, y, z = np.meshgrid(xrng, yrng, zrng, indexing="ij")
    grid = pv.StructuredGrid(x, y, z)

    # add cell data
    grid.cell_data["mean"] = np.arange(grid.n_cells)

    # save to vtk file
    filename = str(tmpdir.join("original_structured.vtk"))
    grid.save(filename)

    # save to vtk file
    filename = str(tmpdir.join("original.vtk"))
    grid.save(filename)

    reader = StructuredGridReader(filename=filename)
    dolfinx_function = reader.create_dolfinx_function()

    assert isinstance(dolfinx_function, fem.Function)
