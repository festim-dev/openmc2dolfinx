import numpy as np
import pyvista as pv
from dolfinx import fem

from openmc2dolfinx import UnstructuredMeshReader


def test_read_and_generation_of_dolfinx_function_from_unstructured_mesh(tmpdir):
    """Test UnstructuredMeshReader"""

    # Define the points
    points = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [2.0, 2.0, -1.0],  # Additional point for the second tetrahedron
    ]

    # Define the cells (two tetrahedra sharing a face)
    cells = [4, 0, 1, 2, 3, 4, 0, 1, 2, 4]  # First tetrahedron  # Second tetrahedron

    # Define the cell types
    celltypes = [pv.CellType.TETRA, pv.CellType.TETRA]

    grid = pv.UnstructuredGrid(cells, celltypes, points)
    # grid.plot(show_edges=True, show_axes=True)

    grid.cell_data["mean"] = np.arange(grid.n_cells)

    # save to vtk file
    filename = str(tmpdir.join("original_unstructured.vtk"))
    grid.save(filename)

    reader = UnstructuredMeshReader(filename=filename)
    dolfinx_function = reader.create_dolfinx_function()

    assert isinstance(dolfinx_function, fem.Function)
