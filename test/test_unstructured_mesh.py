import numpy as np
import pyvista as pv
import dolfinx
from dolfinx import fem
from mpi4py import MPI
import pytest

from openmc2dolfinx import UnstructuredMeshReader, VTKHDFUnstructuredMeshReader


@pytest.fixture
def unstructured_mesh():
    points = [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [2.0, 2.0, -1.0],  # Additional point for the second tetrahedron
    ]

    cells = [4, 0, 1, 2, 3, 4, 0, 1, 2, 4]  # First tetrahedron  # Second tetrahedron

    celltypes = [pv.CellType.TETRA, pv.CellType.TETRA]

    grid = pv.UnstructuredGrid(cells, celltypes, points)
    grid.cell_data["mean"] = np.arange(grid.n_cells)

    return grid


def test_read_and_generation_of_dolfinx_function_from_unstructured_mesh(
    tmpdir, unstructured_mesh
):
    """Test UnstructuredMeshReader"""

    # save to vtk file
    filename = str(tmpdir.join("original_unstructured.vtk"))
    unstructured_mesh.save(filename)

    reader = UnstructuredMeshReader(filename)
    dolfinx_function = reader.create_dolfinx_function()

    assert isinstance(dolfinx_function, fem.Function)


def test_cell_type_raises_error_if_not_defined(tmpdir, unstructured_mesh):
    # save to vtk file
    filename = str(tmpdir.join("original_unstructured.vtk"))
    unstructured_mesh.save(filename)

    from openmc2dolfinx.core import OpenMC2dolfinx

    class Temp(OpenMC2dolfinx):
        pass

    with pytest.raises(
        AttributeError, match="cell_type must be defined in the child class"
    ):
        my_temp = Temp(filename)
        my_temp.create_dolfinx_mesh()


def test_cell_connectivity_raises_error_if_not_defined(tmpdir, unstructured_mesh):
    # save to vtk file
    filename = str(tmpdir.join("original_unstructured.vtk"))
    unstructured_mesh.save(filename)

    from openmc2dolfinx.core import OpenMC2dolfinx

    class Temp(OpenMC2dolfinx):
        cell_type = "tetrahedron"

    with pytest.raises(
        AttributeError, match="cell_connectivity must be defined in the child class"
    ):
        my_temp = Temp(filename)
        my_temp.create_dolfinx_mesh()


def test_download_from_pyvista_examples(tmpdir):
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
    writer = dolfinx.io.VTXWriter(MPI.COMM_WORLD, tmpdir + "/out.bp", u, "BP5")
    writer.write(t=0)


def test_vtkhdf_unstructured_mesh_reader(tmpdir, unstructured_mesh):
    """Test VTKHDFUnstructuredMeshReader with VTKHDF format files."""
    # save to vtkhdf file
    filename = str(tmpdir.join("test_mesh.vtkhdf"))
    unstructured_mesh.save(filename)

    reader = VTKHDFUnstructuredMeshReader(filename)
    dolfinx_function = reader.create_dolfinx_function()

    assert isinstance(dolfinx_function, fem.Function)


def test_vtkhdf_reader_with_pyvista_example(tmpdir):
    """Test VTKHDFUnstructuredMeshReader with a larger example mesh."""
    # download an example tetmesh
    filename = pv.examples.download_tetrahedron(load=False)

    grid = pv.read(filename)
    grid.cell_data["mean"] = np.arange(grid.n_cells)

    # save as vtkhdf
    vtkhdf_file = str(tmpdir.join("example.vtkhdf"))
    grid.save(vtkhdf_file)

    # read the vtkhdf file
    reader = VTKHDFUnstructuredMeshReader(vtkhdf_file)

    # make a dolfinx function
    u = reader.create_dolfinx_function("mean")

    assert isinstance(u, fem.Function)
    assert u.function_space.mesh.topology.dim == 3
