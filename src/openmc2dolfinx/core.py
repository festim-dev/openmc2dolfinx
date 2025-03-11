from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pyvista
import pyvista.core.pointset
import ufl
from dolfinx.mesh import create_mesh

__all__ = ["StructuredGridReader", "UnstructuredMeshReader"]


class OpenMC2dolfinx(pyvista.VTKDataSetReader):
    """
    Base OpenMC2Dolfinx Mesh Reader

    Converts OpenMC results data into a dolfinx.fem.Function

    Args:
        path: the path to the OpenMC .vtk file

    Attributes:
        data: the mesh and results from the OpenMC .vtk file
        connectivity: The OpenMC mesh cell connectivity
        dolfinx_mesh: the dolfinx mesh
    """

    data: pyvista.core.pointset.UnstructuredGrid | pyvista.core.pointset.StructuredGrid
    connectivity: np.ndarray
    dolfinx_mesh: dolfinx.mesh.Mesh = None

    def create_dolfinx_mesh(self):
        """Creates the dolfinx mesh depending on the type of cell provided

        args:
            cell_type: the cell type for the dolfinx mesh, defaults to "tetrahedron"
        """

        # TODO find a way to fix this with abstractmethod and property
        if not hasattr(self, "cell_type"):
            raise AttributeError("cell_type must be defined in the child class")

        self.data = self.read()
        # if not hasattr(self, "cell_connectivity"):
        #     raise AttributeError("cell_connectivity must be defined in the child class")

        degree = 1  # Set polynomial degree

        cell = ufl.Cell(f"{self.cell_type}")
        mesh_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=(3,)
        )

        # Create dolfinx Mesh
        mesh_ufl = ufl.Mesh(mesh_element)
        self.dolfinx_mesh = create_mesh(
            MPI.COMM_WORLD, self.cell_connectivity, self.data.points, mesh_ufl
        )

    def create_dolfinx_function(self, data: str = "mean") -> dolfinx.fem.Function:
        """reads the filename of the OpenMC file

        Arguments:
            data: the name of the data to extract from the vtk file

        Returns:
            dolfinx function with openmc results mapped
        """

        if not self.dolfinx_mesh:
            self.create_dolfinx_mesh()

        function_space = dolfinx.fem.functionspace(self.dolfinx_mesh, ("DG", 0))
        u = dolfinx.fem.Function(function_space)

        u.x.array[:] = self.data.cell_data[f"{data}"][
            self.dolfinx_mesh.topology.original_cell_index
        ]

        return u


class UnstructuredMeshReader(OpenMC2dolfinx):
    """
    Unstructured Mesh Reader

    Reads an OpenMC .vtk results file with unstructured meshes and converts the data
    into a dolfinx.fem.Function

    Args:
        path: the path to the OpenMC .vtk file

    Example:
    .. code-block:: python
        reader = UnstructuredMeshReader("path/to/file.vtk")
        dolfinx_function = reader.create_dolfinx_function()
    """

    cell_type = "tetrahedron"

    @property
    def cell_connectivity(self):
        return self.data.cells_dict[10]


class StructuredGridReader(OpenMC2dolfinx):
    """
    Structured Mesh Reader

    Reads an OpenMC .vtk results file with Structured meshes and converts the data
    into a dolfinx.fem.Function

    Args:
        path: the path to the OpenMC .vtk file

    Example:
    .. code-block:: python
        reader = StructuredGridReader("path/to/file.vtk")
        dolfinx_function = reader.create_dolfinx_function()
    """

    cell_type = "hexahedron"
    _cell_connectivity = None

    def get_connectivity(self):
        num_cells = self.data.GetNumberOfCells()
        assert self.data.GetCellType(0) == 12, "Only hexahedron cells are supported"

        # Extract connectivity information
        ordering = [0, 1, 3, 2, 4, 5, 7, 6]

        self._cell_connectivity = []

        # TODO numpify this
        # Extract all cell connectivity data at once
        for i in range(num_cells):
            cell = self.data.GetCell(i)  # Get the i-th cell
            point_ids = [cell.GetPointId(j) for j in ordering]  # Extract connectivity
            self._cell_connectivity.append(point_ids)

    @property
    def cell_connectivity(self):
        if self._cell_connectivity is None:
            self.get_connectivity()
        return self._cell_connectivity
