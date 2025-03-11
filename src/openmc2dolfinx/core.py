from abc import ABC, abstractmethod

from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pyvista
import pyvista.core.pointset
import ufl
from dolfinx.mesh import create_mesh

__all__ = ["StructuredGridReader", "UnstructuredMeshReader"]


class OpenMC2dolfinx(ABC):
    """
    Base OpenMC2Dolfinx Mesh Reader

    Converts OpenMC results data into a dolfinx.fem.Function

    Attributes:
        grid: the mesh and results from the OpenMC .vtk file
        connectivity: The OpenMC mesh cell connectivity
        dolfinx_mesh: the dolfinx mesh
    """

    grid: pyvista.core.pointset.UnstructuredGrid | pyvista.core.pointset.StructuredGrid
    connectivity: np.ndarray
    dolfinx_mesh: dolfinx.mesh.Mesh

    @abstractmethod
    def read_vtk_file(self):
        """Abstract method that must be implemented by subclasses"""
        pass

    def create_dolfinx_mesh(self, cell_type: str = "tetrahedron"):
        """Creates the dolfinx mesh depending on the type of cell provided

        args:
            cell_type: the cell type for the dolfinx mesh, defaults to "tetrahedron"
        """
        degree = 1  # Set polynomial degree
        cell = ufl.Cell(f"{cell_type}")
        mesh_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=(3,)
        )

        # Create dolfinx Mesh
        mesh_ufl = ufl.Mesh(mesh_element)
        self.dolfinx_mesh = create_mesh(
            MPI.COMM_WORLD, self.cell_connectivity, self.grid.points, mesh_ufl
        )

    def create_dolfinx_function(self, data: str = "mean"):
        """reads the filename of the OpenMC file

        args:
            data: the name of the data to extract from the vtk file

        returns:
            dolfinx function with openmc results mapped
        """

        function_space = dolfinx.fem.functionspace(self.dolfinx_mesh, ("DG", 0))
        u = dolfinx.fem.Function(function_space)

        u.x.array[:] = self.grid.cell_data[f"{data}"][
            self.dolfinx_mesh.topology.original_cell_index
        ]

        return u


class UnstructuredMeshReader(OpenMC2dolfinx, pyvista.VTKDataSetReader):
    """
    Unstructured Mesh Reader

    Reads an OpenMC .vtk results file with unstructured meshes and converts the data
    into a dolfinx.fem.Function

    Args:
        filename: the filename

    Attributes:
        filename: the filename
        grid: the mesh and results from the OpenMC .vtk file
        connectivity: The OpenMC mesh cell connectivity
        dolfinx_mesh: the dolfinx mesh
    """

    def __init__(self, path):
        super().__init__(path)
        self.read_vtk_file()

    @property
    def cell_connectivity(self):
        return self.grid.cells_dict[10]

    def read_vtk_file(self):
        """reads the filename of the OpenMC file, extracts the data, creates the cell
        connectivity between the openmc mesh and the dolfinx mesh and finally creates
        the dolfinx mesh"""

        self.grid = self.read()

        self.create_dolfinx_mesh(cell_type="tetrahedron")


class StructuredGridReader(OpenMC2dolfinx, pyvista.VTKDataSetReader):
    """
    Structured Mesh Reader

    Reads an OpenMC .vtk results file with Structured meshes and converts the data
    into a dolfinx.fem.Function

    Args:
        filename: the filename

    Attributes:
        filename: the filename
        grid: the mesh and results from the OpenMC .vtk file
        connectivity: The OpenMC mesh cell connectivity
        dolfinx_mesh: the dolfinx mesh
    """

    def __init__(self, path):
        super().__init__(path)
        self.read_vtk_file()

    def get_connectivity(self):
        num_cells = self.grid.GetNumberOfCells()

        # Extract connectivity information
        ordering = [0, 1, 3, 2, 4, 5, 7, 6]

        self.cell_connectivity = []

        # Extract all cell connectivity data at once
        for i in range(num_cells):
            cell = self.grid.GetCell(i)  # Get the i-th cell
            point_ids = [cell.GetPointId(j) for j in ordering]  # Extract connectivity
            self.cell_connectivity.append(point_ids)

    def read_vtk_file(self):
        """reads the filename of the OpenMC file, extracts the data, creates the cell
        connectivity between the openmc mesh and the dolfinx mesh and finally creates
        the dolfinx mesh"""

        self.grid = self.read()

        self.get_connectivity()
        self.create_dolfinx_mesh(cell_type="hexahedron")
