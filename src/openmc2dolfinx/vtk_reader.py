from mpi4py import MPI

import basix
import dolfinx
import numpy as np
import pyvista
import ufl
from dolfinx.mesh import create_mesh

__all__ = ["StructuredGridReader", "UnstructuredMeshReader"]


class UnstructuredMeshReader:
    """
    Reads an OpenMC .vtk results file with an unstructured mesh and converts the
    velocity data into a dolfinx.fem.Function

        Args:
            filename: the filename

        Attributes:
            filename: the filename
            grid: the mesh and results from the OpenMC .vtk file
            connectivity: The OpenMC mesh cell connectivity
    """

    def __init__(self, filename):
        self.filename = filename

        self.grid = None
        self.cell_connectivity = None

        self.read_vtk_file()

    def read_vtk_file(self):
        """reads the filename of the OpenMC file"""

        self.grid = pyvista.read(self.filename)

        # Extract connectivity information
        self.cell_connectivity = self.grid.cells_dict[10]

    def create_dolfinx_fucntion(self):
        """reads the filename of the OpenMC file"""

        degree = 1  # Set polynomial degree
        cell = ufl.Cell("tetrahedron")
        mesh_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=(3,)
        )

        # Create dolfinx Mesh
        mesh_ufl = ufl.Mesh(mesh_element)
        dolfinx_mesh = create_mesh(
            MPI.COMM_WORLD, self.cell_connectivity, self.grid.points, mesh_ufl
        )
        function_space = dolfinx.fem.functionspace(dolfinx_mesh, ("DG", 0))
        u = dolfinx.fem.Function(function_space)

        return u


class StructuredGridReader:
    """
    Reads an OpenMC .vtk results file with an structured mesh and converts the
    velocity data into a dolfinx.fem.Function

        Args:
            filename: the filename

        Attributes:
            filename: the filename
            grid: the mesh and results from the OpenMC .vtk file
            connectivity: The OpenMC mesh cell connectivity
    """

    def __init__(self, filename):
        self.filename = filename

        self.grid = None
        self.cell_connectivity = []

        self.read_vtk_file()

    def read_vtk_file(self):
        """reads the filename of the OpenMC file"""

        self.grid = pyvista.read(self.filename)

        num_cells = self.grid.GetNumberOfCells()

        # Extract connectivity information
        ordering = [0, 1, 3, 2, 4, 5, 7, 6]

        # Extract all cell connectivity data at once
        cell_connectivity_raw = np.array(
            [
                [
                    self.grid.GetCell(i).GetPointId(j)
                    for j in range(self.grid.GetCell(i).GetNumberOfPoints())
                ]
                for i in range(num_cells)
            ],
            dtype=int,
        )

        # Apply ordering
        self.cell_connectivity = cell_connectivity_raw[:, ordering]

    def create_dolfinx_function(self):
        degree = 1  # Set polynomial degree
        cell = ufl.Cell("hexahedron")
        mesh_element = basix.ufl.element(
            "Lagrange", cell.cellname(), degree, shape=(3,)
        )

        # Create dolfinx Mesh
        mesh_ufl = ufl.Mesh(mesh_element)
        self.dolfinx_mesh = create_mesh(
            MPI.COMM_WORLD, self.cell_connectivity, self.grid.points, mesh_ufl
        )

        function_space = dolfinx.fem.functionspace(self.dolfinx_mesh, ("DG", 0))
        u = dolfinx.fem.Function(function_space)

        u.x.array[:] = self.grid.cell_data["mean"][
            self.dolfinx_mesh.topology.original_cell_index
        ]

        return u
