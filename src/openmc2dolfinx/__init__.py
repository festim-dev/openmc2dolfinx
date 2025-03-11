from importlib import metadata

try:
    __version__ = metadata.version("openmc2dolfinx")
except Exception:
    __version__ = "unknown"


from .core import StructuredGridReader, UnstructuredMeshReader

__all__ = ["StructuredGridReader, UnstructuredMeshReader"]
