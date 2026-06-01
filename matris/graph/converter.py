"""
    This code is referenced from: https://github.com/CederGroupHub/chgnet/blob/main/chgnet/graph/converter.py
    The original implementation can be found at the link above.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from torch import nn
import gc

from .radiusgraph import Graph, Node, RadiusGraph
from pymatgen.core import Structure

try:
    from .cygraph import line_graph_adjacency_list_fast, make_graph
except (ImportError, AttributeError):
    make_graph = None
    line_graph_adjacency_list_fast = None

datatype = torch.float32

class GraphConverter(nn.Module):
    """Convert a pymatgen.core.Structure to a RadiusGraph"""

    def __init__(
        self,
        atom_graph_cutoff: float = 6,
        line_graph_cutoff: float = 4,
        neighbor_backend: str | None = None,
        neighbor_device: str | None = None,
        neighbor_method: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the Graph Converter.
        
        Args:
            atom_graph_cutoff (float): cutoff radius in atom graph.
            line_graph_cutoff (float): bond length threshold in line graph.
            neighbor_backend (str): neighbor list backend. Defaults to
                MATRIS_GRAPH_BACKEND or "pymatgen".
            neighbor_device (str): device for nvalchemi neighbor list. Defaults to
                MATRIS_NV_GRAPH_DEVICE or cuda when available.
            neighbor_method (str): nvalchemi neighbor list method. Defaults to
                MATRIS_NV_GRAPH_METHOD or "naive".
            verbose (bool): whether to print the GraphConverter.
        """
        super().__init__()
        self.atom_graph_cutoff = atom_graph_cutoff
        self.line_graph_cutoff = (
            atom_graph_cutoff if line_graph_cutoff is None else line_graph_cutoff
        )
        self.neighbor_backend = (
            neighbor_backend or os.environ.get("MATRIS_GRAPH_BACKEND", "pymatgen")
        ).strip().lower()
        self.neighbor_device = neighbor_device or os.environ.get("MATRIS_NV_GRAPH_DEVICE")
        self.neighbor_method = (
            neighbor_method or os.environ.get("MATRIS_NV_GRAPH_METHOD", "naive")
        ).strip()
        
        if make_graph is not None:
            self.create_graph = self._create_graph_fast
            self.algorithm = 'fast'
        else: 
            self.create_graph = self._create_graph_legacy
            self.algorithm = 'legacy'
            print("fast graph converter algorithm import error, using legacy")
        
        if verbose:
            print(self)

    def __repr__(self) -> str:
        """String representation of the GraphConverter."""
        atom_graph_cutoff = self.atom_graph_cutoff
        line_graph_cutoff = self.line_graph_cutoff
        algorithm = self.algorithm
        neighbor_backend = self.neighbor_backend
        neighbor_method = self.neighbor_method
        cls_name = type(self).__name__
        return (
            f"{cls_name}({algorithm=}, {neighbor_backend=}, {neighbor_method=}, "
            f"{atom_graph_cutoff=}, {line_graph_cutoff=})"
        )

    def forward(
        self,
        structure: Structure,
        graph_id=None,
        mp_id=None,
    ) -> RadiusGraph:
        """Convert a structure, return a RadiusGraph.

        Args:
            structure (pymatgen.core.Structure): structure to convert
            graph_id (str): an id to keep track of this crystal graph
                Default = None
            mp_id (str): Materials Project id of this structure
                Default = None
        
        """
        n_atoms = len(structure)
        atomic_number = torch.tensor( [site.specie.Z for site in structure], dtype=torch.int32 )
        
        atom_frac_coord = torch.tensor( structure.frac_coords, dtype=datatype )
        lattice = torch.tensor( structure.lattice.matrix, dtype=datatype )
        
        center_index, neighbor_index, image, distance = self._get_neighbor_list(structure)
        # Ceate atom graph
        graph = self.create_graph(
            n_atoms, center_index, neighbor_index, image, distance
        )
        atom_graph, directed2undirected = graph.adjacency_list()
        atom_graph = torch.tensor(atom_graph, dtype=torch.int32)
        directed2undirected = torch.tensor(directed2undirected, dtype=torch.int32)
        undirected2directed = graph.undirected2directed()
        undirected2directed = torch.tensor(undirected2directed, dtype=torch.int32)
        
        line_graph = []
        disable_gc_for_line_graph = os.environ.get(
            "MATRIS_DISABLE_GC_DURING_LINE_GRAPH", "1"
        ) != "0"
        use_fast_line_graph = (
            line_graph_adjacency_list_fast is not None
            and os.environ.get("MATRIS_USE_FAST_LINE_GRAPH", "0") == "1"
        )
        gc_was_enabled = gc.isenabled()
        if disable_gc_for_line_graph and gc_was_enabled:
            gc.disable()
        try:
            try:
                if use_fast_line_graph:
                    line_graph = line_graph_adjacency_list_fast(
                        graph.nodes,
                        graph.undirected_edges_list,
                        self.line_graph_cutoff,
                    )
                else:
                    line_graph = graph.line_graph_adjacency_list(
                        cutoff=self.line_graph_cutoff
                    )
            except Exception as exc:
                structure.to(filename="error_graph.cif")

            line_graph = torch.tensor(line_graph, dtype=torch.int32)
        finally:
            if disable_gc_for_line_graph and gc_was_enabled:
                gc.enable()

        # For isolated atom, we stop this calculation
        n_isolated_atoms = len({*range(n_atoms)} - {*center_index})
        if n_isolated_atoms:
            atom_graph_cutoff = self.atom_graph_cutoff
            error = f"Error: Detected {n_isolated_atoms} isolated atom. Calculation stopped"
            raise ValueError(error) # or print(error)
        
        return RadiusGraph(
            atomic_number=atomic_number,
            atom_frac_coord=atom_frac_coord,
            atom_graph=atom_graph,
            neighbor_image=torch.tensor(image, dtype=datatype),
            directed2undirected=directed2undirected,
            undirected2directed=undirected2directed,
            line_graph=line_graph,
            lattice=lattice,
            graph_id=graph_id,
            mp_id=mp_id,
            composition=structure.composition.formula,
            atom_graph_cutoff=self.atom_graph_cutoff,
            line_graph_cutoff=self.line_graph_cutoff,
        )

    def _get_neighbor_list(self, structure: Structure):
        if self.neighbor_backend in ("pymatgen", "pmg"):
            return structure.get_neighbor_list(
                r=self.atom_graph_cutoff, sites=structure.sites, numerical_tol=1e-8
            )
        if self.neighbor_backend in ("nvalchemi", "nv", "nvidia"):
            return self._get_neighbor_list_nvalchemi(structure)
        raise ValueError(
            f"Unknown neighbor_backend={self.neighbor_backend!r}. "
            "Expected 'pymatgen' or 'nvalchemi'."
        )

    def _get_neighbor_list_nvalchemi(self, structure: Structure):
        try:
            from nvalchemiops.torch.neighbors import neighbor_list
        except ImportError as exc:
            raise ImportError(
                "MATRIS_GRAPH_BACKEND=nvalchemi requires nvalchemi-toolkit-ops "
                "and warp-lang to be installed."
            ) from exc

        device = self.neighbor_device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        positions = torch.tensor(
            structure.cart_coords,
            dtype=datatype,
            device=device,
        )
        lattice = torch.tensor(
            structure.lattice.matrix,
            dtype=datatype,
            device=device,
        )
        pbc = torch.ones((1, 3), dtype=torch.bool, device=device)

        edge_index, _, shifts = neighbor_list(
            positions,
            self.atom_graph_cutoff,
            cell=lattice.unsqueeze(0),
            pbc=pbc,
            return_neighbor_list=True,
            method=self.neighbor_method,
        )
        if positions.is_cuda:
            torch.cuda.synchronize(positions.device)

        center_index = edge_index[0].detach().cpu().numpy().astype(np.int_, copy=False)
        neighbor_index = edge_index[1].detach().cpu().numpy().astype(np.int_, copy=False)
        image = shifts.detach().cpu().numpy().astype(np.int_, copy=False)

        shift_vectors = shifts.to(dtype=datatype) @ lattice
        edge_vectors = positions[edge_index[0].long()] - (
            positions[edge_index[1].long()] + shift_vectors
        )
        distance = torch.linalg.norm(edge_vectors, dim=1)
        distance = distance.detach().cpu().numpy().astype(np.float64, copy=False)
        return center_index, neighbor_index, image, distance

    @staticmethod
    def _create_graph_legacy(
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> Graph:
        """Given structure information, create a Graph structure to be used to
        create Crystal_Graph using pure python implementation.

        Args:
            n_atoms (int): the number of atoms in the structure
            center_index (np.ndarray): np array of indices of center atoms.
                [num_undirected_bonds]
            neighbor_index (np.ndarray): np array of indices of neighbor atoms.
                [num_undirected_bonds]
            image (np.ndarray): np array of images for each edge.
                [num_undirected_bonds, 3]
            distance (np.ndarray): np array of distances.
                [num_undirected_bonds]

        Return:
            Graph data structure used to create Crystal_Graph object
        """
        
        graph = Graph([Node(index=idx) for idx in range(n_atoms)])
        for ii, jj, img, dist in zip(center_index, neighbor_index, image, distance):
            graph.add_edge(center_index=ii, neighbor_index=jj, image=img, distance=dist)
      
        return graph

    @staticmethod
    def _create_graph_fast(
        n_atoms: int,
        center_index: np.ndarray,
        neighbor_index: np.ndarray,
        image: np.ndarray,
        distance: np.ndarray,
    ) -> Graph:
        """Given structure information, create a Graph structure to be used to
        create Crystal_Graph using C implementation.

        NOTE: this is the fast version of _create_graph_legacy optimized
            in c (~3x speedup).

        Args:
            n_atoms (int): the number of atoms in the structure
            center_index (np.ndarray): np array of indices of center atoms.
                [num_undirected_bonds]
            neighbor_index (np.ndarray): np array of indices of neighbor atoms.
                [num_undirected_bonds]
            image (np.ndarray): np array of images for each edge.
                [num_undirected_bonds, 3]
            distance (np.ndarray): np array of distances.
                [num_undirected_bonds]
        
        Return:
            Graph data structure used to create Crystal_Graph object
        """
        center_index = np.ascontiguousarray(center_index)
        neighbor_index = np.ascontiguousarray(neighbor_index)
        image = np.ascontiguousarray(image, dtype=np.int_)
        distance = np.ascontiguousarray(distance)
        gc_saved = gc.get_threshold()
        gc.set_threshold(0)
        (
            nodes,
            directed_edges_list,
            undirected_edges_list,
            undirected_edges,
        ) = make_graph(
            center_index, len(center_index), neighbor_index, image, distance, n_atoms
        )
        
        graph = Graph(nodes=nodes)
        graph.directed_edges_list = directed_edges_list
        graph.undirected_edges_list = undirected_edges_list
        graph.undirected_edges = undirected_edges
        gc.set_threshold(gc_saved[0])
        
        return graph

    def as_dict(self) -> dict[str, float]:
        """Save the args of the graph converter."""
        return {
            "atom_graph_cutoff": self.atom_graph_cutoff,
            "line_graph_cutoff": self.line_graph_cutoff,
            "algorithm": self.algorithm,
            "neighbor_backend": self.neighbor_backend,
            "neighbor_device": self.neighbor_device,
            "neighbor_method": self.neighbor_method,
        }

    @classmethod
    def from_dict(cls, dict) -> GraphConverter:
        """Create converter from dictionary."""
        kwargs = dict.copy()
        kwargs.pop("algorithm", None)
        return GraphConverter(**kwargs)
