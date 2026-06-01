"""
    This code is referenced from: https://github.com/CederGroupHub/chgnet/blob/main/chgnet/graph/converter.py
    The original implementation can be found at the link above.
"""

# cython: language_level=3
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False
# distutils: language = c


from . import radiusgraph
import numpy as np
from libc.stdlib cimport free

cdef extern from 'fast_converter_libraries/create_graph.c':
    ctypedef struct Node:
        long index
        LongToDirectedEdgeList* neighbors
        long num_neighbors

    ctypedef struct NodeIndexPair:
        long center
        long neighbor

    ctypedef struct UndirectedEdge:
        NodeIndexPair nodes
        long index
        long* directed_edge_indices
        long num_directed_edges
        double distance

    ctypedef struct DirectedEdge:
        NodeIndexPair nodes
        long index
        const long* image
        long undirected_edge_index
        double distance

    ctypedef struct LongToDirectedEdgeList:
        long key
        DirectedEdge** directed_edges_list
        int num_directed_edges_in_group

    ctypedef struct ReturnElems2:
        long num_nodes
        long num_directed_edges
        long num_undirected_edges
        Node* nodes
        UndirectedEdge** undirected_edges_list
        DirectedEdge** directed_edges_list

    ReturnElems2* create_graph(
        long* center_index,
        long n_e,
        long* neighbor_index,
        long* image,
        double* distance,
        long num_atoms)

    void free_LongToDirectedEdgeList_in_nodes(Node* nodes, long num_nodes)


    LongToDirectedEdgeList** get_neighbors(Node* node)

def make_graph(
        const long[::1] center_index,
        const long n_e,
        const long[::1] neighbor_index,
        const long[:, ::1] image,
        const double[::1] distance,
        const long num_atoms
    ):
    cdef ReturnElems2* returned
    returned = <ReturnElems2*> create_graph(<long*> &center_index[0], n_e, <long*> &neighbor_index[0], <long*> &image[0][0], <double*> &distance[0], num_atoms)
    
    chg_DirectedEdge = radiusgraph.DirectedEdge
    chg_Node = radiusgraph.Node
    chg_UndirectedEdge = radiusgraph.UndirectedEdge
    
    image_np = np.asarray(image)

    cdef LongToDirectedEdgeList** node_neighbors
    cdef Node this_node
    cdef LongToDirectedEdgeList this_entry
    py_nodes = []
    cdef DirectedEdge* this_DE

    # Handling nodes + directed edges
    for idx in range(returned[0].num_nodes):
        this_node = returned[0].nodes[idx]
        this_py_node = chg_Node(index=idx)

        node_neighbors = get_neighbors(&this_node)

        # Iterate through all neighbors and populate our py_node.neighbors dict
        for j in range(this_node.num_neighbors):
            this_entry = node_neighbors[j][0]
            directed_edges = []

            for k in range(this_entry.num_directed_edges_in_group):
                this_DE = this_entry.directed_edges_list[k]
                directed_edges.append(this_DE[0].index)

            this_py_node.neighbors[this_entry.key] = directed_edges

        py_nodes.append(this_py_node)

        free(node_neighbors)

    # Handling directed edges
    py_directed_edges_list = []

    for idx in range(returned[0].num_directed_edges):
        this_DE = returned[0].directed_edges_list[idx]
        py_DE = chg_DirectedEdge(nodes = [this_DE[0].nodes.center, this_DE[0].nodes.neighbor], index=this_DE[0].index, info = {"distance": this_DE[0].distance, "image": image_np[this_DE[0].index], "undirected_edge_index": this_DE[0].undirected_edge_index})

        py_directed_edges_list.append(py_DE)


    # Handling undirected edges
    py_undirected_edges_list = []
    cdef UndirectedEdge* UDE

    for idx in range(returned[0].num_undirected_edges):
        UDE = returned[0].undirected_edges_list[idx]
        py_undirected_edge = chg_UndirectedEdge([UDE[0].nodes.center, UDE[0].nodes.neighbor], index = UDE[0].index, info =  {"distance": UDE[0].distance, "directed_edge_index": []})

        for j in range(UDE[0].num_directed_edges):
            py_undirected_edge.info["directed_edge_index"].append(UDE[0].directed_edge_indices[j])

        py_undirected_edges_list.append(py_undirected_edge)


    # Create Undirected_Edges hashmap
    py_undirected_edges = {}
    for undirected_edge in py_undirected_edges_list:
        this_set = frozenset(undirected_edge.nodes)
        if this_set not in py_undirected_edges:
            py_undirected_edges[this_set] = [undirected_edge]
        else:
            py_undirected_edges[this_set].append(undirected_edge)

    # # Update the nodes list to have pointers to DirectedEdges instead of indices
    for node_index in range(returned[0].num_nodes):
        this_neighbors = py_nodes[node_index].neighbors
        for this_neighbor_index in this_neighbors:
            replacement = [py_directed_edges_list[edge_index] for edge_index in this_neighbors[this_neighbor_index]]
            this_neighbors[this_neighbor_index] = replacement


    # Free everything unneeded
    for idx in range(returned[0].num_directed_edges):
        free(returned[0].directed_edges_list[idx])

    for idx in range(returned[0].num_undirected_edges):
        free(returned[0].undirected_edges_list[idx].directed_edge_indices)
        free(returned[0].undirected_edges_list[idx])


    # Free node LongToDirectedEdgeList
    free_LongToDirectedEdgeList_in_nodes(returned[0].nodes, returned[0].num_nodes)

    free(returned[0].directed_edges_list)
    free(returned[0].undirected_edges_list)
    free(returned[0].nodes)

    free(returned)
    
    return py_nodes, py_directed_edges_list, py_undirected_edges_list, py_undirected_edges


def line_graph_adjacency_list_fast(nodes, undirected_edges_list, double cutoff):
    """Build the MatRIS line graph using the same ordering as Graph.line_graph_adjacency_list.

    This keeps the existing Python graph object model intact, but moves the
    deeply nested loop out of Python bytecode. It is intentionally conservative:
    output rows and cutoff semantics match radiusgraph.Graph.line_graph_adjacency_list.
    """
    cdef list line_graph = []
    cdef object u_edge
    cdef object directed_edges
    cdef object directed_edge
    cdef object center
    cdef object de_index
    cdef object edge_indices
    cdef long n_directed
    cdef long i
    cdef double u_distance
    cdef double directed_distance

    assert len(undirected_edges_list) * 2 >= 0

    for u_edge in undirected_edges_list:
        u_distance = u_edge.info["distance"]
        if u_distance > cutoff:
            continue

        edge_indices = u_edge.info["directed_edge_index"]
        assert len(edge_indices) == 2, (
            "Did not find 2 Directed_edges !!!"
            f"undirected edge {u_edge} has:"
            f"edge.info['directed_edge_index'] = "
            f"{edge_indices}"
        )

        for i in range(2):
            center = u_edge.nodes[i]
            de_index = edge_indices[i]
            for directed_edges in nodes[center].neighbors.values():
                for directed_edge in directed_edges:
                    if directed_edge.index == de_index:
                        continue
                    directed_distance = directed_edge.info["distance"]
                    if directed_distance < cutoff:
                        line_graph.append(
                            [
                                center,
                                u_edge.index,
                                de_index,
                                directed_edge.info["undirected_edge_index"],
                                directed_edge.index,
                            ]
                        )
    return line_graph
