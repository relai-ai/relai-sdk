from typing import Any


class Graph:
    """
    A simple directed graph representation using adjacency lists.

    Nodes are stored as keys in a dictionary, and their values are sets
    of neighbors (i.e., edges pointing to other nodes).
    """

    def __init__(self):
        """
        Initializes an empty graph with no edges.
        """
        self.edges = {}

    def clear(self):
        """
        Removes all nodes and edges from the graph.
        """
        self.edges = {}

    def add_edge(self, s: Any, t: Any):
        """
        Adds a directed edge from node `s` to node `t`.

        If the source node `s` does not exist in the graph, it is added.

        Args:
            s (Any): Source node.
            t (Any): Target node.
        """
        if s not in self.edges:
            self.edges[s] = set()
        self.edges[s].add(t)

    def neighbors(self, node: Any):
        """
        Returns the set of neighbors (outgoing edges) for the given node.

        If the node does not exist, it is added with an empty neighbor set.

        Args:
            node (Any): The node whose neighbors are to be retrieved.

        Returns:
            set: A set of neighboring nodes that `node` has edges to.
        """
        if node not in self.edges:
            self.edges[node] = set()
        return self.edges[node]

    def export(self) -> dict:
        """
        Export the graph as in the format expected by agent backend

        Returns:
            dict: A dictionary representing the graph, where keys are node identifiers
                  and values are lists of neighboring node identifiers.
        """
        return {"edges": {str(node): list(neighbors) for node, neighbors in self.edges.items()}}


param_graph = Graph()
"""Global instance of `Graph` used for parameter dependency tracking."""
