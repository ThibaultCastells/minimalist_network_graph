"""
Code based on HiddenLayer, by Waleed Abdulla
"""

from __future__ import absolute_import, division, print_function
import os
import re
from random import getrandbits
import inspect
import numpy as np
import copy

REMOVED_NODES = ["Constant", "FeatureDropout", "Unsqueeze", "Transpose"]


###########################################################################
# Utility Functions
###########################################################################

def detect_framework(value):
    # Get all base classes
    classes = inspect.getmro(value.__class__)
    for c in classes:
        if c.__module__.startswith("torch"):
            return "torch"
        elif c.__module__.startswith("tensorflow"):
            return "tensorflow"


###########################################################################
# Node
###########################################################################

class Node():
    """Represents a framework-agnostic neural network layer in a directed graph."""

    def __init__(self, uid, name, op, output_shape=None, params=None):
        """
        uid: unique ID for the layer that doesn't repeat in the computation graph.
        name: Name to display
        op: Framework-agnostic operation name.
        """
        self.id = uid
        self.name = name  # TODO: clarify the use of op vs name vs title
        self.op = op
        self.repeat = 1
        if output_shape:
            assert isinstance(output_shape, (tuple, list)), \
                "output_shape must be a tuple or list but received {}".format(type(output_shape))
        self.output_shape = output_shape
        self.params = params if params else {}

    @property
    def title(self):
        # Default
        title = self.name or self.op

        if "kernel_shape" in self.params:
            # Kernel
            kernel = self.params["kernel_shape"]
            title += "x".join(map(str, kernel))
        if "stride" in self.params:
            stride = self.params["stride"]
            if np.unique(stride).size == 1:
                stride = stride[0]
            if stride != 1:
                title += "/s{}".format(str(stride))
        #         # Transposed
        #         if node.transposed:
        #             name = "Transposed" + name
        return title

    def __repr__(self):
        args = (self.op, self.name, self.id, self.title, self.repeat)
        f = "<Node: op: {}, name: {}, id: {}, title: {}, repeat: {}"
        if self.output_shape:
            args += (str(self.output_shape),)
            f += ", shape: {:}"
        if self.params:
            args += (str(self.params),)
            f += ", params: {:}"
        f += ">"
        return f.format(*args)


###########################################################################
# Graph
###########################################################################


class Graph():
    """Tracks nodes and edges of a directed graph and supports basic operations on them."""

    def __init__(self, model=None, args=None, input_names=None,
                 transforms="default", framework_transforms="default"):
        from collections import OrderedDict
        self.nodes = OrderedDict({})
        self.edges = []

        if model:
            # Detect framwork
            framework = detect_framework(model)
            if framework == "torch":
                from .pytorch_builder import import_graph, FRAMEWORK_TRANSFORMS
                assert args is not None, "Argument args must be provided for Pytorch models."
                import_graph(self, model, args)
            elif framework == "tensorflow":
                from .tf_builder import import_graph, FRAMEWORK_TRANSFORMS
                import_graph(self, model)

            # Apply Transforms
            if framework_transforms:
                if framework_transforms == "default":
                    framework_transforms = FRAMEWORK_TRANSFORMS
                for t in framework_transforms:
                    t.apply(self)
            if transforms:
                if transforms == "default":
                    from .transforms import SIMPLICITY_TRANSFORMS
                    transforms = SIMPLICITY_TRANSFORMS
                for t in transforms:
                    t.apply(self)

        # remove nodes we don't want:
        first = True
        for k, n in list(self.nodes.items()):
            if n.op in REMOVED_NODES:
                self.remove(n)
            if len(self.incoming(n)) == 0 and not first:
                self.remove(n)
                first = False
        # removed multi-input operations with only one input (may happen )
        for k, n in list(self.nodes.items()):
            if n.op in ["Add", "Mul", "Concat", "Div", "Cast"]:
                if len(self.incoming(n)) <= 1:
                    self.remove(n)

        name_map = dict([(k, f"{i:03d}") for i, k in enumerate(self.nodes)])
        remaped_nodes = {}
        for k, n in self.nodes.items():
            n.id = name_map[k]
            remaped_nodes[name_map[k]] = n
        self.nodes = remaped_nodes
        for i in range(len(self.edges)):
            self.edges[i] = (name_map[self.edges[i][0]], name_map[self.edges[i][1]], self.edges[i][2])

        self.node_input, self.node_output = None, None
        for k, n in self.nodes.items():
            if len(self.incoming(n)) == 0 and self.node_input is None: self.node_input = n
            if len(self.outgoing(n)) == 0 and self.node_output is None: self.node_output = n
            if self.node_output is not None and self.node_input is not None: break
        print(f"node_input: {self.node_input.id}")
        print(f"node_output: {self.node_output.id}")

        self.get_topological_sort()

    def id(self, node):
        """Returns a unique node identifier. If the node has an id
        attribute (preferred), it's used. Otherwise, the hash() is returned."""
        return node.id if hasattr(node, "id") else hash(node)

    def add_node(self, node):
        id = self.id(node)
        # assert(id not in self.nodes)
        self.nodes[id] = node

    def add_edge(self, node1, node2, label=None):
        # If the edge is already present, don't add it again.
        # TODO: If an edge exists with a different label, still don't add it again.
        edge = (self.id(node1), self.id(node2), label)
        if edge not in self.edges:
            self.edges.append(edge)

    def add_edge_by_id(self, vid1, vid2, label=None):
        self.edges.append((vid1, vid2, label))

    def outgoing(self, node):
        """Returns nodes connecting out of the given node (or list of nodes)."""
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        # Find edges outgoing from this group but not incoming to it
        outgoing = [self[e[1]] for e in self.edges
                    if e[0] in node_ids and e[1] not in node_ids]
        return outgoing

    def incoming(self, node):
        """Returns nodes connecting to the given node (or list of nodes)."""
        nodes = node if isinstance(node, list) else [node]
        node_ids = [self.id(n) for n in nodes]
        # Find edges incoming to this group but not outgoing from it
        incoming = [self[e[0]] for e in self.edges
                    if e[1] in node_ids and e[0] not in node_ids]
        return incoming

    def siblings(self, node):
        """Returns all nodes that share the same parent (incoming node) with
        the given node, including the node itself.
        """
        incoming = self.incoming(node)
        # TODO: Not handling the case of multiple incoming nodes yet
        if len(incoming) == 1:
            incoming = incoming[0]
            siblings = self.outgoing(incoming)
            return siblings
        else:
            return [node]

    def __getitem__(self, key):
        if isinstance(key, list):
            return [self.nodes.get(k) for k in key]
        else:
            return self.nodes.get(key)

    def remove(self, nodes):
        """Remove a node and its edges."""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        for node in nodes:
            k = self.id(node)
            in_edges = list(filter(lambda e: e[1] == k, self.edges))
            out_edges = list(filter(lambda e: e[0] == k, self.edges))
            self.edges = list(filter(lambda e: e[0] != k and e[1] != k, self.edges))
            for e_in in in_edges:
                for e_out in out_edges:
                    if len(list(filter(lambda e: e[0] == e_in[0] and e[1] == e_out[1], self.edges))) == 0:
                        self.edges.append((e_in[0], e_out[1], ''))

            del self.nodes[k]

    def replace(self, nodes, node):
        """Replace nodes with node. Edges incoming to nodes[0] are connected to
        the new node, and nodes outgoing from nodes[-1] become outgoing from
        the new node."""
        nodes = nodes if isinstance(nodes, list) else [nodes]
        # Is the new node part of the replace nodes (i.e. want to collapse
        # a group of nodes into one of them)?
        collapse = self.id(node) in self.nodes
        # Add new node and edges
        if not collapse:
            self.add_node(node)
        for in_node in self.incoming(nodes):
            # TODO: check specifically for output_shape is not generic. Consider refactoring.
            self.add_edge(in_node, node, in_node.output_shape if hasattr(in_node, "output_shape") else None)
        for out_node in self.outgoing(nodes):
            self.add_edge(node, out_node, node.output_shape if hasattr(node, "output_shape") else None)
        # Remove the old nodes
        for n in nodes:
            if collapse and n == node:
                continue
            self.remove(n)

    def search(self, pattern):
        """Searches the graph for a sub-graph that matches the given pattern
        and returns the first match it finds.
        """
        print(pattern)
        for node in self.nodes.values():
            match, following = pattern.match(self, node)
            if match:
                return match, following
        return [], None

    def sequence_id(self, sequence):
        return getrandbits(64)

    def get_topological_sort(self, recompute=False):
        if not hasattr(self, 'topological_sort') or recompute:
            V = len(self.nodes)
            visited = [False for i in range(V + 1)]
            adj = [[] for i in range(V + 1)]
            for e in self.edges: adj[int(e[0])].append(int(e[1]))
            self.topological_sort = []

            def topologicalSortUtil(v):
                visited[v] = True
                for i in adj[v]:
                    if not visited[i]:
                        topologicalSortUtil(i)
                self.topological_sort.append(v)

            for i in range(V):
                if (visited[i] == False):
                    topologicalSortUtil(i)
        return copy.deepcopy(self.topological_sort)

    def rec_print(self, node):
        """Recursively print a node and its children."""
        i = 0
        next_visits = []
        for child in self.outgoing(node):
            # if child.op not in ['Reshape', 'Unsqueeze', 'Shape'] and 
            if str(child.id) not in self.visited:

                if i == 0:
                    print('=============')
                else:
                    print('-')

                print(f"Node id  : {str(child.id)}")
                print(f"Parent   : {[str(p.id) for p in self.incoming(child)]}")
                print(f"Children : {[str(p.id) for p in self.outgoing(child)]}")
                print(f"Node op  : {str(child.op)}")
                print(f"Node params: {str(child.params)}")

                self.visited.append(str(child.id))
                next_visits.append(child)
                i += 1
        for child in next_visits: self.rec_print(child)

    def show_connections(self):
        # Node: ['id', 'name', 'op', 'repeat', 'output_shape', 'params', '_caption']
        print()
        if self.node_input is None:
            print("Input node not found")
        else:
            self.visited = [str(self.node_input.id)]
            print(f"Node id: {str(self.node_input.id)}")
            print(f"Parent : None")
            print(f"Children : {[str(p.id) for p in self.outgoing(self.node_input)]}")
            print(f"Node op: {str(self.node_input.op)}")
            print(f"Node params: {str(self.node_input.params)}")

            self.rec_print(self.node_input)
