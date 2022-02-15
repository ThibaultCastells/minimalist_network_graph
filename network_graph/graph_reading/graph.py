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
from typing import Any, Union, List
import torch.nn as nn
import torch
from network_graph.graph_reading.module_info import ModuleInfo, ModulesInfo

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

    def __init__(self, uid, name, op, shape=None, params=None):
        """
        uid: unique ID for the layer that doesn't repeat in the computation graph.
        name: Name to display
        op: Framework-agnostic operation name.
        """
        self.id = uid
        self.name = name  # TODO: clarify the use of op vs name vs title
        self.op = op
        self.repeat = 1
        if shape:
            assert isinstance(shape, (tuple, list)), \
                "shape must be a tuple or list but received {}".format(type(shape))
        self.shape = shape
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
        if self.shape:
            args += (str(self.shape),)
            f += ", shape: {:}"
        if self.params:
            args += (str(self.params),)
            f += ", params: {:}"
        f += ">"
        return f.format(*args)

    def equal(self, node):
        # done in 3 steps, to avoid useless operations
        op_equal = (self.op == node.op)
        if not op_equal:
            return False
        shape_equal = str(self.shape) == str(node.shape)
        if not shape_equal:
            return False
        shared_items = {k: self.params[k] for k in self.params if k in node.params and str(self.params[k]) == str(node.params[k])}
        params_equal = len(shared_items) == len(self.params) and len(shared_items) == len(node.params)
        return params_equal

###########################################################################
# Graph
###########################################################################


class Graph():
    """Tracks nodes and edges of a directed graph and supports basic operations on them."""

    def __init__(self, model=None, input=None, transforms="default", framework_transforms="default"):
        from collections import OrderedDict
        self.nodes = OrderedDict({})
        self.edges = []

        if model:
            # Detect framwork
            framework = detect_framework(model)
            if framework == "torch":
                from .pytorch_builder import import_graph, FRAMEWORK_TRANSFORMS
                assert input is not None, "Argument input must be provided for Pytorch models."
                import_graph(self, model, input)
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

        # remap nodes to be number between 0 and len(nodes)-1
        name_map = dict([(k, f"{i:03d}") for i, k in enumerate(self.nodes)])
        remaped_nodes = {}
        for k, n in self.nodes.items():
            n.id = name_map[k]
            remaped_nodes[name_map[k]] = n
        self.nodes = remaped_nodes
        for i in range(len(self.edges)):
            self.edges[i] = (name_map[self.edges[i][0]], name_map[self.edges[i][1]], self.edges[i][2])

        # find model input and output
        self.node_input, self.node_output = None, None
        for k, n in self.nodes.items():
            if len(self.incoming(n)) == 0 and self.node_input is None: self.node_input = n
            if len(self.outgoing(n)) == 0 and self.node_output is None: self.node_output = n
            if self.node_output is not None and self.node_input is not None: break

        # find topological order
        self.get_topological_sort()

        # reorder nodes to be in topological order
        self.nodes = OrderedDict( [ (f'{v:03d}', self.nodes[f'{v:03d}']) for v in self.topological_sort[::-1] ] )

        # remap nodes so that the indexes fit the new order
        name_map = dict([(k, f"{i:03d}") for i, k in enumerate(self.nodes)])
        remaped_nodes = {}
        for k, n in self.nodes.items():
            n.id = name_map[k]
            remaped_nodes[name_map[k]] = n
        self.nodes = remaped_nodes
        for i in range(len(self.edges)):
            self.edges[i] = (name_map[self.edges[i][0]], name_map[self.edges[i][1]], self.edges[i][2])

        # update topological order (no need to recompute, since the nodes are in topological order)
        self.topological_sort = list(np.arange(len(self.nodes))[::-1])

        # update input and output nodes
        self.node_output = self.nodes[f'{len(self.nodes)-1:03d}']
        if isinstance(input, tuple):
            self.node_input = [self.nodes[f'{i:03d}'] for i in range(len(input))]
        else:
            self.node_input = self.nodes[f'{0:03d}']


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
            # TODO: check specifically for shape is not generic. Consider refactoring.
            self.add_edge(in_node, node, in_node.shape if hasattr(in_node, "shape") else None)
        for out_node in self.outgoing(nodes):
            self.add_edge(node, out_node, node.shape if hasattr(node, "shape") else None)
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

    def search_node(self, node: Node, candidates: list = None):
        """ 
            Return all the nodes of the graph equal to the input node, in between start and end.
            Args:
                - node: the node to search for.
                - candidates: the nodes id of the subgraph in which the node is searched (if None, all nodes are considered). Must be sorted!
            Returns: a list of nodes id.
        """
        if candidates is None: candidates = list(self.nodes.keys())
        elif len(candidates) == 0: return []
        # look for the pattern in the graph
        curr_node_index = 0
        curr_node_id = candidates[curr_node_index]
        if not isinstance(curr_node_id, str): curr_node_id = f'{curr_node_id:03d}'
        curr_node = self.nodes[curr_node_id]
        matched = []

        while curr_node_index != len(candidates):
            if curr_node.equal(node):
                matched.append(curr_node_id)
        
            curr_node_index += 1
            if curr_node_index != len(candidates):
                curr_node_id = candidates[curr_node_index]
                if not isinstance(curr_node_id, str): curr_node_id = f'{curr_node_id:03d}'
                curr_node = self.nodes[curr_node_id]
        return matched
            

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
        # Node: ['id', 'name', 'op', 'repeat', 'shape', 'params', '_caption']
        self.visited = []
        node_input = self.node_input if isinstance(self.node_input, tuple) else [self.node_input]
        for n in node_input:
            self.visited.append(str(n.id))
            print('=============')
            print(f"Node id: {str(n.id)}")
            print(f"Parent : None")
            print(f"Children : {[str(p.id) for p in self.outgoing(n)]}")
            print(f"Node op: {str(n.op)}")
            print(f"Node params: {str(n.params)}")
            self.rec_print(n)

###########################################################################
# get_pytorch_names
###########################################################################

def get_pytorch_names(model: torch.nn.Module, graph: Graph, input: Union[torch.Tensor, List[torch.Tensor], Any], verbose=False):
    """
        Return the names in the input Pytorch model of the nodes of the input graph.
        As it requires to create a graph for each module in the model, it may take some time for big models.
    """

    # create a graph for each node of the model (in order to be able to find the match between pytorch modules and graph nodes)
    modules = []
    def rec_visit(module, root=''):
        for name_block, block in module.named_children():
            name = name_block if root == '' else root+'.'+name_block 
            if not isinstance(block, (nn.ModuleList, nn.Dropout)):
                modules.append(ModuleInfo(block, name))
            rec_visit(block, name)
    rec_visit(model)

    device = next(model.parameters()).device
    if isinstance(input, tuple):
        input_name = inspect.getfullargspec(model.forward)[0][1:]
        input = {input_name[i]: input[i].to(device) for i in range(len(input))}
    else:
        input = input.to(device)

    modules = ModulesInfo(model, modules, input)

    if verbose:
        print("\nList of modules:")
        for e in modules.modules():
            print(e.name)
        print()

    # put the modules in a dictionary that we can visit recursively
    model_dict = {}
    for m in modules:
        names = m.name.split('.')
        depth = len(names)
        curr_dict = model_dict
        for i, name in enumerate(names):
            if name not in curr_dict:
                curr_dict[name] = [None, {}]
            if i == depth-1:
                curr_dict[name][0] = m
            curr_dict = curr_dict[name][1]

    # recursively visit the model and find the matching pytorch modules
    pytorch_names = []
    def rec_find_match(modules_dict, candidates=None):
        if candidates is None: candidates = np.arange(len(graph.nodes))
        for key in modules_dict:
            m = modules_dict[key][0]
            m_children = modules_dict[key][1]
            if m is not None:
                input_dim = m.info['input_dim']
                m_graph = Graph(m.module, torch.zeros(input_dim))

                nb_nodes = len(m_graph.nodes)
                candidate_inputs = graph.search_node(m_graph.node_input, candidates)
                candidate_outputs = graph.search_node(m_graph.node_output, candidates)

                if len(candidate_inputs) == 0 or len(candidate_outputs) == 0:
                    if verbose: 
                        print("Warning: no matching node found for module %s" % m.name)
                    continue

                start_point = np.min([int(e) for e in candidate_inputs])
                theo_end_point = start_point + nb_nodes
                # look for the closest point to theo_end_point
                dist = [abs(int(e) - theo_end_point) for e in candidate_outputs]
                end_point = int(candidate_outputs[np.argmin(dist)])
                
                if verbose:
                    print("---")
                    print(m.name)
                    print(f"start_point: {start_point:03d}")
                    print(f"end_point: {end_point:03d}")
                pytorch_names.append([start_point, end_point, m.name])
            else:
                start_point = candidates[0]
                end_point = candidates[-1]

            if len(m_children) > 0:
                children_candidates = [e for e in candidates if int(e) >= start_point and int(e) <= end_point]
                rec_find_match(m_children, children_candidates)

            if m is not None:
                candidates = [e for e in candidates if int(e) < start_point or int(e) > end_point]
            
    rec_find_match(model_dict)
    return pytorch_names
