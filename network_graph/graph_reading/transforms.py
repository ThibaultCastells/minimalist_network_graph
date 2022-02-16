"""
Code based on HiddenLayer, by Waleed Abdulla
Licensed under the MIT License
"""

import re
import copy
from .graph import Node
from . import ge


###########################################################################
# Transforms
###########################################################################

class Fold():
    def __init__(self, pattern, op):
        # TODO: validate that op and name are valid
        self.pattern = ge.GEParser(pattern).parse()
        self.op = op

    def apply(self, graph):
        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break

            # Replace pattern with new node
            if self.op == "__first__":
                combo = matches[0]
            elif self.op == "__last__":
                combo = matches[-1]
            else:
                uid = '_'.join([matches[i].id for i in range(len(matches))])
                combo = Node(uid=uid,
                             op=self.op or self.pattern,
                             shape=matches[-1].shape)
            graph.replace(matches, combo)


class FoldId():
    def __init__(self, id_regex, op):
        # TODO: validate op and name are valid
        self.id_regex = re.compile(id_regex)
        self.op = op

    def apply(self, graph):
        # Group nodes by the first matching group of the regex
        groups = {}
        for node in graph.nodes.values():
            m = self.id_regex.match(node.id)
            if not m:
                continue

            assert m.groups(), "Regular expression must have a matching group to avoid folding unrelated nodes."
            key = m.group(1)
            if key not in groups:
                groups[key] = []
            groups[key].append(node)

        # Fold each group of nodes together
        for key, nodes in groups.items():
            # Replace with a new node
            # TODO: Find last node in the sub-graph and get the output shape from it
            combo = Node(uid=key,
                         op=self.op)
            graph.replace(nodes, combo)


class Prune():
    def __init__(self, pattern):
        self.pattern = ge.GEParser(pattern).parse()

    def apply(self, graph):
        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            # Remove found nodes
            graph.remove(matches)
        # return graph


class PruneBranch():
    def __init__(self, pattern):
        self.pattern = ge.GEParser(pattern).parse()

    def tag(self, node, tag, graph, conditional=False):
        # Return if the node is already tagged
        if hasattr(node, "__tag__") and node.__tag__ == "tag":
            return
        # If conditional, then tag the node if and only if all its
        # outgoing nodes already have the same tag.
        if conditional:
            # Are all outgoing nodes already tagged?
            outgoing = graph.outgoing(node)
            tagged = filter(lambda n: hasattr(n, "__tag__") and n.__tag__ == tag,
                            outgoing)
            if len(list(tagged)) != len(outgoing):
                # Not all outgoing are tagged
                return
        # Tag the node
        node.__tag__ = tag
        # Tag incoming nodes
        for n in graph.incoming(node):
            self.tag(n, tag, graph, conditional=True)

    def apply(self, graph):
        while True:
            matches, _ = graph.search(self.pattern)
            if not matches:
                break
            # Tag found nodes and their incoming branches
            for n in matches:
                self.tag(n, "delete", graph)
            # Find all tagged nodes and delete them
            tagged = [n for n in graph.nodes.values()
                      if hasattr(n, "__tag__") and n.__tag__ == "delete"]
            graph.remove(tagged)


class FoldDuplicates():
    def apply(self, graph):
        matches = True
        while matches:
            for node in graph.nodes.values():
                pattern = ge.SerialPattern([ge.NodePattern(node.op), ge.NodePattern(node.op)])
                matches, _ = pattern.match(graph, node)
                if matches:
                    # Use op from the first node, and shape from the last
                    combo = Node(uid=graph.sequence_id(matches),
                                 op=node.op,
                                 shape=matches[-1].shape)
                    combo.repeat = sum([n.repeat for n in matches])
                    graph.replace(matches, combo)
                    break


class Rename():
    def __init__(self, op=None, to=None):
        assert op, "op must be provided"
        assert bool(to), "The to parameter is required"
        self.to = to
        self.op = re.compile(op) if op else None

    def apply(self, graph):
        for node in graph.nodes.values():
            if self.op:
                node.op = self.op.sub(self.to, node.op)


# Transforms to simplify graphs by folding layers that tend to be 
# used together often, such as Conv/BN/Relu.
# These transforms are used AFTER the framework specific transforms
# that map TF and PyTorch graphs to a common representation.
SIMPLICITY_TRANSFORMS = [
    # Fold("((Sigmoid > Mul) | ())", "Swish"),
    # Fold(" > Sigmoid > Mul | ", "Swish"),
    FoldDuplicates()
]
