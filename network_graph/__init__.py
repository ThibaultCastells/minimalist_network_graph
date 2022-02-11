minimum_required = '1.0.0'


# Ensure Pytorch is importable and its version is sufficiently recent. This
# needs to happen before anything else, since the imports below will try to
# import Pytorch, too.
def _ensure_pt_install():  # pylint: disable=g-statement-before-imports
    """Attempt to import Pytorch, and ensure its version is sufficient.
    Raises:
    ImportError: if either Pytorch is not importable or its version is
    inadequate.
    """
    try:
        import torch
    except ImportError:
        # Print more informative error message, then reraise.
        print('\n\nFailed to import Pytorch. '
              'To use this package, please install '
              'Pytorch (> %s) by following instructions at '
              'https://pytorch.org/get-started/locally/.\n\n' % minimum_required)
        raise

    del torch


_ensure_pt_install()

# Cleanup symbols to avoid polluting namespace.
del minimum_required

from . import graph_reading
from . import graph_drawing
from . import models
import torch
from typing import Any


def draw_net(model: torch.nn.Module, input: Any, debug: bool = False):
    model = model.eval()
    graph = graph_reading.Graph(model, input)
    if debug:
        print("-" * 20)
        print(model)
        print("-" * 20)
        graph.show_connections()  # output a written version of the graph (useful for debugging)
        print("-" * 20)

    draw = graph_drawing.DrawGraph(graph, debug=debug)
    draw.draw_graph()


import sys as _sys

for symbol in ['torch', 'Any', '_ensure_pt_install', '_sys']:
    delattr(_sys.modules[__name__], symbol)
