import pytest
from network_graph import graph_reading
from network_graph import models
from network_graph.models import *
import inspect
import torch

size_32 = {'vgg_16_bn', 'googlenet', 'densenet_40', 'resnet_56', 'resnet_110'}


@pytest.mark.parametrize(
    "model", [s[0] for s in inspect.getmembers(models, inspect.isfunction)]
)
def test_onnx_graph(model):
    size = 224
    if model in size_32:
        size = 32

    input = torch.empty(1, 3, size, size)
    model = eval(model)().eval()
    graph = graph_reading.Graph(model, input)
    print("-" * 20)
    print(model)
    print("-" * 20)
    graph.show_connections()  # output a written version of the graph (useful for debugging)
    print("-" * 20)

    pytorch_names = graph_reading.get_pytorch_names(model, graph, input, verbose=False)
    print(model)
