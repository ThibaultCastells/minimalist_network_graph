
import graph_reading as gr
import graph_drawing as gd
import torch
import torchvision.models

from models.mixnet import mixnet_s, mixnet_m, mixnet_l
from models.atomnas import atomnas_a
from models.vgg import vgg_16_bn
# from models.resnet_cifar10 import resnet_56, resnet_110 # not working because of ::2
from models.resnet_imgnet import resnet_50
from models.googlenet import googlenet
from models.densenet import densenet_40
from models.mobilenetv1 import mobilenet_v1
from models.mobilenetv2 import mobilenet_v2
from models.shufflenetv2_plus import shufflenetv2plus_small

import os
import argparse
import sys

# ======================= PARSER =======================
def get_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Parser')

    parser.add_argument(
        '--input',
        type=int,
        default=224,
        help='Input data size')

    parser.add_argument(
        '--arch',
        type=str,
        default='resnet_50',
        help='The architecture of the model')

    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='Mode to display debug information')

    args = parser.parse_args(args)
    return args

# ======================== MAIN ========================
if __name__ == '__main__':
    args = get_args()
    model = eval(args.arch)()
    input_size = args.input

    model = model.eval()
    
    graph = gr.Graph(model, torch.zeros([1, 3, input_size, input_size]))
    if args.debug: 
        print("-"*20)
        print(model)
        print("-"*20)
        graph.show_connections() # output a written version of the graph (useful for debugging)
        print("-"*20)
    draw = gd.DrawGraph(graph, debug=args.debug)
    draw.draw_graph()