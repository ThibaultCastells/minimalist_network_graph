import sys
[sys.path.append(i) for i in ['.', '..']]

from network_graph import draw_net
import torch
import torchvision.models

from network_graph.models import *

import os
import argparse
import ast


# ======================= PARSER =======================
def get_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Parser')

    parser.add_argument(
        '--input',
        type=str,
        default=224,
        help='Input data size.\n\
            if int: the input will be a square image of dim [1,3,input,input]\n\
            if list: the input will use the list as dimensions\n\
            if tuple: each element in the tuple will be considered as an independent input')

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

    parser.add_argument(
        '--hide_pytorch_names',
        default=False,
        action='store_true',
        help='Hide the pytorch names of the modules in which nodes are coming from. This can improve the graph generation speed.')

    args = parser.parse_args(args)
    return args


# ======================== MAIN ========================
if __name__ == '__main__':
    args = get_args()
    model = eval(args.arch)()

    input = ast.literal_eval(args.input)
    if isinstance(input, list):
        input = torch.empty(input)
    elif isinstance(input, tuple):
        # tuple of inputs (if multiple distinct inputs)
        tmp_input = []
        for elem in input:
            if isinstance(elem, list):
                tmp_input.append(torch.empty(elem))
            else:
                tmp_input.append(torch.empty([1, 3, elem, elem]))
        input = tuple(tmp_input)
    else:
        # by default, we assume a square input with 3 channels
        input = torch.empty([1, 3, int(args.input), int(args.input)])


    draw_net(model, input, debug=args.debug, match_pytorch_graph=not args.hide_pytorch_names)
