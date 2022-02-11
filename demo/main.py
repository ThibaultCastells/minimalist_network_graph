from network_graph import draw_net
import torch
import torchvision.models

from network_graph.models import *

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
    draw_net(model, torch.empty([1, 3, input_size, input_size]), debug=args.debug)
