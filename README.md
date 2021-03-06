# A tool to get a minimalist view of any architecture

This tool has only be tested with the models included in this repo.
Therefore, I can't guarantee that it will work with other architectures, maybe you will have to adapt it a bit if your architecture is too complex or unusual.

The code to get the graph edges and nodes is a modified version of [this repo](https://github.com/waleedka/hiddenlayer). It does it by using the torch.jit._get_trace_graph functions of Pytorch.

The code to draw the graph is my own code, and I used [Turtle graphics](https://docs.python.org/3/library/turtle.html). I didn't use an existing library as my objective was to have something minimalist (i.e. no need to install anything, and the drawn graph only contains the essential info).

![overview](./demo/main_demo.gif)

## Quick start

```
PYTHONPATH=. python3 demo/main.py --arch arch_name --input input_size
```
By default, `--arch` is resnet_50 and `--input` is 224.
If your model doesn't use square images or 3 input channels, you can specify the exact input shape as a list (example: `--input '[1, 4, 224, 224]'`).
If you have a model that requires multiple inputs, you can write them as tuple (example: `--input '(224, 224)'` or `--input '([1, 3, 224, 224], [1, 3, 224, 224])'`).

Options for `--arch` (feel free to add more in *[models](network_graph/models/)*): 

input 224:
- mixnet_s, mixnet_m, mixnet_l
- atomnas_a
- resnet_50
- mobilenet_v1
- mobilenet_v2
- shufflenetv2plus_small

input 32:
- vgg_16_bn
- googlenet
- densenet_40
- resnet_56, resnet_110

## Installation

```
pip install git+https://github.com/ThibaultCastells/minimalist_network_graph
```
or clone the project and then 
```
cd minimalist_network_graph
python setup.py install
```
To run the demo, execute in the root folder
```
python demo/main.py --arch arch_name --input input_size
```
with optional arguments described above.

## Explanation of the view

The info printed at the top left corner appears when the mouse is over an operation. It indicates the node id, the operation type (as well as the kernel size and number of groups for Convolutions), the node input and output shape, the parents and children nodes, and the corresponding name in Pytorch (can be removed for optimization purposes with `--hide_pytorch_names`).

The legend isn't printed (since we can get the info by hovering the mouse over the nodes), but the most important things to know are: yellow with a dot is conv (different shades for different kernel size), purple-ish is ReLU, green is BN, pink with a dot is Linear.

Example: MixNet large (*mixnet_l*):
![mixnet_l](./demo/mixnet_l.png)

## Mouse commands

Left click will draw a big dot. Right click will erase all the dots. Mouse scroll will change the color (the selected color will be shown at the top left of the screen: by default, 5 different colors are available).

Demo:

![Use color](./demo/color_demo.gif)

## Modify the code

[The list of available operations](https://github.com/onnx/onnx/blob/main/docs/Operators.md) being really long, I didn't implement a specific drawing for all of them. If you feel like one of them should be added, this can be done easily in *[op.py](network_graph/graph_drawing/op.py)*. The one that are not implemented will be displayed in dark grey by default.

If you want to add a model, put the architecture file in *[models](network_graph/models/)*, import it in *[main.py](demo/main.py)*, and you are good to go.

If there is a specific operation that you don't want to see, you can add it in the *REMOVED_NODES* list in *[graph.py](network_graph/graph_reading/graph.py)*.

Also, if you have improvement ideas or if you want to contribute, you are welcome to open an issue or a pull request!

## Testing

To run the tests, execute the following in the project root

```
make test
```

## Known issues

- For complex connections (such as in atomnas model), some connections are drawn on top of each other, so it may be hard to understand. In this situation, you can use the text info (top left) to know the children and parents of each nodes.

- Models that contain completely independent graphes won't work (however, it isn't a common scenario). Example:

``` python
class FOO(nn.Module):
    def __init__(self, in_channels):
        super(FOO, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, 3)
        self.conv1 = torch.nn.Conv2d(in_channels, 2, 3)
        self.conv2 = torch.nn.Conv2d(in_channels, 2, 3)

    def forward(self, input1, input2):
        input1 = self.conv(input1)
        input2 = self.conv(input2)
        return self.conv1(input1), self.conv2(input2)
```

In this scenario, we have 2 distinct graphs: 1) input1 => conv => conv1 and 2) input2 => conv => conv2. therefore, it won't work.
However, if we add an interaction between these 2 graphs, there won't be any issue (example: if conv1 and conv2 take input1+input2 as input)

## Requirements :wrench:
* pytorch