# A tool to get a minimalist view of any architecture

This tool has only be tested with the models included in this repo.
Therefore, I can't guarantee that it will work with other architectures, maybe you will have to adapt it a bit if your architecture is too complex or unusual.

The code to get the graph edges and nodes is a modified version of [this repo](https://github.com/waleedka/hiddenlayer). It does it by using the torch.jit._get_trace_graph functions of Pytorch.

The code to draw the graph is my own code, and I used [Turtle graphics](https://docs.python.org/3/library/turtle.html). I didn't use an existing library as my objective was to have something minimalist (i.e. no need to install anything, and the drawn graph only contains the essential info).


## Quick start

```
python3 main.py --arch arch_name --input input_size
```
By default, `--arch` is resnet_50 and `--input` is 224.

Options for `--arch` (feel free to add more in *[models](models/)*): 

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

## Explanation of the view

The info printed at the top left corner appears when the mouse is over an operation. It indicates the node id, the operation type, the parents and children nodes and the position of the node in the screen (useful when we want to modify the code).

The legend isn't printed (since we can get the info by hovering the mouse over the nodes), but the most important things to know are: yellow with a dot is conv (different shades for different kernel size), purple-ish is ReLU, green is BN, pink with a dot is Linear.

ResNet 50 (*resnet_50*):
![resnet_50](/demo/resnet50.png)

MixNet large (*mixnet_l*):
![mixnet_l](/demo/mixnet_l.png)

## Modify the code

[The list of available operations](https://github.com/onnx/onnx/blob/main/docs/Operators.md) being really long, I didn't implement a specific drawing for all of them. If you feel like one of them should be added, this can be done easily in *[op.py](graph_drawing/op.py)*. The one that are not implemented will be displayed in dark grey by default.

If you want to add a model, put the architecture file in *[models](models/)*, import it in *[main.py](main.py)*, and you are good to go.

If there is a specific operation that you don't want to see, you can add it in the *REMOVED_NODES* list in *[graph.py](graph_reading/graph.py)*.

Also, if you have improvement ideas or if you want to contribute, you can send me a message :)

## Known issues

- If you use a model that contains slices with step>1, then you will get the following error: 

```
RuntimeError: step!=1 is currently not supported
```

This is due too the *torch.onnx._optimize_trace* function that doesn't support step>1 slices (so for instance, you can't do *x[::2]*).

- For complex conenctions (such as in atomnas model), some connections are drawn on top of each other, so it may be hard to understand. In this situation, you can use the text info (top left) to know the children and parents of each nodes.

## Requirements :wrench:
* pytorch