from turtle import *
import turtle as t
import math

SUPPORTED_OPERATIONS = ['Add', 'Mul', 'Concat', 'Split', 'BatchNorm', 'Conv', 'Linear', 'Relu', 'Sigmoid',
                        'GlobalAveragePool', 'AveragePool', 'MaxPool']


class Operation():
    def __init__(self, op_size, x, y, params):
        self.x, self.y = x, y
        self.params = params
        self.op_size = op_size
        self.diag = math.sqrt(2) * op_size

    def draw_square(self, color=None):
        self.goto(self.x - round(self.op_size / 2), self.y - round(self.op_size / 2))
        if color is not None:
            t.fillcolor(color)
            t.begin_fill()
        for _ in range(4):
            t.forward(round(self.op_size))  # Forward turtle by op_size units
            t.left(90)  # Turn turtle by 90 degree
        if color is not None: t.end_fill()
        self.goto(self.x + round(self.op_size / 2), self.y)

    def draw_circle(self, color=None):
        # self.goto(self.x, self.y)
        self.goto(self.x, self.y - round(self.op_size / 2))
        if color is not None:
            t.fillcolor(color)
            t.begin_fill()
        t.circle(round(self.op_size / 2))
        if color is not None: t.end_fill()
        self.goto(self.x + round(self.op_size / 2), self.y)

    def goto(self, x, y):
        t.up()
        t.goto(x, y)
        t.down()
        t.setheading(0)


class Unspecified(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square(color='dim gray')


class Conv(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        if 'kernel_shape' in params:
            kernel_shape = params['kernel_shape']
            if 1 in kernel_shape:
                self.draw_square(color='#FAF0B3')
            elif 3 in kernel_shape:
                self.draw_square(color='#F1D526')
            elif 5 in kernel_shape:
                self.draw_square(color='#D6B900')
            elif 7 in kernel_shape:
                self.draw_square(color='#A79000')
            elif 9 in kernel_shape:
                self.draw_square(color='#776700')
            else:
                self.draw_square(color='#473E00')
        else:
            self.draw_square(color='#FAF0B3')
        self.goto(self.x, self.y)
        t.dot(round(self.op_size / 8))
        self.goto(self.x + round(self.op_size / 2), self.y)


class Linear(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square(color='#d79eba')
        self.goto(self.x, self.y)
        t.dot(round(self.op_size / 8))
        self.goto(self.x + round(self.op_size / 2), self.y)


class Add(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_circle()
        t.setheading(180)  # 270 south, 0 east, 180 west, 90 north
        t.forward(self.op_size)
        self.goto(x, y + round(self.op_size / 2))
        t.setheading(270)
        t.forward(self.op_size)
        self.goto(self.x + round(self.op_size / 2), self.y)


class Mul(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_circle()
        r = round(self.op_size / 2)
        # go to top left
        x_delta = math.cos(math.radians(135)) * r
        y_delta = math.sin(math.radians(135)) * r
        self.goto(x + x_delta, y + y_delta)
        # go to bottom right
        t.setheading(270 + 45)  # south east
        x_delta = math.cos(math.radians(-45)) * r
        y_delta = math.sin(math.radians(-45)) * r
        t.goto(x + x_delta, y + y_delta)
        # go to top right
        x_delta = math.cos(math.radians(45)) * r
        y_delta = math.sin(math.radians(45)) * r
        self.goto(x + x_delta, y + y_delta)
        # go to bottom left
        t.setheading(270 - 45)  # south west
        x_delta = math.cos(math.radians(270 - 45)) * r
        y_delta = math.sin(math.radians(270 - 45)) * r
        t.goto(x + x_delta, y + y_delta)

        self.goto(self.x + round(self.op_size / 2), self.y)


class Concat(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square()
        self.goto(x - round(self.op_size / 2), y + round(self.op_size / 2))
        t.setheading(270 + 45)  # south east
        t.forward(self.diag)
        self.goto(x - round(self.op_size / 2), y - round(self.op_size / 2))
        t.setheading(0 + 45)  # north east
        t.forward(self.diag)
        self.goto(self.x + round(self.op_size / 2), self.y)


class Split(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square()
        t.setheading(180)  # 270 south, 0 east, 180 west, 90 north
        t.forward(self.op_size)
        self.goto(self.x + round(self.op_size / 2), self.y)


class BatchNorm(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square(color='#CC9FD9')


class Sigmoid(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square(color='#A7CBDC')


class Relu(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square(color='#8EDEC1')


class GlobalAveragePool(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square(color='#7c4710')


AveragePool = GlobalAveragePool


class MaxPool(Operation):
    def __init__(self, op_size, x, y, params):
        super().__init__(op_size, x, y, params)
        self.draw_square(color='#9f7243')
