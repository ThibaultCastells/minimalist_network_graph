from turtle import Screen, Turtle
import turtle as t
from .op import *
import math


class DrawGraph:

    def __init__(self, graph, debug=False, pytorch_names=None):
        t.tracer(0)
        self.debug = debug
        self.graph = graph
        self.drawn = [[None, None] for i in range(len(self.graph.nodes))]  # x,y position, index = id

        t.mode("standard")
        t.speed('fastest')
        t.hideturtle()

        self.highlighted = [False for i in range(len(self.graph.nodes))] # highlight active or not of each node
        self.highlight_turtle = []
        for _ in range(len(self.graph.nodes)):
            # each highlight has it's own turtle (allows to clear one specific highlight)
            t_highlight = t.Turtle()
            t_highlight.speed('fastest')
            t_highlight.hideturtle()
            self.highlight_turtle.append(t_highlight)
        self.highlight_status = [0 for i in range(len(self.graph.nodes))] # highlight status of each node
        self.highlight_status_choices = [('#2164a0', 4), ('#cc0000', 4), ('#388c14', 4), ('#725db0', 4)] # blue red green purple

        self.drawn_op = set()  # not used currently, but may be used to draw the legend
        self.pytorch_names = pytorch_names

        self.w, self.h = 1700, 1200
        self.w_canvas, self.h_canvas = 1700, 800
        self.op_size = 20

        self.create_canvas()

        print(f"h:{t.window_width()}, w:{t.window_height()}, op_size:{self.op_size}")

        t.title("Minimalist Network Graph")

        self.init_events()

    def create_canvas(self):
        self.screen = Screen()
        self.screen.screensize(self.w, self.h)  # this is the size of the screen
        self.screen.setup(self.w_canvas, self.h_canvas)  # this is the size of the window (=seen part of the screen)
        self.canvas = self.screen.getcanvas()

    def draw_node(self, node, x, y):
        """
            Draw a node at the given position
            Args:
                node: the node to draw
                x: x position (of the center of the node)
                y: y position (of the center of the node)
        """
        self.drawn_op.add(node.op)
        if node.op in SUPPORTED_OPERATIONS:
            eval(f"{node.op}(self.op_size, {x}, {y}, {node.params})")
        else:
            Unspecified(self.op_size, x, y, node.params)

    def coord_equal(self, coord1, coord2, eps=0.01):
        return abs(coord1[0] - coord2[0]) < eps and abs(coord1[1] - coord2[1]) < eps

    def draw_nodes_and_edges(self, start_x=0, start_y=0):
        start_x, start_y = round(start_x), round(start_y)
        end_skip_connection = []

        self.goto(start_x, start_y)

        node_input = self.graph.node_input if isinstance(self.graph.node_input, list) else [self.graph.node_input]
        coord = self.draw_branches(len(node_input), pen_up=True)
        curr_branches = [(node_input[i], coord[i], ([] if len(node_input) == 1 else [start_y]), None) for i in
                         range(len(node_input))]
        topological_sort = self.graph.get_topological_sort()

        # go through the graph in topological order
        while (len(topological_sort) > 0):
            u = topological_sort[-1]
            del topological_sort[-1]
            node_id = f'{u:03d}'
            node = self.graph[node_id]

            node_coords = [e for e in curr_branches if e[0].id == node_id]  # [(next_node, coord, center, node_id)]

            if len(node_coords) == 0:
                # this is an imperfect way to solve the problem of a branch not starting from a node.
                # it simply removes this branch
                self.drawn[u] = [0, 0]
                # print(f"{u}: {node.op}")
                continue
            elif len(node_coords) == 1:
                node_coord = node_coords[0]
                x, y = node_coord[1]
                center = node_coord[2]
            else:
                node_coords = sorted(node_coords, key=lambda e: e[1][0], reverse=True)
                node_coord = node_coords[0]
                center = node_coord[2]
                x = node_coord[1][0]
                parent = node_coord[3]
                parent_nb_children = len(self.graph.outgoing(self.graph[parent]))
                if parent_nb_children == 1:
                    y = center[-1] if len(center) > 0 else 0
                else:
                    y = node_coord[1][1]
                center = node_coord[2][:-1]

            # remove the branches that are already drawn
            for i in range(len(curr_branches), 0, -1):
                if curr_branches[i - 1][0].id == node_id:
                    if len(node_coords) > 1: end_skip_connection.append(curr_branches[i - 1])
                    del curr_branches[i - 1]

            self.draw_node(node, x + round(self.op_size / 2), y)
            self.drawn[u] = [x, y]

            node_next = self.graph.outgoing(node)
            if len(node_next) == 0:
                continue  # for the last node, there is no next node
            if len(node_next) > 1:
                center = center.copy()
                center.append(round(t.pos()[1]))  # if >1 branches, memorize the y-coord of the last branch
            coord = self.draw_branches(len(node_next), len(center))
            for i in range(len(node_next)):
                curr_branches.append((node_next[i], coord[i], center, node.id))

        # complete the skip connections
        for node, coord, _, parent_id in end_skip_connection:
            x_prev, y_prev = coord
            x_parent, y_parent = self.drawn[int(parent_id)]
            if self.coord_equal([x_prev, y_prev], [x_parent, y_parent]):
                x_prev += round(self.op_size / 2)
            x, y = self.drawn[int(node.id)]

            self.goto(x_prev, y_prev)
            if self.coord_equal([x, y_prev], [x, y]):  # if prev is not higher/lower
                t.goto(x, y_prev)
            else:
                t.goto(x + round(self.op_size / 2), y_prev)
                t.goto(x + round(self.op_size / 2),
                       y + (round(self.op_size / 2) if y_prev > y else - round(self.op_size / 2)))
            self.goto(x, y)

        # update window size to fit the graph:
        x_sort_drawn = sorted(self.drawn, key=lambda e: e[0], reverse=True)
        x_min = x_sort_drawn[-2][0]
        x_max = x_sort_drawn[0][0]
        w = x_max - x_min + 10 * int(self.op_size / 2)
        if w < self.w_canvas:
            w = self.w_canvas
        self.screen.screensize(w, self.h)
        delta_w = round((self.w - w) / 2)
        self.w = w
        for element_id in self.canvas.find_all():
            self.canvas.move(element_id, delta_w, 0)
        for element_id in range(len(self.drawn)):
            self.drawn[element_id][0] += delta_w
        # scroll at the begining of the model (left)
        self.canvas.xview_scroll(-25, "page")

    def draw_legend(self):
        # draw a legend for the op (only legend the op that are present in this graph)
        return

    def draw_branches(self, n, depth=0, pen_up=False):
        # known issue: if branches in branches, then things are drawn on top of each other
        def int_tuple(tuple): 
            return (round(tuple[0]), round(tuple[1]))
        t.setheading(0)
        self.forward(round(self.op_size / 2), draw=(not pen_up))
        coord = []
        if n > 1:
            if not pen_up:
                t.dot(round(self.op_size / 4))
            gap = max(round(8 * self.op_size - 1.5 * (depth * self.op_size)), round(2 * self.op_size))
            (x0,y0) = int_tuple(t.pos())
            start = round(y0 + ((n / 2 - 1 / 2) * gap))
            self.goto(x0, start)
            for i in range(n):
                t.setheading(0)
                self.forward(round(self.op_size / 2), draw=(not pen_up))
                coord.append(int_tuple(t.pos()))
                if i + 1 < n:
                    self.forward(round(self.op_size / 2), 180, draw=False)
                    if i > 0: t.dot(round(self.op_size / 4))
                    t.setheading(270)
                    self.forward(gap, draw=(not pen_up))
            t.setheading(0)
        else:
            coord.append(int_tuple(t.pos()))
        return coord

    def forward(self, dist, heading=None, draw=True):
        if heading is not None: t.setheading(heading)
        if not draw: t.up()
        t.forward(dist)
        if not draw: t.down()
        t.setheading(0)

    def goto(self, x, y, turtle=None):
        if turtle is None: turtle = t
        turtle.up()
        turtle.goto(x, y)
        turtle.down()
        turtle.setheading(0)

    def draw_graph(self):
        self.draw_nodes_and_edges(start_x=-self.w / 2 + 0.5 * self.op_size + 10)
        self.draw_legend()
        t.update()
        t.done()

    def init_events(self):
        # create an independent turtle to draw the text info
        t_info = t.Turtle()
        t_info.speed('fastest')
        t_info.hideturtle()

        # mouse click event
        self.last_seen = None

        def motion(event):
            mouse_x, mouse_y = self.canvas.canvasx(event.x), -self.canvas.canvasy(event.y)
            for item in self.canvas.find_all():
                # compare mouse pos to all widgets in the canvas, to find the ones that are under the mouse
                obj_coords = self.canvas.coords(item)
                x = [obj_coords[i] for i in range(0, len(obj_coords), 2)]
                y = [-obj_coords[i] for i in range(1, len(obj_coords), 2)]
                min_x, max_x = min(x), max(x)
                min_y, max_y = min(y), max(y)
                if mouse_x >= min_x and mouse_x <= max_x and mouse_y >= min_y and mouse_y <= max_y:
                    # if the mouse is over a widget, check if this widget is a node
                    for i, e in enumerate(self.drawn):
                        if e[0] + self.op_size / 2 >= min_x and e[0] <= max_x and e[1] >= min_y and e[1] <= max_y:
                            self.curr_id = f"{i:03d}"
                            curr_id = self.curr_id # just for readability
                            self.on_node = True
                            if curr_id != self.last_seen:
                                # print(f"{curr_id}: {e[0]}, {e[1]} ({self.graph[curr_id].op})")
                                t_info.clear()

                                x_txt = (self.canvas.xview()[0] - 0.5) * self.w
                                y_txt = (0.5 - self.canvas.yview()[0]) * self.h
                                delta_y = 23
                                font = ("Arial", 13, "normal")
                                curr_y = y_txt - 30

                                self.goto(x_txt + 5, curr_y, turtle=t_info)
                                
                                info_conv = "  "
                                if self.graph[curr_id].op == 'Conv':
                                    if 'kernel_shape' in self.graph[curr_id].params:
                                        info_conv += f"  |  k: {self.graph[curr_id].params['kernel_shape']}"
                                    if 'group' in self.graph[curr_id].params:
                                        info_conv += f"  |  group: {self.graph[curr_id].params['group']}"
                                t_info.write(f"{curr_id}: {self.graph[curr_id].op}" + info_conv, font=font)

                                shape = self.graph[curr_id].shape
                                if shape is not None:
                                    curr_y -= delta_y
                                    self.goto(x_txt + 5, curr_y, turtle=t_info)
                                    info_in = f"input shape: {shape[0]}" if shape[0] is not None and len(shape[0])>0 else ""
                                    info_out = f"output shape: {shape[1]}" if shape[1] is not None and len(shape[1])>0 else ""
                                    separator = " | " if info_in != "" and info_out != "" else ""
                                    t_info.write(f"{info_in}{separator}{info_out}", font=font)

                                curr_y -= delta_y
                                self.goto(x_txt + 5, curr_y, turtle=t_info)
                                t_info.write(f"parents: {[e.id for e in self.graph.incoming(self.graph[curr_id])]}",
                                            font=font)
                                curr_y -= delta_y
                                self.goto(x_txt + 5, curr_y, turtle=t_info)
                                t_info.write(f"children: {[e.id for e in self.graph.outgoing(self.graph[curr_id])]}",
                                            font=font)

                                if self.pytorch_names is not None:
                                    id_matches = [i for i in self.pytorch_names if i[0] <= int(curr_id) and i[1] >= int(curr_id)]
                                    if len(id_matches) > 0:
                                        curr_y -= delta_y
                                        self.goto(x_txt + 5, curr_y, turtle=t_info)
                                        t_info.write(f"PyTorch name: {id_matches[-1][2]}", font=font)


                                if self.debug:
                                    curr_y -= delta_y
                                    self.goto(x_txt + 5, curr_y, turtle=t_info)
                                    t_info.write(f"pos: {self.drawn[i]}", font=font)

                                self.last_seen = curr_id
                            return
            self.on_node = False

        self.canvas.bind('<Motion>', motion)

        # create an independent turtle to draw dots
        t_draw = t.Turtle()
        t_draw.speed('fastest')
        t_draw.hideturtle()
        self.colors = ["#B31C26", "#222268", "#008F00", "#E28100", "#9920E6"]
        self.current_color = 0
        t_draw.color(self.colors[self.current_color])

        def draw_highlight(x, y, turtle):
            turtle.clear()
            self.goto(x, y - round(self.op_size / 2), turtle=turtle)
            for _ in range(4):
                turtle.forward(round(self.op_size))  # Forward turtle by op_size units
                turtle.left(90)  # Turn turtle by 90 degree
            self.goto(x, y, turtle=turtle)

        def update_highlight_turtle(node_idx):
            turtle = self.highlight_turtle[node_idx]
            highlight_status = self.highlight_status[node_idx]
            color, thickness = self.highlight_status_choices[highlight_status]
            turtle.color(color)
            turtle.width(thickness)
            return turtle

        def update_highlight_status(node_idx, status=None):
            # update the highlight status of a node
            if status is None:
                prev_status = self.highlight_status[node_idx]
                status = (prev_status + 1) % len(self.highlight_status_choices)
            self.highlight_status[node_idx] = status

        def update_highlight_activation_status(node_idx):
            # update highlight activation status of a node
            self.highlighted[node_idx] = not self.highlighted[node_idx]
            if not self.highlighted[node_idx]:
                # if not highlighted anymore, remove the highlight
                self.highlight_turtle[node_idx].clear()
            else:
                # else, add the highlight
                turtle = update_highlight_turtle(node_idx)
                x,y = self.drawn[node_idx]
                draw_highlight(x, y, turtle)
           
        def left_click(event):
            if self.on_node:
                # if a node is left clicked
                node_idx = int(self.curr_id)
                if self.highlighted[node_idx]:
                    # if it was highlighted, change the highlight color
                    update_highlight_status(node_idx)
                    turtle = update_highlight_turtle(node_idx)
                    x,y = self.drawn[node_idx]
                    draw_highlight(x, y, turtle)
                else:
                    # if it wasn't highlighted, highlight it
                   update_highlight_activation_status(node_idx)
            else:
                x_canvas = (self.canvas.xview()[0] - 0.5) * self.w
                y_canvas = (0.5 - self.canvas.yview()[0]) * self.h
                x, y = event.x, event.y
                self.goto(x_canvas + x, y_canvas - y, turtle=t_draw)
                t_draw.dot(self.op_size)

        self.canvas.bind("<Button-1>", left_click)

        def right_click(event):
            if self.on_node:
                # if a node is right clicked, un-highlight it
                node_idx = int(self.curr_id)
                update_highlight_activation_status(node_idx)
            else:
                t_draw.clear()

        self.canvas.bind("<Button-2>", right_click)
        self.canvas.bind("<Button-3>", right_click)

        def mouse_wheel(event):
            self.current_color = (self.current_color + 1) % len(self.colors)
            t_draw.color(self.colors[self.current_color])

            t_info.clear()
            t_info.color(self.colors[self.current_color])
            x_txt = (self.canvas.xview()[0] - 0.5) * self.w
            y_txt = (0.5 - self.canvas.yview()[0]) * self.h
            self.goto(x_txt + self.op_size / 2 + 5, y_txt - self.op_size / 2 - 5, turtle=t_info)
            t_info.dot(self.op_size)
            t_info.color("black")

        # with Windows OS
        self.canvas.bind("<MouseWheel>", mouse_wheel)
        # with Linux OS
        self.canvas.bind("<Button-4>", mouse_wheel)
        self.canvas.bind("<Button-5>", mouse_wheel)
