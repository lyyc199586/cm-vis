"""
Flowchart Creation Tools
========================

This module provides classes and methods for creating and annotating flowcharts
in CM-VIS. It defines various node shapes (Rectangle, Cube, Diamond, Ellipse, 
Parallelogram) and the FlowScheme class for managing and drawing flowcharts.

The module supports creating professional flowcharts with customizable node
styles, connections, and layouts suitable for technical documentation and
process visualization.

Classes
-------
Node : Base class for flowchart nodes
Rectangle : Rectangular node shape
Cube : 3D cube-style node  
Diamond : Diamond-shaped decision node
Ellipse : Elliptical start/end node
Parallelogram : Parallelogram input/output node
FlowScheme : Main flowchart management class

Examples
--------
>>> import matplotlib.pyplot as plt
>>> from cm_vis.scheme import FlowScheme, Rectangle, Diamond
>>> 
>>> fig, ax = plt.subplots()
>>> flow = FlowScheme(ax)
>>> 
>>> # Create nodes
>>> start = Rectangle((1, 3), 1.5, 0.6, "Start Process")
>>> decision = Diamond((1, 2), 1.2, 0.8, "Decision?")
>>> 
>>> # Add to flowchart
>>> flow.add_node(start)
>>> flow.add_node(decision)
>>> flow.add_connection(start, decision, "bottom", "top")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Optional, Tuple, Dict
from .basic import Scheme


class Node:
    """
    Base class for flow chart node.
    Represents a generic node with a center, width, height, and optional text.
    """

    def __init__(self, center: Tuple[float, float], width: float, height: float, text: str = "", fc="white", ec="black", textc="black") -> None:
        self.center = center
        self.width = width
        self.height = height
        self.text = text
        self.fc = fc
        self.ec = ec
        self.textc = textc

    def get_anchor(self, direction: str) -> Tuple[float, float]:
        """
        Get anchor point on node boundary for connection placement.
        
        Parameters
        ----------
        direction : str
            Direction for anchor point location.
            Options: 'top', 'bottom', 'left', 'right'
            
        Returns
        -------
        tuple of float
            The (x, y) coordinates of the anchor point
            
        Examples
        --------
        >>> node = Rectangle((0, 0), 2, 1, "Test")
        >>> top_anchor = node.get_anchor('top')
        >>> print(top_anchor)
        (0.0, 0.5)
        """
        x, y = self.center
        dx, dy = self.width / 2, self.height / 2
        return {"top": (x, y + dy), "bottom": (x, y - dy), "left": (x - dx, y), "right": (x + dx, y)}[direction]

    def draw(self, scheme: "FlowScheme") -> None:
        """
        Draw the node on the flowchart scheme.
        
        Abstract method that must be implemented by subclasses to define
        how each specific node type should be rendered.
        
        Parameters
        ----------
        scheme : FlowScheme
            The flowchart scheme object to draw on
            
        Raises
        ------
        NotImplementedError
            If called on base Node class directly
        """
        raise NotImplementedError("Each subclass must implement draw.")


class Rectangle(Node):
    """
    Rectangle node for flowcharts.
    """

    def draw(self, scheme: "FlowScheme") -> None:
        x, y = self.center
        dx = self.width / 2
        dy = self.height / 2
        rect = [(x - dx, y - dy), (x + dx, y - dy), (x + dx, y + dy), (x - dx, y + dy)]
        scheme.ax.add_patch(mpatches.Polygon(rect, closed=True, facecolor=self.fc, edgecolor=self.ec, lw=scheme.lw))
        if self.text:
            scheme.add_text(x, y, self.text, loc="center")


class Cube(Node):
    """
    Cube node for flowcharts (drawn as a 2.5D cube projection).
    """

    def __init__(self, center, width, height, text="", fc="white", ec="black", textc="black", side_color="lightgray", top_color="gray", depth=0.15):
        super().__init__(center, width, height, text, fc, ec, textc)
        self.side_color = side_color
        self.top_color = top_color
        self.depth = depth

    def get_anchor(self, direction: str) -> Tuple[float, float]:
        """
        Get the anchor point on the cube node for a given direction.
        Args:
            direction (str): One of 'top', 'bottom', 'left', 'right'.
        Returns:
            Tuple[float, float]: The (x, y) coordinates of the anchor.
        """
        x, y = self.center
        dx = self.width / 2
        dy = self.height / 2
        offset = self.depth * self.width
        return {"top": (x, y + dy), "bottom": (x, y - dy - offset / 2), "left": (x - dx - offset / 2, y), "right": (x + dx, y)}[direction]

    def draw(self, scheme: "FlowScheme") -> None:
        """
        Draw the cube node as a 2.5D projection.
        Args:
            scheme (FlowScheme): The flow scheme to draw on.
        """
        x, y = self.center
        dx = self.width / 2
        dy = self.height / 2
        offset = self.depth * self.width
        x = x - offset / 2
        y = y - offset / 2

        front = [(x - dx, y - dy), (x + dx, y - dy), (x + dx, y + dy), (x - dx, y + dy)]
        top = [(x - dx, y + dy), (x + dx, y + dy), (x + dx + offset, y + dy + offset), (x - dx + offset, y + dy + offset)]
        side = [(x + dx, y - dy), (x + dx + offset, y - dy + offset), (x + dx + offset, y + dy + offset), (x + dx, y + dy)]

        style = {"edgecolor": self.ec, "lw": scheme.lw, "joinstyle": "bevel"}
        scheme.ax.add_patch(mpatches.Polygon(top, closed=True, facecolor=self.top_color, **style))
        scheme.ax.add_patch(mpatches.Polygon(side, closed=True, facecolor=self.side_color, **style))
        scheme.ax.add_patch(mpatches.Polygon(front, closed=True, facecolor=self.fc, **style))

        if self.text:
            scheme.add_text(x, y, self.text, textc=self.textc, loc="center")


class Diamond(Node):
    """
    Diamond node for flowcharts.
    """

    def draw(self, scheme: "FlowScheme") -> None:
        x, y = self.center
        dx = self.width / 2
        dy = self.height / 2
        diamond = [(x, y + dy), (x + dx, y), (x, y - dy), (x - dx, y)]
        scheme.ax.add_patch(mpatches.Polygon(diamond, closed=True, facecolor=self.fc, edgecolor=self.ec, lw=scheme.lw))
        if self.text:
            scheme.add_text(x, y, self.text, loc="center")


class Ellipse(Node):
    """
    Ellipse node for flowcharts.
    """

    def draw(self, scheme: "FlowScheme") -> None:
        x, y = self.center
        ellipse = mpatches.Ellipse((x, y), width=self.width, height=self.height, facecolor=self.fc, edgecolor=self.ec, lw=scheme.lw)
        scheme.ax.add_patch(ellipse)
        if self.text:
            scheme.add_text(x, y, self.text, loc="center")


class Parallelogram(Node):
    """
    Parallelogram node for flowcharts.
    """

    def draw(self, scheme: "FlowScheme") -> None:
        x, y = self.center
        dx = self.width / 2
        dy = self.height / 2
        shift = 0.1 * self.width

        poly = [(x - dx - shift, y - dy), (x + dx - shift, y - dy), (x + dx + shift, y + dy), (x - dx + shift, y + dy)]

        scheme.ax.add_patch(mpatches.Polygon(poly, closed=True, facecolor=self.fc, edgecolor=self.ec, lw=scheme.lw))
        if self.text:
            scheme.add_text(x, y, self.text, loc="center")


shape_registry: Dict[str, type] = {"cube": Cube, "rectangle": Rectangle, "diamond": Diamond, "ellipse": Ellipse, "parallelogram": Parallelogram}


class FlowScheme(Scheme):
    """
    FlowScheme for managing and drawing flowchart diagrams.
    Inherits from Scheme and manages nodes and their connections.
    """

    def __init__(self, ax: plt.Axes, lw: float = 0.4) -> None:
        super().__init__(ax, lw)
        self.nodes: Dict[str, Node] = {}

    def add_node(self, name: str, shape: str, center: Tuple[float, float], width: float, height: float, text: str = "", **kwargs) -> None:
        """
        Add a node to the flowchart.
        Args:
            name (str): Node identifier.
            shape (str): Shape type (cube, rectangle, diamond, ellipse, parallelogram).
            center (Tuple[float, float]): Center coordinates.
            width (float): Node width.
            height (float): Node height.
            text (str, optional): Node label. Defaults to "".
        """
        shape = shape.lower()
        if shape not in shape_registry:
            raise ValueError(f"Unsupported shape '{shape}'. Available: {list(shape_registry.keys())}")

        node_cls = shape_registry[shape]
        node = node_cls(center=center, width=width, height=height, text=text, **kwargs)
        self.nodes[name] = node
        node.draw(self)

    def connect(
        self,
        name_from: str,
        name_to: str,
        from_dir: str = "right",
        to_dir: str = "left",
        label: Optional[str] = None,
        label_offset: float = 0.1,
        via: Optional[List[Tuple[float, float]]] = None,
        type: str = "-latex",
        fc: Optional[str] = None,
        ec: Optional[str] = None,
        head_offset: float = 0.0,
        tail_offset: float = 0.0,
    ):
        node_from = self.nodes[name_from]
        node_to = self.nodes[name_to]
        pt0 = np.array(node_from.get_anchor(from_dir))
        pt1 = np.array(node_to.get_anchor(to_dir))

        # arrow direction
        delta = pt1 - pt0
        norm = np.linalg.norm(delta)
        if norm > 1e-12:
            unit = delta / norm
        else:
            unit = np.zeros_like(delta)

        # apply offsets
        pt0_adj = pt0 + unit * tail_offset
        pt1_adj = pt1 - unit * head_offset

        # build arrow path
        if via is None:
            path_pts = [pt0_adj.tolist(), pt1_adj.tolist()]
            self.add_arrow(type=type, xy=path_pts, fc=fc, ec=ec)
        else:
            path_pts = [pt0_adj] + via + [pt1_adj]
            self.add_path_arrow(path_pts, type=type, fc=fc, ec=ec)

        # add label
        if label is not None:
            idx = (len(path_pts) - 1) // 2
            x0, y0 = path_pts[idx]
            x1, y1 = path_pts[idx + 1]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx, dy = x1 - x0, y1 - y0
            norm = np.hypot(dx, dy)
            if norm == 0:
                offset_x = offset_y = 0
            else:
                offset_x = -dy / norm * label_offset * self.max_len
                offset_y = dx / norm * label_offset * self.max_len
            self.add_text(mx + offset_x, my + offset_y, label, loc="center", fs="small")
