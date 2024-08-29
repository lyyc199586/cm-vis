import numpy as np
from matplotlib.patches import FancyArrowPatch, ArrowStyle

def axis_line(ax, lw=0.4, origin=[0, 0], labels=None, arrowfc='k', offset=0.05):
    """
    Set the axes of the given matplotlib Axes object to intersect at the specified position (x, y)
    and point in the specified directions given by vectors.
    
    Parameters:
    ax (matplotlib.axes.Axes): The axes object to modify.
    origin [float, float]: The x and y-coordinate of the intersection point.
    lw (float): Line width of the arrows.
    arrowfc (str): Face color of the arrows.
    offset (float): Ratio to extend the axis limits (0.0, 1.0).
    """
    # Set the spines to be at the specified position
    ax.spines['left'].set_position(('data', origin[0]))
    ax.spines['bottom'].set_position(('data', origin[1]))
    ax.spines['left'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    
    # Move the ticks to the bottom and left
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate the extension based on the current axis limits
    x_ext = (xlim[1] - xlim[0]) * offset
    y_ext = (ylim[1] - ylim[0]) * offset
    
    # Adjust axis limits to include the origin point
    xlim = (min(xlim[0], origin[0]) - x_ext, max(xlim[1], origin[0]) + x_ext)
    ylim = (min(ylim[0], origin[1]) - y_ext, max(ylim[1], origin[1]) + y_ext)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    xy_from = [[min(xlim[0], origin[0]), origin[1]], [origin[0], min(ylim[0], origin[1])]]
    xy_to = [[max(xlim[1], origin[0]), origin[1]], [origin[0], max(ylim[1], origin[1])]]
    
    # Create arrows
    for i in range(2):
        start = xy_from[i]
        end = xy_to[i]
        arrow_props = dict(posA=start, posB=end, shrinkA=0, shrinkB=0, lw=lw,
                           joinstyle="miter", capstyle="butt", fc=arrowfc, ec=arrowfc)
        style = ArrowStyle('-|>', head_length=6*lw, head_width=2*lw)
        arrow_props.update(arrowstyle=style)
        arr = FancyArrowPatch(**arrow_props)
        ax.add_patch(arr)
    
    # set labels
    if labels is not None:
        x_length = xlim[1] - xlim[0]
        y_length = ylim[1] - ylim[0]
        ax.set_xlabel(labels[0])
        ax.xaxis.set_label_coords(1.05, (origin[1] - ylim[0])/y_length)
        ax.set_ylabel(labels[1], rotation=0)
        ax.yaxis.set_label_coords((origin[0] - xlim[0])/x_length, 1.05)