#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()

    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    # want the arrow to appear at the end
    xstart = xdata[end_ind]
    ystart = ydata[end_ind]
    xend = xdata[end_ind] + 0.5 * (xdata[end_ind] - xdata[start_ind])
    yend = ydata[end_ind] + 0.5 * (ydata[end_ind] - ydata[start_ind])

    line.axes.annotate('',
                       xytext=(xstart, ystart),
                       xy=(xend, yend),
                       arrowprops=dict(arrowstyle="-|>", color=color),
                       size=size
                       )