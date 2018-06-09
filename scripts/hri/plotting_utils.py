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


def turn_off_axes_labels(ax):
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off') # labels along the bottom edge are off


def visualize_explicit_model():
    A = 100.0
    M = 2.4

    t_c = 0.0
    t = np.arange(0, 20, 0.1)

    P = (A * np.exp(t_c-t / M) + 1.0) / (A * np.exp(t_c-t / M) + 2.0)

    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax.plot(t, P, '-', lw=1.5, color=(0.3, 0.3, 0.8))
    ax.set_yticks([0.5, 0.75, 1.0])

    plt.show()

if __name__ == "__main__":
    visualize_explicit_model()
