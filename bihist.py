#!/usr/bin/env python
# Time-stamp: <2012-04-03 23:47 ycopin@...4058...>
# Author: Yannick Copin <y.copin@...671...>

def bihist(ax, upatches, lpatches, orientation='vertical'):
    """Convert upper/lower (or right/left) patches from previous calls
    to ax.hist to bi-histogram.

    Reference: http://www.itl.nist.gov/div898/handbook/eda/section3/bihistog.htm
    """

    if orientation.startswith('v'):             # Vertical orientation
        for p in lpatches:
            try:
                p._height *= -1                 # matplotlib.patches.Rectangle
            except AttributeError:
                p._path.vertices[:,1] *= -1     # matplotlib.patches.Polygon
    elif orientation.startswith('h'):           # Horizontal orientation
        for p in upatches:
            try:
                p._width *= -1                  # matplotlib.patches.Rectangle
            except AttributeError:
                p._path.vertices[:,0] *= -1     # matplotlib.patches.Polygon
    else:
        raise ValueError("Unknown orientation '%s'" % orientation)

    ax.relim()
    ax.autoscale_view()

if __name__=='__main__':

    import numpy as N
    import matplotlib.pyplot as P

    x1 = N.random.randn(1000)
    x2 = N.random.randn(1000) + 1
    bins = N.linspace(-5,5,21)

    fig = P.figure()

    ax1 = fig.add_subplot(2,2,1,
                          title="Overplotted histograms",
                          ylabel="Vertical")

    orientation = 'vertical'
    ax1.hist(x1, bins=bins,
             histtype='bar',
             color='b', alpha=0.5, orientation=orientation,
             label="x1")
    ax1.hist(x2, bins=bins,
             histtype='bar',
             color='r', alpha=0.5, orientation=orientation,
             label="x2")
    ax1.legend()

    ax2 = fig.add_subplot(2,2,2,
                          title="Bi-histogram")
    n1,b,p1 = ax2.hist(x1, bins=bins,
                       histtype='bar',
                       color='b', alpha=0.5, orientation=orientation,
                       label="x1")
    n2,b,p2 = ax2.hist(x2, bins=bins,
                       histtype='bar',
                       color='r', alpha=0.5, orientation=orientation,
                       label="x2")
    bihist(ax2, p1, p2, orientation=orientation)
    ax2.legend()

    ax3 = fig.add_subplot(2,2,3,
                          ylabel="Step[filled], horizontal")

    orientation = 'horizontal'
    ax3.hist(x1, bins=bins,
             histtype='step',
             color='b', alpha=0.5, orientation=orientation,
             label="x1")
    ax3.hist(x2, bins=bins,
             histtype='stepfilled',
             color='r', alpha=0.5, orientation=orientation,
             label="x2")
    ax3.legend()

    ax4 = fig.add_subplot(2,2,4)
    n1,b,p1 = ax4.hist(x1, bins=bins,
                       histtype='step',
                       color='b', alpha=0.5, orientation=orientation,
                       label="x1")
    n2,b,p2 = ax4.hist(x2, bins=bins,
                       histtype='stepfilled',
                       color='r', alpha=0.5, orientation=orientation,
                       label="x2")
    bihist(ax4, p1, p2, orientation=orientation)
    ax4.legend()

    P.show()
