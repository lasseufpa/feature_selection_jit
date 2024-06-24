def bihist(ax, upatches, lpatches, orientation='vertical'):
    if orientation.startswith('v'):  # Vertical orientation
        for p in lpatches:
            try:
                p._height *= -1  # matplotlib.patches.Rectangle
            except AttributeError:
                p._path.vertices[:, 1] *= -1  # matplotlib.patches.Polygon
    elif orientation.startswith('h'):  # Horizontal orientation
        for p in upatches:
            try:
                p._width *= -1  # matplotlib.patches.Rectangle
            except AttributeError:
                p._path.vertices[:, 0] *= -1  # matplotlib.patches.Polygon
    else:
        raise ValueError("Unknown orientation '%s'" % orientation)

    ax.relim()
    ax.autoscale_view()



if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv('src/features.csv')
    x1 = df[df['label'] == 1].drop(columns=['label', 'commit'])
    x2 = df[df['label'] == 0].drop(columns=['label', 'commit'])

    for column in x1.columns:
        orientation = 'vertical'
        fig = plt.figure()
        
        min_val = min(min(x1[column]), min(x2[column]))
        max_val = max(max(x1[column]), max(x2[column]))
        
        bins = np.linspace(min_val, max_val, 30)
        
        ax2 = fig.add_subplot(1, 1, 1, title=column + " Bi-histogram")
        n1, b, p1 = ax2.hist(x2[column], bins=bins,
                             histtype='bar',
                             color='b', alpha=0.5, orientation=orientation,
                             label="Not Bug",
                             density=True)
        n2, b, p2 = ax2.hist(x1[column], bins=bins,
                             histtype='bar',
                             color='r', alpha=0.5, orientation=orientation,
                             label="Bug",
                             density=True)
        bihist(ax2, p1, p2, orientation=orientation)
        ax2.legend()
        plt.savefig(f"{column}_bihistogram.png", format="png", bbox_inches="tight")
        plt.close(fig)
        # plt.show()
        # break
        # Save the figure
