"""
This module provide visualization tools
which displays process of fitting.

"""

from itertools import cycle


class SampleDisplayMixin:
    """
    Mixin class, which adds
    sample visualization functionality.

    """

    def draw_sample(self, xs, ys, ax):
        """
        Visualizes the sample on 2D-plane.

        Parameters
        ----------
        xs: feature data (2D numpy-like array)

        ys: label data (2D numpy-like array)

        ax: axes to plot on (axes-object matplotlib)

        """

        # initialize color array
        colors = list('bgrcmyk') + ['purple', 'lime', 'olive', 'pink',
                                    'coral', 'indigo', 'gold', 'chocolate']

        # cycle array of colors
        cycler = cycle(colors)
        colors = [next(cycler) for _ in range(self.classes)]

        # set figure size for plotting
        ax.figure(figsize=(7, 5))

        xs = xs.reshape(self.classes, -1, 2)
        ys = ys.reshape(self.classes, -1)

        # plot each class, pointing label and color values
        for x, y, clr in zip(xs, ys, colors):
            ax.scatter(x[:, 0], x[:, 1], label=f'class {y[0]}', color=clr)

        ax.legend()
        ax.show()