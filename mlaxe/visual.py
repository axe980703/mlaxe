"""
This module provide visualization tools
which displays process of fitting.

"""

import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib import animation, rc


class SampleDisplayMixin:
    """
    Mixin class, which adds
    sample visualization functionality.

    """

    @staticmethod
    def draw_sample(xs, ys, ax):
        """
        Visualizes the sample on 2D-plane.
        The input sample must have equal
        number of objects in each class.

        Parameters
        ----------
        xs: feature data (2D numpy-like array)

        ys: label data (1D numpy-like array)

        ax: axes to plot on (axes-object matplotlib)

        """

        # determine number of classes
        n_classes = np.unique(ys).shape[0]

        # initialize color array
        colors = list('bgrcmyk') + ['purple', 'lime', 'olive', 'pink',
                                    'coral', 'indigo', 'gold', 'chocolate']

        # cycle array of colors
        cycler = cycle(colors)
        colors = [next(cycler) for _ in range(n_classes)]

        xs = xs.reshape(n_classes, -1, 2)
        ys = ys.reshape(n_classes, -1)

        # plot each class, pointing label and color values
        for x, y, clr in zip(xs, ys, colors):
            ax.scatter(x[:, 0], x[:, 1], label=f'class {y[0]}', color=clr)


    @staticmethod
    def zoom_axes(ax, ratio):
        xe, xb = ax.get_xlim()
        ye, yb = ax.get_ylim()

        dx = (xe - xb) * (ratio - 1) / 2
        dy = (ye - yb) * (ratio - 1) / 2

        ax.set_xlim(xb - dx, xe + dx)
        ax.set_ylim(yb - dy, ye + dy)


class Animation(SampleDisplayMixin):
    """
    This class implements the animation rendering
    of process of fitting.

    Parameters
    ----------
    classes: int (default: 2)
        Number of classes to create.

    radius: float (deafult: 5)
        The radius of circumference on which the centers of classes lie.

    mean: float (default: 0)
        Mean parameter of normal distributed variable.

    fig_size: 2-elements tuple (default: (9, 6))
        Size of window for plotting.

    save_gif: bool (default: False)
        Whether to save animation as .gif file.

    verbose: bool (default: False)
        Whether to display process of animating or not.

    """

    def __init__(self, sample, weights, grads,
                 verbose=False, save_gif=False):
        """
        Initializes attributes of the class with
        corresponding values (if specified) from arguments.

        """

        self.xs, self.ys = sample
        self.weights = weights
        self.grads = grads

        self.verbose = verbose
        self.save_gif = save_gif
        self.congif()


    def congif(self):
        rc('animation', html='html5')
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.x_grid = np.linspace(-10, 10, 5)

        self.line = self.ax.plot([], [], lw=2, color='red')[0]
        self.text = self.ax.text(
            0.95, 0.05, '', verticalalignment='bottom',
            horizontalalignment='right', transform=self.ax.transAxes,
            color='brown', fontsize=12
        )

        self.n_frames = len(self.weights)
        self.iter = 0


    def pre_anim(self):
        self.draw_sample(self.xs, self.ys, self.ax)
        self.line.set_xdata(self.x_grid)

        return (self.line,)


    def animate(self, i: int):
        w = self.weights[i]
        g = self.grads[i]
        xs = self.x_grid
        ys = (w[2] + w[0] * xs) / -w[1]

        self.iter += 1

        if self.iter <= self.n_frames:
            status = f'Animating {self.iter / self.n_frames * 100:.2f}%'
            print(status + '\r', end='')

        info = (f'weights: {w[0]:.3f}  {w[1]:.3f}  {w[2]:.3f}\n'
                f'gradient: {g[0]:.3f}  {g[1]:.3f}  {g[2]:.3f}')
        self.text.set_text(info)

        self.line.set_ydata(ys)

        return (self.line, self.text)


    def build_animation(self):
        anim = animation.FuncAnimation(
            self.fig, self.animate, init_func=self.pre_anim,
            interval=7, blit=True, frames=self.n_frames,
            repeat=False
        )

        self.draw_sample(self.xs, self.ys, self.ax)
        self.ax.set_title(f'Process of fitting')
        self.ax.legend(loc='upper right')
        self.zoom_axes(self.ax, 1.2)
        plt.show()

        if self.save_gif:
            writergif = animation.PillowWriter(fps=30)
            anim.save('out.gif', writer=writergif)
