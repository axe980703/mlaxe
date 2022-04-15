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
        xs: 2D numpy-like array
            Feature data.

        ys: 1D numpy-like array
            Dabel data.

        ax: Axes
            Axes to plot on.

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
        """
        Zooms in/out limits of axes, according
        to ratio value.

        Parameters
        ----------
        ax: Axes
            Axes to plot on.

        ratio: float
            The coefficient of zoom.
            if ratio >= 1: the greater, the closer.
            if ratio < 1: the smaller, the farther.

        """

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

    Attributes
    ----------
    sample: tuple of (2D, 1D) numpy-like arrays
        Sample data.

    weights: 2D numpy-like array
        The history of weights saved while fitting.

    grads: 2D numpy-like array
        The history of gradients saved while fitting.

    save_gif: bool (default: False)
        Whether to save animation as .gif file.

    verbose: bool (default: False)
        Whether to display process of animating or not.

    """

    def __init__(self, sample, weights, grads,
                 save_gif=False, verbose=False):
        """
        Initializes attributes of the class with
        corresponding values (if specified) from arguments.

        """

        self.xs, self.ys = sample
        self.weights = weights
        self.grads = grads
        self.verbose = verbose
        self.save_gif = save_gif

        # setup plotting parameters
        self.congif()


    def congif(self):
        """ Setting up all necessary objects for plotting """

        rc('animation', html='html5')

        # creating subplots and grid
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.x_grid = np.linspace(-10, 10, 5)

        # creating objects for line and text
        self.line = self.ax.plot([], [], lw=2, color='red')[0]
        self.text = self.ax.text(
            0.95, 0.05, '', verticalalignment='bottom',
            horizontalalignment='right', transform=self.ax.transAxes,
            color='brown', fontsize=12
        )

        # saving frames count and initializing iterations count
        self.n_frames = len(self.weights)
        self.iter = 0


    def pre_anim(self):
        """ The method which is called before animating. """

        self.draw_sample(self.xs, self.ys, self.ax)
        self.line.set_xdata(self.x_grid)

        return (self.line,)


    def animate(self, i: int):
        """
        The method which is called to render each frame
        of animation.

        Parameters
        ----------
        i: int
            Iteration (frame) number.

        """

        # getting simple links to variables
        w = self.weights[i]
        g = self.grads[i]

        # getting the abscissas grid
        x = self.x_grid

        # calculating the ordinate of decision function
        y = (w[2] + w[0] * x) / -w[1]

        # printing and updating status of animation
        self.iter += 1

        if self.iter <= self.n_frames:
            status = f'Animating {self.iter / self.n_frames * 100:.2f}%'
            print(status + '\r', end='')

        # displaying of weights/grads changes through fit iterations
        info = (f'weights: {w[0]:.3f}  {w[1]:.3f}  {w[2]:.3f}\n'
                f'gradient: {g[0]:.3f}  {g[1]:.3f}  {g[2]:.3f}')
        self.text.set_text(info)

        self.line.set_ydata(y)

        return (self.line, self.text)


    def build_animation(self):
        """
        The method that launches the animating process.

        """

        # creating the instance of FuncAnimation
        # using optimal parameters
        anim = animation.FuncAnimation(
            self.fig, self.animate, init_func=self.pre_anim,
            interval=7, blit=True, frames=self.n_frames,
            repeat=False
        )

        # plotting sample
        self.draw_sample(self.xs, self.ys, self.ax)
        self.ax.set_title(f'Process of fitting')
        self.ax.legend(loc='upper right')

        # zooming limits of plot
        self.zoom_axes(self.ax, 1.2)
        plt.show()

        # save .gif file of animation
        # if corresponding parameter is true.
        if self.save_gif:
            writergif = animation.PillowWriter(fps=30)
            anim.save('out.gif', writer=writergif)
