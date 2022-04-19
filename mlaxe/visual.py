"""
This module provide visualization tools
which displays process of fitting.

"""

from msilib.schema import Error
from turtle import width
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

        xb, xe = ax.get_xlim()
        yb, ye = ax.get_ylim()

        dx = (xe - xb) * (ratio - 1) / 2
        dy = (ye - yb) * (ratio - 1) / 2

        ax.set_xlim(xb - dx, xe + dx)
        ax.set_ylim(yb - dy, ye + dy)


class Animation2D(SampleDisplayMixin):
    """
    This class implements the animation rendering
    of process of fitting for 2D sample.

    Attributes
    ----------
    sample: tuple of (2D, 1D) numpy-like arrays
        Sample data.

    weights: 2D numpy-like array
        The history of weights saved while fitting.

    grads: 2D numpy-like array
        The history of gradients saved while fitting.

    losses: 1D numpy-like array
        The history of losses saved while fitting.

    risks: 1D numpy-like array
        The history of emp risks saved while fitting.

    save_gif: bool (default: False)
        Whether to save animation as .gif file.

    verbose: bool (default: False)
        Whether to display process of animating or not.

    """

    def __init__(self, sample, weights, grads, losses,
                 risks, save_gif=False, verbose=False):
        """
        Initializes attributes of the class with
        corresponding values (if specified) from arguments.

        """

        self.xs, self.ys = sample
        self.weights = weights
        self.grads = grads
        self.risks = risks
        self.losses = losses
        self.verbose = verbose
        self.save_gif = save_gif

        # setup plotting parameters
        self.congif()


    def congif(self):
        """ Setting up all necessary objects for plotting """

        rc('animation', html='html5')

        # setting constants
        self.WEIGHT_PRECISION = 3
        self.LOSS_PRECISION = 5

        # creating subplots and grid
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            1, 2, figsize=(11, 4),
            gridspec_kw={'width_ratios': [2, 1]}
        )

        # calculating limits for hyperplane grid
        xl, xr = np.min(self.xs[:, 0]), np.max(self.xs[:, 0])
        self.hyp_grid = np.linspace(xl - 1, xr + 1, 5)
        self.risk_grid = np.arange(1, len(self.risks) + 1)

        # setting limits
        self.ax2.set_xlim(1, len(self.risks) + 1)
        self.ax2.set_ylim(-0.2, max(self.risks) + 0.1)

        # saving frames count and initializing iterations count
        self.n_frames = len(self.weights)
        self.iter = 0


    def pre_anim(self):
        """ The method which is called before animating. """

        # plotting lower bound of loss
        x_lower = np.linspace(*self.ax2.get_xlim(), 5)
        self.ax2.plot(x_lower, np.zeros(x_lower.shape[0]), lw=0.5)

        # creating objects for hyperplane, risk_graph and text
        self.hyper_plane = self.ax1.plot([], [], lw=2, color='black')[0]
        self.hyper_plane.set_xdata(self.hyp_grid)

        bbox_style = dict(facecolor='black', fill=False)

        self.grad_text = self.ax1.text(
            0.1, 0.01, '', verticalalignment='bottom',
            horizontalalignment='center', transform=self.ax1.transAxes,
            color='brown', fontsize=9, bbox=bbox_style
        )

        self.weig_text = self.ax1.text(
            0.9, 0.01, '', verticalalignment='bottom',
            horizontalalignment='center', transform=self.ax1.transAxes,
            color='brown', fontsize=9, bbox=bbox_style
        )

        self.risk_text = self.ax2.text(
            0.85, 0.9, '', verticalalignment='center',
            horizontalalignment='center', transform=self.ax2.transAxes,
            color='brown', fontsize=9, bbox=bbox_style
        )

        self.risk_graph = self.ax2.plot([], [], lw=1, color='black')[0]

        return (self.hyper_plane, self.grad_text, self.weig_text,
                self.risk_text, self.risk_graph)


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
        loss = self.losses[i]
        risk = self.risks[i]

        # getting the abscissas grid
        x = self.hyp_grid

        # calculating the ordinate of decision function
        y = (w[2] + w[0] * x) / -w[1]

        # printing and updating status of animation
        self.iter += 1

        if self.iter <= self.n_frames:
            status = f'Animating {self.iter / self.n_frames * 100:.2f}%'
            print(status + '\r', end='')

        # formatting and displaying text
        w_prec = self.WEIGHT_PRECISION
        l_prec = self.LOSS_PRECISION

        def formatter(s: str, p=2):
            return '\n'.join(map(lambda x: f'{x:.{p}f}', s))

        info_weig = ' weights: \n' + formatter(w, w_prec)
        info_grad = '  grads:  \n' + formatter(g, w_prec)
        info_risk = '   loss:   \n' + formatter([loss], l_prec)
        info_risk += '\n   risk:   \n' + formatter([risk], l_prec)

        self.weig_text.set_text(info_weig)
        self.grad_text.set_text(info_grad)
        self.risk_text.set_text(info_risk)
        self.hyper_plane.set_ydata(y)

        # updating risk graph
        self.risk_graph.set_xdata(self.risk_grid[:i + 1])
        self.risk_graph.set_ydata(self.risks[:i + 1])

        return (self.weig_text, self.grad_text, self.risk_text,
                self.hyper_plane, self.risk_graph)


    def build_animation(self):
        """
        This method launches the animating process.

        """

        # creating the instance of FuncAnimation
        # using optimal parameters
        anim = animation.FuncAnimation(
            self.fig, self.animate,
            interval=20, frames=self.n_frames,
            repeat=False, blit=True, init_func=self.pre_anim
        )

        # plot sample
        self.draw_sample(self.xs, self.ys, self.ax1)

        # set titles
        self.ax1.set_title('Process of fitting')
        self.ax1.legend(loc='upper right')

        self.ax2.set_title('Empirical risk graph')

        # zooming limits of plot
        self.zoom_axes(self.ax1, 1.2)

        plt.show()
        print()

        if self.save_gif:
            writergif = animation.PillowWriter(fps=30)
            anim.save('out.gif', writer=writergif)
