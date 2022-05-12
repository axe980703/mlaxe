"""
This module provide visualization tools
which displays process of fitting.

"""

import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mlaxe.standards import Config


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

        # sort by class labels
        order = np.argsort(ys)
        xs, ys = xs[order], ys[order]

        # initialize color array
        colors = Config().colors

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
                 risks, class_num, save_gif=False, verbose=False):
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
        self.class_num = class_num

        # setup plotting parameters
        self.congif()


    def congif(self):
        """ Setting up all necessary staff for plotting """

        rc('animation', html='html5')

        # setting constants
        self.WEIGHT_PRECISION = 3
        self.LOSS_PRECISION = 5

        self.colors = Config().colors

        # creating subplots and grid
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            1, 2, figsize=(11, 4),
            gridspec_kw={'width_ratios': [2, 1]}
        )

        # calculating limits for hyperplane grid
        xl, xr = np.min(self.xs[:, 0]), np.max(self.xs[:, 0])
        self.hyp_grid = np.linspace(xl - 1, xr + 1, 3)

        # saving frames count and initializing iterations count
        self.iter_class = 0
        self.iter_count = [len(w) for w in reversed(self.weights)]
        self.n_frames = sum(self.iter_count)
        self.cur_class = 0


    def pre_anim(self):
        """ The method which is called before animating. """

        # plotting and tracking hyperplane objects
        self.hyper_plane = []

        for n_class in range(self.class_num):
            # use black color for plane if it's binary classification
            color = self.colors[n_class] if self.class_num > 2 else 'k'
            plane = self.ax1.plot([], [], lw=2, color=color)[0]
            self.hyper_plane.append(plane)

        # plotting and tracking text and risk graph objects
        bbox_style = dict(facecolor='white', edgecolor='brown')

        text_style = dict(
            verticalalignment='bottom', horizontalalignment='center',
            color='brown', fontsize=7, bbox=bbox_style
        )

        self.grad_text = self.ax1.text(
            0.1, 0.01, '', transform=self.ax1.transAxes, **text_style
        )

        self.weig_text = self.ax1.text(
            0.9, 0.01, '', transform=self.ax1.transAxes, **text_style
        )

        self.risk_text = self.ax2.text(
            0.85, 0.83, '', transform=self.ax2.transAxes, **text_style
        )

        self.risk_graph = self.ax2.plot([], [], lw=1, color='black')[0]

        return (self.weig_text, self.grad_text, self.risk_text,
                *self.hyper_plane, self.risk_graph)


    def animate(self, iter):
        """
        The method which is called to render each frame
        of animation.

        Parameters
        ----------
        iter: int
            Iteration (frame) number.

        """

        # dealing with class transitions
        if self.iter_class == 0:

            # setting risk grid
            risks = self.risks[self.cur_class]
            self.risk_grid = np.arange(1, len(risks) + 1)

            # setting limits
            self.ax2.set_xlim(1, len(risks) + 1)
            self.ax2.set_ylim(-0.2, max(risks) + 0.1)

            # redrawing canvas to update ticks
            self.fig.canvas.draw_idle()

            # setting grid
            self.hyper_plane[self.cur_class].set_xdata(self.hyp_grid)

        # getting simple links to variables
        cls = self.cur_class
        i = self.iter_class

        w = self.weights[cls][i]
        g = self.grads[cls][i]
        loss = self.losses[cls][i]
        risk = self.risks[cls][i]

        # getting the abscissas grid
        x = self.hyp_grid

        # calculating the ordinates of decision function
        y = (w[2] + w[0] * x) / -w[1]

        # showing status of animation
        status = f'Animating {(iter + 1) / self.n_frames * 100:.2f}%'
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
        self.hyper_plane[cls].set_ydata(y)

        # updating risk graph
        self.risk_graph.set_xdata(self.risk_grid[:i + 1])
        self.risk_graph.set_ydata(self.risks[cls][:i + 1])

        # updating iteration count
        self.iter_class += 1
        if self.iter_class == self.iter_count[-1]:
            self.iter_count.pop()
            self.cur_class += 1
            self.iter_class = 0

        return (*self.hyper_plane, self.risk_graph,
                self.weig_text, self.grad_text, self.risk_text)


    def build_animation(self):
        """
        This method launches the animating process.

        """

        # plot sample
        self.draw_sample(self.xs, self.ys, self.ax1)

        # set titles
        self.ax1.set_title('Process of fitting')
        self.ax1.legend(prop={'size': 7})
        self.ax2.set_title('Empirical risk graph')

        # creating the animation
        anim = animation.FuncAnimation(
            self.fig, self.animate, init_func=self.pre_anim,
            interval=0, frames=self.n_frames,
            repeat=False, blit=True
        )

        # zooming limits of plot
        self.zoom_axes(self.ax1, 1.2)

        plt.show()
        print()

        if self.save_gif:
            writergif = animation.PillowWriter(fps=30)
            anim.save('out.gif', writer=writergif)
