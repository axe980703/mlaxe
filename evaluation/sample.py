import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


class Sample2D:
    """
    The Class creates a sintetic sample of normal distrubuted
    double-dimensional random variable.
    The parameters of normal distribution(mean and stdev) can
    be manually specified.
    The centers of classes are equdistant from coordinate plane
    origin, and they lie on the circumference with radius,
    which specified in parameters.

    Parameters
    ----------
    classes: int (default: 2)
        Number of classes to create.

    radius: float (deafult:5)
        The radius of circumference on which the centers of classes lie.

    mean: float (default: 0)
        Mean parameter of normal distributed variable.

    stdev: float (default: 2)
        Standard deviation parameter of normal distributed variable.

    seed: int (default: 322)
        Seed value for random value generator.

    cl_size: int (default: 100)
        The size of the class(number of objects per class).

    """

    def __init__(self, classes=2, radius=5, mean=0,
                 stdev=2, seed=322, cl_size=100):

        self.classes = classes
        self.radius = radius
        self.mean = mean
        self.stdev = stdev
        self.seed = seed
        self.cl_size = cl_size


    def gen(self):
        rnd = np.random.RandomState(self.seed)

        phi = np.linspace(0, 2 * np.pi, self.classes + 1)[:-1]
        y = self.radius * np.sin(phi)
        x = self.radius * np.cos(phi)
        centers = np.stack((x, y), axis=1)

        noise = rnd.standard_normal((self.classes, self.cl_size, 2))

        centers = np.repeat(centers, self.cl_size, axis=0)
        centers = centers.reshape(-1, self.cl_size, 2)
        xs = (centers + noise * self.stdev + self.mean).reshape(-1, 2)

        if self.classes == 2:
            ys = np.array([-1] * self.cl_size + [1] * self.cl_size)
        else:
            ys = np.repeat(np.arange(self.classes), self.cl_size)

        self.draw_sample(xs, ys, plt)

        return xs, ys


    def draw_sample(self, xs, ys, ax):
        colors = list('bgrcmyk') + ['purple', 'lime', 'olive', 'pink',
                                    'coral', 'indigo', 'gold', 'chocolate']
        cycler = cycle(colors)
        colors = [next(cycler) for _ in range(self.classes)]
        ax.figure(figsize=(7, 5))

        xs = xs.reshape(self.classes, -1, 2)
        ys = ys.reshape(self.classes, -1)

        for x, y, clr in zip(xs, ys, colors):
            ax.scatter(x[:, 0], x[:, 1], label=f'class {y[0]}', color=clr)

        ax.legend()
        ax.show()
