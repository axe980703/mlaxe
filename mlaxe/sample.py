"""
This module provide tools for creating
random sample.
You can specify parameters that suit
your task.

"""

import numpy as np
import matplotlib.pyplot as plt
from mlaxe.visual import SampleDisplayMixin


class Sample2D(SampleDisplayMixin):
    """
    The Class creates a sintetic sample of normal distrubuted
    double-dimensional random variable.
    The parameters of normal distribution(mean and stdev) can
    be manually specified.
    The centers of classes are equdistant from coordinate plane
    origin, and they lie on the circumference with radius,
    which specified in parameters.

    Attributes
    ----------
    classes: int (default: 2)
        Number of classes to create.

    radius: float (deafult: 5)
        The radius of circumference on which the centers of classes lie.

    mean: float (default: 0)
        Mean parameter of normal distributed variable.

    stdev: float (default: 2)
        Standard deviation parameter of normal distributed variable.

    seed: int (default: 322)
        Seed value for random value generator.

    cl_size: int (default: 100)
        The size of the class(number of objects per class).

    show: bool (default: True)
        Whether to show generated sample right after creating.

    """

    def __init__(self, classes=2, radius=5, mean=0,
                 stdev=2, seed=322, cl_size=100, show=True,
                 standard=True):
        """
        Initializes attributes of the class with
        corresponding values (if specified) from arguments.

        """

        self.classes = classes
        self.radius = radius
        self.mean = mean
        self.stdev = stdev
        self.seed = seed
        self.cl_size = cl_size
        self.show = show
        self.standard = standard


    def gen(self):
        """
        Generates random 2-dimensional sample with
        parameters from class attributes.

        Returns
        ----------
        xs, ys: tuple of (2D, 1D) numpy-like arrays
            Feature and label data.

        """

        # create random generator with corresponding seed.
        rnd = np.random.RandomState(self.seed)

        # create uniform grid of angles in a circumference.
        phi = np.linspace(0, 2 * np.pi, self.classes + 1)[:-1]

        # get coordinates of points on a plane, that have
        # such angles as values in 'phi' array (in polar
        # coordinte system) and have radius equal to 'self.radius'.
        y = self.radius * np.sin(phi)
        x = self.radius * np.cos(phi)

        # found points will be used as centers of each class.
        centers = np.stack((x, y), axis=1)

        # the class will be formed as a set of points, where
        # each point is the center of class added with some noise.
        noise = rnd.standard_normal((self.classes, self.cl_size, 2))

        centers = np.repeat(centers, self.cl_size, axis=0)
        centers = centers.reshape(-1, self.cl_size, 2)

        # form sample and then add dispersion and bias to it.
        xs = (centers + noise * self.stdev + self.mean).reshape(-1, 2)

        if self.standard:
            xs = self.standardize(xs)

        # create labels for classes
        ys = np.repeat(np.arange(self.classes), self.cl_size)

        # visualize created sample
        if self.show:
            plt.figure(figsize=(7, 5))
            self.draw_sample(xs, ys, plt)
            plt.title(f'Generated sample for {self.classes} classes')
            plt.legend()
            plt.show()

        return xs, ys


    @staticmethod
    def standardize(a):
        means = a.mean(axis=0)
        stds = a.std(axis=0)
        return (a - means) / stds
