import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from mlaxe.math import _Loss, _MovAvg


class SGDLinearClassifier:
    """
    The Class implements binary classification with a linear decision function.
    It provides set of loss functions to choose.
    Weights by default initialized with standard gaussian random values.
    Learning rate by default decreases.
    Description is expanding..

    Parameters
    ----------
    lr_init: float (default: 0.006)
        Initial learning rate of sgd.

    upd_rate: float (deafult: 0.1)
        Update rate of empirical risk on each iteration (0 <= val <= 1).

    max_iter: int (default: 10000)
        Upper bound of maximum number of iterations.

    loss_eps: float (default: 1e-7)
        Stop criterion. Fitting will be stopped when
        condition: (loss - min_loss <= loss_eps) is true
        for (tol_iter) consecutive iterations.

    tol_iter: int (default: 5)
        Number of iterations to wait before fitting will be stopped.

    add_bias: bool (default: True)
        Whether to add bias coefficient to weights vector or not.

    save_hist: bool (default: True)
        Whether to log all weights and gradients states during fitting or not.

    verbose: bool (default: False)
        Whether to print out the process of fitting or not.

    shuffle: bool (default: False)
        Whether to shuffle feature data after each iteration or not.

    mov_avg: str (default: 'exp')
        Type of moving average to use ('exp' - exponential,
        'mean'- mean) for emperical risk re-calculation.

    loss_func: str (default: 'relu')
        Target loss function, for which we minimize empirical risk.
        If parameter set to 'relu', f(x) = max(0, -x) will be used.
        If parameter set to 'log', f(x) = log2(1 + e^(-x)) will be used.

    """

    def __init__(self, lr_init=0.006, upd_rate=0.1, max_iter=10000,
                 loss_eps=1e-5, tol_iter=5, add_bias=True,
                 save_hist=True, verbose=False, shuffle=False,
                 mov_avg='exp', loss_func='relu', seed=322):

        self._lr_init = lr_init
        self._upd_rate = upd_rate
        self._max_iter = max_iter
        self._loss_eps = loss_eps
        self._tol_iter = tol_iter
        self._add_bias = add_bias
        self._save_hist = save_hist
        self._verbose = verbose
        self._shuffle = shuffle
        self._seed = seed
        self._weights_hist = []
        self._grad_hist = []

        # select functions, corresponding to received parameters
        self._update_risk = _MovAvg(mov_avg).get_update_method()
        self._loss, self._grad = _Loss(loss_func).get_loss_grad_methods()


    def fit(self, x, y):
        """
        Fits linear classifier using Stochastic Gradient Descent.
        Empirical risk is updating its value by moving average.
        The partial derivatives for each loss function already
        specified in _Loss class and stored as _grad attribute.

        Parameters
        ----------
        x: feature data (2D numpy-like array)

        y: label data (1D numpy-like array)

        Returns
        ----------
        self: the current instance of class

        """

        self._n_objects = x.shape[0]
        self._n_features = x.shape[1]
        self._converge_streak = 0
        self._rand_gen = np.random.RandomState(self._seed)

        if self._add_bias:
            x = np.hstack([x, np.ones((self._n_objects, 1))])

        # init shuffle
        x, y = self._shuffle_objects(x, y)

        # init weights with gaussian standard distribution
        w = self._rand_gen.randn(self._n_features + 1)


        # init local variables
        iter_step = 0
        min_risk = cur_risk = self._empirical_risk(x, y, w)
        l_rate = self._lr_init


        while iter_step < self._max_iter:

            if self._shuffle:
                x, y = self._shuffle_objects(x, y)

            # picking random index for current object
            i = self._rand_gen.randint(0, self._n_objects)

            # calulating loss on current object
            iter_loss = self._loss(x[i], y[i], w)

            # calculating gradient
            grad = self._grad(x[i], y[i], w)

            # changing weights according to the gradient value
            w = w - l_rate * grad

            # updating empirical risk
            cur_risk = self._update_risk(cur_risk, iter_loss, self._upd_rate)

            if self._verbose:
                print(f'loss: {cur_risk}')

            # logging fitting's data
            if self._save_hist:
                self._weights_hist.append(w)
                self._grad_hist.append(grad)

            # processing stop criterion
            if self._time_to_stop(cur_risk, min_risk):
                break

            # changing learning rate
#             l_rate = self._update_l_rate(l_rate, iter_step)

            # updating iteration's variables
            min_risk = min(min_risk, cur_risk)
            iter_step += 1

        # saving final weights as an attribute
        self.weights = w
        self.iters = iter_step

        return self


    def predict(self, x):
        """
        Predicts class labels for provided feature data.
        If _add_bias flag is specified, column of ones will be added.

        Parameters
        ----------
        x: feature data (2D numpy-like array)

        Returns
        ----------
        y: predicted labels of classes (1D numpy-like array)

        """

        # check if fit was called before, raise error otherwise

        if self._add_bias:
            x = np.hstack([x, np.ones((self._n_objects, 1))])

        y = np.sign(np.sum(x * self.weights, axis=1))

        return y


    def evaluate(self, x, y):
        """
        Calculates accuracy (the ratio of the number of objects
        with correct predicted classes to the total number of objects).

        Parameters
        ----------
        x: feature data (2D numpy-like array)

        y: label data (1D numpy-like array)

        Returns
        ----------
        acc: accuracy of number of correct predicted classes (float)

        """

        # check if fit was called before, raise error otherwise

        y_pred = self.predict(x)
        acc = np.sum(y_pred == y) / y.shape[0]

        return acc


    def _empirical_risk(self, x, y, w):
        """
        Calculates the value of empirical risk for
        specified sample data.

        Parameters
        ----------
        x: feature data (2D numpy-like array)

        y: label data (1D numpy-like array)

        w: weights (coefficients) of linear decision
           function (1D numpy-like array)

        Returns
        ----------
        risk: value of empirical risk (float)

        """

        ws, y = np.expand_dims(w, axis=0), np.expand_dims(y, axis=-1)
        ws = np.repeat(ws, self._n_objects, axis=0)
        loss = np.vectorize(self._loss)
        risk = np.sum(loss(x, y, ws)) / self._n_objects

        return risk


    def _time_to_stop(self, cur_risk, min_risk):
        """
        Checks if the convergence condition is already satisfied or not.
        Also, it updates convergence-control variables.

        Parameters
        ----------
        cur_risk: the risk value on the current iteration (float)

        min_risk: the minimal risk value during fitting process (float)

        Returns
        ----------
        to_stop: stop indicator (bool)

        """

        if abs(cur_risk - min_risk) < self._loss_eps:
            self._converge_streak += 1
        else:
            self._converge_streak = 0

        to_stop = (self._converge_streak == self._tol_iter)

        return to_stop


    def _shuffle_objects(self, x, y):
        """
        Shuffles fitting data randomly.

        Parameters
        ----------
        x: feature data (2D numpy-like array)

        y: label data (1D numpy-like array)

        Returns
        ----------
        x, y: shuffled sample data (tuple)

        """

        shuffle_perm = self._rand_gen.permutation(self._n_objects)
        x, y = x[shuffle_perm], y[shuffle_perm]

        return x, y


    @staticmethod
    def _update_l_rate(l_rate, iter_step):
        """
        Updates learning rate according to the rules
        specified at the parameters.

        Parameters
        ----------
        l_rate: current learning rate (float)

        iter_step: number of iteration (int)

        Returns
        ----------
        next_lr: updated learning rate (float)

        """

        next_lr = min(l_rate, 1 / (1 + iter_step))

        return next_lr
