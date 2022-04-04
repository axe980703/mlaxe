import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc


class SGDLinearClassifier:
    """
    The Class implements binary classification with a linear decision function.
    It provides set of loss functions to choose.
    Weights by default initialized with standard gaussian random values.
    Learning rate by default decreases.
    Description is expanding..

    Parameters
    ----------
    lr_init : float (default: 0.006)
        Initial learning rate of sgd.

    upd_rate : float (deafult: 0.1)
        Update rate of empirical risk on each iteration (0 <= val <= 1).

    max_iter : int (default: 10000)
        Upper bound of maximum number of iterations.

    loss_eps : float (default: 1e-7)
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
                 mov_avg='exp', loss_func='relu'):

        self._lr_init = lr_init
        self._upd_rate = upd_rate
        self._max_iter = max_iter
        self._loss_eps = loss_eps
        self._tol_iter = tol_iter
        self._add_bias = add_bias
        self._save_hist = save_hist
        self._verbose = verbose
        self._shuffle = shuffle
        self._weights_hist = []
        self._grad_hist = []

        # select functions, corresponding to received parameters

        self._update_risk = self._MovAvg(mov_avg).get_update_method()
        self._loss, self._grad = self._Loss(loss_func).get_loss_grad_methods()


    def fit(self, x, y):
        """
        Parameters
        ----------
        x: feature_data (2D numpy-like array)

        y: label_data (1D numpy-like array)

        """

        self._n_objects = x.shape[0]
        self._n_features = x.shape[1]
        self._converge_streak = 0

        if self._add_bias:
            x = np.hstack([x, np.ones((self._n_objects, 1))])

        # init weights with gaussian standard distribution
        w = np.random.randn(self._n_features + 1)


        # init local variables
        iter_step = 0
        min_risk = cur_risk = self._empirical_risk(x, y, w)
        l_rate = self._lr_init


        while iter_step < self._max_iter:

            if self._shuffle:
                shuffle_perm = np.random.permutation(self._n_objects)
                x, y = x[shuffle_perm], y[shuffle_perm]

            # picking random index for current object
            i = np.random.randint(0, self._n_objects)

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


    def predict(self, x):
        """
        Parameters
        ----------
        x: feature_data (2D numpy-like array)

        Returns
        ----------
        y: predicted labels of classes (1D numpy-like array)

        """

        # check if fit was called before, raise error otherwise

        if self._add_bias:
            x = np.hstack([x, np.ones((self._n_objects, 1))])

        return np.sign(np.sum(x * self.weights, axis=1))



    def evaluate(self, x, y):
        """
        Parameters
        ----------
        x: feature_data (2D numpy-like array)

        y: label_data (1D numpy-like array)

        Returns
        ----------
        acc: accuracy of number of correct predicted classes

        """

        # check if fit was called before, raise error otherwise

        y_pred = self.predict(x)
        acc = np.sum(y_pred == y) / y.shape[0]

        return acc


    def _empirical_risk(self, x, y, w):
        ws, y = np.expand_dims(w, axis=0), np.expand_dims(y, axis=-1)
        ws = np.repeat(ws, self._n_objects, axis=0)
        loss = np.vectorize(self._loss)

        return np.sum(loss(x, y, ws))


    def _time_to_stop(self, cur_risk, min_risk):
        if abs(cur_risk - min_risk) < self._loss_eps:
            self._converge_streak += 1
        else:
            self._converge_streak = 0

        if self._converge_streak == self._tol_iter:
            return True
        return False


    @staticmethod
    def _update_l_rate(l_rate, iter_step):
        next_lr = 1 / (1 + iter_step)

        return min(l_rate, next_lr)



    class _Loss:
        def __init__(self, loss_name: str):
            self.func = getattr(self, f'func_{loss_name}')
            self.grad = getattr(self, f'grad_{loss_name}')


        def get_loss_grad_methods(self):
            return self.func, self.grad


        @staticmethod
        def func_relu(x, y, w):
            margin = np.dot(x, w) * y

            if margin >= 0:
                return 0
            return -margin


        @staticmethod
        def grad_relu(x, y, w):
            n_features = x.shape[0]
            margin = np.dot(x, w) * y

            if margin >= 0:
                return np.zeros(n_features)
            return x * (-y)


        @staticmethod
        def func_log(x, y, w):
            pass


        @staticmethod
        def grad_log(x, y, w):
            pass



    class _MovAvg:
        def __init__(self, mavg_name: str):
            self.update_risk = getattr(self, f'update_{mavg_name}')


        def get_update_method(self):
            return self.update_risk


        @staticmethod
        def update_exp(risk, loss, upd_rate):
            return upd_rate * loss + (1 - upd_rate) * risk


        @staticmethod
        def update_mean(x, y, w):
            pass
