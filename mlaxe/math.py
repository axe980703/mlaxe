"""
This module provides tools for
various calculation tasks.
It also allows you, to select necessary
functions by specifying their names.

"""

import numpy as np


class _Loss:
    """
    This class stores implementations of several loss
    functions, and their gradients. By specifying name
    of the loss function while creating instance of
    the class, you will be able to get corresponding
    functions by using 'get_loss(grad)_func' methods.

    Parameters
    ----------
    loss_name: str
        The name of the loss function.
        The list of valid names: ['relu', 'log']

    """

    def __init__(self, loss_name: str):
        """
        Initializes class public attributes
        with static methods, using getattr method.

        """

        self._loss = getattr(self, f'loss_{loss_name}')
        self._grad = getattr(self, f'grad_{loss_name}')


    def get_loss_func(self):
        """ Getter of loss function"""
        return self._loss


    def get_grad_func(self):
        """ Getter of grad function"""
        return self._grad


    @staticmethod
    def loss_relu(x, y, w):
        """
        This is implementation of 'relu' loss function.
        Formula: f(m) = max(0, -m)

        """

        margin = np.dot(x, w) * y

        if margin >= 0:
            return 0
        return -margin


    @staticmethod
    def grad_relu(x, y, w):
        """
        This function calculates gradient vector
        for 'relu' loss function.

        """

        n_features = x.shape[0]
        margin = np.dot(x, w) * y

        if margin >= 0:
            return np.zeros(n_features)
        return x * (-y)


    @staticmethod
    def loss_log(x, y, w):
        """
        This is implementation of 'log' loss function.
        Formula: f(m) = ...

        """

        pass


    @staticmethod
    def grad_log(x, y, w):
        """
        This function calculates gradient vector
        for 'log' loss function.

        """

        pass



class _MovAvg:
    """
    This class stores implementations of several moving
    averages (MA). By specifying name of MA while
    creating instance of the class, you will be able to
    get the function by using 'get_update_func' method.

    Parameters
    ----------
    mavg_name: str
        The name of the moving average.
        The list of valid names: ['exp', 'mean']

    """

    def __init__(self, mavg_name: str):
        """
        Initializes class public attributes
        with static methods, using getattr method.

        """

        self._update_risk = getattr(self, f'update_{mavg_name}')


    def get_update_func(self):
        """ Getter of moving average function"""
        return self._update_risk


    @staticmethod
    def update_exp(risk, loss, upd_rate):
        """
        This is implementation of 'exp' moving average.
        Formula: (for fixed 'i' and 'l'):
        f_i(x) = l * x + (1 - l) * f_(i-1)

        """

        return upd_rate * loss + (1 - upd_rate) * risk


    @staticmethod
    def update_mean(x, y, w):
        """
        This is implementation of 'mean' moving average.
        Formula: (for fixed 'i' and 'l'):
        f_i(x) = ...

        """

        pass
