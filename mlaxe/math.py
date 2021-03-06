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

    """

    def __init__(self, loss_name: str):
        """
        Initializes class public attributes
        with static methods, using getattr method.

        Parameters
        ----------
        loss_name: str
            The name of the loss function.
            The list of valid names: ['hebb', 'hinge', 'log', 'exp]

        """

        self.loss = getattr(self, f'loss_{loss_name}')
        self.grad = getattr(self, f'grad_{loss_name}')


    def get_loss_func(self):
        """ Getter of loss function """

        return self.loss


    def get_grad_func(self):
        """ Getter of grad function """

        return self.grad


    @staticmethod
    def loss_hebb(x, y, w):
        """
        This is implementation of 'hebb' loss function.
        Formula: f(m) = max(0, -m)

        """

        margin = np.dot(x, w) * y

        if margin >= 0:
            return 0
        return -margin


    @staticmethod
    def grad_hebb(x, y, w):
        """
        This function calculates gradient vector
        for 'hebb' loss function.

        """

        n_features = x.shape[0]
        margin = np.dot(x, w) * y

        if margin >= 0:
            return np.zeros(n_features)
        return x * (-y)


    @staticmethod
    def loss_hinge(x, y, w):
        """
        This is implementation of 'hinge' loss function.
        Formula: f(m) = max(0, 1 - m)

        """

        margin = np.dot(x, w) * y

        if margin >= 1:
            return 0
        return 1 - margin


    @staticmethod
    def grad_hinge(x, y, w):
        """
        This function calculates gradient vector
        for 'hinge' loss function.

        """

        n_features = x.shape[0]
        margin = np.dot(x, w) * y

        if margin >= 1:
            return np.zeros(n_features)
        return x * (-y)


    @staticmethod
    def loss_log(x, y, w):
        """
        This is implementation of 'log' loss function.
        Formula: f(m) = log2(1 + e^(-m))

        """

        margin = np.dot(x, w) * y

        return np.log2(1 + np.exp(-margin))


    @staticmethod
    def grad_log(x, y, w):
        """
        This function calculates gradient vector
        for 'log' loss function.

        """

        margin = np.dot(x, w) * y

        return -x * y / (np.exp(margin) + 1) / np.log(2)


class _MovAvg:
    """
    This class stores implementations of several moving
    averages (MA). By specifying name of MA while
    creating instance of the class, you will be able to
    get the function by using 'get_update_func' method.

    """

    def __init__(self, mavg_name: str):
        """
        Initializes class public attributes
        with static methods, using getattr method.

        Parameters
        ----------
        mavg_name: str
            The name of the moving average.
            The list of valid names: ['exp', 'mean'].

        """

        self.update_risk = getattr(self, f'update_{mavg_name}')


    def get_update_func(self):
        """ Getter of moving average function"""

        return self.update_risk


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


class _Regularizer:
    """
    This class stores implementations of regularizers.
    By specifying name of regularizer while creating
    instance of the class, you will be able to get
    the function by using 'get_regul_func' method.

    """

    def __init__(self, reg_name):
        """
        Initializes class public attributes with
        static methods, using getattr method.

        Parameters
        ----------
        reg_name: str
            The name of the regularizer.
            The list of valid names: ['l1', 'l2']

        """

        if reg_name is None:
            reg_name = 'zero'
        self.regularizer = getattr(self, f'{reg_name}_norm')


    def get_regul_func(self):
        """ Getter of moving average function"""

        return self.regularizer


    @staticmethod
    def l1_norm(w, tau):
        """
        This method calculates the gradient vector
        for 'l1' norm.
        Formula of norm: ||w|| = sum(|w_i|), for each i.
        Formula of gradient: tau * sign(w),
        where: w - input vector, tau - variance rate.

        """

        return tau * np.sign(w)


    @staticmethod
    def l2_norm(w, tau):
        """
        This method calculates the gradient vector
        for 'l2' norm.
        Formula of norm: ||w|| = sum(w_i^2), for each i.
        Formula of gradient: tau * w,
        where: w - input vector, tau - variance rate.

        """

        return tau * w


    @staticmethod
    def zero_norm(w, tau):
        """
        Method which is used when none of the norms
        is specified. Just returns zero vector.

        """

        return np.zeros(w.shape[0])
