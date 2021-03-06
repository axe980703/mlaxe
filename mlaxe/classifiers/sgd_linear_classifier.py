"""
This module contains classifiers with
linear decision function.

"""

import numpy as np
from mlaxe.standards import BaseLinearClassifier
from mlaxe.math import _Loss, _MovAvg, _Regularizer
from mlaxe.visual import Animation2D


class SGDLinearClassifier(BaseLinearClassifier):
    """
    The class implements binary classification with a linear decision function.
    It provides set of loss functions to choose.
    Weights by default initialized with standard gaussian random values.
    The value of empirical risk is updated using the moving average.
    Learning rate by default decreases.
    Description is expanding..

    Attributes
    ----------
    weights: numpy-like array
        Vector of coeffitients of linear decision function.
        Shape of array equal to (n_classes, n_features + 1).

    iter_spent: int
        Number of iterations spent before convergence.

    """

    def __init__(self, lr_init=0.01, upd_rate=0.01, max_iter=750,
                 loss_eps=1e-3, tol_iter=4, add_bias=True,
                 save_hist=True, shuffle=False, alt_class=False,
                 mov_avg='exp', loss_func='hinge', seed=322,
                 regul=None, reg_coef=0.001, decr_lrate=False):
        """
        Initializes parameters of the model.

        Parameters
        ----------
        lr_init: float (default: 0.01)
            Initial learning rate of sgd.

        upd_rate: float (deafult: 0.07)
            Update rate of empirical risk on each iteration.
            In interval : 0 <= val <= 1.

        max_iter: int (default: 10000)
            Upper bound of maximum number of iterations.

        loss_eps: float (default: 1e-4)
            Stop criterion. Fitting will be stopped when
            condition: (loss - min_loss <= loss_eps) is true
            for (tol_iter) consecutive iterations.

        tol_iter: int (default: 5)
            Number of iterations to wait before stop fitting.

        add_bias: bool (default: True)
            Whether to add bias to sample by concatenating
            column of ones or not.

        save_hist: bool (default: True)
            Whether to log all weights and gradients states
            during fitting or not.

        shuffle: bool (default: False)
            Whether to shuffle feature data after each iteration or not.

        mov_avg: str (default: 'exp')
            Type of moving average to use ('exp' - exponential,
            'mean'- mean) for emperical risk re-calculation.

        loss_func: str (default: 'hinge')
            Target loss function, for which we minimize empirical risk.
            For 'hebb', f(x) = max(0, -x),
            For 'hinge', f(x) = max(0, 1 - x),
            For 'log', f(x) = log2(1 + e^(-x)),
            For 'exp', f(x) = e^(-x),
            will be used.

        regul: str (default: None)
            Type of regularization function to use.
            For 'l1 norm', f(X) = sum(|x_i|), for each i,
            For 'l2 norm', f(X) = sum(x_i^2), for each i.

        reg_coef: float (default: 0.001)
            The rate of regularization. The more the value,
            the less the variance of weights vector.

        alt_class: bool (default: False)
            Whether to alternate classes while fitting.
            Objects inside class are chosen equiprobably.

        decr_lrate: bool (default: False)
            Updates learning rate on each iteration.
            Inversely proportional to the number of iteration.

        """

        self._lr_init = lr_init
        self._upd_rate = upd_rate
        self._max_iter = max_iter
        self._loss_eps = loss_eps
        self._tol_iter = tol_iter
        self._add_bias = add_bias
        self._save_hist = save_hist
        self._shuffle = shuffle
        self._seed = seed
        self._reg_coef = reg_coef
        self._alt_class = alt_class
        self._loss_func = loss_func
        self._decr_lrate = decr_lrate

        self.iter_spent = 0
        self.weights = []

        # select functions, corresponding to received parameters
        loss_select, mavg_select = _Loss(loss_func), _MovAvg(mov_avg)
        reg_select = _Regularizer(regul)
        self._update_risk = mavg_select.get_update_func()
        self._loss = loss_select.get_loss_func()
        self._grad = loss_select.get_grad_func()
        self._regul = reg_select.get_regul_func()


    def _fit_binary(self, x, y, class_num):
        """
        Fits linear classifier using Stochastic Gradient Descent.
        This method only solves binary classification task.

        Parameters
        ----------
        x: 2D numpy-like array
            Feature data.

        y: 1D numpy-like array
            Label data.

        class_num: int
            For binary classification it equals to 0.
            Otherwise it indicates decision function number.

        Returns
        ----------
        weights: 1D numpy-like array
            Coefficients of decision function.

        """

        self._converge_streak = 0

        if self._alt_class:
            # separate objects of two classes
            sort_ord = np.argsort(y)
            x, y = x[sort_ord], y[sort_ord]

            # set bounds for classes
            cl1_size = y[y == -1].shape[0]
            bounds = [(0, cl1_size), (cl1_size, self._n_objects)]
        else:
            # init shuffle
            x, y = self._shuffle_objects(x, y)

        # init weights with gaussian standard distribution
        w = self._rand_gen.randn(self._n_features + int(self._add_bias))

        # init local variables
        iter_step = 0
        cur_class = 0
        min_risk = cur_risk = self._empirical_risk(x, y, w)
        l_rate = self._lr_init

        # set threshold if loss_func is appropriate
        threshold = None
        if self._loss_func in ['hinge', 'hebb']:
            threshold = 1 if self._loss_func == 'hinge' else 0


        while iter_step < self._max_iter:

            # shuffling the whole sample
            if self._shuffle:
                x, y = self._shuffle_objects(x, y)

            # picking random index for current object
            if self._alt_class:
                i = self._rand_gen.randint(*bounds[cur_class])
                cur_class ^= 1
            else:
                i = self._rand_gen.randint(0, self._n_objects)

            # calulating loss on current object
            iter_loss = self._loss(x[i], y[i], w)

            # updating empirical risk
            cur_risk = self._update_risk(cur_risk, iter_loss, self._upd_rate)

            # processing stop criterion
            if self._time_to_stop(cur_risk, min_risk):
                break

            min_risk = min(min_risk, cur_risk)

            # heuristic: skip object if margin is good
            if threshold is not None:
                margin = np.dot(x[i], w) * y[i]
                if margin >= threshold:
                    continue

            # calculating gradient of loss function
            loss_grad = self._grad(x[i], y[i], w)

            # calculating regularization gradient
            reg_grad = self._regul(w, self._reg_coef)

            # changing weights according to the gradient value
            w = w - l_rate * (loss_grad + reg_grad)

            # saving history data
            if self._save_hist:
                self._weig_hist[class_num].append(w)
                self._grad_hist[class_num].append(loss_grad)
                self._risk_hist[class_num].append(cur_risk)
                self._loss_hist[class_num].append(iter_loss)

            # changing learning rate
            if self._decr_lrate:
                l_rate = self._update_l_rate(l_rate, iter_step)

            # updating iteration number
            iter_step += 1

        # sum up iteration number
        self.iter_spent += iter_step + 1

        return w


    def fit(self, x, y):
        """
        Fits linear classifier using Stochastic Gradient Descent.
        For multiclass classification One-Versus-All strategy
        is used.

        Parameters
        ----------
        x: 2D numpy-like array
            Feature data.

        y: 1D numpy-like array
            Label data.

        Returns
        ----------
        self: SGDLinearClassifier
            The current instance of the classifier.

        """

        self._n_classes = np.unique(y).shape[0]

        # if sample contains two classes, then
        # we only need one decision function
        self._dec_num = self._n_classes - (self._n_classes == 2)

        # init private attributes, for history storing and
        # other needs
        self._weig_hist = [[] for _ in range(self._dec_num)]
        self._grad_hist = [[] for _ in range(self._dec_num)]
        self._risk_hist = [[] for _ in range(self._dec_num)]
        self._loss_hist = [[] for _ in range(self._dec_num)]

        if self._save_hist:
            self._xs, self._ys = x, y

        # save sample information
        self._n_objects = x.shape[0]
        self._n_features = x.shape[1]
        self._rand_gen = np.random.RandomState(self._seed)

        if self._add_bias:
            x = np.hstack([x, np.ones((self._n_objects, 1))])

        # choose strategy, depending on binary or
        # multiclass classification
        if self._n_classes == 2:
            y = np.copy(y)
            y[y == 0] = -1
            weights = self._fit_binary(x, y, 0)
            self.weights.append(weights)
        else:
            for n_class in range(self._n_classes):
                yi = np.copy(y)
                mask_eq, mask_neq = (yi == n_class, yi != n_class)
                yi[mask_eq], yi[mask_neq] = 1, -1
                weights = self._fit_binary(x, yi, n_class)
                self.weights.append(weights)

        return self


    def predict(self, x):
        """
        Predicts class labels for provided feature data.
        If _add_bias flag is specified, column of ones will be added.

        Parameters
        ----------
        x: 2D numpy-like array
            Feature data.

        Returns
        ----------
        y: 1D numpy-like array
            Predicted labels of classes.

        """

        # check if fit was called before, otherwise raise error
        ...

        n_objects = x.shape[0]

        if self._add_bias:
            x = np.hstack([x, np.ones((n_objects, 1))])

        # choose strategy, depending on binary or
        # multiclass classification
        if self._n_classes == 2:
            w = self.weights[0]
            y = np.sign(np.dot(x, w))
            y[y == -1] = 0
        else:
            # adapt shapes to each other
            x = np.repeat(x, self._n_classes, axis=0)
            w = np.tile(np.array(self.weights), reps=(n_objects, 1))

            # calculate dot product for each object and class
            dots = np.sum(x * w, axis=1).reshape(-1, self._n_classes)

            # the answer is the class with maximal dot product
            y = np.argmax(dots, axis=1)

        return y


    def evaluate(self, x, y):
        """
        Calculates accuracy (the ratio of the number of objects
        with correct predicted classes to the total number of objects).

        Parameters
        ----------
        x: 2D numpy-like array
            Feature data.

        y: 1D numpy-like array
            Label data.

        Returns
        ----------
        acc: float
            Accuracy of number of correct predicted classes.

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
        x: 2D numpy-like array
            Feature data.

        y: 1D numpy-like array
            Label data.

        w: 1D numpy-like array
            Weights of linear decision function.

        Returns
        ----------
        risk: float
            Value of empirical risk.

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
        cur_risk: float
            The risk value on the current iteration.

        min_risk: float
            The minimal risk value during fitting process.

        Returns
        ----------
        to_stop: bool
            Stop indicator.

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
        x: 2D numpy-like array
            Feature data.

        y: 1D numpy-like array
            Label data.

        Returns
        ----------
        x, y: tuple of (2D, 1D) numpy-like arrays
            Shuffled sample data.

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
        l_rate: float
            Current learning rate.

        iter_step: float
            Number of iteration.

        Returns
        ----------
        next_lr: float
            Updated learning rate.

        """

        next_lr = min(l_rate, 2 / (1 + iter_step))

        return next_lr


    def get_anim(self, save_gif=True, verbose=True):
        """
        Creates animation of fitting process.

        Parameters
        ----------
        save_gif: bool
            Whether to save .gif file or not.

        verbose: bool
            Whether to display process of animating or not.

        """

        anim = Animation2D(
            sample=(self._xs, self._ys), class_num=self._dec_num,
            weights=self._weig_hist, grads=self._grad_hist,
            risks=self._risk_hist, losses=self._loss_hist,
            verbose=verbose,
            save_gif=save_gif
        )

        anim.build_animation()
