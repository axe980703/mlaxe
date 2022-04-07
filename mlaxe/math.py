import numpy as np


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
