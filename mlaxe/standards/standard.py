"""
This module stores patterns of creating
and defining estimators and others objects.

"""

from abc import ABC, abstractmethod


class BaseLinearClassifier(ABC):

    @abstractmethod
    def fit(self, x, y):
        raise NotImplementedError


    @abstractmethod
    def predict(self, x):
        raise NotImplementedError
