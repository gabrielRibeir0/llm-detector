import numpy as np
from abc import abstractmethod

class LossFunction:
    @abstractmethod
    def loss(self, y_true, y_pred): pass
    @abstractmethod
    def derivative(self, y_true, y_pred): pass

class CategoricalCrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(y_pred)) / len(y_true)

    def derivative(self, y_true, y_pred):
        # A combinação Softmax + CCE simplifica a derivada para (y_pred - y_true)
        return (y_pred - y_true) / len(y_true)