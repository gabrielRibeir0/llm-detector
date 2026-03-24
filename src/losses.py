import numpy as np
from abc import abstractmethod

class LossFunction:
    @abstractmethod
    def loss(self, y_true, y_pred): pass
    @abstractmethod
    def derivative(self, y_true, y_pred): pass

class CategoricalCrossEntropy(LossFunction):
    def __init__(self, class_weights=None):
        self.class_weights = None if class_weights is None else np.asarray(class_weights, dtype=np.float32)

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        per_sample = -np.sum(y_true * np.log(y_pred), axis=1)
        if self.class_weights is not None:
            weights = np.sum(y_true * self.class_weights, axis=1)
            per_sample = per_sample * weights
        return np.mean(per_sample)

    def derivative(self, y_true, y_pred):
        # A combinação Softmax + CCE simplifica a derivada para (y_pred - y_true)
        grad = (y_pred - y_true)
        if self.class_weights is not None:
            weights = np.sum(y_true * self.class_weights, axis=1, keepdims=True)
            grad = grad * weights
        return grad / len(y_true)