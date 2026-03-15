import numpy as np
import copy
from abc import ABCMeta, abstractmethod

class Layer(metaclass=ABCMeta):
    def set_input_shape(self, input_shape): self._input_shape = input_shape
    def input_shape(self): return self._input_shape
    
class DenseLayer(Layer):
    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self._input_shape = input_shape
        
    def initialize(self, optimizer):
        self.weights = np.random.randn(self.input_shape()[0], self.n_units) * np.sqrt(2. / self.input_shape()[0])
        self.biases = np.zeros((1, self.n_units))
        self.w_opt = copy.deepcopy(optimizer)
        self.b_opt = copy.deepcopy(optimizer)
        
    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, inputs, training):
        self.input = inputs
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output
 
    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)
        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)
        return input_error
 
    def output_shape(self): return (self.n_units,) 

class DropoutLayer(Layer):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate
        self.mask = None
    def initialize(self, optimizer): pass
    def parameters(self): return 0
    def output_shape(self): return self._input_shape

    def forward_propagation(self, inputs, training):
        self.input = inputs
        if training:
            self.mask = np.random.binomial(1, 1 - self.drop_rate, size=inputs.shape) / (1 - self.drop_rate)
            self.output = inputs * self.mask
        else:
            self.output = inputs
        return self.output

    def backward_propagation(self, output_error):
        return output_error * self.mask