import numpy as np
from layers import Layer

class ActivationLayer(Layer):
    def forward_propagation(self, inputs, training):
        self.input = inputs
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.derivative(self.input) * output_error

    def output_shape(self): return self._input_shape
    def parameters(self): return 0

class ReLUActivation(ActivationLayer):
    def activation_function(self, inputs):
        return np.maximum(0, inputs)
    def derivative(self, inputs):
        return np.where(inputs > 0, 1, 0)

class SoftmaxActivation(ActivationLayer):
    def activation_function(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    def derivative(self, inputs):
        # A derivada matemática já está tratada no CategoricalCrossEntropy
        return np.ones_like(inputs)