import numpy as np
from src.optimizer import Optimizer
from src.losses import CategoricalCrossEntropy
from src.metrics import accuracy_score

class NeuralNetwork:
    def __init__(self, epochs=50, batch_size=32, learning_rate=0.01, momentum=0.9):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = Optimizer(learning_rate=learning_rate, momentum=momentum)
        self.loss_func = CategoricalCrossEntropy()
        self.layers = []

    def add(self, layer):
        if self.layers:
            layer.set_input_shape(input_shape=self.layers[-1].output_shape())
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)
        self.layers.append(layer)

    def forward_propagation(self, X, training):
        output = X
        for layer in self.layers:
            output = layer.forward_propagation(output, training)
        return output

    def backward_propagation(self, output_error):
        error = output_error
        for layer in reversed(self.layers):
            error = layer.backward_propagation(error)

    def fit(self, X, y):
        n_samples = X.shape[0]
        for epoch in range(1, self.epochs + 1):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch, y_batch = X[indices[start:end]], y[indices[start:end]]
                
                output = self.forward_propagation(X_batch, training=True)
                error = self.loss_func.derivative(y_batch, output)
                self.backward_propagation(error)

            preds = self.forward_propagation(X, training=False)
            loss = self.loss_func.loss(y, preds)
            acc = accuracy_score(y, preds)
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.4f} - Acc: {acc:.4f}")

    def predict(self, X):
        return self.forward_propagation(X, training=False)