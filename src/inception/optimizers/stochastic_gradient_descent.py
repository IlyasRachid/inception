import numpy as np  # type: ignore
from ..base import BaseOptimizer
import random

class StochasticGradientDescent(BaseOptimizer):
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
    """

    def __init__(self, learning_rate=0.01, max_iter=5000, epochs=10, tolerance=1e-6, verbose=False, seed=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.epochs = epochs
        self.tolerance = tolerance
        self.verbose = verbose
        self.history_ = []
        self.opt_point_ = None
        self.opt_value_ = None
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def fit(self, func, grad, x0, data):
        theta = np.array(x0, dtype=float)
        self.history_ = []
        n = len(data)
        step_counts = 0

        # Handle case where max_iter is zero
        if self.max_iter == 0:
            if self.verbose:
                print("[Max Iterations Reached] max_iter = 0, stopping immediately.")
            self.opt_point_ = theta
            self.opt_value_ = func(theta, *data[0])  # Use the first data point for loss
            return self
        for epoch in range(self.epochs): 
            if step_counts >= self.max_iter:
                if self.verbose:
                    print(f"[Max Iterations Reached] Stopping at epoch {epoch}")
                break

            # Shuffle dataset indices each epoch
            indices = np.random.permutation(n)
            for i in indices:
                x_i, y_i = data[i]
                g = grad(theta, x_i, y_i)

                if np.linalg.norm(g) < self.tolerance:
                    if self.verbose:
                        print(f"[Converged] Epoch {epoch}, data index {i}, ||grad|| = {np.linalg.norm(g):.2e}")
                    self.opt_point_ = theta
                    self.opt_value_ = func(theta, x_i, y_i)
                    return self

                theta -= self.learning_rate * g
                loss_value = func(theta, x_i, y_i)
                self.history_.append((theta.copy(), loss_value))
                step_counts += 1

                if self.verbose and (epoch * n + i) % 20 == 0:
                    print(f"[Epoch {epoch} Step {i}] loss = {loss_value:.6f}, θ = {theta}")

        self.opt_point_ = theta
        self.opt_value_ = loss_value
        return self


    
    def predict(self):
        if self.opt_point_ is None:
            raise ValueError("Optimizer has not been fitted yet.")
        return self.opt_point_

    def score(self):
        if self.opt_value_ is None:
            raise ValueError("Optimizer has not been fitted yet.")
        return self.opt_value_
    
    def get_history(self):
        return np.array(self.history_, dtype=object)
    
