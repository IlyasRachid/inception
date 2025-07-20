import numpy as np
from ..base import BaseOptimizer

class GradientDescent(BaseOptimizer):
    """
    Implements the Gradient Descent optimization algorithm.
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, tolerance=1e-6, verbose=False):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose
        self.history_ = []
        self.opt_point_ = None
        self.opt_value_ = None

    def fit(self, func, grad, x0):
        """
        Optimize the function using gradient descent.

        Parameters:
        - func: callable, the function to minimize
        - grad: callable, the gradient of the function
        - x0: np.ndarray, initial point for optimization
        """
        x = np.array(x0, dtype=float)
        self.history_ = [(x.copy(), func(x))]

        for i in range(self.max_iter):
            g = grad(x)
            if np.linalg.norm(g) < self.tolerance:
                if self.verbose:
                    print(f"[Converged] Step {i}, ||grad|| = {np.linalg.norm(g):.2e}")
                break
            x -= self.learning_rate * g
            self.history_.append((x.copy(), func(x)))

            if self.verbose and i % 50 == 0:
                print(f"[Step {i}] f(x) = {func(x):.6f}, x = {x}")
            
        self.opt_point_ = x
        self.opt_value_ = func(x)
        return self
    
    def predict(self):
        """
        Returns the optimal point found during optimization.
        """
        if self.opt_point_ is None:
            raise ValueError("Optimizer has not been fitted yet.")
        return self.opt_point_
    
    def score(self):
        """
        Returns the optimal value of the function at the optimal point.
        """
        if self.opt_value_ is None:
            raise ValueError("Optimizer has not been fitted yet.")
        return self.opt_value_
    
    def get_history(self):
        """
        Returns the history of optimization steps.
        """
        return np.array(self.history_, dtype=object)