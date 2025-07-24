import numpy as np #type: ignore
from ..base import BaseOptimizer
from typing import Callable, List, Tuple

class Momentum(BaseOptimizer):
    def __init__(
            self, 
            learning_rate: float = 0.01,
            momentum: float = 0.9,
            max_iter: int = 1000,
            tolerance: float = 1e-6,
            verbose: bool = False
            ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.verbose = verbose
        self.history_ = []
        self.opt_point_ = None
        self.opt_value_ = None
    
    def fit(
            self,
            loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
            grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
            x0: np.ndarray,
            data: List[Tuple[np.ndarray, np.ndarray]]
    ):
        theta = np.array(x0, dtype=float)
        v = np.zeros_like(theta)

        for i in range(self.max_iter):
            grads = [grad_fn(theta, x, y) for x, y in data]
            grad_avg = np.mean(grads, axis=0)
            loss_avg = np.mean([loss_fn(theta, x, y) for x, y in data])

            if np.linalg.norm(grad_avg) < self.tolerance:
                if self.verbose:
                    print(f"Convergence reached at iteration {i}.")
                break
            v = self.momentum * v - self.learning_rate * grad_avg
            theta += v

            self.history_.append((theta.copy(), loss_avg))

            if self.verbose and i % 50 == 0:
                print(f"Iteration {i}: Loss = {loss_avg:.4f}, theta = {theta}")

        self.opt_point_ = theta
        self.opt_value_ = loss_avg
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