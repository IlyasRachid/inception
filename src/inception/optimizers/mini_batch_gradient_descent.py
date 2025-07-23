import numpy as np #type:ignore
from ..base import BaseOptimizer
from typing import Callable, List, Tuple

class MiniBatchGradientDescent(BaseOptimizer):
    def __init__(
            self,
            learning_rate: float = 0.01,
            max_iter: int = 1000,
            batch_size: int = 32,
            tolerance: float = 1e-6,
            verbose: bool = False,
            shuffle: bool = True,
            random_state: int = None
            ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.verbose = verbose
        self.shuffle = shuffle
        self.random_state = random_state
        self.history_ = []
        self.opt_point_ = None
        self.opt_value_ = None
        self._rng = np.random.default_rng(random_state) # _rng is a random number generator used for shyffling data

    def fit(
            self,
            loss_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
            grad_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray], 
            x0: np.ndarray,
            data: List[Tuple[np.ndarray, np.ndarray]]
    ):
        theta = np.array(x0, dtype=float)
        n_samples = len(data)
        for epoch in range(self.max_iter):
            if self.shuffle:
                self._rng.shuffle(data)
            
            for i in range(0, n_samples, self.batch_size):
                batch = data[i:i+self.batch_size]
                X_batch, y_batch = zip(*batch)
                X_batch = np.stack(X_batch)
                y_batch = np.stack(y_batch)

                # Compute the gradient over the batch
                grads = [grad_fn(theta, x_i, y_i) for x_i, y_i in zip(X_batch, y_batch)]
                grad_avg = np.mean(grads, axis=0)
                loss_avg = np.mean([loss_fn(theta, x_i, y_i) for x_i, y_i in zip(X_batch, y_batch)])
                # Check for convergence
                if np.linalg.norm(grad_avg) < self.tolerance:
                    if self.verbose:
                        print(f"Convergence reached at epoch {epoch}, batch {i // self.batch_size}")
                    self.opt_point_ = theta
                    self.opt_value_ = loss_avg
                    return self
                
                # Update the parameters
                theta -= self.learning_rate * grad_avg
                self.history_.append((theta.copy(), loss_avg))

                if self.verbose and i % (self.batch_size * 2) == 0:
                    print(f"Epoch {epoch}, Batch {i // self.batch_size}, Loss: {loss_avg:.4f}, theta: {theta}")

        self.opt_point_ = theta
        self.opt_value_ = np.mean([loss_fn(theta, x_i, y_i) for x_i, y_i in data])
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