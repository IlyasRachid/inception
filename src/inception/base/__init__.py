from abc import ABC, abstractmethod

class BaseOptimizer(ABC):

    @abstractmethod
    def fit(self, func, grad, x0):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def score(self):
        pass

class BaseLoss(ABC):
    """Abstract base class for loss functions."""
    @abstractmethod
    def __call__(self, y_true, y_pred) -> float:
        """Calculate the loss value."""
        pass

    @abstractmethod
    def gradient(self, y_true, y_pred, X):
        """Calculate the gradient of the loss function."""
        pass