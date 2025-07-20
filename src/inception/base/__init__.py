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