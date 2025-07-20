from inception.optimizers import GradientDescent
import numpy as np

def test_gradient_descent():
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])

    gd = GradientDescent(learning_rate=0.1, max_iter=1000)
    gd.fit(f, grad_f, x0=np.array([5.0, -3.0]))
    result = gd.predict()

    # Check convergence near origin
    assert np.allclose(result, np.array([0.0, 0.0]), atol=1e-2)