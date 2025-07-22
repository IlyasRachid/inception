from inception.optimizers import GradientDescent
from inception.optimizers import StochasticGradientDescent
import numpy as np # type: ignore

def test_gradient_descent():
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])

    gd = GradientDescent(learning_rate=0.1, max_iter=1000)
    gd.fit(f, grad_f, x0=np.array([5.0, -3.0]))
    result = gd.predict()

    # Check convergence near origin
    assert np.allclose(result, np.array([0.0, 0.0]), atol=1e-2)


def test_sgd_optimizer_converges():
    # Create a toy dataset for y = 2x
    X = [np.array([x]) for x in range(-10, 11)] # shape: (21, 1)
    y = [2 * x for x in range(-10, 11)] # shape: (21,)
    data = list(zip(X, y))

    # Loss: MSE for a single point
    def loss(theta, x, y):
        return (theta@x - y) ** 2
    
    # Gradient: derivative of MSE
    def grad(theta, x, y):
        return 2*(theta@x - y) * x
    
    # Initial parameter
    x0 = np.array([0.0]) # starts at 0

    # Train useing SGD
    sgd = StochasticGradientDescent(learning_rate=0.01, max_iter=1000, epochs=10, tolerance=1e-6, seed=42, verbose=True)
    sgd.fit(loss, grad, x0, data)
    theta_opt = sgd.predict()

    # The optimal parameter should be close to 2
    assert np.allclose(theta_opt, [2.0], atol=1e-3), f"SGD did not converge properly: {theta_opt}"