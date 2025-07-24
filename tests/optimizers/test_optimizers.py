from inception.optimizers import GradientDescent
from inception.optimizers import StochasticGradientDescent
from inception.optimizers import MiniBatchGradientDescent
from inception.optimizers import Momentum
import numpy as np # type: ignore

def test_gradient_descent():
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])

    gd = GradientDescent(learning_rate=0.1, max_iter=1000)
    gd.fit(f, grad_f, x0=np.array([5.0, -3.0]))
    result = gd.predict()

    # Check convergence near origin
    assert np.allclose(result, np.array([0.0, 0.0]), atol=1e-2)


def test_stochastic_gradient_descent():
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


def test_mini_batch_gradient_descent():
    rng = np.random.default_rng(42)
    X = np.array([[x] for x in range(-5,5)])  # shape: (100, 1)
    true_theta = np.array([3.0])
    y = X@true_theta + rng.normal(0, 0.1, size=X.shape[0])

    # Define loss function: MSE for one sample
    def loss_fn(theta, x_i, y_i):
        return (theta @ x_i - y_i) ** 2

    # Define gradient of the loss
    def grad_fn(theta, x_i, y_i):
        return 2 * (theta @ x_i - y_i) * x_i
    
    # Initial guess
    x0 = np.array([0.0])

    # Train with MBGD
    mbgd = MiniBatchGradientDescent(
        learning_rate=0.01,
        max_iter=100,
        batch_size=5,
        tolerance=1e-6,
        verbose=False,
        random_state=42
    )

    mbgd.fit(loss_fn, grad_fn, x0, list(zip(X, y)))
    theta_opt = mbgd.predict()

    # check if the optimal parameter is close to the true value
    assert np.allclose(theta_opt, true_theta, atol=5e-2), f"MBGD did not converge properly: {theta_opt}"

def test_momentum():
    # Generate toy data: y = 3x + noise
    rng = np.random.default_rng(42)
    X = np.array([[x] for x in np.linspace(-10, 10, 500)])
    true_theta = np.array([3.0]) 
    y = X @ true_theta +  rng.normal(0, 0.5, size=X.shape[0])
    data = list(zip(X, y))

    # Define loss function (MSE)
    def loss_fn(theta, x_i, y_i):
        return (theta @ x_i - y_i)**2
    
    # Define gradient of the loss
    def grad_fn(theta, x_i, y_i):
        return 2 * (theta @ x_i - y_i) * x_i
    
    # Initial parameter guess
    x0 = np.array([0.0])

    # Run momentum optimizer
    opt = Momentum(
        learning_rate=0.01,
        momentum=0.9,
        max_iter=300,
        tolerance=1e-6,
        verbose=True
    )
    opt.fit(loss_fn, grad_fn, x0, data)
    theta_opt = opt.predict()

    # Check if the optimal parameter is close to the true value
    assert np.allclose(theta_opt, true_theta, atol=1e-2), f"Momentum did not converge properly: {theta_opt}"
