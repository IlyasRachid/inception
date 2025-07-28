from inception.optimizers import GradientDescent
from inception.optimizers import StochasticGradientDescent
from inception.optimizers import MiniBatchGradientDescent
from inception.optimizers import Momentum
from inception.utils import SurfacePlotter3D
from inception.optimizers import Nesterov
from inception.optimizers import RMSProp
import numpy as np # type: ignore

def test_gradient_descent():
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])

    gd = GradientDescent(learning_rate=0.1, max_iter=1000)
    gd.fit(f, grad_f, x0=np.array([5.0, -3.0]))
    result = gd.predict()

    # Check convergence near origin
    assert np.allclose(result, np.array([0.0, 0.0]), atol=1e-2)

    # Prepare history for plotting
    history = gd.get_history()
    trajectory = [(point, eval) for (point, eval, _) in history]
    vectors = [(theta, -grad) for (theta, _, grad) in history]

    # Visualize
    def scalar_f(x, y): return f(np.array([x, y]))

    plotter = SurfacePlotter3D(scalar_f, x_range=(-6, 6), y_range=(-6, 6), resolution=50)
    fig = plotter.plot_surface(title="Gradient Descent Path")
    fig = plotter.add_path(fig, trajectory)
    fig = plotter.add_vectors(fig, vectors, color='blue')
    fig.show()


def test_stochastic_gradient_descent():
    # Create a toy dataset for y = 2x1 + 3x2 + noise
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(200, 2))  # shape: (100, 2)
    true_theta = np.array([2.0, 3.0])
    y = X @ true_theta + rng.normal(0, 0.2, size=X.shape[0])  # Add some noise
    
    data = list(zip(X, y))

    # Loss: MSE for a single point
    def loss(theta, x, y):
        return (theta@x - y) ** 2
    
    # Gradient: derivative of MSE
    def grad(theta, x, y):
        return 2*(theta@x - y) * x
    
    # Initial parameter
    x0 = np.array([0.0, 0.0]) # starts at 0

    # Train useing SGD
    sgd = StochasticGradientDescent(learning_rate=0.01, max_iter=500, epochs=10, tolerance=1e-6, seed=42, verbose=False)
    sgd.fit(loss, grad, x0, data)
    theta_opt = sgd.predict()

    # The optimal parameter should be close to 2
    assert np.allclose(theta_opt, true_theta, atol=1e-2), f"SGD did not converge properly: {theta_opt}"

    def full_loss(theta):
        return np.mean([loss(theta, x, y) for x, y in data])
    
    history = sgd.get_history()
    trajectory = [(point, eval) for (point, eval, _) in history]
    vectors = [(theta, -grad) for (theta, _, grad) in history]

    # Visualize the path of SGD
    def scalar_f(x, y): return full_loss(np.array([x, y]))

    plotter = SurfacePlotter3D(scalar_f, x_range=(-12, 12), y_range=(-12, 12), resolution=100)
    fig = plotter.plot_surface(title="SGD Path")
    fig = plotter.add_path(fig, trajectory)
    fig = plotter.add_vectors(fig, vectors, color='blue')
    fig.show()


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
    # Generate toy data: y = 2x1 + 3x2 + noise
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 2))  # shape: (500, 2)
    true_theta = np.array([2.0, 3.0])
    y = X @ true_theta + rng.normal(0, 0.2, size=X.shape[0])
    data = [(X,y)]

    # Define loss function (MSE)
    def loss_fn(theta, X, y):
        return np.mean((X @ theta - y)**2)

    # Define gradient of the loss
    def grad_fn(theta, X, y):
        return 2 * (X @ theta - y) @ X / X.shape[0]
    
    # Initial parameter guess
    x0 = np.array([0.0, 0.0])

    # Run momentum optimizer
    opt = Momentum(
        learning_rate=0.01,
        momentum=0.9,
        max_iter=500,
        tolerance=1e-6,
        verbose=True
    )
    opt.fit(loss_fn, grad_fn, x0, data)
    theta_opt = opt.predict()

    # Check if the optimal parameter is close to the true value
    assert np.allclose(theta_opt, true_theta, atol=1e-1), f"Momentum did not converge properly: {theta_opt}"

    # Prepare history for plotting
    history = opt.get_history()
    trajectory = [(point, eval) for (point, eval, _, _) in history]
    vectors = [(theta, -grad) for (theta, _, grad, _) in history]
    momentum = [(point, v) for (point, _, _, v) in history]

    # Visualize the path of Momentum
    def scalar_f(x, y0):
        return loss_fn(np.array([x, y0]), X, y)

    plotter = SurfacePlotter3D(scalar_f, x_range=(-12, 12), y_range=(-12, 12), resolution=100)
    fig = plotter.plot_surface(title="Momentum Path")
    fig = plotter.add_path(fig, trajectory)
    fig = plotter.add_vectors(fig, vectors, color='blue')
    fig = plotter.add_vectors(fig, momentum, color='green', scale=0.5)
    fig = plotter.add_2d_projection(fig, trajectory)
    fig.show()

def test_nesterov():
    # Generate toy data: y = 2x1 + 3x2 + noise
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 2))
    true_theta = np.array([2.0, 3.0])
    y = X @ true_theta + rng.normal(0, 0.2, size=X.shape[0])
    data = [(X, y)]

    # Mean Squared Error Loss
    def loss_fn(theta, X, y):
        return np.mean((X @ theta - y)**2)

    # Gradient of the loss
    def grad_fn(theta, X, y):
        return 2 * (X @ theta - y) @ X / X.shape[0]

    # Initial guess
    x0 = np.array([0.0, 0.0])

    # Instantiate and fit with Nesterov
    opt = Nesterov(
        learning_rate=0.01,
        momentum=0.9,
        max_iter=500,
        tolerance=1e-6,
        verbose=True
    )
    opt.fit(loss_fn, grad_fn, x0, data)
    theta_opt = opt.predict()

    # Validate convergence
    assert np.allclose(theta_opt, true_theta, atol=1e-1), f"Nesterov did not converge properly: {theta_opt}"

    # Prepare history for visualization
    history = opt.get_history()
    trajectory = [(point, eval) for (point, eval, _, _) in history]
    vectors = [(theta, -grad) for (theta, _, grad, _) in history]
    momentum = [(point, v) for (point, _, _, v) in history]

    # Visualization
    def scalar_f(x, y0):
        return loss_fn(np.array([x, y0]), X, y)

    plotter = SurfacePlotter3D(scalar_f, x_range=(-12, 12), y_range=(-12, 12), resolution=100)
    fig = plotter.plot_surface(title="Nesterov Path")
    fig = plotter.add_path(fig, trajectory, name="Nesterov Path", color="blue")
    fig = plotter.add_vectors(fig, vectors, color='purple')
    fig = plotter.add_vectors(fig, momentum, color='orange', scale=0.5)
    fig = plotter.add_2d_projection(fig, trajectory, name="Nesterov 2D", color='cyan')
    fig.show()

def test_rmsprop():
    # Generate toy data
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(100, 2))
    true_theta = np.array([2.0, 3.0])
    y = X @ true_theta + rng.normal(0, 0.2, size=X.shape[0])
    data = [(X, y)]

    # Loss and gradient functions
    def loss_fn(theta, X, y):
        return np.mean((X @ theta - y) ** 2)

    def grad_fn(theta, X, y):
        return 2 * (X @ theta - y) @ X / X.shape[0]

    x0 = np.array([0.0, 0.0])

    opt = RMSProp(
        learning_rate=0.01,
        beta=0.9,
        epsilon=1e-8,
        max_iter=500,
        tolerance=1e-6,
        verbose=True
    )
    opt.fit(loss_fn, grad_fn, x0, data)
    theta_opt = opt.predict()

    assert np.allclose(theta_opt, true_theta, atol=1e-1), f"RMSProp did not converge properly: {theta_opt}"

    history = opt.get_history()
    trajectory = [(theta, loss) for (theta, loss, _, _) in history]
    grad_vectors = [(theta, -grad) for (theta, _, grad, _) in history]
    updates = [(theta, update) for (theta, _, _, update) in history]

    def scalar_f(x, y0):
        return loss_fn(np.array([x, y0]), X, y)

    plotter = SurfacePlotter3D(scalar_f, x_range=(-12, 12), y_range=(-12, 12), resolution=100)
    fig = plotter.plot_surface(title="RMSProp Path")
    fig = plotter.add_path(fig, trajectory)
    fig = plotter.add_vectors(fig, grad_vectors, color='blue')
    fig = plotter.add_vectors(fig, updates, color='orange', scale=0.5)
    fig = plotter.add_2d_projection(fig, trajectory)
    fig.show()
