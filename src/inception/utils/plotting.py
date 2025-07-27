import numpy as np # type: ignore
import plotly.graph_objects as go # type: ignore
from typing import Callable, List, Tuple, Optional

class SurfacePlotter3D:
    def __init__(
            self,
            func: Callable[[float, float], float],
            x_range: Tuple[float, float] = (-3, 3),
            y_range: Tuple[float, float] = (-3, 3),
            resolution: int = 50
    ):
        self.func = func
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.X, self.Y, self.Z = self._generate_surface()

    def _generate_surface(self):
        x_vals = np.linspace(*self.x_range, self.resolution)
        y_vals = np.linspace(*self.y_range, self.resolution)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([[self.func(x, y) for x, y in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
        return X, Y, Z
    
    def plot_surface(self, title: str = "3D Surface Plot"):
        surface = go.Surface(x=self.X, y=self.Y, z=self.Z, colorscale='Viridis', showscale=False)
        fig = go.Figure(data=[surface], )
        fig.update_layout(title=title, scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='f(x, y)',
        ))
        return fig
    
    def add_path(self, fig, trajectory: List[np.ndarray]):
        path_x, path_y, path_z = [], [], []
        for point, eval in trajectory:
            x, y = point
            path_x.append(x)
            path_y.append(y)
            path_z.append(eval)
        path_trace = go.Scatter3d(
            x=path_x, y=path_y, z=path_z,
            mode="lines+markers",
            marker=dict(size=4, color="red"),
            line=dict(color="red", width=4),
            name="Path"
        )
        fig.add_trace(path_trace)
        return fig
    
    def add_vectors(self, fig, vectors: List[Tuple[np.ndarray, np.ndarray]], color='red', scale=0.05, vector_offset=-0.1):
        for point, direction in vectors:
            x0, y0 = point
            dx, dy = direction*scale
            x1, y1 = x0 + dx, y0 + dy
            z0 = self.func(x0, y0) + vector_offset
            z1 = self.func(x1, y1) + vector_offset
            fig.add_trace(go.Scatter3d(
                x = [x0, x1],
                y = [y0, y1],
                z = [z0, z1],
                mode = "lines+markers",
                marker=dict(size=2, color=color),
                line=dict(color=color, width=3),
                name="Vector"
            ))
        return fig
    

if __name__ == "__main__":

    def func(x, y):
        return np.sin(np.sqrt(x**2 + y**2))
    

    # Example trajectory
    trajectory = [np.array([1.5, 1.5]), np.array([1.0, 1.0]), np.array([0.5, 0.5])]
    vectors = [(pt, -pt) for pt in trajectory]

    # Plotting the surface
    plotter = SurfacePlotter3D(func, x_range=(-4, 4), y_range=(-4, 4), resolution=500)
    fig = plotter.plot_surface("bowling ball function")
    fig = plotter.add_path(fig, trajectory)
    #fig = plotter.add_vectors(fig, vectors)
    fig.show()