import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pandas as pd

centers = np.array([
    [769.73, 1528.8],  
    [631.82, 1734.60],
    [267.86, 2363.75],
    [461.38, 1906.73],
    [669.84, 995.82],
    [342.67, 1239.11]
]) # shape (7, 2)
labels = ["/æ/", "/ɛ/", "/i/", "/ɪ/", "/ɒ/", "/u/"]

# making the expanded convex hull
hull = ConvexHull(centers)
hull_points = centers[hull.vertices]
centroid = hull_points.mean(axis=0)
scale = 1.10
expanded_vertices = centroid + scale * (hull_points - centroid)
expanded_hull = ConvexHull(expanded_vertices)

# Grid Limits
F1_min, F1_max = expanded_vertices[:,0].min(), expanded_vertices[:,0].max()
F2_min, F2_max = expanded_vertices[:,1].min(), expanded_vertices[:,1].max()

# Make grid
F1_vals = np.linspace(F1_min, F1_max, 11)
F2_vals = np.linspace(F2_min, F2_max, 11)
F1_grid, F2_grid = np.meshgrid(F1_vals, F2_vals)

# Flatten both grids for convex hull operations later
grid_points = np.column_stack([F1_grid.ravel(), F2_grid.ravel()])

# test which points from the grid are inside the the convex hull
def in_hull(points):
    A = expanded_hull.equations[:, :-1] # all rows and all columns except last
    b = expanded_hull.equations[:, -1]  # last column only
    return np.all(A @ points.T + b[:, None] <= 1e-12, axis=0)

mask = in_hull(grid_points)
inside_points = grid_points[mask]
outside_points = grid_points[~mask]

df = pd.DataFrame(inside_points, columns=["F1", "F2"])
df.to_csv("inside_points.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5))
ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)

ax.scatter(grid_points[:,1], grid_points[:,0], color='lightgray')
ax.scatter(inside_points[:,1], inside_points[:,0], color='blue')

ax.scatter(centers[:, 1], centers[:, 0], color="red", s=40)
for (F1, F2), label in zip(centers, labels):
    ax.text(F2, F1 - 10, label,
            fontsize=14,
            ha='center',  # horizontal alignment
            va='bottom')  # vertical alignment slightly above point
        
ax.set_xlabel("F2 Hz")
ax.set_ylabel("F1 Hz")
ax.set_title("Vowel chart grid")
plt.show()