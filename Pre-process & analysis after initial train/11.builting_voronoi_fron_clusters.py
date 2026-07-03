import numpy as np
import matplotlib.pyplot as plt
import time
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union
from scipy.spatial import Voronoi, distance
from matplotlib.ticker import MultipleLocator
import json
import os

# -----------------------------
# Input data
# -----------------------------

# Coordinates of the 30 points from the experimrnt
coordinates = [(0.280523, -0.099261), (0.174975, 0.022145), (0.04511, 0.335246), (-0.255583, -0.319737), (-0.288784, 0.265462), 
               (0.168919, 0.282576), (0.282104, 0.229513), (-0.023534, 0.223568), (-0.228066, 0.102234), (-0.214418, 0.348996), 
               (0.014801, -0.060823), (-0.141068, 0.21052), (-0.02711, 0.046829), (-0.29154, 0.087415), (0.068046, 0.093169), 
               (0.275452, 0.143683), (-0.092086, -0.046179), (0.115288, -0.349798), (0.10159, 0.132903), (-0.17289, -0.290315), 
               (-0.109355, -0.100594), (0.043824, -0.048605), (0.287881, -0.307627), (-0.176845, -0.264763), (0.09559, -0.191672), 
               (-0.020933, -0.201218), (-0.20756, -0.300347), (0.096825, -0.278365), (-0.181221, -0.102943), (0.192615, -0.318763)]
pts = np.array(coordinates)

#seed=0

# anneal 1000 iters

# groups_1based = {
#     'GroupOrange': [5,1,6,8,7,4,2,3,16,9],
#     'GroupGreen' : [24,23,22,27,26,28,25,12,11,10,21,30],
#     'GroupRed'   : [15,14,13,29,17],
#     'GroupPurple': [19,18,20]
#     }
# groups = {k: [i-1 for i in v] for k, v in groups_1based.items()}


# 0.8 1000 iters

# groups_1based = {
#     'GroupOrange': [15,13,16,14,17,22,21,24,30,29,10,9],
#     'GroupGreen' : [8,7,6,5,4,3,2,1],
#     'GroupRed'   : [27,23,26,25,12,11,28,18,20,19]
#     }

#seed=42

#anneal 1000 iters

# groups_1based = {
#     'GroupOrange': [17,15,13,16,14],
#     'GroupGreen' : [8,7,6,5,4,3,2,1],
#     'GroupRed'   : [12,11,23,18,20,19,30,10,29,21,24,22,28,9],
#     'GroupPurple': [25,26,27]
#     }

# 0.8 1000 iters

# groups_1based = {
#     'GroupOrange': [22,18,21,19,12,17,28,24],
#     'GroupGreen' : [27,1,23,9,6],
#     'GroupRed'   : [26,25,11,30,29],
#     'GroupPurple': [20,3,4,7,2,5,8],
#     'GroupBrown': [15,14,16,13,10]
#     }

groups_1based = {
    'GroupOrange': [13,3,2,7,5],
    'GroupGreen' : [14,10,16,15],
    'GroupRed'   : [12,4,22,18,19,8,20,9,17,6,1],
    'GroupPurple': [23,21,27,26,25,28,24],
    'GroupBrown': [30,29,11]
    }

groups = {k: [i-1 for i in v] for k, v in groups_1based.items()}

# -----------------------------
# Helper functions
# -----------------------------

# Convert Voronoi diagram to finite polygons (clip infinite rays to a bounding radius)
def voronoi_finite_polygons_2d(vor, radius=None):
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions, new_vertices = [], vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        # Set radius large enough to contain all points
        radius = np.ptp(vor.points, axis=0).max() * 2

    # Map each point to its ridges, builds a dictionary of all neighboring edges for each input point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    
    # Construct new bounded regions
    for p1, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx] # Gives a list of vertex indices for the region
        if all(v >= 0 for v in region): # Already bounded, if all vertex indices are valid (≥ 0), reuse it
            new_regions.append(region)
            continue
        
        # Patch regions with infinite edges
        ridges = all_ridges[p1]
        # Copying over the finite part
        new_region = [v for v in region if v >= 0]
        # For each edge involving this point
        for p2, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0: 
                continue
            # Compute direction of infinite ray
            t = vor.points[p2] - vor.points[p1] # Direction vector of the ridge
            t /= np.linalg.norm(t) # Normalize
            n = np.array([-t[1], t[0]]) # Normal vector 
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far = vor.vertices[v1 if v1 >= 0 else v2] + direction * radius
            new_vertices.append(far.tolist())
            new_region.append(len(new_vertices) - 1)

        # Sort the new polygon's vertices counter-clockwise, ensures the polygon vertices are in order
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_regions.append([new_region[i] for i in np.argsort(angles)])

    return new_regions, np.asarray(new_vertices)

# -----------------------------
# Build bounded Voronoi inside the fixed region
# -----------------------------

# Create Voronoi diagram
vor = Voronoi(pts)
radius = 2 * max(0.3, 0.39)
regions, vertices = voronoi_finite_polygons_2d(vor, radius)

# Define operation region bounds (fixed rectangle)
operation_region = box(-0.3, -0.39, 0.3, 0.39)

# Clip Voronoi cells to operation region, conversion to polygon
cell_polys = {}
for i, region_idx in enumerate(vor.point_region):
    region = regions[i]
    poly = Polygon(vertices[region]).intersection(operation_region)
    if not poly.is_empty: poly = poly.buffer(0)
    cell_polys[i] = poly

# Union by groups, keep strictly inside operation region
group_unions = {}
for name, idxs in groups.items():
    cells = [cell_polys[i] for i in idxs if not cell_polys[i].is_empty] # Collect all cells belonging to that groups's points
    union_poly = unary_union(cells).intersection(operation_region).buffer(0) if cells else Polygon() # Merge all those cells into one big shape or multiple disconnected shapes
    group_unions[name] = union_poly

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 9))

# Plot operation region boundary
ox, oy = operation_region.exterior.xy
ax.plot(ox, oy, lw=2, color='k', alpha=0.6, label='Operation region')

# Plot Voronoi cells (gray background)
for poly in cell_polys.values():
    if not poly.is_empty:
        x, y = poly.exterior.xy
        ax.fill(x, y, facecolor='#dddddd', edgecolor='white', linewidth=0.6, alpha=0.5)

# Group colored unions and sampled points
#colors = {'GroupOrange':'#ff8c00','GroupGreen':'#2ca02c','GroupRed':'#d62728','GroupPurple':'#9467bd'}
colors = {'GroupOrange':'#ff8c00','GroupGreen':'#2ca02c','GroupRed':'#d62728','GroupPurple':'#9467bd', 'GroupBrown':'#8B4513'}
for name, poly in group_unions.items():
    if poly.is_empty: continue
    polys = [poly] if poly.geom_type == 'Polygon' else list(poly.geoms)
    for p in polys:
        x, y = p.exterior.xy
        ax.fill(x, y, facecolor=colors[name], edgecolor='k', linewidth=1.0, alpha=0.35)
    

# Plot input points and annotate with labels (P1 = coordinates[0])
ax.scatter(pts[:,0], pts[:,1], c='black', s=25, zorder=5)
for i, (x, y) in enumerate(pts):
    ax.text(x, y, f"P{i + 1}", fontsize=8, ha='left', va='bottom')

# Configure axes and grid
ax.set_aspect('equal', adjustable='box')
ax.set_title("Voronoi with Polygon-based Area-Weighted Sampling")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, alpha=0.2)

# Fix the axis limits
ax.set_xlim(-0.3, 0.3)
ax.set_ylim(-0.39, 0.39)

# Set tick spacing to 0.1
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# Draw x=0 and y=0 axes in black
ax.axhline(0, color='black', linewidth=1.2)
ax.axvline(0, color='black', linewidth=1.2)

plt.tight_layout()
plt.show()

save_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\group_unions_wkt_PLR_1000.json"

wkt_dict = {name: geom.wkt for name, geom in group_unions.items()}

with open(save_path, "w") as f:
    json.dump(wkt_dict, f, indent=2)

print("Saving JSON to:", os.path.abspath(save_path))

