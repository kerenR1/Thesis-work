import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R


# ------------------------------------------3D GMMs before change in distance--------------------------------------

# Function to generate 3D data from a Gaussian distribution
def generate_data(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)


# Generate synthetic data for GMM 1 and GMM 2 (3D data)
np.random.seed(42)  # Random seed
data1 = np.vstack([  # Combine three Gaussian clusters into a single dataset
    generate_data([2, 2, 2], [[1, 0.5, 0.1], [0.5, 1, 0.2], [0.1, 0.2, 1]], 100),
    generate_data([5, 5, 5], [[1, -0.3, 0], [-0.3, 1, 0.1], [0, 0.1, 1]], 100),
    generate_data([8, 2, 2], [[1, 0, 0], [0, 1, 0.1], [0, 0.1, 1]], 100)
])

data2 = np.vstack([
    generate_data([1, 1, 1], [[1, 0.2, 0.1], [0.2, 1, 0.3], [0.1, 0.3, 1]], 100),
    generate_data([6, 6, 6], [[1, 0.4, 0], [0.4, 1, 0.2], [0, 0.2, 1]], 100),
    generate_data([9, 1, 1], [[1, 0.1, 0], [0.1, 1, 0.1], [0, 0.1, 1]], 100)
])

# Fit GMMS to the data (3D data)
gmm1 = GaussianMixture(n_components=3, covariance_type='full')
gmm1.fit(data1)

gmm2 = GaussianMixture(n_components=3, covariance_type='full')
gmm2.fit(data2)

# Print GMM attributes
print("GMM 1:")
print("Means:\n", gmm1.means_)
print("Covariances:\n", gmm1.covariances_)
print("Weights:\n", gmm1.weights_)
print("\nGMM 2:")
print("Means:\n", gmm2.means_)
print("Covariances:\n", gmm2.covariances_)
print("Weights:\n", gmm2.weights_)


# Function to calculate the Hellinger distance between two GMMs
def hellinger_distance_gmm(gmm_1, gmm_2):
    distance = 0
    for i in range(gmm_1.n_components):
        for j in range(gmm_2.n_components):
            # Extract the mean and covariance of the components
            mean1, cov1 = gmm_1.means_[i], gmm_1.covariances_[i]
            mean2, cov2 = gmm_2.means_[j], gmm_2.covariances_[j]

            # Compute the average covariance matrix
            cov_avg = (cov1 + cov2) / 2

            # Calculate the coefficient using determinants
            coeff = (np.linalg.det(cov1) ** 0.25 * np.linalg.det(cov2) ** 0.25) / (np.linalg.det(cov_avg) ** 0.5)

            # Calculate the exponential term based on the difference in means
            mean_diff = mean1 - mean2
            exp_term = -0.125 * mean_diff @ np.linalg.inv(cov_avg) @ mean_diff.T

            # Add contribution to the total distance
            distance += np.sqrt(coeff * np.exp(exp_term))

    # Final Hellinger distance calculation
    return np.sqrt(1 - distance / (gmm_1.n_components * gmm_2.n_components))


# Calculate and print the Hellinger distance
hellinger_dist = hellinger_distance_gmm(gmm1, gmm2)
print("\nHellinger Distance between GMM 1 and GMM 2:", hellinger_dist)


# Function to plot GMM components in 3D
def plot_gmm_3d_with_ellipsoids(gmm, axis, gmm_label, color):
    """
    Plots the Gaussian components in 3D space with confidence ellipsoids.
    Parameters:
        gmm (GaussianMixture): The GMM to plot.
        axis (Axes3D): The matplotlib 3D Axes object to draw on.
        gmm_label (str): Label prefix for the GMM components.
        color (str): Color for the ellipses and markers.
    """
    for i in range(gmm.n_components):
        # Extract the mean and covariance of the i-th component
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]

        # Plot the mean as a marker with a specific label
        axis.scatter(mean[0], mean[1], mean[2], c=color, s=150, label=f"{gmm_label} - Component {i + 1}", marker='x')

        # Generate confidence ellipsoid
        plot_ellipsoid_3D(mean, cov, axis, color=color, alpha=0.1)


def plot_ellipsoid_3D(mean, cov, axis, color, alpha=0.1):
    """
    Draws a 3D confidence ellipsoid for a Gaussian distribution.
    Parameters:
        mean (ndarray): Mean of the Gaussian distribution (3D vector).
        cov (ndarray): Covariance matrix (3x3).
        axis (Axes3D): The matplotlib 3D Axes object to draw on.
        color (str): Color for the ellipsoid.
        alpha (float): Transparency of the ellipsoid.
    """
    # Eigen decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Scale eigenvectors by sqrt of eigenvalues to get principal axes lengths
    radii = 2 * np.sqrt(eigenvalues)  # 2 standard deviations

    # Create a grid of points for a unit sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Combine sphere points into a single array
    sphere = np.stack([x, y, z], axis=-1)

    # Transform the sphere by the eigenvectors and scale by radii
    rotation = R.from_matrix(eigenvectors).as_matrix()
    ellipsoid = sphere @ (rotation.T * radii)

    # Translate to the mean
    ellipsoid += mean

    # Create a 3D polygon collection for the ellipsoid
    poly = Poly3DCollection([list(zip(ellipsoid[:, :, 0].flatten(),
                                      ellipsoid[:, :, 1].flatten(),
                                      ellipsoid[:, :, 2].flatten()))],
                            alpha=alpha, color=color)
    axis.add_collection3d(poly)


# Create the plot figure (3D)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c='blue', s=10, alpha=0.5, label='Data Points (GMM 1)')
ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='green', s=10, alpha=0.5, label='Data Points (GMM 2)')

# Plot each GMM with labeled components and ellipsoids
plot_gmm_3d_with_ellipsoids(gmm1, ax, gmm_label='GMM 1', color='red')
plot_gmm_3d_with_ellipsoids(gmm2, ax, gmm_label='GMM 2', color='orange')

# Configure the plot
ax.set_title("3D Gaussian Mixture Models with Confidence Ellipsoids", fontsize=16)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")
ax.legend(loc='upper left', fontsize=10)

# Add the Hellinger distance as text on the graph
hellinger_text = f"Hellinger Distance: {hellinger_dist:.4f}"
ax.text2D(0.95, 0.05, hellinger_text, transform=ax.transAxes, fontsize=12,
          verticalalignment='bottom', horizontalalignment='right',
          bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Display the plot
plt.show()
