import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt


# Function to calculate the Hellinger distance between two PDFs.
def hellinger_distance(pdf1, pdf2):
    """
        Calculate the Hellinger distance between two PDFs.

        Parameters:
        - pdf1: PDF values from the first GMM.
        - pdf2: PDF values from the second GMM.

        Returns:
        - Hellinger distance.
        """
    pdf1 /= np.sum(pdf1)  # Normalize pdf1
    pdf2 /= np.sum(pdf2)  # Normalize pdf2
    return np.sqrt(1 - np.sum(np.sqrt(pdf1 * pdf2)))


# Generate synthetic data for two GMMs
data1 = np.random.normal(loc=0, scale=1, size=(100, 3))  # Data for GMM 1
data2 = np.random.normal(loc=2, scale=1.5, size=(100, 3))  # Data for GMM 2

# Fit GMMs with three components to the data.
gmm1 = GaussianMixture(n_components=3)
gmm1.fit(data1)

print("GMM 1:")
print("Weights:", gmm1.weights_)
print("Means:", gmm1.means_)
print("Covariances:", gmm1.covariances_)


gmm2 = GaussianMixture(n_components=3)
gmm2.fit(data2)

print("\nGMM 2:")
print("Weights:", gmm2.weights_)
print("Means:", gmm2.means_)
print("Covariances:", gmm2.covariances_)


# Evaluate PDFs of the GMMs on a 3D grid
X = np.linspace(-5, 7, 50)  # Define a range for each dimension
grid = np.array(np.meshgrid(X, X, X)).T.reshape(-1, 3)  # Create a 3D grid of points
PDF1 = np.exp(gmm1.score_samples(grid)) + 1e-10  # Add epsilon to stabilize calculations for PDF1
PDF2 = np.exp(gmm2.score_samples(grid)) + 1e-10  # Add epsilon to stabilize calculations for PDF2

# Calculate the Hellinger distance.
distance = hellinger_distance(PDF1, PDF2)
print("\nHellinger Distance between GMM 1 and GMM 2:", distance)


# Function to plot 3D ellipsoid for each GMM component
def plot_ellipsoid(mean, cov, ax, color, alpha=0.3):
    """Plot a 3D ellipsoid representing a GMM component"""
    u, s, vh = np.linalg.svd(cov)
    radii = np.sqrt(s)
    # Generate sphere points
    phi = np.linspace(0, 2 * np.pi, 100)
    theta = np.linspace(0, np.pi, 50)
    x = radii[0] * np.outer(np.cos(phi), np.sin(theta))
    y = radii[1] * np.outer(np.sin(phi), np.sin(theta))
    z = radii[2] * np.outer(np.ones_like(phi), np.cos(theta))

    for i in range(len(x)):
        for j in range(len(x[i])):
            xyz = np.dot(u, np.array([x[i][j], y[i][j], z[i][j]])) + mean
            x[i][j], y[i][j], z[i][j] = xyz

    ax.plot_surface(x, y, z, color=color, alpha=alpha)


# Plot GMMs and ellipsoids
fig = plt.figure(figsize=(10, 10))
axis = fig.add_subplot(111, projection='3d')

# Plot data and ellipsoids for GMM 1
axis.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c='blue', label='GMM 1 Data', alpha=0.5)
for means, covariance in zip(gmm1.means_, gmm1.covariances_):
    plot_ellipsoid(means, covariance, axis, color='blue')

# Plot data and ellipsoids for GMM 2
axis.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='red', label='GMM 2 Data', alpha=0.5)
for means, covariance in zip(gmm2.means_, gmm2.covariances_):
    plot_ellipsoid(means, covariance, axis, color='red')

# Annotate with Hellinger distance.
axis.text2D(0.05, 0.95, f"Hellinger Distance: {distance:.4f}", transform=axis.transAxes)

axis.set_title("GMM 3D Plot with Ellipsoids")
axis.set_xlabel("X")
axis.set_ylabel("Y")
axis.set_zlabel("Z")
axis.legend(loc='upper right')
plt.show()
