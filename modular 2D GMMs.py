import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse


# Function to generate 2D data from a Gaussian distribution
def generate_data(mean, cov, size):
    # Generate multivariate normal data with specified mean and covariance matrix
    return np.random.multivariate_normal(mean, cov, size)


# Function to fit a GMM to the data
def fit_gmm(data, n_components=3, covariance_type='full'):
    # Fit a GMM to the data with specified number of components and covariance type
    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    gmm.fit(data)
    print("Means:\n", gmm.means_)  # Centers of the Gaussian components
    print("Covariances:\n", gmm.covariances_)  # Covariance matrices of the components
    print("Weights:\n", gmm.weights_)  # Mixing weights of the components
    return gmm


# Function to calculate the Hellinger distance between two GMMs
def hellinger_distance_gmm(gmm_1, gmm_2):
    # Hellinger distance calculation involves averaging covariances and computing determinants
    # These operations require covariances to be in a consistent form (full matrix)
    # converting diagonal or spherical covariances to their full equivalents
    def get_full_covariance(cov, covariance_type):
        if covariance_type == 'diag':
            return np.diag(cov)
        elif covariance_type == 'spherical':
            return np.eye(len(gmm_1.means_[0])) * cov
        return cov

    distance = 0
    for i in range(gmm_1.n_components):
        for j in range(gmm_2.n_components):
            # Extract the mean and covariance of the components
            mean1 = gmm_1.means_[i]
            mean2 = gmm_2.means_[j]

            # Handle covariance types for each GMM component
            cov1 = get_full_covariance(gmm_1.covariances_[i], gmm_1.covariance_type) \
                if gmm_1.covariance_type != 'tied' else gmm_1.covariances_
            cov2 = get_full_covariance(gmm_2.covariances_[j], gmm_2.covariance_type) \
                if gmm_2.covariance_type != 'tied' else gmm_2.covariances_

            # Compute average covariance and coefficients for the Hellinger formula
            cov_avg = (cov1 + cov2) / 2
            # Calculate the coefficient using determinants
            coeff = (np.linalg.det(cov1) ** 0.25 * np.linalg.det(cov2) ** 0.25) / (np.linalg.det(cov_avg) ** 0.5)
            # Calculate the exponential term based on the difference in means
            mean_diff = mean1 - mean2
            exp_term = -0.125 * mean_diff @ np.linalg.inv(cov_avg) @ mean_diff.T
            # Add contribution to the total distance
            distance += np.sqrt(coeff * np.exp(exp_term))

    # Normalize the distance by the number of components in each GMM
    return np.sqrt(1 - distance / (gmm_1.n_components * gmm_2.n_components))


# Function to plot GMM components and their ellipses
def plot_gmm(gmm, axis, label, color):
    """
        Parameters:
            gmm (GaussianMixture): The GMM to plot.
            axis (Axes): The matplotlib Axes object to draw on.
            label (str): Label for the GMM components.
            color (str): Color for the ellipses and markers.
    """
    def get_full_covariance(cov, covariance_type):
        """Convert covariance to full matrix if needed."""
        if covariance_type == 'diag':
            return np.diag(cov)
        elif covariance_type == 'spherical':
            return np.eye(len(gmm.means_[0])) * cov
        return cov

    for i in range(gmm.n_components):
        # Extract the mean and covariance of the i-th component
        mean = gmm.means_[i]
        if gmm.covariance_type == 'tied':
            cov = gmm.covariances_
        else:
            cov = get_full_covariance(gmm.covariances_[i], gmm.covariance_type)

        # Compute the eigenvalues and eigenvectors of the covariance matrix, Spread determined by the eigenvalues
        # of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Compute the angle of rotation using the arctangent of the eigenvectors' components
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi
        # Width and height of the ellipse based on eigenvalues
        width, height = 2 * np.sqrt(eigenvalues)
        # Create and add the ellipse to the plot
        ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                          edgecolor=color, facecolor='none', lw=2)
        # Center (xy): the mean of the component
        axis.add_patch(ellipse)  # Plot the center of the Gaussian component (mean) as an 'x' marker
    # Plot GMM component means
    axis.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c=color, s=100, label=label, marker='x')
    # gmm.means_[:, 0], gmm.means_[:, 1] = x (row) and y (col) coordinates of the center, s= size of the marker


# Function to execute and visualize a GMM scenario
def gmm_scenario(data1_params, data2_params, title, covariance_type='full'):
    # Generate data for two GMMs based on provided parameters
    data1 = np.vstack([generate_data(*params) for params in data1_params])
    data2 = np.vstack([generate_data(*params) for params in data2_params])

    # Fit GMMs to the generated data
    print("GMM 1:")
    gmm1 = fit_gmm(data1, covariance_type=covariance_type)
    print("\nGMM 2:")
    gmm2 = fit_gmm(data2, covariance_type=covariance_type)

    # Compute Hellinger distance between the two GMMs
    hellinger_dist = hellinger_distance_gmm(gmm1, gmm2)

    # Plot the data and GMM components
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data1[:, 0], data1[:, 1], c='blue', s=10, alpha=0.5, label='Data GMM 1')
    ax.scatter(data2[:, 0], data2[:, 1], c='green', s=10, alpha=0.5, label='Data GMM 2')
    # data[:, 0]: selects the first column, corresponding to the first feature (x-axis).
    # data[:, 1]: selects the second column, corresponding to the second feature (y-axis).
    # s=10: defines the size of the points in the scatter plot (a small size of 10 is used for visual clarity)
    # alpha=0.5: sets the transparency level of the points. A value of 0.5 ensures the points are semi-transparent,
    # helping avoid clutter when points overlap
    plot_gmm(gmm1, ax, label='GMM 1 Components', color='red')
    plot_gmm(gmm2, ax, label='GMM 2 Components', color='orange')

    # Add title, labels, legend, and Hellinger distance annotation
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    # Add the Hellinger distance as text on the graph
    hellinger_text = f"Hellinger Distance: {hellinger_dist:.4f}"
    ax.text(0.05, 0.95, hellinger_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))
    plt.grid(True)
    plt.show()


# Define parameters for scenarios
# Combine three Gaussian clusters into a single dataset, each param has 2 data sets
original_params = [
    ([[2, 2], [[1, 0.5], [0.5, 1]], 100],
     [[5, 5], [[1, -0.3], [-0.3, 1]], 100],
     [[8, 2], [[1, 0], [0, 1]], 100]),
    ([[1, 1], [[1, 0.2], [0.2, 1]], 100],
     [[6, 6], [[1, 0.4], [0.4, 1]], 100],
     [[9, 1], [[1, 0.1], [0.1, 1]], 100])
]

further_apart_params = [
    ([[2, 2], [[1, 0.5], [0.5, 1]], 100],
     [[5, 5], [[1, -0.3], [-0.3, 1]], 100],
     [[8, 2], [[1, 0], [0, 1]], 100]),
    ([[10, 10], [[1, 0.2], [0.2, 1]], 100],
     [[15, 15], [[1, 0.4], [0.4, 1]], 100],
     [[18, 5], [[1, 0.1], [0.1, 1]], 100])
]

closer_params = [
    ([[2, 2], [[1, 0.5], [0.5, 1]], 100],
     [[5, 5], [[1, -0.3], [-0.3, 1]], 100],
     [[8, 2], [[1, 0], [0, 1]], 100]),
    ([[2.5, 2.5], [[1, 0.2], [0.2, 1]], 100],
     [[5.5, 5.5], [[1, 0.4], [0.4, 1]], 100],
     [[8.5, 2.5], [[1, 0.1], [0.1, 1]], 100])
]

# Execute scenarios
gmm_scenario(*original_params, title="Original GMMs - Full Covariance", covariance_type='full')
gmm_scenario(*original_params, title="Original GMMs - Spherical Covariance", covariance_type='spherical')
gmm_scenario(*original_params, title="Original GMMs - Diagonal Covariance", covariance_type='diag')
gmm_scenario(*original_params, title="Original GMMs - Tied Covariance", covariance_type='tied')
gmm_scenario(*further_apart_params, title="Original GMMs - Full Covariance and Further Apart")
gmm_scenario(*closer_params, title="Original GMMs - Full Covariance and Closer Together")
