import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# ------------------------------------------2D GMMs before change in distance--------------------------------------


# Function to generate 2D data from a Gaussian distribution
def generate_data(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)


# Generate synthetic data for GMM 1 and GMM 2
np.random.seed(42)  # Random seed
data1 = np.vstack([  # Combine three Gaussian clusters into a single dataset
    generate_data([2, 2], [[1, 0.5], [0.5, 1]], 100),
    generate_data([5, 5], [[1, -0.3], [-0.3, 1]], 100),
    generate_data([8, 2], [[1, 0], [0, 1]], 100)
])
data2 = np.vstack([
    generate_data([1, 1], [[1, 0.2], [0.2, 1]], 100),
    generate_data([6, 6], [[1, 0.4], [0.4, 1]], 100),
    generate_data([9, 1], [[1, 0.1], [0.1, 1]], 100)
])

# Fit GMMS to the data
# covariance_type='full'- each Gaussian component has its own full covariance matrix
# when having sufficient data and need to model complex shapes or correlations
gmm1 = GaussianMixture(n_components=3, covariance_type='full')
gmm1.fit(data1)

gmm2 = GaussianMixture(n_components=3, covariance_type='full')
gmm2.fit(data2)

# Print GMM attributes
print("GMM 1:")
print("Means:\n", gmm1.means_)  # Centers of the Gaussian components
print("Covariances:\n", gmm1.covariances_)  # Covariance matrices of the components
print("Weights:\n", gmm1.weights_)  # Mixing weights of the components
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


# Function to plot GMM components
def plot_gmm(gmm, axis, label, color):
    """
       Parameters:
           gmm (GaussianMixture): The GMM to plot.
           axis (Axes): The matplotlib Axes object to draw on.
           label (str): Label for the GMM components.
           color (str): Color for the ellipses and markers.
       """
    for i in range(gmm.n_components):
        # Extract the mean and covariance of the i-th component
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]

        # Compute the eigenvalues and eigenvectors of the covariance matrix, Spread determined by the eigenvalues
        # of the covariance matrix

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Compute the angle of rotation using the arctangent of the eigenvectors' components
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi

        # Width and height of the ellipse based on eigenvalues
        width, height = 2 * np.sqrt(eigenvalues)  # 2 standard deviations along the axes ellipse

        # Create and add the ellipse to the plot
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            edgecolor=color, facecolor='none', lw=2
        )  # Center (xy): the mean of the component

        axis.add_patch(ellipse)  # Plot the center of the Gaussian component (mean) as an 'x' marker

    # Plot the means as markers
    axis.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c=color, s=100, label=label, marker='x')
    # gmm.means_[:, 0], gmm.means_[:, 1] = x (row) and y (col) coordinates of the center, s= size of the marker


# Create the plot figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data points and the GMM components
ax.scatter(data1[:, 0], data1[:, 1], c='blue', s=10, alpha=0.5, label='Data GMM 1')
ax.scatter(data2[:, 0], data2[:, 1], c='green', s=10, alpha=0.5, label='Data GMM 2')
# data[:, 0]: selects the first column, corresponding to the first feature (x-axis).
# data[:, 1]: selects the second column, corresponding to the second feature (y-axis).
# s=10: defines the size of the points in the scatter plot (a small size of 10 is used for visual clarity)
# alpha=0.5: sets the transparency level of the points. A value of 0.5 ensures the points are semi-transparent,
# helping avoid clutter when points overlap

plot_gmm(gmm1, ax, label='GMM 1 Components', color='red')
plot_gmm(gmm2, ax, label='GMM 2 Components', color='orange')

# Configure the plot
ax.set_title("GMMs")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Add the Hellinger distance as text on the graph
hellinger_text = f"Hellinger Distance: {hellinger_dist:.4f}"
ax.text(0.05, 0.95, hellinger_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Display the grid and plot
plt.grid(True)
plt.show()


# ------------------------------------------2D GMMs after moving the data sets further apart-------------------


# Function to generate 2D data from a Gaussian distribution
def generate_data(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)


# Generate synthetic data for GMM 1 and GMM 2
np.random.seed(42)  # Random seed
data1 = np.vstack([  # Combine three Gaussian clusters into a single dataset
    generate_data([2, 2], [[1, 0.5], [0.5, 1]], 100),
    generate_data([5, 5], [[1, -0.3], [-0.3, 1]], 100),
    generate_data([8, 2], [[1, 0], [0, 1]], 100)
])
# Modify the means in data2 to increase separation from data1
data2 = np.vstack([
    generate_data([10, 10], [[1, 0.2], [0.2, 1]], 100),  # Move further in the upper-right direction
    generate_data([15, 15], [[1, 0.4], [0.4, 1]], 100),  # Move further diagonally
    generate_data([18, 5], [[1, 0.1], [0.1, 1]], 100)   # Move further right
])


# Fit GMMS to the data
# covariance_type='full'- each Gaussian component has its own full covariance matrix
# when having sufficient data and need to model complex shapes or correlations
gmm1 = GaussianMixture(n_components=3, covariance_type='full')
gmm1.fit(data1)

gmm2 = GaussianMixture(n_components=3, covariance_type='full')
gmm2.fit(data2)

# Print GMM attributes
print("GMM 1:")
print("Means:\n", gmm1.means_)  # Centers of the Gaussian components
print("Covariances:\n", gmm1.covariances_)  # Covariance matrices of the components
print("Weights:\n", gmm1.weights_)  # Mixing weights of the components
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


# Function to plot GMM components
def plot_gmm(gmm, axis, label, color):
    """
       Parameters:
           gmm (GaussianMixture): The GMM to plot.
           axis (Axes): The matplotlib Axes object to draw on.
           label (str): Label for the GMM components.
           color (str): Color for the ellipses and markers.
       """
    for i in range(gmm.n_components):
        # Extract the mean and covariance of the i-th component
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]

        # Compute the eigenvalues and eigenvectors of the covariance matrix, Spread determined by the eigenvalues
        # of the covariance matrix

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Compute the angle of rotation using the arctangent of the eigenvectors' components
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi

        # Width and height of the ellipse based on eigenvalues
        width, height = 2 * np.sqrt(eigenvalues)  # 2 standard deviations along the axes ellipse

        # Create and add the ellipse to the plot
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            edgecolor=color, facecolor='none', lw=2
        )  # Center (xy): the mean of the component

        axis.add_patch(ellipse)  # Plot the center of the Gaussian component (mean) as an 'x' marker

    # Plot the means as markers
    axis.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c=color, s=100, label=label, marker='x')
    # gmm.means_[:, 0], gmm.means_[:, 1] = x (row) and y (col) coordinates of the center, s= size of the marker


# Create the plot figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data points and the GMM components
ax.scatter(data1[:, 0], data1[:, 1], c='blue', s=10, alpha=0.5, label='Data GMM 1')
ax.scatter(data2[:, 0], data2[:, 1], c='green', s=10, alpha=0.5, label='Data GMM 2')
# data[:, 0]: selects the first column, corresponding to the first feature (x-axis).
# data[:, 1]: selects the second column, corresponding to the second feature (y-axis).
# s=10: defines the size of the points in the scatter plot (a small size of 10 is used for visual clarity)
# alpha=0.5: sets the transparency level of the points. A value of 0.5 ensures the points are semi-transparent,
# helping avoid clutter when points overlap

plot_gmm(gmm1, ax, label='GMM 1 Components', color='red')
plot_gmm(gmm2, ax, label='GMM 2 Components', color='orange')

# Configure the plot
ax.set_title("GMMs of more further apart data sets")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Add the Hellinger distance as text on the graph
hellinger_text = f"Hellinger Distance: {hellinger_dist:.4f}"
ax.text(0.05, 0.95, hellinger_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Display the grid and plot
plt.grid(True)
plt.show()

# ------------------------------------------2D GMMs after moving the data sets further apart-------------------


# Function to generate 2D data from a Gaussian distribution
def generate_data(mean, cov, size):
    return np.random.multivariate_normal(mean, cov, size)


# Generate synthetic data for GMM 1 and GMM 2
np.random.seed(42)  # Random seed
data1 = np.vstack([  # Combine three Gaussian clusters into a single dataset
    generate_data([2, 2], [[1, 0.5], [0.5, 1]], 100),
    generate_data([5, 5], [[1, -0.3], [-0.3, 1]], 100),
    generate_data([8, 2], [[1, 0], [0, 1]], 100)
])
# Modify the means in data2 to decrease separation from data1
data2 = np.vstack([
    generate_data([2.5, 2.5], [[1, 0.2], [0.2, 1]], 100),  # Close to the first component of data1
    generate_data([5.5, 5.5], [[1, 0.4], [0.4, 1]], 100),  # Close to the second component of data1
    generate_data([8.5, 2.5], [[1, 0.1], [0.1, 1]], 100)   # Close to the third component of data1
])


# Fit GMMS to the data
# covariance_type='full'- each Gaussian component has its own full covariance matrix
# when having sufficient data and need to model complex shapes or correlations
gmm1 = GaussianMixture(n_components=3, covariance_type='full')
gmm1.fit(data1)

gmm2 = GaussianMixture(n_components=3, covariance_type='full')
gmm2.fit(data2)

# Print GMM attributes
print("GMM 1:")
print("Means:\n", gmm1.means_)  # Centers of the Gaussian components
print("Covariances:\n", gmm1.covariances_)  # Covariance matrices of the components
print("Weights:\n", gmm1.weights_)  # Mixing weights of the components
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


# Function to plot GMM components
def plot_gmm(gmm, axis, label, color):
    """
       Parameters:
           gmm (GaussianMixture): The GMM to plot.
           axis (Axes): The matplotlib Axes object to draw on.
           label (str): Label for the GMM components.
           color (str): Color for the ellipses and markers.
       """
    for i in range(gmm.n_components):
        # Extract the mean and covariance of the i-th component
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]

        # Compute the eigenvalues and eigenvectors of the covariance matrix, Spread determined by the eigenvalues
        # of the covariance matrix

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Compute the angle of rotation using the arctangent of the eigenvectors' components
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]) * 180 / np.pi

        # Width and height of the ellipse based on eigenvalues
        width, height = 2 * np.sqrt(eigenvalues)  # 2 standard deviations along the axes ellipse

        # Create and add the ellipse to the plot
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            edgecolor=color, facecolor='none', lw=2
        )  # Center (xy): the mean of the component

        axis.add_patch(ellipse)  # Plot the center of the Gaussian component (mean) as an 'x' marker

    # Plot the means as markers
    axis.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c=color, s=100, label=label, marker='x')
    # gmm.means_[:, 0], gmm.means_[:, 1] = x (row) and y (col) coordinates of the center, s= size of the marker


# Create the plot figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the data points and the GMM components
ax.scatter(data1[:, 0], data1[:, 1], c='blue', s=10, alpha=0.5, label='Data GMM 1')
ax.scatter(data2[:, 0], data2[:, 1], c='green', s=10, alpha=0.5, label='Data GMM 2')
# data[:, 0]: selects the first column, corresponding to the first feature (x-axis).
# data[:, 1]: selects the second column, corresponding to the second feature (y-axis).
# s=10: defines the size of the points in the scatter plot (a small size of 10 is used for visual clarity)
# alpha=0.5: sets the transparency level of the points. A value of 0.5 ensures the points are semi-transparent,
# helping avoid clutter when points overlap

plot_gmm(gmm1, ax, label='GMM 1 Components', color='red')
plot_gmm(gmm2, ax, label='GMM 2 Components', color='orange')

# Configure the plot
ax.set_title("GMMs of more closer data sets")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

# Add the Hellinger distance as text on the graph
hellinger_text = f"Hellinger Distance: {hellinger_dist:.4f}"
ax.text(0.05, 0.95, hellinger_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

# Display the grid and plot
plt.grid(True)
plt.show()

