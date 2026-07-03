import os
os.environ["LOKY_MAX_CPU_COUNT"] = "5"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
import seaborn as sns

def load_and_prepare_data(file_path, joint_features):
    df = pd.read_excel(file_path)
    # Convert 'sample' column to time in seconds (0.01s per sample)
    df["Time"] = df["sample"] * 0.01
    features_to_normalize = joint_features + ['Time']

    # Normalize features within each position (POS group)
    def normalize_group(group):
        scaler = StandardScaler()
        group[features_to_normalize] = scaler.fit_transform(group[features_to_normalize])
        return group

    df = df.groupby('POS', group_keys=False).apply(normalize_group)
    return df

def fit_gmm_per_pos(df, joint_features, components_range, covariance_type='full', plot=True, plot_dir="bic_plots"):
    pos_gmms = {}  # Store the best model per POS
    bic_scores_all = []  # To store all BIC scores for global plotting
    os.makedirs(plot_dir, exist_ok=True)  # Ensure the plot directory exists
    features = joint_features + ['Time']  # Features used for GMM

    # Process each position (POS) separately
    for pos, group in df.groupby('POS'):
        X = group[features].values
        bic_scores = []      # BIC values for this POS
        models = []          # Fitted GMMs for each n_components

        # Try different number of GMM components
        for n in components_range:
            gmm = GaussianMixture(n_components=n, covariance_type=covariance_type, random_state=42)
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            models.append(gmm)

        bic_scores_all.append(bic_scores)

        # Find the elbow in the BIC curve (or fallback to min BIC)
        knee = KneeLocator(components_range, bic_scores, curve='convex', direction='decreasing')
        elbow = knee.knee or components_range[np.argmin(bic_scores)]

        # Save the selected model for this POS
        pos_gmms[pos] = models[components_range.index(elbow)]

        print(f"POS {pos}: Selected components = {elbow}, BIC = {bic_scores[components_range.index(elbow)]:.2f}")

        # Plot the BIC curve for this POS
        if plot:
            plt.figure(figsize=(6, 4))
            plt.plot(components_range, bic_scores, marker='o', label='BIC Score')
            if knee.knee:
                plt.axvline(x=elbow, color='red', linestyle='--', label=f'Elbow = {elbow}')
            plt.xlabel('Number of Components')
            plt.ylabel('BIC Score')
            plt.title(f'POS {pos} - BIC vs Components')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/POS_{pos}_bic_plot.png")
            plt.close()

    return pos_gmms, np.array(bic_scores_all)


def plot_global_bic(bic_scores_all, components_range, output_path):
    # Average BIC across positions for each component count
    avg_bic = np.mean(bic_scores_all, axis=0)

    # Use knee detection on the average BIC curve
    knee = KneeLocator(components_range, avg_bic, curve='convex', direction='decreasing')
    elbow = knee.knee or components_range[np.argmin(avg_bic)]

    # Plot the global BIC curve
    plt.figure(figsize=(8, 5))
    plt.plot(components_range, avg_bic, marker='o', label='Average BIC Score')
    if knee.knee:
        plt.axvline(x=elbow, color='red', linestyle='--', label=f'Elbow = {elbow}')
    plt.xlabel("Number of Components")
    plt.ylabel("Average BIC Score")
    plt.title("Average BIC Score vs GMM Components (All Positions)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


# === CONFIG ===
#joint_features = ['Joint 1 Pos','Joint 2 Pos','Joint 3 Pos','Joint 5 Pos']
#joint_features = ['Joint 1 Pos','Joint 3 Pos','Joint 5 Pos']
joint_features = ['Joint 1 Pos','Joint 4 Pos', 'Joint 5 Pos']
file_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\data_until_lift_start.xlsx"
components_range = range(3, 11)
covariance_type = 'diag'


# === RUN PIPELINE ===
df = load_and_prepare_data(file_path, joint_features)
pos_gmms, bic_scores_all = fit_gmm_per_pos(df, joint_features, components_range, covariance_type)
plot_global_bic(bic_scores_all, components_range, r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\global_average_bic_plot.png")
