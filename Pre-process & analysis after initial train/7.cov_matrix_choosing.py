# Limit CPU usage to avoid overload when using joblib/multiprocessing
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "5"

# Standard data science imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import seaborn as sns  # Imported but not used; can be removed if unused

# === Data Preprocessing ===
def load_and_prepare_data(file_path, joint_features):
    df = pd.read_excel(file_path)  # Load Excel file
    df["Time"] = df["sample"] * 0.01  # Convert sample index to time in seconds

    features_to_normalize = joint_features + ['Time']

    # Normalize joint features + time per POS group
    def normalize_group(group):
        scaler = StandardScaler()
        group[features_to_normalize] = scaler.fit_transform(group[features_to_normalize])
        return group

    df = df.groupby('POS', group_keys=False).apply(normalize_group)
    return df

# === GMM Fitting and BIC Scoring ===
def fit_gmm_per_pos(df, joint_features, components_range, cov_types):
    bic_results = {}

    # Fit GMM for each POS separately
    for pos, group in df.groupby('POS'):
        X = group[joint_features + ['Time']].values  # Feature matrix
        bic_scores = []

        for cov_type in cov_types:
            for n_components in components_range:
                try:
                    # Fit GMM with given number of components and covariance type
                    gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=0)
                    gmm.fit(X)
                    bic = gmm.bic(X)
                    bic_scores.append((bic, n_components, cov_type))
                except Exception as e:
                    # Catch fitting errors (e.g., singular covariance) and continue
                    print(f"Failed to fit GMM for POS {pos}, n_components={n_components}, cov_type={cov_type}: {e}")

        # Sort by BIC (lower is better)
        bic_scores.sort()
        best_bic, best_n, best_cov = bic_scores[0]
        print(f"POS {pos}: Best cov_type = {best_cov}, components = {best_n}, BIC = {best_bic:.2f}")

        bic_results[pos] = bic_scores

    return bic_results

# === BIC Plotting ===
def plot_global_bic(bic_results, components_range, output_path, cov_types):
    plt.figure(figsize=(15, 25))
    num_positions = len(bic_results)
    cols = 5
    rows = int(np.ceil(num_positions / cols))

    for idx, (pos, scores) in enumerate(sorted(bic_results.items()), 1):
        plt.subplot(rows, cols, idx)
        data = pd.DataFrame(scores, columns=['BIC', 'n_components', 'cov_type'])

        # Plot a BIC curve for each covariance type
        for cov_type in cov_types:
            subset = data[data['cov_type'] == cov_type].sort_values(by='n_components')
            if not subset.empty:
                plt.plot(subset['n_components'], subset['BIC'], marker='o', label=cov_type)

        plt.title(f'POS {pos}')
        plt.xlabel('n_components')
        plt.ylabel('BIC')
        plt.legend()

    plt.tight_layout()
    plt.suptitle("BIC Scores per POS for Different Covariance Types", y=1.02, fontsize=16)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save and show the figure
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


# === CONFIG ===
# joint_features = ['Joint 1 Pos','Joint 2 Pos','Joint 3 Pos', 'Joint 5 Pos']
#joint_features = ['Joint 1 Pos','Joint 3 Pos', 'Joint 5 Pos']
joint_features = ['Joint 1 Pos','Joint 4 Pos', 'Joint 5 Pos']
file_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\data_until_lift_start.xlsx"
# cov_types = ['tied', 'diag', 'spherical','full']
cov_types = ['tied', 'diag', 'spherical']
components_range = range(3, 11)

# === RUN PIPELINE ===
df = load_and_prepare_data(file_path, joint_features)
bic_results = fit_gmm_per_pos(df, joint_features, components_range, cov_types)
plot_global_bic(bic_results, components_range, r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\bic_resoults", cov_types)

