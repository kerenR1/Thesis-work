import os
os.environ['LOKY_MAX_CPU_COUNT'] = '5'

import pandas as pd
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# ----------------------
# Data Preparation
# ----------------------
def load_and_normalize_data(file_path, joint_features):
    df = pd.read_excel(file_path)
    df["Time"] = df["sample"] * 0.01
    features_to_normalize = joint_features + ['Time']

    def normalize_group(group):
        scaler = StandardScaler()
        group[features_to_normalize] = scaler.fit_transform(group[features_to_normalize])
        return group

    df = df.groupby('POS', group_keys=False).apply(normalize_group)
    return df

# ----------------------
# Fit GMM for each POS
# ----------------------
def fit_gmms(df, joint_features, n_components=5, covariance_type='tied', pos_list=None):
    gmm_models = {}
    features = joint_features + ['Time']
    positions = sorted(df['POS'].unique()) if pos_list is None else pos_list

    for pos in positions:
        data = df[df['POS'] == pos][features].values
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=0)
        gmm.fit(data)

        if gmm.converged_:
            gmm_models[pos] = gmm
        else:
            print(f"Warning: GMM did not converge for POS {pos}")
    
    return gmm_models, positions

# ----------------------
# Compute Hellinger Distance
# ----------------------
def gmm_hellinger_distance(gmm1, gmm2, n_samples=1000):
    X1, _ = gmm1.sample(n_samples)
    p1 = np.exp(gmm1.score_samples(X1))
    p2 = np.exp(gmm2.score_samples(X1))

    p1 /= p1.sum()
    p2 /= p2.sum()

    return np.sqrt(0.5 * np.sum((np.sqrt(p1) - np.sqrt(p2))**2))

def compute_hellinger_matrix(gmm_models, positions):
    N = len(positions)
    matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(i + 1, N):
            d = gmm_hellinger_distance(gmm_models[positions[i]], gmm_models[positions[j]])
            matrix[i, j] = matrix[j, i] = d
    
    return matrix

# ----------------------
# Plot Dendrogram
# ----------------------
def plot_dendrogram(distance_matrix, positions, title="Hierarchical Clustering using Hellinger Distance"):
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')

    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=[f'POS {p}' for p in positions], orientation='top', distance_sort='descending')
    plt.title(title)
    plt.xlabel("Position")
    plt.ylabel("Hellinger Distance")
    plt.tight_layout()
    plt.show()

    # Extract cluster assignments
    #cluster_assignments = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    #pos_to_cluster = {pos: cluster for pos, cluster in zip(positions, cluster_assignments)}

    #return pos_to_cluster



# ----------------------
# Export Results
# ----------------------
def export_results(hellinger_matrix, positions, pos_to_cluster, output_excel_path):
    hellinger_df = pd.DataFrame(hellinger_matrix, index=positions, columns=positions)
    cluster_df = pd.DataFrame({
        'POS': positions,
        'Cluster': [pos_to_cluster[pos] for pos in positions]
    })

    with pd.ExcelWriter(output_excel_path) as writer:
        hellinger_df.to_excel(writer, sheet_name='Hellinger Matrix')
        cluster_df.to_excel(writer, sheet_name='Cluster Assignments', index=False)

    print(f"Results exported to: {output_excel_path}")

# def compute_within_cluster_hd(hellinger_matrix, positions, pos_to_cluster):
#     cluster_to_positions = {}
#     for pos, cluster in pos_to_cluster.items():
#         cluster_to_positions.setdefault(cluster, []).append(pos)
    
#     results = {}

#     for cluster_id, pos_list in cluster_to_positions.items():
#         if len(pos_list) < 2:
#             results[cluster_id] = np.nan
#             continue

#         indices = [positions.index(p) for p in pos_list]
#         dists = [hellinger_matrix[i, j] for i in indices for j in indices if i < j]

#         results[cluster_id] = {
#             'mean_HD': np.mean(dists),
#             'std_HD': np.std(dists),
#             'count': len(dists)
#         }
    
#     return pd.DataFrame.from_dict(results, orient='index').rename_axis('Cluster')


# def compute_within_group_hd(hellinger_matrix, positions, grouping):
#     results = {}

#     for group_name, pos_list in grouping.items():
#         indices = [positions.index(p) for p in pos_list if p in positions]
#         if len(indices) < 2:
#             results[group_name] = {'mean_HD': np.nan, 'std_HD': np.nan, 'count': 0}
#             continue

#         dists = [hellinger_matrix[i, j] for i in indices for j in indices if i < j]
#         results[group_name] = {
#             'mean_HD': np.mean(dists),
#             'std_HD': np.std(dists),
#             'count': len(dists)
#         }

#     return pd.DataFrame.from_dict(results, orient='index').rename_axis('Group')

# def compute_between_group_hd(hellinger_matrix, positions, grouping):
#     group_names = list(grouping.keys())
#     results = {}

#     for i in range(len(group_names)):
#         for j in range(i + 1, len(group_names)):
#             group1 = grouping[group_names[i]]
#             group2 = grouping[group_names[j]]

#             indices1 = [positions.index(p) for p in group1 if p in positions]
#             indices2 = [positions.index(p) for p in group2 if p in positions]

#             dists = [hellinger_matrix[i1, i2] for i1 in indices1 for i2 in indices2]
#             results[f"{group_names[i]} vs {group_names[j]}"] = {
#                 'mean_HD': np.mean(dists),
#                 'std_HD': np.std(dists),
#                 'count': len(dists)
#             }

#     return pd.DataFrame.from_dict(results, orient='index').rename_axis('Group Comparison')

# within_group_data = {
#     'Green': {'mean_HD': 0.295, 'std_HD': 0.078},
#     'Purple': {'mean_HD': 0.327, 'std_HD': 0.127},
#     'Orange': {'mean_HD': 0.376, 'std_HD': 0.115},
#     'Red': {'mean_HD': 0.513, 'std_HD': 0.132}
# }

# between_group_data = {
#     'Orange vs Green': {'mean_HD': 0.476, 'std_HD': 0.075},
#     'Green vs Purple': {'mean_HD': 0.487, 'std_HD': 0.125}, 
#     'Orange vs Purple': {'mean_HD': 0.546, 'std_HD': 0.12},  
#     'Orange vs Red': {'mean_HD': 0.561, 'std_HD': 0.127},  
#     'Green vs Red': {'mean_HD': 0.565, 'std_HD': 0.081},  
#     'Red vs Purple': {'mean_HD': 0.708, 'std_HD': 0.122}
# }

# # Create the visualization
# def plot_behavioral_groups_hd_analysis():
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
#     # Define colors for consistency
#     group_colors = {
#         'Green': '#2E7D32',    # Dark green
#         'Purple': '#7B1FA2',   # Dark purple
#         'Orange': '#F57C00',   # Dark orange
#         'Red': '#C62828'       # Dark red
#     }
    
#     # ============== Within-Group Analysis ==============
#     groups = ['Green', 'Purple', 'Orange', 'Red']  # Ordered by homogeneity
#     means = [within_group_data[g]['mean_HD'] for g in groups]
#     stds = [within_group_data[g]['std_HD'] for g in groups]
#     colors = [group_colors[g] for g in groups]
    
#     bars1 = ax1.bar(range(len(groups)), means, yerr=stds, 
#                     capsize=8, alpha=0.8, color=colors,
#                     edgecolor='black', linewidth=1.2)
    
#     ax1.set_xlabel('Behavioral Groups', fontsize=14, fontweight='bold')
#     ax1.set_ylabel('Hellinger Distance', fontsize=14, fontweight='bold')
#     ax1.set_title('Within-Group Analysis\n(Internal Homogeneity)', fontsize=16, fontweight='bold')
#     ax1.set_xticks(range(len(groups)))
#     ax1.set_xticklabels(groups, fontsize=12, fontweight='bold')
#     ax1.grid(True, alpha=0.3, axis='y')
#     ax1.set_ylim(0, 0.7)
    
#     # Add value annotations on bars
#     for i, (bar, mean, std) in enumerate(zip(bars1, means, stds)):
#         height = bar.get_height()
#         ax1.annotate(f'{mean:.3f} ± {std:.3f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01),
#                     ha='center', va='bottom', fontsize=11, fontweight='bold')
    
#     # Add ranking annotation
#     ax1.text(0.02, 0.98, 'Homogeneity Ranking:\nGreen > Purple > Orange > Red', 
#              transform=ax1.transAxes, fontsize=12, fontweight='bold',
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
#     comparisons = list(between_group_data.keys())
#     between_means = [between_group_data[c]['mean_HD'] for c in comparisons]
#     between_stds = [between_group_data[c]['std_HD'] for c in comparisons]
    
#     # Create gradient colors for between-group comparisons
#     between_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(comparisons)))
    
#     bars2 = ax2.bar(range(len(comparisons)), between_means, yerr=between_stds,
#                     capsize=6, alpha=0.8, color=between_colors,
#                     edgecolor='black', linewidth=1)
    
#     ax2.set_xlabel('Group Comparisons', fontsize=14, fontweight='bold')
#     ax2.set_ylabel('Hellinger Distance', fontsize=14, fontweight='bold')
#     ax2.set_title('Between-Group Analysis\n(Behavioral Distinctiveness)', fontsize=16, fontweight='bold')
#     ax2.set_xticks(range(len(comparisons)))
#     ax2.set_xticklabels(comparisons, rotation=45, ha='right', fontsize=11)
#     ax2.grid(True, alpha=0.3, axis='y')
#     ax2.set_ylim(0, 0.85)
    
#     # Add value annotations on bars
#     for i, (bar, mean, std) in enumerate(zip(bars2, between_means, between_stds)):
#         height = bar.get_height()
#         ax2.annotate(f'{mean:.3f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01),
#                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
#     # Highlight most similar and most distinct pairs
#     ax2.axhline(y=min(between_means), color='green', linestyle='--', alpha=0.7, linewidth=2)
#     ax2.axhline(y=max(between_means), color='red', linestyle='--', alpha=0.7, linewidth=2)
    
#     ax2.text(0.02, 0.98, 'Most Similar: Orange vs Green\nMost Distinct: Red vs Purple', 
#              transform=ax2.transAxes, fontsize=12, fontweight='bold',
#              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
#     plt.tight_layout()
#     plt.show()
    
#     # Print detailed summary
#     print("=" * 80)
#     print("BEHAVIORAL GROUPS HELLINGER DISTANCE ANALYSIS")
#     print("=" * 80)
#     print("\nWITHIN-GROUP ANALYSIS (Internal Homogeneity):")
#     print("-" * 50)
#     for group in groups:
#         data = within_group_data[group]
#         print(f"{group:>8}: HD = {data['mean_HD']:.3f} ± {data['std_HD']:.3f}")
    
#     print(f"\nHomogeneity Ranking: {' > '.join(groups)}")
    
#     print("\nBETWEEN-GROUP ANALYSIS (Behavioral Distinctiveness):")
#     print("-" * 50)
#     sorted_comparisons = sorted(between_group_data.items(), key=lambda x: x[1]['mean_HD'])
#     for comp, data in sorted_comparisons:
#         print(f"{comp:>20}: HD = {data['mean_HD']:.3f} ± {data['std_HD']:.3f}")
    
#     print(f"\nMost Similar Pair: {sorted_comparisons[0][0]} (HD = {sorted_comparisons[0][1]['mean_HD']:.3f})")
#     print(f"Most Distinct Pair: {sorted_comparisons[-1][0]} (HD = {sorted_comparisons[-1][1]['mean_HD']:.3f})")
    
#     # Calculate separation index
#     avg_within = np.mean([data['mean_HD'] for data in within_group_data.values()])
#     avg_between = np.mean([data['mean_HD'] for data in between_group_data.values()])
#     separation_index = avg_between / avg_within
    
#     print(f"\nSEPARATION ANALYSIS:")
#     print(f"Average Within-Group HD: {avg_within:.3f}")
#     print(f"Average Between-Group HD: {avg_between:.3f}")
#     print(f"Separation Index (Between/Within): {separation_index:.2f}")
#     print("=" * 80)

# ----------------------
# Run Full Pipeline
# ----------------------
def run_pipeline(file_path, joint_features, pos_limit=30):
    df = load_and_normalize_data(file_path, joint_features)
    gmm_models, positions = fit_gmms(df, joint_features, pos_list=sorted(df['POS'].unique())[:pos_limit])
    hellinger_matrix = compute_hellinger_matrix(gmm_models, positions)
    plot_dendrogram(hellinger_matrix, positions)
#     grouping = {
#     'GroupOrange': [ 27, 7, 12, 17, 2, 14, 10],
#     'GroupGreen': [8, 3, 21, 30, 16, 4, 1, 22, 20],
#     'GroupRed': [9, 5, 19, 18, 6, 11, 26],
#     'GroupPurple': [28, 13, 25, 29, 15, 24, 23]
# }
#     within_HD = compute_within_group_hd(hellinger_matrix, positions, grouping)
#     print(within_HD)
#     between_HD = compute_between_group_hd(hellinger_matrix, positions, grouping)
#     print(between_HD)

#     plot_behavioral_groups_hd_analysis()

    # ----------------------
# Config and Execute
# ----------------------
file_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=0\PLR anneal 1000\data_until_lift_start.xlsx"
joint_features = ['Joint 1 Pos','Joint 2 Pos','Joint 3 Pos','Joint 5 Pos']
#joint_features = ['Joint 1 Pos','Joint 3 Pos','Joint 5 Pos']
#joint_features = ['Joint 1 Pos','Joint 3 Pos', 'Joint 5 Pos']

run_pipeline(file_path, joint_features)