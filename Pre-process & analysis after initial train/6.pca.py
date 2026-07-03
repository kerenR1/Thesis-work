import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_pdf import PdfPages

# Load original dataset
df = pd.read_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\joint_positions_after_cut.xlsx")

# Load segmentation results
segmentation_results = pd.read_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\segmentation_results_re.xlsx")

# Merge segmentation info into the original data
df_merged = df.merge(segmentation_results, on=["POS", "REP"], how="left")

# Create two subsets: one until lift start (exclusive), one until motion end (exclusive)
df_lift = []
df_motion_end = []

for (pos, rep), group in df_merged.groupby(["POS", "REP"]):
    group = group.sort_values("sample").reset_index(drop=True)

    lift_start_sample = group["Lift Start Sample"].iloc[0]
    motion_end_sample = group["Motion End Sample"].iloc[0]

    if not np.isnan(lift_start_sample):
        lift_start_sample = int(lift_start_sample)
        df_lift.append(group.iloc[:lift_start_sample])  # exclusive

    if not np.isnan(motion_end_sample):
        motion_end_sample = int(motion_end_sample)
        df_motion_end.append(group.iloc[:motion_end_sample])  # exclusive

# Concatenate all repetitions
df_lift = pd.concat(df_lift, ignore_index=True)
df_motion_end = pd.concat(df_motion_end, ignore_index=True)

# Save segmented files
df_lift.to_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\data_until_lift_start.xlsx", index=False)
df_motion_end.to_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\data_until_motion_end.xlsx", index=False)

# Define joint columns
joint_cols = ['Joint 1 Pos', 'Joint 2 Pos', 'Joint 3 Pos', 'Joint 4 Pos',
              'Joint 5 Pos', 'Joint 6 Pos', 'Joint 7 Pos']

# PCA function

def perform_pca_and_save(df, output_excel, output_pdf):
    X = df[joint_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Create results DataFrames
    pca_scores = pd.DataFrame(X_pca[:, :3], columns=["PC1", "PC2", "PC3"])
    pca_loadings = pd.DataFrame(pca.components_[:3, :], columns=joint_cols, index=["PC1", "PC2", "PC3"])
    pca_variance = pd.DataFrame({
        "Explained Variance Ratio": pca.explained_variance_ratio_,
        "Cumulative Variance": np.cumsum(pca.explained_variance_ratio_)
    })

    # Write to single Excel file
    with pd.ExcelWriter(output_excel) as writer:
        pca_scores.to_excel(writer, sheet_name="PCA Scores", index=False)
        pca_loadings.to_excel(writer, sheet_name="PCA Loadings")
        pca_variance.to_excel(writer, sheet_name="Variance Analysis")

    # Generate PDF with bar plots only
    with PdfPages(output_pdf) as pdf:
        for pc in pca_loadings.index:
            plt.figure(figsize=(8, 4))
            sns.barplot(x=joint_cols, y=pca_loadings.loc[pc].values)
            plt.title(f"{pc} - Joint Contributions")
            plt.ylabel("Loading Value")
            plt.xlabel("Joint")
            plt.xticks(rotation=45)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

# Perform PCA and save results
perform_pca_and_save(df_lift, r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\pca_results_until_lift_start_StandardScaler.xlsx",
                      r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\pca_barplots_until_lift_start_StandardScaler.pdf")
perform_pca_and_save(df_motion_end, r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\pca_results_until_motion_end_StandardScaler.xlsx"
                     , r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\pca_barplots_until_motion_end_StandardScaler.pdf")

