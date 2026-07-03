import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# === Step 1: Provide the 2 Excel file paths ===
excel_file_paths = [
    r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\pca_results_until_lift_start_StandardScaler.xlsx",
    r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\pca_results_until_motion_end_StandardScaler.xlsx",
]

# === Step 2: Provide matching titles for plots (in the same order) ===
titles = [
    "Until Lift Start - 1000 iterations model",
    "StandardScaler - Until Motion End"
]

# === Step 3: Load data from each file's "PCA Loadings" sheet ===
all_data = {}
for path, title in zip(excel_file_paths, titles):
    df = pd.read_excel(path, sheet_name="PCA Loadings", index_col=0)
    all_data[title] = df

# === Step 4: Plotting settings ===
pc_colors = ['red', 'blue', 'green']

pdf_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\pca_joint_comparison_colored.pdf"
with PdfPages(pdf_path) as pdf:
    for title, df in all_data.items():
        ax = df.T.plot(kind="bar", figsize=(10, 6), color=pc_colors)
        plt.title(f"PCA Loadings - {title}")
        plt.ylabel("Loading Value")
        plt.xlabel("Joints")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend(title="Principal Component")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"PDF saved to: {pdf_path}")
