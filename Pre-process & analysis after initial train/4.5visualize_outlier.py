import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load the data
file_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\joint_positions_after_cut.xlsx"

df = pd.read_excel(file_path)

# Joint column names
joint_cols = [f"Joint {i} Pos" for i in range(1, 8)]

# Unique positions
unique_positions = sorted(df["POS"].unique())

# Create a PDF file for the plots
pdf_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\positions_joint_analysis_cut.pdf"
with PdfPages(pdf_path) as pdf:
    for pos in unique_positions:
        position_data = df[df["POS"] == pos]
        unique_reps = sorted(position_data["REP"].unique())

        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, joint in enumerate(joint_cols):
            ax = axes[idx]
            for rep in unique_reps:
                rep_data = position_data[position_data["REP"] == rep]
                ax.plot(rep_data["sample"], rep_data[joint], label=f"Rep {rep}", alpha=0.7)
            ax.set_title(joint)
            ax.set_xlabel("Sample")
            ax.set_ylabel("Angle (degrees)")
            ax.legend()
            ax.grid(True)

        # Hide unused subplots if any
        for j in range(len(joint_cols), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Joint Angles Over Time - Position {pos}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

pdf_path
