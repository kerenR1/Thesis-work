import pandas as pd
import numpy as np

# Load your CSV file
df = pd.read_csv(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR\joint_positions.csv")

# Header row values (Joint 1-9 positions at the beginning of each position)
header_values = [0, -0.569, 0, -2.81, 0, 3.037, 0.741, 0.04, 0.04]

# Get column names for joint positions (first 9 columns should be Joint 1-9)
joint_cols = df.columns[:9].tolist()

pos_list = []
sample_list = []

current_pos = 0
sample_count = 0

for idx, row in df.iterrows():
    # Check if this is a header row by comparing joint position values
    row_values = row[joint_cols].values

    # Compare with tolerance for floating point
    is_header = np.allclose(row_values, header_values, atol=0.001)

    if is_header:
        # This is a header row - it marks the start of a new position
        current_pos += 1  # Increment position
        sample_count = 1  # Reset sample count for the new position
        pos_list.append(current_pos)
        sample_list.append(1)  # First sample of the new position
    else:
        # This is an actual data sample
        sample_count += 1
        pos_list.append(current_pos)
        sample_list.append(sample_count)

# Add POS and sample columns to the DataFrame
df["POS"] = pos_list
df["sample"] = sample_list

# Reorder columns so POS and sample are first
cols = ["POS", "sample"] + [col for col in df.columns if col not in ["POS", "sample"]]
df = df[cols]

# Save the new DataFrame as xlsx
df.to_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR\labeled_joint_positions.xlsx", index=False)
