import pandas as pd
import numpy as np

# Load your CSV file
df = pd.read_csv(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\joint_positions.csv")

# Header row values (Joint 1-9 positions at the beginning of each rep)
header_values = [0, -0.569, 0, -2.81, 0, 3.037, 0.741, 0.04, 0.04]


# Get column names for joint positions (first 9 columns should be Joint 1-9)
joint_cols = df.columns[:9].tolist()
# Get cube position columns (3)
cube_cols = df.columns[-3:].tolist()

pos_list = []
rep_list = []
sample_list = []

current_pos = 1
current_rep = 1
rep_sample_count = 1

for idx, row in df.iterrows():
    # Check if this is a header row by comparing joint position values
    row_values = row[joint_cols].values
    cube_values = row[cube_cols].values
    
    # Compare with tolerance for floating point
    # Header rows have specific joint values AND Cube X/Y/Z = 0
    is_header = np.allclose(row_values, header_values, atol=0.001) and np.allclose(cube_values, [0, 0, 0], atol=0.001)
    
    if is_header:
        # This is a header row - it marks the start of a new rep
        # Skip labeling this row or label it as sample 0
        pos_list.append(current_pos)
        rep_list.append(current_rep)
        sample_list.append(0)  # 0 indicates header row
        rep_sample_count = 0
    else:
        # This is an actual data sample
        rep_sample_count += 1
        pos_list.append(current_pos)
        rep_list.append(current_rep)
        sample_list.append(rep_sample_count)
        

# Identify all header rows first
df['is_header'] = df[joint_cols].apply(
    lambda row: np.allclose(row.values, header_values, atol=0.001), 
    axis=1
)

# Now iterate through with header information
pos_list = []
rep_list = []
sample_list = []

current_pos = 1
current_rep = 1
rep_sample_count = 1

for idx, row in df.iterrows():
    if row['is_header']:
        # Header row - this is the start of a new rep
        if current_rep > 1 or idx > 0:  # Not the very first header
            # We've completed a rep, move to next rep
            current_rep += 1
            
            # Check if we've completed all 10 reps for this position
            if current_rep > 10:
                current_pos += 1
                current_rep = 1
        
        # Header row is sample 1
        rep_sample_count = 1
        pos_list.append(current_pos)
        rep_list.append(current_rep)
        sample_list.append(rep_sample_count)
    else:
        # Data sample
        rep_sample_count += 1
        pos_list.append(current_pos)
        rep_list.append(current_rep)
        sample_list.append(rep_sample_count)

df["POS"] = pos_list
df["REP"] = rep_list
df["sample"] = sample_list

# Remove the helper column
df = df.drop('is_header', axis=1)

# Save the new DataFrame
df.to_csv(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\labeled_joint_positions.csv", index=False)
