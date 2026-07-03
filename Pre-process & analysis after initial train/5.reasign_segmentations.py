import pandas as pd

# Load your data
df = pd.read_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\segmentation_results.xlsx")

# Step 1: Drop the first repetition of POS=1 and REP=1
df = df[~((df['POS'] == 1) & (df['REP'] == 1))].reset_index(drop=True)

# Step 2: Reassign new POS and REP
# Total remaining rows = 299
df['POS'] = (df.index // 10) + 1
df['REP'] = (df.index % 10) + 1

# The last position (POS=30) should only have 9 reps
df = df[~((df['POS'] == 30) & (df['REP'] == 10))].reset_index(drop=True)

# Save the cleaned data
df.to_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\segmentation_results_re.xlsx", index=False)

