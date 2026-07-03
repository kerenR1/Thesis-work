import pandas as pd

# === CONFIGURATION ===
input_file = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\joint_positions_after_conv_trim.xlsx"
output_file = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\joint_positions_after_cut.xlsx"

# === STEP 1: Load data ===
df = pd.read_excel(input_file)

# Sort to ensure correct order
df = df.sort_values(by=["POS", "REP", "sample"]).reset_index(drop=True)

# === STEP 2: Remove first repetition of first position ===
df = df[~((df["POS"] == 1) & (df["REP"] == 1))].copy()

# === STEP 3: Reassign global repetition number ===
df["global_rep"] = df.groupby(["POS", "REP"]).ngroup()

# === STEP 4: Assign new POS and REP based on corrected order ===
df["new_POS"] = (df["global_rep"] // 10) + 1
df["new_REP"] = (df["global_rep"] % 10) + 1

# Optional: Drop old columns and rename for clarity
df = df.drop(columns=["POS", "REP", "global_rep"])
df = df.rename(columns={"new_POS": "POS", "new_REP": "REP"})

# === STEP 5: Save to new Excel file ===
df.to_excel(output_file, index=False)

print(f"Adjusted data saved to:\n{output_file}")
