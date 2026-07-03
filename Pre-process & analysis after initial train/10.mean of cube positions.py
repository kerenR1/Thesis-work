import pandas as pd

df = pd.read_csv(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\cube_positions.csv")

# Drop first 4 rows (after the header)
df2 = df.iloc[4:].reset_index(drop=True)

# Keep only the first of each block of 4 identical positions
df_unique = df2.iloc[::4].reset_index(drop=True)

print("Original rows:", len(df))
print("After dropping first 4:", len(df2))
print("After keeping 1 of each 4:", len(df_unique))

# Average by blocks of 10 positions
block_size = 10
blocks = []
coords = []

for start in range(0, len(df_unique), block_size):
    end = min(start + block_size, len(df_unique))
    block = df_unique.iloc[start:end]
    avg = block.mean()
    blocks.append({
        "block_index": start // block_size + 1,
        "start_row": start,
        "end_row": end - 1,
        "avg_X": avg["X"],
        "avg_Y": avg["Y"],
        "avg_Z": avg["Z"],
    })
    coords.append((round(avg["X"],6), round(avg["Y"],6)))

avg_df = pd.DataFrame(blocks)
clean_coords = [(float(x), float(y)) for (x, y) in coords]
print(clean_coords)

# Save both the cleaned positions and the block averages
df_unique.to_csv(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\cube_positions_cleaned.csv", index=False)
avg_df.to_csv(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\cube_avg_cleaned.csv", index=False)