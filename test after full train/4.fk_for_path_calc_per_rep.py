import pandas as pd
import numpy as np

# 1. Load full dataset
file_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\joint_positions_after_conv_trim.xlsx"
df = pd.read_excel(file_path)

# 2. Convert joint angles from degrees to radians
joint_cols = [f'Joint {i} Pos' for i in range(1, 8)]
df_rad = df[joint_cols].apply(np.radians)

# 3. Define FK functions
def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta
    return np.array([[np.cos(q), -np.sin(q), 0, a],
                     [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                     [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                     [0, 0, 0, 1]])

def get_fk_solution(joint_angles):
    dh_params = [[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi/2, joint_angles[1]],
                 [0, 0.316, np.pi/2, joint_angles[2]],
                 [0.0825, 0, np.pi/2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi/2, joint_angles[4]],
                 [0, 0, np.pi/2, joint_angles[5]],
                 [0.088, 0, np.pi/2, joint_angles[6]],
                 [0, 0.107, 0, 0],
                 [0, 0, 0, -np.pi/4],
                 [0.0, 0.1034, 0, 0]]
    T = np.eye(4)
    for i in range(10):
        T = T @ get_tf_mat(i, dh_params)
    return T

# 4. Apply FK to each row
ee_positions = []
for i in range(len(df_rad)):
    joint_angles = df_rad.iloc[i].values
    T = get_fk_solution(joint_angles)
    ee_positions.append(T[:3, 3])  # Extract x, y, z

# 5. Add to original dataframe
ee_array = np.array(ee_positions)
df["ee_x"] = ee_array[:, 0]
df["ee_y"] = ee_array[:, 1]
df["ee_z"] = ee_array[:, 2]

# 6. Save the result
df.to_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\robot_with_fk_positions_rad.xlsx", index=False)

# 7. Load data
traj_df = pd.read_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\robot_with_fk_positions_rad.xlsx")
seg_df = pd.read_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\segmentation_results.xlsx")

# 8. Compute distance
def compute_distance(path):
    diffs = np.diff(path, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

# Updated to calculate distances and speeds for the entire motion from start to motion end
# 9. Loop over each POS and calculate distances and speeds
dist_start_to_lift = []
speed_lift = []
speed_motion = []
dist_motion = []

for _, row in seg_df.iterrows():
    pos = row['POS']
    lift_sample = row['Lift Start Sample']
    lift_time = row['Lift Start Time [s]']
    end_sample = row['Motion End Sample']
    motion_time = row['Motion End Time [s]']

    sub_df = traj_df[(traj_df['POS'] == pos)]

    # a) From start to lift start
    part1 = sub_df[(sub_df['sample'] >= 1) & (sub_df['sample'] <= lift_sample)]
    path1 = part1[['ee_x', 'ee_y', 'ee_z']].values
    dist1 = compute_distance(path1)
    dist_start_to_lift.append(dist1)
    speed_lift.append(dist1 / lift_time if lift_time > 0 else np.nan)

    # b) From start to motion end
    part2 = sub_df[(sub_df['sample'] >= 1) & (sub_df['sample'] <= end_sample)]
    path2 = part2[['ee_x', 'ee_y', 'ee_z']].values
    dist2 = compute_distance(path2)
    dist_motion.append(dist2)
    speed_motion.append(dist2 / motion_time if motion_time > 0 else np.nan)

# 10. Add to DataFrame
seg_df['distance_from_start_to_lift_start_m'] = dist_start_to_lift
seg_df['speed_lift_m_per_s'] = speed_lift
seg_df['speed_motion_m_per_s'] = speed_motion
seg_df['motion_path_distance_m'] = dist_motion

# 11. Save final result
seg_df.to_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\segmentation_with_distances_and_speeds.xlsx", index=False)
