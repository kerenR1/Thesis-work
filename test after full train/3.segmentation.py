import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


columns = ['POS', 'sample',
           'Joint 1 Pos', 'Joint 2 Pos', 'Joint 3 Pos', 'Joint 4 Pos',
           'Joint 5 Pos', 'Joint 6 Pos', 'Joint 7 Pos']


import numpy as np

def find_lift_or_reach_index(xyz_positions, lift_threshold=0.04, z_stability_threshold=1e-9, window=2):
    """
    Detects the first time step when the cube is lifted (Z >= threshold),
    or if not lifted, the time it was likely reached (Z becomes stable).

    Parameters:
        xyz_positions (np.ndarray): Array of shape (N, 3) for cube X, Y, Z positions
        lift_threshold (float): Z value considered as lifted
        z_stability_threshold (float): Max Z variation to define stable contact
        window (int): Number of samples to check for Z stability

    Returns:
        int: Index of lift if found, else index of stable contact (reach), else -1
    """
    z_positions = xyz_positions[:, 2]

    # First: try to find the lift point
    lift_indices = np.where(z_positions >= lift_threshold)[0]
    if lift_indices.size > 0:
        return int(lift_indices[0])

    # If not lifted: try to find the point where Z stabilizes
    for i in range(len(z_positions) - window):
        z_window = z_positions[i:i+window]
        if np.max(z_window) - np.min(z_window) < z_stability_threshold:
            return i

    return 240  # Neither lifted nor clearly reached


def detect_motion_end(
    df,
    lift_step,
    position_columns=['Cube X Position', 'Cube Y Position', 'Cube Z Position'],
    threshold=1e-3,
    window=5,
    plot=False
):
    """
    Detects when the cube stops moving significantly after the lifting begins.

    Parameters:
        df (pd.DataFrame): DataFrame containing cube position data
        lift_step (int): Index where lift starts
        position_columns (list): Columns to track motion
        threshold (float): Max delta to consider as 'stopped'
        window (int): Number of still frames to confirm stop
        plot (bool): Plot motion analysis

    Returns:
        int or None: Index of motion stop in original data or None if not found.
    """

    # If lift never happened or too close to the end, can't analyze
    if lift_step == 240 or lift_step >= len(df) - window:
        return 240

    # Analyze only data after lift
    post_lift = df.iloc[lift_step:].reset_index(drop=True)
    # Compute absolute differences between successive positions (velocity approximation)
    deltas = post_lift[position_columns].diff().abs()
    # Add a new column representing the maximum delta across all 3 axes at each time step
    deltas['max_delta'] = deltas.max(axis=1)

    # Use a rolling window to count how many consecutive frames had small enough motion
    # A window is considered a "motion stop" if all values in the window are under the threshold
    rolling_still = (deltas['max_delta'] < threshold).rolling(window).sum()
    # Find the first index where all 'window' values were under threshold
    stop_index = rolling_still[rolling_still == window].index.min()

    # Convert stop index to original index
    if pd.notna(stop_index):
        motion_end_step = int(stop_index + lift_step)
        if motion_end_step > lift_step:
            if plot:
                plt.figure(figsize=(10, 4))
                plt.plot(deltas['max_delta'], label='Max Delta (Motion)')
                plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
                plt.axvline(stop_index, color='green', linestyle='--', label='Motion Stops')
                plt.legend()
                plt.title("Motion Detection After Lift")
                plt.xlabel("Step After Lift")
                plt.ylabel("Max Delta")
                plt.tight_layout()
                plt.show()
            return motion_end_step

    return 240 # No valid motion stop detected


def check_goal_reached(xyz_positions, goal_position=(0.45, 0.0, 0.35), tolerance=0.05):
    """
    Check if the final cube position is within a small radius (tolerance) of the goal.

    Parameters:
        xyz_positions (array-like): array of cube positions over time (X, Y, Z)
        goal_position (tuple): The target (X, Y, Z) coordinates of the goal
        tolerance (float): Maximum allowed distance from the goal to consider it "reached" (5 mm)

    Returns:
        (bool, float): 
            - True and the distance if the final position is within tolerance.
            - False and the distance otherwise.
    """

    # Convert the goal position to a NumPy array
    goal = np.array(goal_position)

    # Get the final cube position from the repetition
    final_position = xyz_positions[-1]

    # Compute distance from the final position to the goal
    final_distance = np.linalg.norm(final_position - goal)

    # Check if it's within the specified radius
    return final_distance <= tolerance, final_distance



# Updated to handle potential tuple issue with POS
def analyze_all_positions(df):
    results = []
    grouped = df.groupby(['POS'])

    for pos, group in grouped:
        # Ensure POS is treated as an integer if it's not already
        if isinstance(pos, tuple):
            pos = pos[0]  # Extract the first element if it's a tuple
        pos = int(pos)  # Convert to integer explicitly

        group = group.sort_values(by='sample').reset_index(drop=True)
        xyz_positions = group[['Cube X Position', 'Cube Y Position', 'Cube Z Position']].to_numpy()

        if len(group) < 240:
            lift_step = -1
            motion_end_step = -1
            goal_reached = False
            min_dist = -1
        else:
            lift_step = find_lift_or_reach_index(xyz_positions)
            motion_end_step = detect_motion_end(group, lift_step, plot=False)
            goal_reached, min_dist = check_goal_reached(xyz_positions)

        results.append({
            'POS': pos,
            'Lift Start Sample': lift_step,
            'Motion End Sample': motion_end_step,
            'Goal Reached': goal_reached,
            'Min Distance to Goal': min_dist
        })

    return pd.DataFrame(results)

# Load full dataset
df = pd.read_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\joint_positions_after_conv_trim.xlsx")

# Analyze all positions
result_df = analyze_all_positions(df)

# Add time columns
result_df['Lift Start Time [s]'] = (result_df['Lift Start Sample'] + 10) * 0.01
result_df['Motion End Time [s]'] = (result_df['Motion End Sample'] + 10) * 0.01

# Save the results to a new Excel file
result_df.to_excel(r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\segmentation_results.xlsx", index=False)
