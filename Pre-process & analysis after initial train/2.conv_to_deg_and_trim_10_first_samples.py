import pandas as pd
import numpy as np

def convert_to_deg_and_trim_joint_data(input_excel_path, output_excel_path):
    # Load Excel file
    df = pd.read_excel(input_excel_path)

    # Drop the first 10 samples in each repetition
    df = df[df['sample'] > 10].copy()

    # Re-index 'sample' from 1 to 240 per (POS, REP)
    df['sample'] = df.groupby(['POS', 'REP']).cumcount() + 1

    # Define the joint position column names
    joint_columns = [
        "Joint 1 Pos", "Joint 2 Pos", "Joint 3 Pos",
        "Joint 4 Pos", "Joint 5 Pos", "Joint 6 Pos",
        "Joint 7 Pos"
    ]

    # Convert from radians to degrees
    df[joint_columns] = np.degrees(df[joint_columns])

    # Save the result to a new Excel file
    df.to_excel(output_excel_path, index=False)
    print(f"Processed and converted joint data saved to: {output_excel_path}")


convert_to_deg_and_trim_joint_data(
    r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\labeled_joint_positions.xlsx",
    r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\PLR 0.8 1000\joint_positions_after_conv_trim.xlsx"
)
