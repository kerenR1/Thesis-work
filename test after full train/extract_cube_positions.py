import pandas as pd

def extract_unique_coordinates(
    csv_path: str,
    drop_first_rows: int = 4,
    reps_per_position: int = 4,
    n_positions: int = 100,
    x_col: str | None = None,
    y_col: str | None = None,
    z_col: str | None = None,
):
    """
    Reads a CSV that starts with `drop_first_rows` dummy poses, then contains
    `n_positions` positions where each position is repeated `reps_per_position` times.

    Returns:
      coordinates_xy: list[tuple[float, float]]  length = n_positions
      unique_df:      DataFrame with x,y,z for the unique positions
    """

    df = pd.read_csv(csv_path)

    # Drop the dummy poses
    df = df.iloc[drop_first_rows:].reset_index(drop=True)

    # Auto-detect x,y,z columns if not provided
    if x_col is None or y_col is None or z_col is None:
        cols_lower = {c.lower(): c for c in df.columns}
        if all(k in cols_lower for k in ("x", "y", "z")):
            x_col, y_col, z_col = cols_lower["x"], cols_lower["y"], cols_lower["z"]
        else:
            # Fall back to first 3 columns
            x_col, y_col, z_col = df.columns[:3]

    needed_rows = n_positions * reps_per_position
    if len(df) < needed_rows:
        raise ValueError(
            f"After dropping {drop_first_rows} rows, expected at least {needed_rows} rows "
            f"({n_positions} positions × {reps_per_position} reps), but got {len(df)}."
        )

    # Keep only the expected block (ignore any extra rows if exist)
    df = df.iloc[:needed_rows].copy()

    # Take the 1st repetition of each position group
    unique_df = df.iloc[::reps_per_position][[x_col, y_col, z_col]].reset_index(drop=True)

    # Convert to Python list of (x,y) tuples
    coordinates_xy = [(float(r[x_col]), float(r[y_col])) for _, r in unique_df.iterrows()]

    return coordinates_xy, unique_df


def format_coordinates_list(coordinates_xy, decimals: int = 6) -> str:
    """
    Formats list of (x,y) into:
      coordinates = [(0.123456, -0.654321), ...]
    """
    items = ", ".join(f"({x:.{decimals}f}, {y:.{decimals}f})" for x, y in coordinates_xy)
    return f"coordinates = [{items}]"


if __name__ == "__main__":
    csv_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\test after train full 3500\cube_positions_100_uniform.csv"

    coordinates_xy, unique_df = extract_unique_coordinates(
        csv_path,
        drop_first_rows=4,
        reps_per_position=4,
        n_positions=100,
        # If your columns are different, set them here, e.g.:
        # x_col="pos_x", y_col="pos_y", z_col="pos_z"
    )

    # Print in the exact format you want
    print(format_coordinates_list(coordinates_xy, decimals=6))

    # Also save the unique positions to a new CSV
    #unique_df.to_csv("unique_positions.csv", index=False)
