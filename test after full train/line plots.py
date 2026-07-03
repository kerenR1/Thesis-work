import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

# Per learning step

# FILE_PATH = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\full 0.8 8000\avg_cum_reward_per_learning_step.csv"

# # --- Read: auto delimiter (tabs/spaces/commas) ---
# df = pd.read_csv(
#     FILE_PATH,
#     sep=r"[\s,]+",     # split on whitespace OR commas
#     engine="python",
#     header=0         
# )

# df = df[["learning_step", "mean_episode_cum_reward"]].copy()

# # Convert to numeric, drop bad rows
# df["learning_step"] = pd.to_numeric(df["learning_step"], errors="coerce")
# df["mean_episode_cum_reward"] = pd.to_numeric(df["mean_episode_cum_reward"], errors="coerce")
# df = df.dropna().sort_values("learning_step")

# # --- Keep only valid step range ---
# df = df[(df["learning_step"] >= 98304) & (df["learning_step"] <= 491520000)]

# # --- IQR outlier removal on reward ---
# k = 5.0  # 1.5 is "classic"; 5.0 is gentler (often better for noisy training curves)
# q1 = df["mean_episode_cum_reward"].quantile(0.25)
# q3 = df["mean_episode_cum_reward"].quantile(0.75)
# iqr = q3 - q1
# low = q1 - k * iqr
# high = q3 + k * iqr

# mask = df["mean_episode_cum_reward"].between(low, high)
# dropped = (~mask).sum()
# df = df[mask]

# print(f"Rows after IQR filter: {len(df)} (dropped {dropped})")
# print(f"IQR bounds: [{low}, {high}]")

# # --- K / M formatter for x-axis ---
# def format_steps(x, pos):
#     if x >= 1_000_000:
#         return f"{x/1_000_000:.0f}M"
#     elif x >= 1_000:
#         return f"{x/1_000:.0f}K"
#     return str(int(x))

# # --- Plot (raw, no smoothing) ---
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(df["learning_step"], df["mean_episode_cum_reward"], linewidth=1.5)

# ax.set_title("Training Curve")
# ax.set_xlabel("Learning step")
# ax.set_ylabel("Mean episode cumulative reward")
# ax.grid(True)

# #ax.set_xlim(98304, 379355136)
# ax.set_xlim(98304, 491520000)
# ax.set_ylim(-50, 150)

# ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
# #ax.xaxis.set_major_locator(MaxNLocator(8))

# plt.tight_layout()
# plt.show()

# Per iteration

# FILE_PATH = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\PLR A 0.8 8000\avg_cum_reward_per_iteration.csv"

# # --- Read: auto delimiter (tabs/spaces/commas) ---
# df = pd.read_csv(
#     FILE_PATH,
#     sep=r"[\s,]+",     # split on whitespace OR commas
#     engine="python",
#     header=0         
# )

# df = df[["iteration", "mean_episode_cum_reward"]].copy()

# # Convert to numeric, drop bad rows
# df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
# df["mean_episode_cum_reward"] = pd.to_numeric(df["mean_episode_cum_reward"], errors="coerce")
# df = df.dropna().sort_values("iteration")

# # --- Keep only valid step range ---
# df = df[(df["iteration"] >= 0) & (df["iteration"] <= 4999)]

# # --- IQR outlier removal on reward ---
# k = 5.0  # 1.5 is "classic"; 5.0 is gentler (often better for noisy training curves)
# q1 = df["mean_episode_cum_reward"].quantile(0.25)
# q3 = df["mean_episode_cum_reward"].quantile(0.75)
# iqr = q3 - q1
# low = q1 - k * iqr
# high = q3 + k * iqr

# mask = df["mean_episode_cum_reward"].between(low, high)
# dropped = (~mask).sum()
# df = df[mask]

# print(f"Rows after IQR filter: {len(df)} (dropped {dropped})")
# print(f"IQR bounds: [{low}, {high}]")

# # --- K / M formatter for x-axis ---
# def format_steps(x, pos):
#     if x >= 1_000_000:
#         return f"{x/1_000_000:.0f}M"
#     elif x >= 1_000:
#         return f"{x/1_000:.0f}K"
#     return str(int(x))

# # --- Plot (raw, no smoothing) ---
# fig, ax = plt.subplots(figsize=(12, 5))
# ax.plot(df["iteration"], df["mean_episode_cum_reward"], linewidth=1.5)

# ax.set_title("Training Curve")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Mean episode cumulative reward")
# ax.grid(True)

# ax.set_xlim(0, 4999)
# ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
# ax.set_ylim(-50, 150)

# ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
# #ax.xaxis.set_major_locator(MaxNLocator(8))

# plt.tight_layout()
# plt.show()


# Comparisom PLR PLR-A by learning step

# --------- Paths ----------
PATH_FULL = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR\avg_cum_reward_per_learning_step.csv"
PATH_PLRA = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\avg_cum_reward_per_learning_step.csv"

# --------- Shared params ----------
#STEP_MIN, STEP_MAX = 98304, 491_520_000
STEP_MIN, STEP_MAX = 98304, 379355136

def load(file_path, step_min=STEP_MIN, step_max=STEP_MAX):
    df = pd.read_csv(file_path, sep=r"[\s,]+", engine="python", header=0)
    df = df[["learning_step", "mean_episode_cum_reward"]].copy()

    df["learning_step"] = pd.to_numeric(df["learning_step"], errors="coerce")
    df["mean_episode_cum_reward"] = pd.to_numeric(df["mean_episode_cum_reward"], errors="coerce")

    return df

def format_steps(x, pos):
    if x >= 1_000_000:
        return f"{x/1_000_000:.0f}M"
    elif x >= 1_000:
        return f"{x/1_000:.0f}K"
    return str(int(x))

# --------- Load both ----------
df_PLR = load(PATH_FULL)
df_plra = load(PATH_PLRA)

print(f"PLR: rows={len(df_PLR)}")
print(f"PLR-A: rows={len(df_plra)}")

# --- Keep only in step range ---
# df_PLR = df_PLR[(df_PLR["learning_step"] >= 98304) & (df_PLR["learning_step"] <= 491_520_000)]
# df_plra = df_plra[(df_plra["learning_step"] >= 98304) & (df_plra["learning_step"] <= 491_520_000)]
df_PLR = df_PLR[(df_PLR["learning_step"] >= 98304) & (df_PLR["learning_step"] <= 379355136)]
df_plra = df_plra[(df_plra["learning_step"] >= 98304) & (df_plra["learning_step"] <= 379355136)]

# --------- Plot together ----------
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df_PLR["learning_step"], df_PLR["mean_episode_cum_reward"],
        linewidth=0.5, label="PLR")
ax.plot(df_plra["learning_step"], df_plra["mean_episode_cum_reward"],
        linewidth=0.5, label="PLR-A")

ax.set_title("Training Curve Comparison Fixed Configuration")
ax.set_xlabel("Learning step")
ax.set_ylabel("Mean episode cumulative reward")
ax.grid(True)

ax.set_xlim(STEP_MIN, STEP_MAX)
ax.set_ylim(-25, 150)

ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
ax.legend()

plt.tight_layout()
plt.show()


# Comparisom PLR PLR-A by iteration

# --------- Paths ----------
PATH_FULL = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR\avg_cum_reward_per_iteration.csv"
PATH_PLRA = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\avg_cum_reward_per_iteration.csv"

# --------- Shared params ----------
#ITER_MIN, ITER_MAX = 0, 4999
ITER_MIN, ITER_MAX = 0, 3850

def load(file_path):
    df = pd.read_csv(file_path, sep=r"[\s,]+", engine="python", header=0)
    df = df[["iteration", "mean_episode_cum_reward"]].copy()

    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df["mean_episode_cum_reward"] = pd.to_numeric(df["mean_episode_cum_reward"], errors="coerce")

    return df

def format_steps(x, pos):
    if x >= 1_000_000:
        v = x / 1_000_000
        return f"{v:.1f}M".rstrip("0").rstrip(".")
    elif x >= 1_000:
        v = x / 1_000
        return f"{v:.1f}K".rstrip("0").rstrip(".")
    return str(int(x))


# --------- Load both ----------
df_PLR = load(PATH_FULL)
df_plra = load(PATH_PLRA)

print(f"PLR: rows={len(df_PLR)}")
print(f"PLR-A: rows={len(df_plra)}")

# --- Keep only in iter range ---
# df_PLR = df_PLR[(df_PLR["iteration"] >= 0) & (df_PLR["iteration"] <= 4999)]
# df_plra = df_plra[(df_plra["iteration"] >= 0) & (df_plra["iteration"] <= 4999)]
df_PLR = df_PLR[(df_PLR["iteration"] >= 0) & (df_PLR["iteration"] <= 3850)]
df_plra = df_plra[(df_plra["iteration"] >= 0) & (df_plra["iteration"] <= 3850)]

# --------- Plot together ----------
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(df_PLR["iteration"], df_PLR["mean_episode_cum_reward"],
        linewidth=0.5, label="PLR")
ax.plot(df_plra["iteration"], df_plra["mean_episode_cum_reward"],
        linewidth=0.5, label="PLR-A")

ax.set_title("Training Curve Comparison")
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean episode cumulative reward")
ax.grid(True)

#ax.set_xlim(ITER_MIN, ITER_MAX)
ax.set_xlim(0, 3850)
ax.xaxis.set_major_locator(MultipleLocator(500))  # 0, 500, 1000, ...
ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
ax.set_ylim(-25, 150)

ax.legend()

plt.tight_layout()
plt.show()
