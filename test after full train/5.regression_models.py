import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from scipy.stats import chi2_contingency

# ============================================================================
# CONFIGURATION: SELECT WHICH METRICS TO ANALYZE
# ============================================================================

# Set to 'lift' or 'motion' to analyze different phases
ANALYSIS_PHASE = 'motion'

# Metric configuration
METRICS_CONFIG = {
    'lift': {
        'time_col': 'lift_time',
        'speed_col': 'lift_speed',
        'distance_col': 'lift_distance',
        'time_label': 'Lift Time (s)',
        'speed_label': 'Lift Speed (m/s)',
        'distance_label': 'Lift Distance (m)',
        'phase_name': 'Lift Phase'
    },
    'motion': {
        'time_col': 'motion_end_time',
        'speed_col': 'motion_speed',
        'distance_col': 'motion_distance',
        'time_label': 'Motion Time (s)',
        'speed_label': 'Motion Speed (m/s)',
        'distance_label': 'Motion Distance (m)',
        'phase_name': 'Motion Phase'
    }
}

# Get current configuration
config = METRICS_CONFIG[ANALYSIS_PHASE]
print("="*80)
print(f"ANALYSIS CONFIGURATION: {config['phase_name'].upper()}")
print("="*80)
print(f"Time metric:     {config['time_col']} → {config['time_label']}")
print(f"Speed metric:    {config['speed_col']} → {config['speed_label']}")
print(f"Distance metric: {config['distance_col']} → {config['distance_label']}")
print("="*80)

# ============================================================================
# LOAD DATA FROM 4 FILES
# ============================================================================

# File paths and their conditions
anneal_PLR_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\anneal\PLR\segmentation_with_distances_and_speeds.xlsx"
anneal_PLR_A_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\anneal\PLR_A\segmentation_with_distances_and_speeds.xlsx"
fixed_PLR_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR\segmentation_with_distances_and_speeds.xlsx"
fixed_PLR_A_path = r"C:\Users\keren\OneDrive - post.bgu.ac.il\תזה\lift_cube seed=42\test after train full 3500\fixed\PLR_A\segmentation_with_distances_and_speeds.xlsx"

files = {
    anneal_PLR_path: {'method': 'PLR', 'schedule': 'Annealed'},
    anneal_PLR_A_path: {'method': 'PLR-A', 'schedule': 'Annealed'},
    fixed_PLR_path: {'method': 'PLR', 'schedule': 'Fixed'},
    fixed_PLR_A_path: {'method': 'PLR-A', 'schedule': 'Fixed'}
}


# Load and combine all data
data_list = []
for filename, conditions in files.items():
    df = pd.read_excel(filename)
    df['method'] = conditions['method']
    df['schedule'] = conditions['schedule']
    df['filename'] = filename
    data_list.append(df)

# Combine all data
data = pd.concat(data_list, ignore_index=True)

# Rename columns for consistency
data.rename(columns={
    'POS': 'position_id',
    'Goal Reached': 'success',
    'Lift Start Time [s]': 'lift_time',
    'speed_lift_m_per_s': 'lift_speed',
    'distance_from_start_to_lift_start_m': 'lift_distance',
    'motion_path_distance_m': 'motion_distance',
    'speed_motion_m_per_s': 'motion_speed',
    'Motion End Time [s]': 'motion_end_time'
}, inplace=True)

# Convert success to binary if it's TRUE/FALSE strings
if data['success'].dtype == 'object':
    data['success'] = data['success'].map({'TRUE': 1, 'True': 1, True: 1, 'FALSE': 0, 'False': 0, False: 0})

print(f"\nTotal trials loaded: {len(data)}")
print(f"\nData breakdown by condition:")
print(data.groupby(['method', 'schedule']).size())

# ============================================================================
# 1. SUCCESS RATE ANALYSIS & CHI-SQUARED TESTS
# ============================================================================

print("\n" + "="*80)
print("SUCCESS RATE ANALYSIS WITH CHI-SQUARED TESTS")
print("="*80)

# Calculate success rate for each condition
success_rates = data.groupby(['method', 'schedule']).agg({
    'success': ['sum', 'count', 'mean']
}).reset_index()

success_rates.columns = ['method', 'schedule', 'successes', 'total', 'success_rate']
success_rates['success_rate_pct'] = success_rates['success_rate'] * 100

print("\nSuccess rates by condition:")
print(success_rates)

# ============================================================================
# CHI-SQUARED TESTS FOR PAIRWISE COMPARISONS
# ============================================================================

def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def chi_squared_test(successes1, total1, successes2, total2):
    """Perform chi-squared test comparing two proportions."""
    failures1 = total1 - successes1
    failures2 = total2 - successes2
    
    contingency_table = np.array([
        [successes1, failures1],
        [successes2, failures2]
    ])
    
    chi2, p_value, dof, expected = chi2_contingency(contingency_table, correction=True)
    stars = get_significance_stars(p_value)
    
    return chi2, p_value, stars

# Get data for each condition
plr_fixed = success_rates[(success_rates['method'] == 'PLR') & 
                          (success_rates['schedule'] == 'Fixed')].iloc[0]
plr_annealed = success_rates[(success_rates['method'] == 'PLR') & 
                             (success_rates['schedule'] == 'Annealed')].iloc[0]
plra_fixed = success_rates[(success_rates['method'] == 'PLR-A') & 
                           (success_rates['schedule'] == 'Fixed')].iloc[0]
plra_annealed = success_rates[(success_rates['method'] == 'PLR-A') & 
                              (success_rates['schedule'] == 'Annealed')].iloc[0]

print("\n" + "="*80)
print("PAIRWISE CHI-SQUARED TESTS")
print("="*80)

# Test 1: PLR vs PLR-A within Fixed
chi2, p_val, stars = chi_squared_test(
    plr_fixed['successes'], plr_fixed['total'],
    plra_fixed['successes'], plra_fixed['total']
)
print(f"\n1. PLR vs PLR-A (Fixed schedule):")
print(f"   χ² = {chi2:.4f}, p = {p_val:.4f} {stars}")
method_fixed_p = p_val

# Test 2: PLR vs PLR-A within Annealed
chi2, p_val, stars = chi_squared_test(
    plr_annealed['successes'], plr_annealed['total'],
    plra_annealed['successes'], plra_annealed['total']
)
print(f"\n2. PLR vs PLR-A (Annealed schedule):")
print(f"   χ² = {chi2:.4f}, p = {p_val:.4f} {stars}")
method_annealed_p = p_val

# Test 3: Fixed vs Annealed within PLR
chi2, p_val, stars = chi_squared_test(
    plr_fixed['successes'], plr_fixed['total'],
    plr_annealed['successes'], plr_annealed['total']
)
print(f"\n3. Fixed vs Annealed (PLR method):")
print(f"   χ² = {chi2:.4f}, p = {p_val:.4f} {stars}")
schedule_plr_p = p_val

# Test 4: Fixed vs Annealed within PLR-A
chi2, p_val, stars = chi_squared_test(
    plra_fixed['successes'], plra_fixed['total'],
    plra_annealed['successes'], plra_annealed['total']
)
print(f"\n4. Fixed vs Annealed (PLR-A method):")
print(f"   χ² = {chi2:.4f}, p = {p_val:.4f} {stars}")
schedule_plra_p = p_val

# ============================================================================
# 2. BAR PLOT WITH CHI-SQUARED SIGNIFICANCE BRACKETS
# ============================================================================

fig, ax = plt.subplots(figsize=(9, 5))

# Define colors to match boxplots
palette = {'PLR': '#1f77b4', 'PLR-A': '#ff7f0e'}

# ── Plot grouped barplot (x=schedule, hue=method) ───────────────────────────
sns.barplot(
    data      = success_rates,
    x         = 'schedule',
    y         = 'success_rate_pct',
    hue       = 'method',
    order     = ['Fixed', 'Annealed'],
    hue_order = ['PLR', 'PLR-A'],
    palette   = palette,
    ax        = ax,
    width     = 0.5,
    edgecolor = 'black',
    linewidth = 1.2,
    errorbar  = None
)

# ── Add percentage labels on bars ───────────────────────────────────────────
# Reworked labeling loop to robustly calculate the bar centers based on 
# conditions and Seaborn grouping rules, avoiding patch indexing bugs.
for schedule_idx, schedule in enumerate(['Fixed', 'Annealed']):
    for method_idx, method in enumerate(['PLR', 'PLR-A']):
        matching_row = success_rates[(success_rates['schedule'] == schedule) & (success_rates['method'] == method)]
        if not matching_row.empty:
            row = matching_row.iloc[0]
            value = row['success_rate_pct']
            
            # The total width of the group is 0.5. PLR is left (-), PLR-A is right (+).
            group_offset = -0.5 / 4 if method == 'PLR' else 0.5 / 4
            x_pos = schedule_idx + group_offset
            height = value
            
            # Use smaller font for labels to fit
            ax.text(x_pos, height + 1,
                    f"{value:.1f}%",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# ── Clean up legend and move it higher (inside plot area like boxplot) ──────
handles, labels = ax.get_legend_handles_labels()
if ax.get_legend():
    ax.get_legend().remove()

# Place legend inside the plot area at bottom-center
ax.legend(handles, labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.2),  # (x=center, y=bottom edge)
            ncol=2,                     # Arrange items horizontally
            title='Method',
            fontsize=10, 
            title_fontsize=11,
            frameon=True)

# ── Formatting ───────────────────────────────────────────────────────────────
ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Schedule', fontsize=12, fontweight='bold')
ax.set_title('Success Rate Analysis', fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ── Get exact bar positions for brackets (requires redrawing to get patches) ────────────────
# Re-plotting or calling draw() is necessary to ensure patches are fully updated before getting x.
plt.draw() 

patch_positions = []
for patch in ax.patches:
    if hasattr(patch, 'get_x') and patch.get_height() > 0:
        cx = patch.get_x() + patch.get_width() / 2
        patch_positions.append(cx)

patch_positions = sorted(patch_positions)
print(f"Bar positions: {patch_positions}")

# Expected order: Fixed_PLR, Fixed_PLR-A, Annealed_PLR, Annealed_PLR-A
if len(patch_positions) == 4:
    pos_fixed_plr     = patch_positions[0]
    pos_fixed_plra    = patch_positions[1]
    pos_annealed_plr  = patch_positions[2]
    pos_annealed_plra = patch_positions[3]
else:
    # This shouldn't happen with the drawn loop, but kept for safety.
    pos_fixed_plr     = -0.125
    pos_fixed_plra    =  0.125
    pos_annealed_plr  =  0.875
    pos_annealed_plra =  1.125

# ── Significance bracket function (BOXPLOT STYLE: stars above, label above) ──
def add_bar_significance_bracket(ax, x1, x2, y_top, p_value, bracket_height, label=None):
    stars = get_significance_stars(p_value)
    if stars == 'ns':
        return False
    
    # Draw bracket
    ax.plot([x1, x1, x2, x2],
            [y_top, y_top + bracket_height, y_top + bracket_height, y_top],
            lw=1.5, color='black', clip_on=False)
    
    # Stars above bracket line and lable
    ax.text((x1 + x2) / 2, y_top + bracket_height * 2.0,
            stars, ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='black', clip_on=False)
    
    # Label below stars
    if label:
        ax.text((x1 + x2) / 2, y_top + bracket_height * 1.0,
                label, ha='center', va='bottom',
                fontsize=11, style='italic', color='black', clip_on=False)
    
    return True

# ── Add significance brackets ────────────────────────────────────────────────
y_max = 100
bh = 3      # bracket height
step = 12   # vertical step between levels

# Level 1: Method effect (PLR vs PLR-A within each schedule)
lvl1 = y_max + 5
drew_lvl1 = False

# Within Fixed
drew_lvl1 |= add_bar_significance_bracket(ax,
    pos_fixed_plr, pos_fixed_plra,
    lvl1, method_fixed_p, bh, label='method')

# Within Annealed 
drew_lvl1 |= add_bar_significance_bracket(ax,
    pos_annealed_plr, pos_annealed_plra,
    lvl1, method_annealed_p, bh, label='method')

# Level 2: Schedule effect (Fixed vs Annealed for each method)
lvl2 = lvl1 + step
drew_lvl2 = False

# For PLR
drew_lvl2 |= add_bar_significance_bracket(ax,
    pos_fixed_plr, pos_annealed_plr,
    lvl2, schedule_plr_p, bh, label='schedule')


# Level 3: # Schedule effect for PLR-A 
lvl3 = lvl2 + step
drew_lvl3 = False
     
drew_lvl3 |= add_bar_significance_bracket(ax,
    pos_fixed_plra, pos_annealed_plra,
    lvl3, schedule_plra_p, bh, label='schedule')

# ── Adjust y-limit ───────────────────────────────────────────────────────────
if drew_lvl3:
    top = lvl3 + step * 0.9
elif drew_lvl2:
    top = lvl2 + step * 0.9
elif drew_lvl1:
    top = lvl1 + step * 0.9
else:
    top = y_max + 10

ax.set_ylim(0, top)

# ── Add significance key at bottom ───────────────────────────────────────────
fig.text(0.5, -0.08,
         'Significance:  * p<0.05   ** p<0.01   *** p<0.001   ns = not significant\n'
         'Chi-squared test with Yates\' continuity correction',
         ha='center', fontsize=10, style='italic', color='dimgray')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

# plt.savefig('success_rates_barplot_with_tests.png', dpi=300, bbox_inches='tight')
# print("\n✓ Success rate plot with chi-squared tests saved")
plt.show()

# ============================================================================
# 3. SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("SUMMARY: CHI-SQUARED TEST RESULTS")
print("="*80)

comparison_summary = pd.DataFrame({
    'Comparison': [
        'PLR vs PLR-A (Fixed)',
        'PLR vs PLR-A (Annealed)',
        'Fixed vs Annealed (PLR)',
        'Fixed vs Annealed (PLR-A)'
    ],
    'p-value': [method_fixed_p, method_annealed_p, schedule_plr_p, schedule_plra_p],
    'Significance': [
        get_significance_stars(method_fixed_p),
        get_significance_stars(method_annealed_p),
        get_significance_stars(schedule_plr_p),
        get_significance_stars(schedule_plra_p)
    ]
})
print(comparison_summary.to_string(index=False))

# ============================================================================
# 2. FILTER FOR SUCCESSFUL TRIALS ONLY
# ============================================================================
print("\n" + "="*80)
print(f"FILTERING FOR SUCCESSFUL TRIALS ONLY ({config['phase_name']})")
print("="*80)

# Filter only successful trials
data_success = data[data['success'] == 1].copy()

print(f"\nTotal successful trials: {len(data_success)} out of {len(data)} ({len(data_success)/len(data)*100:.1f}%)")
print(f"\nSuccessful trials breakdown by condition:")
print(data_success.groupby(['method', 'schedule']).size())

# Check if we have enough data
if len(data_success) < 50:
    print("\n⚠ WARNING: Very few successful trials. Results may be unreliable.")

# ============================================================================
# 3. ENCODE CATEGORICAL VARIABLES FOR REGRESSION
# ============================================================================
data_success['method_dummy'] = (data_success['method'] == 'PLR-A').astype(int)  # 0=PLR, 1=PLR-A
data_success['schedule_dummy'] = (data_success['schedule'] == 'Annealed').astype(int)  # 0=Fixed, 1=Annealed

# ============================================================================
# 4. REGRESSION MODELS (WITHOUT INTERACTION) - SUCCESSFUL TRIALS ONLY
# ============================================================================

def fit_and_report(data, outcome_name, outcome_col):
    print(f"\n{'='*80}")
    print(f"OUTCOME: {outcome_name} (Successful Trials Only)")
    print(f"{'='*80}")
    
    # Remove any NaN values
    valid_data = data[['method_dummy', 'schedule_dummy', 'position_id', outcome_col]].dropna()
    
    if len(valid_data) < 20:
        print(f"⚠ WARNING: Only {len(valid_data)} valid observations. Results may be unreliable.")
    
    X = valid_data[['method_dummy', 'schedule_dummy', 'position_id']]
    y = valid_data[outcome_col]
    
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate statistics
    y_pred = model.predict(X)
    residuals = y - y_pred
    n = len(y)
    k = X.shape[1]
    
    # Standard errors and t-tests
    mse = np.sum(residuals**2) / (n - k - 1)
    var_coef = mse * np.linalg.inv(X.T @ X).diagonal()
    se_coef = np.sqrt(var_coef)
    t_stats = model.coef_ / se_coef
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
    
    # Print results
    print(f"\nModel: {outcome_name} = β₀ + β₁(PLR-A) + β₂(Annealed) + β₃(Position_ID)")
    print(f"\nSample size: {n} successful trials")
    print(f"\nIntercept: {model.intercept_:.4f}")
    print(f"\nCoefficients:")
    predictors = ['PLR-A', 'Annealed', 'Position_ID']
    for i, pred in enumerate(predictors):
        sig = "***" if p_values[i] < 0.001 else "**" if p_values[i] < 0.01 else "*" if p_values[i] < 0.05 else ""
        print(f"  {pred:15s}: β={model.coef_[i]:7.4f}, SE={se_coef[i]:.4f}, t={t_stats[i]:6.3f}, p={p_values[i]:.4f} {sig}")
    
    print(f"\nR² Score: {model.score(X, y):.4f}")
    print(f"Adjusted R²: {1 - (1 - model.score(X, y)) * (n - 1) / (n - k - 1):.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    
    # Interpretation
    print(f"\n{'─'*80}")
    print("INTERPRETATION:")
    print(f"{'─'*80}")
    if p_values[0] < 0.05:
        direction = "increases" if model.coef_[0] > 0 else "decreases"
        print(f"✓ PLR-A {direction} {outcome_name} by {abs(model.coef_[0]):.4f} units vs PLR (p={p_values[0]:.4f})")
    else:
        print(f"○ No significant difference between PLR-A and PLR (p={p_values[0]:.4f})")
    
    if p_values[1] < 0.05:
        direction = "increases" if model.coef_[1] > 0 else "decreases"
        print(f"✓ Annealing {direction} {outcome_name} by {abs(model.coef_[1]):.4f} units vs Fixed (p={p_values[1]:.4f})")
    else:
        print(f"○ No significant difference between Annealed and Fixed (p={p_values[1]:.4f})")
    
    if p_values[2] < 0.05:
        direction = "increases" if model.coef_[2] > 0 else "decreases"
        print(f"✓ Position ID {direction} {outcome_name} by {abs(model.coef_[2]):.4f} units per position (p={p_values[2]:.4f})")
    else:
        print(f"○ Position ID has no significant effect (p={p_values[2]:.4f})")
    
    return model, X, y

print("\n" + "="*80)
print(f"REGRESSION ANALYSIS (WITHOUT INTERACTION) - {config['phase_name'].upper()}")
print("="*80)

# Fit models for each outcome using configured columns
model_time, X_time, y_time = fit_and_report(data_success, config['time_label'], config['time_col'])
model_speed, X_speed, y_speed = fit_and_report(data_success, config['speed_label'], config['speed_col'])
model_distance, X_distance, y_distance = fit_and_report(data_success, config['distance_label'], config['distance_col'])

# ============================================================================
# 5. REGRESSION MODELS (WITH INTERACTION) - SUCCESSFUL TRIALS ONLY
# ============================================================================
print("\n" + "="*80)
print(f"REGRESSION ANALYSIS (WITH INTERACTION) - {config['phase_name'].upper()}")
print("="*80)

data_success['interaction'] = data_success['method_dummy'] * data_success['schedule_dummy']

def fit_with_interaction(data, outcome_name, outcome_col, model_no_int, X_no_int):
    print(f"\n{'='*80}")
    print(f"OUTCOME: {outcome_name} (WITH INTERACTION)")
    print(f"{'='*80}")
    
    # Remove any NaN values
    valid_data = data[['method_dummy', 'schedule_dummy', 'position_id', 'interaction', outcome_col]].dropna()
    
    X_int = valid_data[['method_dummy', 'schedule_dummy', 'position_id', 'interaction']]
    y = valid_data[outcome_col]
    X_no_int_valid = valid_data[['method_dummy', 'schedule_dummy', 'position_id']]
    
    # Fit model with interaction
    model_int = LinearRegression()
    model_int.fit(X_int, y)
    
    # Refit no-interaction model on same data for fair comparison
    model_no_int_refit = LinearRegression()
    model_no_int_refit.fit(X_no_int_valid, y)
    
    # Calculate statistics
    n = len(y)
    k = X_int.shape[1]
    y_pred = model_int.predict(X_int)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - k - 1)
    
    # Standard errors for interaction term
    var_coef = mse * np.linalg.inv(X_int.T @ X_int).diagonal()
    se_coef = np.sqrt(var_coef)
    t_stat_int = model_int.coef_[3] / se_coef[3]
    p_value_int = 2 * (1 - stats.t.cdf(np.abs(t_stat_int), n - k - 1))
    
    # F-test for interaction
    rss_reduced = np.sum((y - model_no_int_refit.predict(X_no_int_valid))**2)
    rss_full = np.sum(residuals**2)
    f_stat = ((rss_reduced - rss_full) / 1) / (rss_full / (n - 5))
    p_value_f = 1 - stats.f.cdf(f_stat, 1, n - 5)
    
    print(f"\nInteraction coefficient (β₄): {model_int.coef_[3]:.4f}")
    print(f"  Standard Error: {se_coef[3]:.4f}")
    print(f"  t-statistic: {t_stat_int:.3f}")
    print(f"  p-value: {p_value_int:.4f}")
    sig = "***" if p_value_int < 0.001 else "**" if p_value_int < 0.01 else "*" if p_value_int < 0.05 else ""
    print(f"  Significance: {sig if sig else 'ns'}")
    
    print(f"\nF-test for interaction: F={f_stat:.4f}, p={p_value_f:.4f}")
    
    print(f"\n{'─'*80}")
    if p_value_int < 0.05:
        print(f"✓ SIGNIFICANT INTERACTION (p={p_value_int:.4f})")
        if model_int.coef_[3] > 0:
            print(f"  → PLR-A benefits MORE from annealing than PLR does")
            print(f"  → The combination (PLR-A + Annealing) produces an extra {model_int.coef_[3]:.4f} units")
        else:
            print(f"  → PLR-A benefits LESS from annealing than PLR does")
            print(f"  → The combination (PLR-A + Annealing) reduces effect by {abs(model_int.coef_[3]):.4f} units")
    else:
        print(f"○ NO SIGNIFICANT INTERACTION (p={p_value_int:.4f})")
        print(f"  → Effects of method and schedule are independent (additive)")
    
    print(f"\nModel Comparison:")
    print(f"  R² without interaction: {model_no_int_refit.score(X_no_int_valid, y):.4f}")
    print(f"  R² with interaction:    {model_int.score(X_int, y):.4f}")
    print(f"  Improvement: {model_int.score(X_int, y) - model_no_int_refit.score(X_no_int_valid, y):.4f}")
    
    return model_int, X_int, y, p_value_int

model_time_int, X_time_int, y_time_int, p_time = fit_with_interaction(data_success, config['time_label'], config['time_col'], model_time, X_time)
model_speed_int, X_speed_int, y_speed_int, p_speed = fit_with_interaction(data_success, config['speed_label'], config['speed_col'], model_speed, X_speed)
model_distance_int, X_distance_int, y_distance_int, p_dist = fit_with_interaction(data_success, config['distance_label'], config['distance_col'], model_distance, X_distance)

# ============================================================================
# 6. VISUALIZATION WITH AUTOMATIC MODEL SELECTION
# ============================================================================
print("\n" + "="*80)
print("MODEL SELECTION FOR VISUALIZATION")
print("="*80)

# Select best model for each outcome based on interaction significance
def select_model(model_no_int, model_int, X_no_int, X_int, y, p_interaction, outcome_name):
    if p_interaction < 0.05:
        print(f"{outcome_name}: Using model WITH interaction (p={p_interaction:.4f})")
        return model_int, X_int, "with interaction"
    else:
        print(f"{outcome_name}: Using model WITHOUT interaction (p={p_interaction:.4f})")
        return model_no_int, X_no_int, "without interaction"

# Refit models on valid data for fair comparison
valid_data_time = data_success[['method_dummy', 'schedule_dummy', 'position_id', 'interaction', config['time_col']]].dropna()
X_time_no_int_valid = valid_data_time[['method_dummy', 'schedule_dummy', 'position_id']]
model_time_refit = LinearRegression().fit(X_time_no_int_valid, valid_data_time[config['time_col']])

valid_data_speed = data_success[['method_dummy', 'schedule_dummy', 'position_id', 'interaction', config['speed_col']]].dropna()
X_speed_no_int_valid = valid_data_speed[['method_dummy', 'schedule_dummy', 'position_id']]
model_speed_refit = LinearRegression().fit(X_speed_no_int_valid, valid_data_speed[config['speed_col']])

valid_data_dist = data_success[['method_dummy', 'schedule_dummy', 'position_id', 'interaction', config['distance_col']]].dropna()
X_dist_no_int_valid = valid_data_dist[['method_dummy', 'schedule_dummy', 'position_id']]
model_dist_refit = LinearRegression().fit(X_dist_no_int_valid, valid_data_dist[config['distance_col']])

selected_model_time, selected_X_time, model_name_time = select_model(
    model_time_refit, model_time_int, X_time_no_int_valid, X_time_int, y_time_int, p_time, config['time_label']
)
selected_model_speed, selected_X_speed, model_name_speed = select_model(
    model_speed_refit, model_speed_int, X_speed_no_int_valid, X_speed_int, y_speed_int, p_speed, config['speed_label']
)
selected_model_dist, selected_X_dist, model_name_dist = select_model(
    model_dist_refit, model_distance_int, X_dist_no_int_valid, X_distance_int, y_distance_int, p_dist, config['distance_label']
)

# ============================================================================
# VISUALIZATION: FIGURE 1 - GROUPED BOXPLOTS WITH SIGNIFICANCE BRACKETS
# ============================================================================

# Define get_significance_stars locally for visual labeling use
def get_significance_stars(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

# Function for boxplots where x-axis is categorical
def add_significance_bracket_SR(ax, x1, x2, y_top, p_value, bracket_height):
    stars = get_significance_stars(p_value)
    if stars == 'ns':
        return False
    ax.plot([x1, x1, x2, x2],
            [y_top, y_top + bracket_height, y_top + bracket_height, y_top],
            lw=1.0, color='black', clip_on=False, )
    ax.text((x1 + x2) / 2, y_top + bracket_height * 1.3,
            stars, ha='center', va='bottom',
            fontsize=10, color='black', clip_on=False)
    return True

# Helper to get model p-values for plotting
def get_model_p_values(data_success, col):
    valid_data = data_success[['method_dummy', 'schedule_dummy',
                                'position_id', 'interaction', col]].dropna()
    X_no_int = valid_data[['method_dummy', 'schedule_dummy', 'position_id']]
    X_int    = valid_data[['method_dummy', 'schedule_dummy', 'position_id', 'interaction']]
    y        = valid_data[col]
    n        = len(y)

    model_no = LinearRegression().fit(X_no_int, y)
    res_no   = y - model_no.predict(X_no_int)
    mse_no   = np.sum(res_no**2) / (n - X_no_int.shape[1] - 1)
    var_no   = mse_no * np.linalg.inv(X_no_int.T @ X_no_int).diagonal()
    t_no     = model_no.coef_ / np.sqrt(var_no)
    p_no     = 2 * (1 - stats.t.cdf(np.abs(t_no), n - X_no_int.shape[1] - 1))

    model_int = LinearRegression().fit(X_int, y)
    res_int   = y - model_int.predict(X_int)
    mse_int   = np.sum(res_int**2) / (n - X_int.shape[1] - 1)
    var_int   = mse_int * np.linalg.inv(X_int.T @ X_int).diagonal()
    t_int     = model_int.coef_ / np.sqrt(var_int)
    p_int     = 2 * (1 - stats.t.cdf(np.abs(t_int), n - X_int.shape[1] - 1))

    return {
        'method':      p_no[0],
        'schedule':    p_no[1],
        'position':    p_no[2],
        'interaction': p_int[3]
    }

palette = {'PLR': '#1f77b4', 'PLR-A': '#ff7f0e'}

outcomes_data = [
    (config['time_col'],     config['time_label'],
     selected_model_time,    selected_X_time,    model_name_time,
     valid_data_time[config['time_col']]),

    (config['speed_col'],    config['speed_label'],
     selected_model_speed,   selected_X_speed,   model_name_speed,
     valid_data_speed[config['speed_col']]),

    (config['distance_col'], config['distance_label'],
     selected_model_dist,    selected_X_dist,    model_name_dist,
     valid_data_dist[config['distance_col']])
]

y_labels = {
    config['time_col']:     'Time [s]',
    config['speed_col']:    'Speed [m/s]',
    config['distance_col']: 'Distance [m]'
}

plot_titles = {
    config['time_col']:     f'{config["phase_name"]} Time',
    config['speed_col']:    f'{config["phase_name"]} Speed',
    config['distance_col']: f'{config["phase_name"]} Distance'
}

# ============================================================================
# FIGURE 1: BOXPLOTS + SUCCESS RATE BAR PLOT (4 PANELS)
# ============================================================================
fig1, axes1 = plt.subplots(1, 4, figsize=(14, 6))

def add_significance_bracket(ax, x1, x2, y_top, p_value, bracket_height, label=None):
    stars = get_significance_stars(p_value)
    if stars == 'ns':
        return False
    
    # Draw bracket lines
    ax.plot([x1, x1, x2, x2],
            [y_top, y_top + bracket_height, y_top + bracket_height, y_top],
            lw=1.0, color='black', clip_on=False) 
    
    # Draw Stars
    star_offset = bracket_height * 2.0 if label else bracket_height * 1.5
    if label=='schedule':
        ax.text(((x1 + x2) / 2)-0.2, y_top + star_offset,
            stars,
            ha='center', va='bottom',
            fontsize=10, fontweight='normal', color='black', clip_on=False)
    else:
        ax.text(((x1 + x2) / 2), y_top + star_offset,
            stars,
            ha='center', va='bottom',
            fontsize=10, fontweight='normal', color='black', clip_on=False)
    
    # Draw Label
    if label=='schedule':
        # NOTE: This label is specific to Time panel but used on all for now; 
        # consider making this logic conditional on panel.
        ax.text(((x1 + x2) / 2)+0.3, y_top + star_offset,
                '* (interaction)' if p_vals['interaction'] < 0.05 else '** (interaction)' if p_vals['interaction'] < 0.01 else '*** (interaction)' if p_vals['interaction'] < 0.001 else '',
                ha='center', va='bottom',
                fontsize=10,              
                color='black',            
                clip_on=False)
    
    return True

global_handles = []
global_labels = []

# PLOT 3 BOXPLOTS
for idx, (col, title, model, X_model, model_name, y_vals) in enumerate(outcomes_data):
    ax = axes1[idx]

    sns.boxplot(
        data      = data_success,
        x         = 'schedule',
        y         = col,
        hue       = 'method',
        order     = ['Fixed', 'Annealed'],
        hue_order = ['PLR', 'PLR-A'],
        palette   = palette,
        width     = 0.5,
        linewidth = 1.2,
        ax        = ax
    )

    if idx == 0:
        global_handles, global_labels = ax.get_legend_handles_labels()

    if ax.get_legend():
        ax.get_legend().remove()

    if 'Time' in plot_titles[col]:
        title = 'Time'
    elif 'Speed' in plot_titles[col]:
        title = 'Speed'
    elif 'Distance' in plot_titles[col]:
        title = 'Distance'
    else:
        title = plot_titles[col]

    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)    
    ax.set_xlabel('Schedule', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_labels[col], fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Need draw() call here to ensure patches are ready to get their locations
    plt.draw()

    # Get exact patch locations for this panel
    patch_positions = sorted(set(round(p.get_x() + p.get_width() / 2, 4) for p in ax.patches if hasattr(p, 'get_x')))
    if len(patch_positions) == 4:
        pos_fixed_plr      = patch_positions[0]
        pos_fixed_plra     = patch_positions[1]
        pos_annealed_plr   = patch_positions[2]
        pos_annealed_plra  = patch_positions[3]
    else:
        # Fallback to defaults if draw failed to produce 4 patches
        pos_fixed_plr      = -0.125
        pos_fixed_plra     =  0.125
        pos_annealed_plr   =  0.875
        pos_annealed_plra  =  1.125

    # SIGNIFICANCE BRACKETS for boxplots
    p_vals  = get_model_p_values(data_success, col)
    y_min   = data_success[col].dropna().min()
    y_max   = data_success[col].dropna().max()
    y_range = y_max - y_min
    bh   = y_range * 0.035
    step = y_range * 0.25

    # Level 1: Method effect
    lvl1 = y_max + y_range * 0.10
    drew_lvl1 = False
    drew_lvl1 |= add_significance_bracket(ax,
        pos_fixed_plr, pos_fixed_plra,
        lvl1, p_vals['method'], bh,
        label='method')                
    drew_lvl1 |= add_significance_bracket(ax,
        pos_annealed_plr, pos_annealed_plra,
        lvl1, p_vals['method'], bh,
        label='method')                      

    # Level 2: Schedule effect (center-to-center)
    center_fixed    = (pos_fixed_plr + pos_fixed_plra) / 2
    center_annealed = (pos_annealed_plr + pos_annealed_plra) / 2
    lvl2 = lvl1 + step
    drew_lvl2 = False
    drew_lvl2 = add_significance_bracket(ax,
        center_fixed, center_annealed,   
        lvl2, p_vals['schedule'], bh,
        label='schedule')                     

    # Adjust y-limit
    if drew_lvl2:
        top = lvl2 + step * 0.85
    elif drew_lvl1:
        top = lvl1 + step * 0.85
    else:
        top = y_max + y_range * 0.15
    ax.set_ylim(bottom=y_min - y_range * 0.05, top=top)

# ============================================================================
# PANEL 4: SUCCESS RATE BARPLOT (RE-USING computated success_rates from top)
# ============================================================================
ax_bar = axes1[3]

sns.barplot(
    data      = success_rates,
    x         = 'schedule',
    y         = 'success_rate_pct',
    hue       = 'method',
    order     = ['Fixed', 'Annealed'],
    hue_order = ['PLR', 'PLR-A'],
    palette   = palette,
    ax        = ax_bar,
    width     = 0.5,
    edgecolor = 'black',
    linewidth = 1.2,
    errorbar  = None
)

if ax_bar.get_legend():
    ax_bar.get_legend().remove()

# Add labels on bars for the Success Rate panel
# Iterating over the dataframe rows directly to find the matching visual conditions 
# and explicitly set the correct height for each label based on its condition.
for schedule_idx, schedule in enumerate(['Fixed', 'Annealed']):
    for method_idx, method in enumerate(['PLR', 'PLR-A']):
        matching_row = success_rates[(success_rates['schedule'] == schedule) & (success_rates['method'] == method)]
        if not matching_row.empty:
            row = matching_row.iloc[0]
            value = row['success_rate_pct']
            
            # Category center is schedule_idx (0,1)
            # Total width is 0.5, so offset is +/- width/4.
            group_offset = -0.5 / 4 if method == 'PLR' else 0.5 / 4
            x_pos = schedule_idx + group_offset
            
            height = value
            
            ax_bar.text(x_pos, height + 1,
                        f"{value:.1f}%",
                        ha='center', va='bottom', fontsize=7)

ax_bar.set_title('Success Rate', fontsize=16, fontweight='bold', pad=10, y=1.15)
ax_bar.set_xlabel('Schedule', fontsize=12, fontweight='bold')
ax_bar.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax_bar.grid(axis='y', alpha=0.3, linestyle='--')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

# Need another draw() to get patch positions for the bar plot panel specifically
plt.draw()

# Calculate specific patch positions for the bar plot
# This is crucial for significance brackets that need exact visual coordinates.
patch_positions = sorted(p.get_x() + p.get_width() / 2 for p in ax_bar.patches if p.get_height() > 0)
if len(patch_positions) == 4:
    pos_fixed_plr     = patch_positions[0]
    pos_fixed_plra    = patch_positions[1]
    pos_annealed_plr  = patch_positions[2]
    pos_annealed_plra = patch_positions[3]
else:
    # Use robust calculated defaults if drawing failed (less precise, but should work)
    pos_fixed_plr     = -0.125
    pos_fixed_plra     =  0.125
    pos_annealed_plr   =  0.875
    pos_annealed_plra  =  1.125

# vertical placement for significance on bar plot
y_max_bar = max(100, success_rates['success_rate_pct'].max() + 5)
bh = 3
step_bar = 8
lvl1_bar = y_max_bar

# Method effect within schedules
add_significance_bracket_SR(ax_bar, pos_fixed_plr, pos_fixed_plra, lvl1_bar, method_fixed_p, bh)
add_significance_bracket_SR(ax_bar, pos_annealed_plr, pos_annealed_plra, lvl1_bar, method_annealed_p, bh)

# Schedule effect (center-to-center)
center_fixed = (pos_fixed_plr + pos_fixed_plra) / 2
center_annealed = (pos_annealed_plr + pos_annealed_plra) / 2
lvl2_bar = lvl1_bar + step_bar*1.3
# Schedule effect for PLR
add_significance_bracket_SR(ax_bar, pos_fixed_plr, pos_annealed_plr, lvl2_bar, schedule_plr_p, bh)
# Schedule effect for PLR-A
lvl3_bar = lvl2_bar + step_bar*1.2
add_significance_bracket_SR(ax_bar, pos_fixed_plra, pos_annealed_plra, lvl3_bar, schedule_plra_p, bh)

ax_bar.set_ylim(0, lvl3_bar + step_bar * 1.2)

# OVERALL TITLE AND FOOTER
if 'Motion' in config["phase_name"]:
    motion_text = 'Full Motion'
else:
    motion_text = 'Lift Phase'

plt.suptitle('Regression Analysis', fontsize=16, fontweight='bold', x=0.4)
fig1.text(0.5, -0.02,
          'Significance:  * p<0.05   ** p<0.01   *** p<0.001   ns = not significant',
          ha='center', fontsize=11, style='italic', color='dimgray')

# ADD GLOBAL LEGEND
fig1.legend(global_handles, global_labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, 0.0),  
            ncol=2,                     
            title='Method',
            fontsize=10, 
            title_fontsize=11,
            frameon=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Make room for the legend

# SAVE 1: BOXPLOTS
output_fig1 = f'boxplots_{ANALYSIS_PHASE}_phase_corrected.png'
plt.savefig(output_fig1, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved boxplots: '{output_fig1}'")
plt.show()

# ============================================================================
# FIGURE 2: RESIDUALS ONLY
# ============================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(10, 5))

for idx, (col, title, model, X_model, model_name, y_vals) in enumerate(outcomes_data):
    ax2 = axes2[idx]

    y_pred    = model.predict(X_model)
    residuals = y_vals - y_pred
    r2        = model.score(X_model, y_vals)

    ax2.scatter(y_pred, residuals,
                alpha=0.6, color='steelblue',
                edgecolors='white', linewidth=0.3, s=50)
    ax2.axhline(y=0, color='crimson', linestyle='--', linewidth=2)

    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title(
        f'{plot_titles[col]}\nModel: {model_name}  |  R²={r2:.3f}',
        fontsize=11, fontweight='bold'
    )
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    res_std = residuals.std()
    ax2.axhspan(-res_std, res_std, alpha=0.08, color='steelblue', label='±1 std')
    ax2.legend(fontsize=9)

plt.suptitle(f'Residual Plots — {config["phase_name"]}',
             fontsize=16, fontweight='bold')
plt.tight_layout()

# SAVE 2: RESIDUALS
output_fig2 = f'residuals_{ANALYSIS_PHASE}_phase.png'
plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
print(f"✓ Saved residuals: '{output_fig2}'")
plt.show()

print(f"\nFiles generated:")
print(f"  1. {output_fig1} — Boxplots with significance brackets (corrected labeling)")
print(f"  2. {output_fig2} — Residual plots")

# ============================================================================
# 7. SUMMARY TABLES
# ============================================================================
print("\n" + "="*80)
print(f"SUMMARY: MEAN PERFORMANCE BY CONDITION ({config['phase_name']} - SUCCESSFUL TRIALS)")
print("="*80)
summary_stats = data_success.groupby(['method', 'schedule']).agg({
    config['time_col']: ['mean', 'std', 'count'],
    config['speed_col']: ['mean', 'std'],
    config['distance_col']: ['mean', 'std']
}).round(4)
print(summary_stats)

print("\n" + "="*80)
print("FINAL MODEL SELECTION SUMMARY")
print("="*80)
print(f"\n{config['time_label']:20s}: {model_name_time.upper():20s}  (interaction p={p_time:.4f})")
print(f"{config['speed_label']:20s}: {model_name_speed.upper():20s}  (interaction p={p_speed:.4f})")
print(f"{config['distance_label']:20s}: {model_name_dist.upper():20s}  (interaction p={p_dist:.4f})")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAnalyzed phase: {config['phase_name']}")
print(f"To switch to {'motion' if ANALYSIS_PHASE == 'lift' else 'lift'} phase, change:")
print(f"  ANALYSIS_PHASE = '{ANALYSIS_PHASE}' → ANALYSIS_PHASE = '{'motion' if ANALYSIS_PHASE == 'lift' else 'lift'}'")