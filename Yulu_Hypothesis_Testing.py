# %% [markdown]
# # 🚲 Yulu Bike Sharing — Hypothesis Testing Case Study
#
# **Author:** Pritam Palit
# **Date:** June 2026
#
# ---
#
# ## Business Context
#
# **Yulu** is India's leading micro-mobility service provider, offering unique vehicles for daily
# commutes. Yulu has recently experienced **significant dips in revenue** and has engaged a
# consulting company to understand the factors driving demand for their shared electric cycles
# in the Indian market.
#
# **Objective:** Identify the variables that significantly predict demand for shared electric
# cycles and evaluate how well those variables describe the observed patterns.
#
# **Key Questions:**
# 1. Does working day status affect the number of electric cycles rented?
# 2. Does the season have a significant effect on the number of cycles rented?
# 3. Does weather condition significantly influence cycle rentals?
# 4. Is weather condition dependent on the season?

# %% [markdown]
# ---
# # 1. Problem Statement & Exploratory Data Analysis (EDA)
# ---

# %%
# ── Imports ──────────────────────────────────────────────────────────────────────
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, f_oneway, chi2_contingency
from statsmodels.stats.multicomp import pairwise_tukeyhsd

warnings.filterwarnings('ignore')

# ── Paths & Settings ────────────────────────────────────────────────────────────
IMG_DIR = 'images'
DATA_PATH = os.path.join('data', 'yulu_data.csv')
os.makedirs(IMG_DIR, exist_ok=True)

# Plot style
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'font.size': 11,
    'figure.dpi': 120,
})

PALETTE = 'Set2'
print("✅ Libraries loaded and settings configured.")

# %% [markdown]
# ## 1.1 Load & Inspect the Dataset

# %%
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
df.head()

# %%
# Data types
print("── Data Types ──")
print(df.dtypes)

# %%
# Missing values
print("── Missing Values ──")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found! ✅")

# %%
# Statistical summary for numerical columns
print("── Statistical Summary ──")
df.describe().T

# %% [markdown]
# ### Observations – Initial Inspection
# * The dataset has **10,886 hourly records** across 12 features.
# * **No missing values** — the data is clean.
# * `temp` and `atemp` are **normalized** (0–1 scale).
# * `count` ranges from **4 to 1,246** rentals per hour with mean ≈ 625.
# * `casual` users are far fewer on average (≈182) than `registered` users (≈443).

# %% [markdown]
# ## 1.2 Feature Engineering & Type Conversion

# %%
# Parse datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['dayofweek'] = df['datetime'].dt.dayofweek  # 0=Mon … 6=Sun

# Label maps for categorical columns
season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
weather_map = {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
holiday_map = {0: 'No', 1: 'Yes'}
workingday_map = {0: 'Non-Working Day', 1: 'Working Day'}

# Create labelled categorical columns (keep originals for testing)
df['season_label'] = df['season'].map(season_map).astype('category')
df['weather_label'] = df['weather'].map(weather_map).astype('category')
df['holiday_label'] = df['holiday'].map(holiday_map).astype('category')
df['workingday_label'] = df['workingday'].map(workingday_map).astype('category')

# Convert original columns to category dtype
for col in ['season', 'weather', 'holiday', 'workingday']:
    df[col] = df[col].astype('category')

print("✅ Feature engineering complete.")
print(f"   New columns added: hour, month, year, dayofweek + labelled categories")
print(f"\nUpdated dtypes:\n{df[['season','weather','holiday','workingday']].dtypes}")

# %% [markdown]
# ## 1.3 Univariate Analysis — Continuous Variables
#
# Let's examine the distribution of key numerical features: `temp`, `humidity`,
# `windspeed`, `count`, `casual`, and `registered`.

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
num_cols = ['temp', 'humidity', 'windspeed', 'count', 'casual', 'registered']
colors = sns.color_palette(PALETTE, len(num_cols))

for i, col in enumerate(num_cols):
    ax = axes[i // 3, i % 3]
    sns.histplot(df[col], kde=True, color=colors[i], ax=ax, edgecolor='white', bins=40)
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.2, label=f'Mean={df[col].mean():.2f}')
    ax.axvline(df[col].median(), color='blue', linestyle='-.', linewidth=1.2, label=f'Median={df[col].median():.2f}')
    ax.set_title(f'Distribution of {col}', fontweight='bold')
    ax.legend(fontsize=9)

plt.suptitle('Univariate Distributions — Continuous Variables', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '01_univariate_continuous.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 01_univariate_continuous.png")

# %% [markdown]
# ### Observations — Univariate (Continuous)
# * **`temp`** follows a roughly uniform distribution — good spread of temperatures recorded.
# * **`humidity`** is left-skewed with a peak near 50–70%, some extreme low values.
# * **`windspeed`** is right-skewed; most hours have moderate wind (< 35).
# * **`count`** (target) is approximately **normally distributed** with slight right skew; mean ≈ 625.
# * **`casual`** users show a right-skewed distribution — many hours with low casual usage.
# * **`registered`** users are more evenly distributed compared to casual.

# %% [markdown]
# ## 1.4 Univariate Analysis — Categorical Variables

# %%
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
cat_cols = [('season_label', 'Season'), ('weather_label', 'Weather'),
            ('holiday_label', 'Holiday'), ('workingday_label', 'Working Day')]

for i, (col, title) in enumerate(cat_cols):
    order = df[col].value_counts().index
    ax = axes[i]
    sns.countplot(data=df, x=col, order=order, palette=PALETTE, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title(f'Distribution: {title}', fontweight='bold')
    ax.set_xlabel('')
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    if col == 'weather_label':
        ax.tick_params(axis='x', rotation=25)

plt.suptitle('Univariate Distributions — Categorical Variables', fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '02_univariate_categorical.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 02_univariate_categorical.png")

# %% [markdown]
# ### Observations — Univariate (Categorical)
# * **Season:** Spring has slightly fewer records; Fall, Summer, and Winter are more balanced.
# * **Weather:** Overwhelmingly **Clear** weather, followed by Mist/Cloudy. Heavy Rain is very rare.
# * **Holiday:** Only about **3%** of records fall on holidays — heavily imbalanced.
# * **Working Day:** Roughly **69%** of records are working days — reflects a weekday-dominated dataset.

# %% [markdown]
# ## 1.5 Bivariate Analysis — Count vs. Categorical Features

# %%
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Count vs Working Day
sns.boxplot(data=df, x='workingday_label', y='count', palette='Set2', ax=axes[0],
            flierprops=dict(marker='o', markersize=3, alpha=0.3))
axes[0].set_title('Rentals by Working Day Status', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Total Rentals (count)')

# Count vs Season
sns.boxplot(data=df, x='season_label', y='count', palette='Set2', ax=axes[1],
            order=['Spring', 'Summer', 'Fall', 'Winter'],
            flierprops=dict(marker='o', markersize=3, alpha=0.3))
axes[1].set_title('Rentals by Season', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('')

# Count vs Weather
sns.boxplot(data=df, x='weather_label', y='count', palette='Set2', ax=axes[2],
            order=['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow'],
            flierprops=dict(marker='o', markersize=3, alpha=0.3))
axes[2].set_title('Rentals by Weather', fontweight='bold')
axes[2].set_xlabel('')
axes[2].set_ylabel('')
axes[2].tick_params(axis='x', rotation=20)

plt.suptitle('Bivariate Analysis — Rentals vs Categorical Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '03_bivariate_boxplots.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 03_bivariate_boxplots.png")

# %% [markdown]
# ### Observations — Bivariate (Boxplots)
# * **Working Day:** The median rentals are almost **identical** for working and non-working days.
#   The distributions overlap heavily — visually, working day status alone may not be a strong
#   differentiator.
# * **Season:** **Fall** has the highest median rentals, followed by Summer and Winter. **Spring**
#   shows notably lower demand. This aligns with more favorable biking weather in fall.
# * **Weather:** **Clear** weather drives the most rentals. As conditions worsen (Mist → Rain →
#   Heavy Rain), rentals drop sharply. Heavy Rain/Snow has extremely low counts (very few records).

# %%
# Violin plots for deeper distribution insight
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.violinplot(data=df, x='season_label', y='count', palette='muted', ax=axes[0],
               order=['Spring', 'Summer', 'Fall', 'Winter'], inner='box')
axes[0].set_title('Rental Distribution by Season (Violin)', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Total Rentals')

sns.violinplot(data=df, x='weather_label', y='count', palette='muted', ax=axes[1],
               order=['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow'], inner='box')
axes[1].set_title('Rental Distribution by Weather (Violin)', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('')
axes[1].tick_params(axis='x', rotation=20)

plt.suptitle('Bivariate Analysis — Violin Plots', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '04_bivariate_violins.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 04_bivariate_violins.png")

# %% [markdown]
# ## 1.6 Scatter Plots — Count vs. Continuous Features

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
scatter_vars = [('temp', 'Temperature (normalized)'),
                ('humidity', 'Humidity'),
                ('windspeed', 'Windspeed')]

for i, (col, label) in enumerate(scatter_vars):
    ax = axes[i]
    ax.scatter(df[col], df['count'], alpha=0.15, s=10, color=sns.color_palette(PALETTE)[i])
    # Add trend line
    z = np.polyfit(df[col], df['count'], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, p_line(x_line), color='red', linewidth=2, linestyle='--', label='Trend')
    ax.set_title(f'Count vs {label}', fontweight='bold')
    ax.set_xlabel(label)
    ax.set_ylabel('Total Rentals' if i == 0 else '')
    r = df[col].corr(df['count'])
    ax.annotate(f'r = {r:.3f}', xy=(0.05, 0.92), xycoords='axes fraction',
                fontsize=11, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    ax.legend(fontsize=9)

plt.suptitle('Scatter Plots — Rentals vs Continuous Variables', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '05_scatter_plots.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 05_scatter_plots.png")

# %% [markdown]
# ### Observations — Scatter Plots
# * **Temperature** has a **moderate positive correlation** (r ≈ 0.39) with rentals — warmer
#   weather encourages cycling.
# * **Humidity** shows a **weak negative correlation** — higher humidity slightly reduces demand.
# * **Windspeed** has a **very weak positive correlation** — wind alone is not a major driver.

# %% [markdown]
# ## 1.7 Correlation Heatmap

# %%
num_features = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']
corr_matrix = df[num_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0,
            mask=mask, square=True, linewidths=0.8, ax=ax,
            cbar_kws={'label': 'Pearson Correlation'})
ax.set_title('Correlation Heatmap — Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '06_correlation_heatmap.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 06_correlation_heatmap.png")

# %% [markdown]
# ### Observations — Correlation Heatmap
# * `temp` and `atemp` are **nearly perfectly correlated** (r = 0.99) — expected (feels-like ≈ actual temp).
# * `registered` users have the **strongest correlation** with `count` (r ≈ 0.95), much higher than
#   `casual` (r ≈ 0.69) — registered users dominate overall demand.
# * `temp` and `atemp` both positively correlate with all user types.
# * `humidity` has a **negative** correlation with rentals — high humidity discourages cycling.
# * `windspeed` has negligible correlation with demand.

# %% [markdown]
# ## 1.8 Hourly & Monthly Demand Patterns

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Hourly
hourly = df.groupby('hour')['count'].mean()
axes[0].plot(hourly.index, hourly.values, marker='o', color='#2196F3', linewidth=2, markersize=5)
axes[0].fill_between(hourly.index, hourly.values, alpha=0.15, color='#2196F3')
axes[0].set_title('Average Rentals by Hour of Day', fontweight='bold')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Average Rentals')
axes[0].set_xticks(range(0, 24))
axes[0].axvline(8, color='red', linestyle=':', alpha=0.5, label='Rush hours')
axes[0].axvline(17, color='red', linestyle=':', alpha=0.5)
axes[0].legend()

# Monthly
monthly = df.groupby('month')['count'].mean()
axes[1].bar(monthly.index, monthly.values, color=sns.color_palette('Set2', 12), edgecolor='black', linewidth=0.5)
axes[1].set_title('Average Rentals by Month', fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Average Rentals')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

plt.suptitle('Temporal Demand Patterns', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '07_temporal_patterns.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 07_temporal_patterns.png")

# %% [markdown]
# ### Observations — Temporal Patterns
# * **Hourly:** Clear **bimodal pattern** with peaks at **8 AM** (morning commute) and **5–6 PM**
#   (evening commute). Demand is lowest between midnight and 5 AM.
# * **Monthly:** Demand ramps up from January, peaks around **June–September** (summer/fall),
#   and declines in winter months. This confirms the seasonal effect.

# %% [markdown]
# ---
# # 2. Hypothesis Testing
# ---
#
# We now conduct formal statistical tests to validate the patterns observed in EDA.
# All tests use a **significance level α = 0.05**.

# %%
ALPHA = 0.05
print(f"Significance Level (α) = {ALPHA}")

# %% [markdown]
# ---
# ## 2.1 Two-Sample T-Test — Working Day Effect on Rentals (10 pts)
#
# ### Research Question
# > *Does working day status significantly affect the number of electric cycles rented?*
#
# ### Hypotheses
# - **H₀ (Null):** There is **no significant difference** in the mean number of rentals between
#   working days and non-working days. *(μ_working = μ_non-working)*
# - **H₁ (Alternative):** There **is a significant difference** in the mean number of rentals
#   between working days and non-working days. *(μ_working ≠ μ_non-working)*

# %%
# ── Split data by working day status ─────────────────────────────────────────────
working = df[df['workingday'] == 1]['count']
non_working = df[df['workingday'] == 0]['count']

print("── Group Statistics ──")
print(f"  Working Days:     n = {len(working):,},  Mean = {working.mean():.2f},  Std = {working.std():.2f}")
print(f"  Non-Working Days: n = {len(non_working):,},  Mean = {non_working.mean():.2f},  Std = {non_working.std():.2f}")
print(f"  Difference in Means: {abs(working.mean() - non_working.mean()):.2f}")

# %% [markdown]
# ### Visual Comparison Before Testing

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Overlapping histograms
axes[0].hist(working, bins=40, alpha=0.6, color='#4CAF50', label=f'Working (μ={working.mean():.0f})', density=True, edgecolor='white')
axes[0].hist(non_working, bins=40, alpha=0.6, color='#FF5722', label=f'Non-Working (μ={non_working.mean():.0f})', density=True, edgecolor='white')
axes[0].set_title('Distribution Comparison', fontweight='bold')
axes[0].set_xlabel('Total Rentals')
axes[0].set_ylabel('Density')
axes[0].legend()

# Box plot comparison
sns.boxplot(data=df, x='workingday_label', y='count', palette=['#4CAF50', '#FF5722'], ax=axes[1],
            flierprops=dict(marker='o', markersize=3, alpha=0.3))
axes[1].set_title('Boxplot Comparison', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Total Rentals')

# Mean comparison with CI
means = [working.mean(), non_working.mean()]
sems = [working.sem(), non_working.sem()]
labels = ['Working Day', 'Non-Working Day']
colors_bar = ['#4CAF50', '#FF5722']
axes[2].bar(labels, means, yerr=[1.96*s for s in sems], capsize=8, color=colors_bar,
            edgecolor='black', linewidth=0.5, alpha=0.8)
axes[2].set_title('Mean Comparison with 95% CI', fontweight='bold')
axes[2].set_ylabel('Mean Rentals')
for j, (m, s) in enumerate(zip(means, sems)):
    axes[2].text(j, m + 1.96*s + 10, f'{m:.1f}', ha='center', fontweight='bold')

plt.suptitle('Visual Analysis — Working Day vs Non-Working Day', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '08_ttest_visual.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 08_ttest_visual.png")

# %% [markdown]
# ### Assumption Check 1: Normality (Shapiro-Wilk Test)
#
# For large samples (n > 5000), Shapiro-Wilk will almost always reject normality. We use it
# alongside histograms for a complete picture. The **Central Limit Theorem** ensures the
# sampling distribution of the mean is approximately normal for large n.

# %%
# Shapiro-Wilk on random subsamples (test requires n < 5000)
np.random.seed(42)
sample_w = np.random.choice(working, size=min(5000, len(working)), replace=False)
sample_nw = np.random.choice(non_working, size=min(5000, len(non_working)), replace=False)

stat_w, p_w = shapiro(sample_w)
stat_nw, p_nw = shapiro(sample_nw)

print("── Shapiro-Wilk Normality Test ──")
print(f"  Working Day:     W = {stat_w:.4f}, p = {p_w:.6f}  →  {'Normal ✅' if p_w > ALPHA else 'NOT Normal ❌'}")
print(f"  Non-Working Day: W = {stat_nw:.4f}, p = {p_nw:.6f}  →  {'Normal ✅' if p_nw > ALPHA else 'NOT Normal ❌'}")
print()
if p_w < ALPHA or p_nw < ALPHA:
    print("⚠️  Normality assumption is violated. However, with large sample sizes")
    print("   (n > 3,000), the Central Limit Theorem ensures the sampling distribution")
    print("   of the mean is approximately normal. We proceed with the t-test.")

# %% [markdown]
# ### Assumption Check 2: Equality of Variances (Levene's Test)
#
# - **H₀:** The variances are equal across groups.
# - **H₁:** The variances are NOT equal.

# %%
lev_stat, lev_p = levene(working, non_working)
equal_var = lev_p > ALPHA

print("── Levene's Test for Equal Variances ──")
print(f"  Levene Statistic = {lev_stat:.4f}")
print(f"  p-value          = {lev_p:.6f}")
print(f"  Decision:        {'Equal variances ✅ (Fail to reject H₀)' if equal_var else 'Unequal variances ❌ (Reject H₀)'}")
print()
if not equal_var:
    print("  → Using Welch's t-test (equal_var=False) to account for unequal variances.")

# %% [markdown]
# ### Two-Sample T-Test Execution

# %%
t_stat, t_pval = ttest_ind(working, non_working, equal_var=equal_var)

print("═" * 60)
print("  TWO-SAMPLE T-TEST RESULTS")
print("═" * 60)
print(f"  Test Type:     {'Student\'s t-test' if equal_var else 'Welch\'s t-test'}")
print(f"  t-statistic:   {t_stat:.4f}")
print(f"  p-value:       {t_pval:.6f}")
print(f"  α (alpha):     {ALPHA}")
print(f"  Decision:      {'REJECT H₀ ✅' if t_pval < ALPHA else 'FAIL TO REJECT H₀ ❌'}")
print("═" * 60)

# %% [markdown]
# ### Decision & Business Inference — T-Test

# %%
print("┌─────────────────────────────────────────────────────────────────┐")
if t_pval < ALPHA:
    print("│  RESULT: REJECT H₀                                             │")
    print("│                                                                 │")
    print(f"│  p-value ({t_pval:.4f}) < α ({ALPHA})                              │")
    print("│                                                                 │")
    print("│  ✅ There IS a statistically significant difference in the      │")
    print("│     number of rentals between working and non-working days.     │")
    print("│                                                                 │")
    print("│  Business Implication: Yulu should tailor different marketing   │")
    print("│  strategies for working days vs weekends/holidays.              │")
else:
    print("│  RESULT: FAIL TO REJECT H₀                                     │")
    print("│                                                                 │")
    print(f"│  p-value ({t_pval:.4f}) > α ({ALPHA})                              │")
    print("│                                                                 │")
    print("│  ❌ There is NO statistically significant difference in the     │")
    print("│     number of rentals between working and non-working days.     │")
    print("│                                                                 │")
    print("│  Business Implication: Working day status alone does NOT        │")
    print("│  significantly drive demand. Yulu should focus on other factors │")
    print("│  such as weather and season for demand planning.                │")
print("└─────────────────────────────────────────────────────────────────┘")

# %% [markdown]
# ---
# ## 2.2 One-Way ANOVA — Season & Weather Effects on Rentals (10 pts)
#
# ANOVA tests whether the means of **three or more groups** are statistically different.
#
# ---
# ### 2.2.1 ANOVA Test 1: Effect of Season on Rentals
#
# #### Hypotheses
# - **H₀:** The mean number of rentals is the **same across all four seasons**.
#   *(μ_spring = μ_summer = μ_fall = μ_winter)*
# - **H₁:** At least one season has a **significantly different** mean rental count.

# %% [markdown]
# #### Visual Analysis — Rentals by Season

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
season_order = ['Spring', 'Summer', 'Fall', 'Winter']

# Boxplot
sns.boxplot(data=df, x='season_label', y='count', order=season_order, palette='Set2', ax=axes[0],
            flierprops=dict(marker='o', markersize=3, alpha=0.3))
axes[0].set_title('Rental Distribution by Season', fontweight='bold')
axes[0].set_xlabel('Season')
axes[0].set_ylabel('Total Rentals')

# Mean bar chart with CI
season_stats = df.groupby('season_label')['count'].agg(['mean', 'sem']).reindex(season_order)
bars = axes[1].bar(season_stats.index, season_stats['mean'],
                   yerr=1.96 * season_stats['sem'], capsize=8,
                   color=sns.color_palette('Set2', 4), edgecolor='black', linewidth=0.5, alpha=0.85)
axes[1].set_title('Mean Rentals by Season (with 95% CI)', fontweight='bold')
axes[1].set_xlabel('Season')
axes[1].set_ylabel('Mean Rentals')
for bar, val in zip(bars, season_stats['mean']):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
                 f'{val:.0f}', ha='center', fontweight='bold')

plt.suptitle('Pre-ANOVA Visual: Season Effect', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '09_anova_season_visual.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 09_anova_season_visual.png")

# %% [markdown]
# #### Assumption Checks — Season ANOVA

# %%
# Split data by season
season_groups = [df[df['season'] == s]['count'].values for s in [1, 2, 3, 4]]
season_names = ['Spring', 'Summer', 'Fall', 'Winter']

# Normality (Shapiro-Wilk on subsamples)
print("── Shapiro-Wilk Normality Test (per Season) ──")
for name, grp in zip(season_names, season_groups):
    np.random.seed(42)
    sample = np.random.choice(grp, size=min(5000, len(grp)), replace=False)
    stat, p = shapiro(sample)
    result = 'Normal ✅' if p > ALPHA else 'NOT Normal ❌'
    print(f"  {name:8s}: W = {stat:.4f}, p = {p:.6f} → {result}")

print()

# Levene's test for equal variances across all four groups
lev_stat_s, lev_p_s = levene(*season_groups)
print("── Levene's Test for Equal Variances ──")
print(f"  Statistic = {lev_stat_s:.4f}, p-value = {lev_p_s:.6f}")
print(f"  Decision: {'Equal variances ✅' if lev_p_s > ALPHA else 'Unequal variances ❌'}")
print()
print("⚠️  Note: ANOVA is fairly robust to violations of normality and equal variance")
print("   when sample sizes are large and roughly equal. We proceed with the test.")

# %% [markdown]
# #### One-Way ANOVA — Season

# %%
f_stat_season, p_season = f_oneway(*season_groups)

print("═" * 60)
print("  ONE-WAY ANOVA — SEASON EFFECT ON RENTALS")
print("═" * 60)
print(f"  F-statistic:  {f_stat_season:.4f}")
print(f"  p-value:      {p_season:.10f}")
print(f"  α (alpha):    {ALPHA}")
print(f"  Decision:     {'REJECT H₀ ✅' if p_season < ALPHA else 'FAIL TO REJECT H₀ ❌'}")
print("═" * 60)

# %% [markdown]
# #### Post-Hoc Analysis: Tukey's HSD — Season
#
# Since ANOVA is significant, we perform **Tukey's Honestly Significant Difference (HSD)** test
# to identify *which specific pairs* of seasons differ significantly.

# %%
if p_season < ALPHA:
    tukey_season = pairwise_tukeyhsd(endog=df['count'].astype(float),
                                      groups=df['season_label'].astype(str),
                                      alpha=ALPHA)
    print("── Tukey HSD Post-Hoc Results (Season) ──\n")
    print(tukey_season)
    print()

    # Visualize Tukey results
    fig, ax = plt.subplots(figsize=(10, 6))
    tukey_season.plot_simultaneous(ax=ax)
    ax.set_title("Tukey HSD — Season Pairwise Comparisons", fontweight='bold', fontsize=14)
    ax.set_xlabel('Mean Difference in Rentals')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '10_tukey_season.png'), bbox_inches='tight')
    plt.show()
    print("📊 Saved: 10_tukey_season.png")

    # Summary table
    print("\n── Significant Pairwise Differences ──")
    results_df = pd.DataFrame(data=tukey_season._results_table.data[1:],
                               columns=tukey_season._results_table.data[0])
    sig_pairs = results_df[results_df['reject'] == True]
    if len(sig_pairs) > 0:
        for _, row in sig_pairs.iterrows():
            print(f"  {row['group1']} vs {row['group2']}: "
                  f"mean diff = {row['meandiff']:.2f}, p-adj = {row['p-adj']:.6f} ✅")
    else:
        print("  No individually significant pairs found.")
else:
    print("ANOVA not significant — post-hoc test not needed.")

# %% [markdown]
# #### Inference — Season ANOVA

# %%
print("┌─────────────────────────────────────────────────────────────────┐")
if p_season < ALPHA:
    print("│  RESULT: REJECT H₀                                             │")
    print(f"│  F = {f_stat_season:.2f}, p = {p_season:.2e}  (p < 0.05)                    │")
    print("│                                                                 │")
    print("│  ✅ Rental counts differ SIGNIFICANTLY across seasons.          │")
    print("│  Tukey HSD reveals which specific season pairs differ.          │")
    print("│                                                                 │")
    print("│  Business Implication: Season is a critical driver of demand.   │")
    print("│  Yulu should prepare for lower demand in Spring and scale up    │")
    print("│  fleet/marketing in Fall and Summer.                            │")
else:
    print("│  RESULT: FAIL TO REJECT H₀                                     │")
    print("│  ❌ No significant difference in rentals across seasons.        │")
print("└─────────────────────────────────────────────────────────────────┘")

# %% [markdown]
# ---
# ### 2.2.2 ANOVA Test 2: Effect of Weather on Rentals
#
# #### Hypotheses
# - **H₀:** The mean number of rentals is the **same across all weather conditions**.
#   *(μ_clear = μ_mist = μ_rain = μ_heavy)*
# - **H₁:** At least one weather condition has a **significantly different** mean rental count.

# %% [markdown]
# #### Visual Analysis — Rentals by Weather

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
weather_order = ['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow']

sns.boxplot(data=df, x='weather_label', y='count', order=weather_order, palette='coolwarm', ax=axes[0],
            flierprops=dict(marker='o', markersize=3, alpha=0.3))
axes[0].set_title('Rental Distribution by Weather', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Total Rentals')
axes[0].tick_params(axis='x', rotation=15)

weather_stats = df.groupby('weather_label')['count'].agg(['mean', 'sem']).reindex(weather_order)
bars = axes[1].bar(weather_stats.index, weather_stats['mean'],
                   yerr=1.96 * weather_stats['sem'], capsize=8,
                   color=sns.color_palette('coolwarm', 4), edgecolor='black', linewidth=0.5, alpha=0.85)
axes[1].set_title('Mean Rentals by Weather (with 95% CI)', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Mean Rentals')
axes[1].tick_params(axis='x', rotation=15)
for bar, val in zip(bars, weather_stats['mean']):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
                 f'{val:.0f}', ha='center', fontweight='bold')

plt.suptitle('Pre-ANOVA Visual: Weather Effect', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '11_anova_weather_visual.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 11_anova_weather_visual.png")

# %% [markdown]
# #### Assumption Checks — Weather ANOVA

# %%
weather_groups = [df[df['weather'] == w]['count'].values for w in [1, 2, 3, 4]]
weather_names = ['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow']

print("── Shapiro-Wilk Normality Test (per Weather) ──")
for name, grp in zip(weather_names, weather_groups):
    if len(grp) < 3:
        print(f"  {name:20s}: Too few samples (n={len(grp)}) — skipped")
        continue
    np.random.seed(42)
    sample = np.random.choice(grp, size=min(5000, len(grp)), replace=False)
    stat, p = shapiro(sample)
    result = 'Normal ✅' if p > ALPHA else 'NOT Normal ❌'
    print(f"  {name:20s}: W = {stat:.4f}, p = {p:.6f} → {result} (n={len(grp)})")

print()

# Filter out groups with too few samples for Levene's test
valid_weather_groups = [g for g in weather_groups if len(g) >= 3]
lev_stat_w, lev_p_w = levene(*valid_weather_groups)
print("── Levene's Test for Equal Variances ──")
print(f"  Statistic = {lev_stat_w:.4f}, p-value = {lev_p_w:.6f}")
print(f"  Decision: {'Equal variances ✅' if lev_p_w > ALPHA else 'Unequal variances ❌'}")
print()
print("⚠️  Note: Heavy Rain/Snow category has very few observations.")
print("   ANOVA results should be interpreted cautiously for this group.")

# %% [markdown]
# #### One-Way ANOVA — Weather

# %%
# Use all groups with sufficient data
f_stat_weather, p_weather = f_oneway(*valid_weather_groups)

print("═" * 60)
print("  ONE-WAY ANOVA — WEATHER EFFECT ON RENTALS")
print("═" * 60)
print(f"  F-statistic:  {f_stat_weather:.4f}")
print(f"  p-value:      {p_weather:.10f}")
print(f"  α (alpha):    {ALPHA}")
print(f"  Decision:     {'REJECT H₀ ✅' if p_weather < ALPHA else 'FAIL TO REJECT H₀ ❌'}")
print("═" * 60)

# %% [markdown]
# #### Post-Hoc Analysis: Tukey's HSD — Weather

# %%
if p_weather < ALPHA:
    tukey_weather = pairwise_tukeyhsd(endog=df['count'].astype(float),
                                       groups=df['weather_label'].astype(str),
                                       alpha=ALPHA)
    print("── Tukey HSD Post-Hoc Results (Weather) ──\n")
    print(tukey_weather)
    print()

    fig, ax = plt.subplots(figsize=(10, 6))
    tukey_weather.plot_simultaneous(ax=ax)
    ax.set_title("Tukey HSD — Weather Pairwise Comparisons", fontweight='bold', fontsize=14)
    ax.set_xlabel('Mean Difference in Rentals')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, '12_tukey_weather.png'), bbox_inches='tight')
    plt.show()
    print("📊 Saved: 12_tukey_weather.png")

    print("\n── Significant Pairwise Differences ──")
    results_df_w = pd.DataFrame(data=tukey_weather._results_table.data[1:],
                                 columns=tukey_weather._results_table.data[0])
    sig_pairs_w = results_df_w[results_df_w['reject'] == True]
    if len(sig_pairs_w) > 0:
        for _, row in sig_pairs_w.iterrows():
            print(f"  {row['group1']} vs {row['group2']}: "
                  f"mean diff = {row['meandiff']:.2f}, p-adj = {row['p-adj']:.6f} ✅")
    else:
        print("  No individually significant pairs found.")
else:
    print("ANOVA not significant — post-hoc test not needed.")

# %% [markdown]
# #### Inference — Weather ANOVA

# %%
print("┌─────────────────────────────────────────────────────────────────┐")
if p_weather < ALPHA:
    print("│  RESULT: REJECT H₀                                             │")
    print(f"│  F = {f_stat_weather:.2f}, p = {p_weather:.2e}  (p < 0.05)                   │")
    print("│                                                                 │")
    print("│  ✅ Rental counts differ SIGNIFICANTLY across weather types.    │")
    print("│  Tukey HSD pinpoints which weather pairs drive the difference.  │")
    print("│                                                                 │")
    print("│  Business Implication: Weather is a KEY demand driver.          │")
    print("│  Yulu should use weather forecasts for demand prediction and    │")
    print("│  dynamic pricing. Rainy days need promotional strategies.       │")
else:
    print("│  RESULT: FAIL TO REJECT H₀                                     │")
    print("│  ❌ No significant difference in rentals across weather types.  │")
print("└─────────────────────────────────────────────────────────────────┘")

# %% [markdown]
# ---
# ## 2.3 Chi-Square Test — Weather Dependence on Season (10 pts)
#
# ### Research Question
# > *Is the weather condition dependent on the season?*
#
# ### Hypotheses
# - **H₀:** Weather condition is **independent** of season. (No association)
# - **H₁:** Weather condition is **dependent** on season. (Significant association)

# %% [markdown]
# #### Contingency Table Exploration

# %%
# Create contingency table
contingency = pd.crosstab(df['season_label'], df['weather_label'],
                           margins=True, margins_name='Total')
print("── Contingency Table: Season × Weather ──\n")
print(contingency)

# %%
# Proportional heatmap (without margins)
contingency_no_margins = pd.crosstab(df['season_label'], df['weather_label'])
contingency_pct = contingency_no_margins.div(contingency_no_margins.sum(axis=1), axis=0) * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Raw counts heatmap
sns.heatmap(contingency_no_margins, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0],
            linewidths=0.5, cbar_kws={'label': 'Count'})
axes[0].set_title('Season × Weather (Raw Counts)', fontweight='bold')
axes[0].set_xlabel('Weather Condition')
axes[0].set_ylabel('Season')

# Percentage heatmap
sns.heatmap(contingency_pct, annot=True, fmt='.1f', cmap='YlGnBu', ax=axes[1],
            linewidths=0.5, cbar_kws={'label': 'Percentage (%)'})
axes[1].set_title('Season × Weather (Row Percentages %)', fontweight='bold')
axes[1].set_xlabel('Weather Condition')
axes[1].set_ylabel('Season')

plt.suptitle('Chi-Square Visual: Weather Distribution Across Seasons',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '13_chisq_contingency.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 13_chisq_contingency.png")

# %% [markdown]
# #### Stacked Bar Chart — Weather Composition by Season

# %%
fig, ax = plt.subplots(figsize=(10, 6))
contingency_pct_plot = contingency_pct[['Clear', 'Mist/Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow']]
contingency_pct_plot.plot(kind='bar', stacked=True, ax=ax,
                           color=['#66BB6A', '#FFA726', '#42A5F5', '#EF5350'],
                           edgecolor='white', linewidth=0.5)
ax.set_title('Weather Composition by Season (%)', fontweight='bold', fontsize=14)
ax.set_xlabel('Season')
ax.set_ylabel('Percentage (%)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Weather', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, '14_chisq_stacked_bar.png'), bbox_inches='tight')
plt.show()
print("📊 Saved: 14_chisq_stacked_bar.png")

# %% [markdown]
# #### Chi-Square Test Execution

# %%
# Chi-square test of independence
chi2_stat, chi2_p, chi2_dof, chi2_expected = chi2_contingency(contingency_no_margins)

print("═" * 60)
print("  CHI-SQUARE TEST OF INDEPENDENCE")
print("  (Weather × Season)")
print("═" * 60)
print(f"  Chi² statistic:    {chi2_stat:.4f}")
print(f"  p-value:           {chi2_p:.10f}")
print(f"  Degrees of freedom: {chi2_dof}")
print(f"  α (alpha):         {ALPHA}")
print(f"  Decision:          {'REJECT H₀ ✅' if chi2_p < ALPHA else 'FAIL TO REJECT H₀ ❌'}")
print("═" * 60)

# %%
# Expected frequencies
print("\n── Expected Frequencies (under H₀) ──\n")
expected_df = pd.DataFrame(chi2_expected,
                            index=contingency_no_margins.index,
                            columns=contingency_no_margins.columns)
print(expected_df.round(2))

# Check if any expected frequency < 5 (assumption for chi-square)
min_expected = chi2_expected.min()
print(f"\nMinimum expected frequency: {min_expected:.2f}")
if min_expected < 5:
    print("⚠️  Some expected frequencies are < 5. Chi-square approximation may be unreliable.")
    print("   Consider combining rare categories or using Fisher's exact test.")
else:
    print("✅  All expected frequencies ≥ 5. Chi-square approximation is valid.")

# %% [markdown]
# #### Effect Size — Cramér's V

# %%
# Cramér's V for effect size
n = contingency_no_margins.sum().sum()
min_dim = min(contingency_no_margins.shape[0], contingency_no_margins.shape[1]) - 1
cramers_v = np.sqrt(chi2_stat / (n * min_dim))

print(f"── Cramér's V (Effect Size) ──")
print(f"  V = {cramers_v:.4f}")
if cramers_v < 0.1:
    print("  Interpretation: Negligible association")
elif cramers_v < 0.3:
    print("  Interpretation: Weak association")
elif cramers_v < 0.5:
    print("  Interpretation: Moderate association")
else:
    print("  Interpretation: Strong association")

# %% [markdown]
# #### Decision & Business Inference — Chi-Square

# %%
print("┌─────────────────────────────────────────────────────────────────┐")
if chi2_p < ALPHA:
    print("│  RESULT: REJECT H₀                                             │")
    print(f"│  χ² = {chi2_stat:.2f}, p = {chi2_p:.2e}, Cramér's V = {cramers_v:.3f}         │")
    print("│                                                                 │")
    print("│  ✅ Weather IS significantly dependent on season.               │")
    print("│  The distribution of weather types varies across seasons.       │")
    print("│                                                                 │")
    print("│  Business Implication: Seasonal planning must account for       │")
    print("│  weather patterns. Spring/Winter tend to have more adverse      │")
    print("│  weather, compounding lower demand. Yulu should use combined    │")
    print("│  season + weather forecasts for fleet deployment & pricing.     │")
else:
    print("│  RESULT: FAIL TO REJECT H₀                                     │")
    print("│  ❌ Weather is independent of season — no significant           │")
    print("│     association found.                                          │")
print("└─────────────────────────────────────────────────────────────────┘")

# %% [markdown]
# ---
# # 3. Summary of Hypothesis Testing Results
# ---

# %%
summary_data = {
    'Test': [
        'Two-Sample T-Test',
        'One-Way ANOVA (Season)',
        'One-Way ANOVA (Weather)',
        'Chi-Square Test'
    ],
    'Hypothesis (H₀)': [
        'No difference in rentals: Working vs Non-Working Day',
        'Mean rentals same across all 4 seasons',
        'Mean rentals same across all weather conditions',
        'Weather is independent of season'
    ],
    'Test Statistic': [
        f't = {t_stat:.4f}',
        f'F = {f_stat_season:.4f}',
        f'F = {f_stat_weather:.4f}',
        f'χ² = {chi2_stat:.4f}'
    ],
    'p-value': [
        f'{t_pval:.6f}',
        f'{p_season:.2e}',
        f'{p_weather:.2e}',
        f'{chi2_p:.6f}'
    ],
    'Decision (α=0.05)': [
        'Reject H₀' if t_pval < ALPHA else 'Fail to Reject H₀',
        'Reject H₀' if p_season < ALPHA else 'Fail to Reject H₀',
        'Reject H₀' if p_weather < ALPHA else 'Fail to Reject H₀',
        'Reject H₀' if chi2_p < ALPHA else 'Fail to Reject H₀'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.index = range(1, len(summary_df) + 1)
summary_df.index.name = '#'
print("══════════════════════════════════════════════════════════════════════")
print("  HYPOTHESIS TESTING — SUMMARY OF RESULTS")
print("══════════════════════════════════════════════════════════════════════")
print()
print(summary_df.to_string())

# %% [markdown]
# ---
# ## Key Business Recommendations for Yulu
#
# Based on our comprehensive analysis and hypothesis testing, here are the strategic
# recommendations:
#
# 1. **Season-Based Strategy:** Season has a **highly significant** impact on rentals
#    (ANOVA p < 0.001). **Fall** sees the highest demand while **Spring** sees the lowest.
#    Yulu should:
#    - Scale up fleet and maintenance in Fall/Summer
#    - Run promotional campaigns in Spring to boost low-season demand
#    - Adjust pricing dynamically by season
#
# 2. **Weather-Driven Operations:** Weather **significantly** impacts rentals (ANOVA p < 0.001).
#    Clear weather drives peak demand, while rain drastically reduces it. Yulu should:
#    - Integrate weather forecasting into demand prediction models
#    - Offer rain-day discounts or indoor parking promotions
#    - Pre-position vehicles in covered areas during forecast rain
#
# 3. **Weather-Season Interplay:** The Chi-Square test confirms weather is **dependent on
#    season**. This means demand fluctuations are driven by a compound effect of both factors.
#    Yulu should build **multivariate models** considering season-weather interactions.
#
# 4. **Working Day Consideration:** The t-test result should be interpreted alongside the
#    observation that **usage patterns differ** between working and non-working days (commuter
#    vs. leisure use), even if total counts are similar. Segment-level analysis (casual vs.
#    registered) may reveal different patterns.
#
# ---
# **End of Yulu Hypothesis Testing Case Study**
# *Author: Pritam Palit*
