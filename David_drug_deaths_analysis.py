"""
================================================================
LD7186 Big Data Analytics — Section 1
Accidental Drug-Related Deaths, Connecticut (2012-2024)

Author : David
Module : LD7186 Big Data Analytics
Date   : May 2026
================================================================

Methodology: CRISP-DM (Wirth & Hipp, 2000).
This script implements the full analytical workflow:
    1. Data acquisition
    2. Exploratory data analysis
    3. Pre-processing and feature engineering
    4. Descriptive analysis answering RQ1-RQ4
    5. Inferential statistical testing (Welch t-test, chi-square)
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Visualisation defaults
# ---------------------------------------------------------------
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]   = 110
plt.rcParams["savefig.dpi"]  = 150
plt.rcParams["savefig.bbox"] = "tight"

OUTPUT_DIR = "./figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================================================
# Phase 1 — Data acquisition
# ===============================================================
print("=" * 70)
print("Phase 1 — Acquiring the dataset")
print("=" * 70)

dataset = pd.read_csv("Accidental_Drug_Related_Deaths_2012-2024.csv")
print(f"Records loaded   : {dataset.shape[0]:,}")
print(f"Attributes loaded: {dataset.shape[1]}")


# ===============================================================
# Phase 2 — Exploratory data analysis
# ===============================================================
print("\n" + "=" * 70)
print("Phase 2 — Exploratory Data Analysis (EDA)")
print("=" * 70)

print("\n[Sample of first five records]")
print(dataset.head())

print("\n[Variable types]")
print(dataset.dtypes)

print("\n[Numerical summary statistics]")
print(dataset.describe())

print("\n[Top-15 columns ranked by missingness]")
missingness = dataset.isnull().sum().sort_values(ascending=False)
print(missingness.head(15))


# ===============================================================
# Phase 3 — Data pre-processing
# ===============================================================
print("\n" + "=" * 70)
print("Phase 3 — Data Pre-processing")
print("=" * 70)

# 3.1 Parse temporal information
dataset["Date"]  = pd.to_datetime(dataset["Date"], errors="coerce")
dataset["Year"]  = dataset["Date"].dt.year
dataset["Month"] = dataset["Date"].dt.month

# 3.2 Convert substance indicator columns from 'Y'/blank to 1/0
SUBSTANCE_COLUMNS = [
    "Heroin", "Cocaine", "Fentanyl", "Fentanyl Analogue", "Oxycodone",
    "Oxymorphone", "Ethanol", "Hydrocodone", "Benzodiazepine", "Methadone",
    "Meth/Amphetamine", "Amphet", "Tramad", "Hydromorphone",
    "Morphine (Not Heroin)", "Xylazine", "Gabapentin", "Opiate NOS",
    "Heroin/Morph/Codeine", "Other Opioid", "Any Opioid",
]

for substance in SUBSTANCE_COLUMNS:
    if substance in dataset.columns:
        dataset[substance] = (
            dataset[substance].astype(str).str.strip().str.upper() == "Y"
        ).astype(int)

# 3.3 Standardise demographic variables
dataset["Age"] = pd.to_numeric(dataset["Age"], errors="coerce")
dataset["Sex"] = dataset["Sex"].astype(str).str.strip().str.title()
dataset.loc[~dataset["Sex"].isin(["Male", "Female"]), "Sex"] = np.nan
dataset["Race"] = dataset["Race"].astype(str).str.strip()

# 3.4 Drop records missing essential analytical fields
print(f"Records before cleanup : {len(dataset):,}")
cleaned = dataset.dropna(subset=["Year", "Age", "Sex"]).copy()
cleaned["Year"] = cleaned["Year"].astype(int)
print(f"Records after cleanup  : {len(cleaned):,}")
print(f"Records discarded      : {len(dataset) - len(cleaned):,} "
      f"({100*(len(dataset)-len(cleaned))/len(dataset):.2f} %)")


# ===============================================================
# Phase 4 — Descriptive analysis
# ===============================================================
print("\n" + "=" * 70)
print("Phase 4 — Descriptive Analysis")
print("=" * 70)

print(f"\nMean age        : {cleaned['Age'].mean():.2f}")
print(f"Median age      : {cleaned['Age'].median():.2f}")
print(f"Standard dev    : {cleaned['Age'].std():.2f}")
print(f"Age range       : {cleaned['Age'].min()}-{cleaned['Age'].max()}")
print(f"\nSex composition:\n{cleaned['Sex'].value_counts()}")


# ---------------------------------------------------------------
# Research Question 1 — Temporal trend
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("RQ1: How have accidental drug-related deaths changed (2012-2024)?")
print("-" * 70)

annual_counts = cleaned.groupby("Year").size()
print(annual_counts)

fig, ax = plt.subplots(figsize=(10, 5))
annual_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Accidental Drug-Related Deaths, Connecticut (2012-2024)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Deaths")
for idx, value in enumerate(annual_counts.values):
    ax.text(idx, value + 15, str(value), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_annual_deaths.png")
plt.close()


# ---------------------------------------------------------------
# Research Question 2 — Substances most frequently involved
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("RQ2: Which substances are most commonly involved?")
print("-" * 70)

substance_totals = cleaned[SUBSTANCE_COLUMNS].sum().sort_values(ascending=False)
print(substance_totals)

fig, ax = plt.subplots(figsize=(10, 6))
substance_totals.head(15).plot(kind="barh", ax=ax,
                                color="crimson", edgecolor="black")
ax.invert_yaxis()
ax.set_title("Top 15 Substances Detected in Overdose Deaths",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Deaths")
for idx, value in enumerate(substance_totals.head(15).values):
    ax.text(value + 50, idx, str(value), va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_top_substances.png")
plt.close()

# Trend of the three most prominent substances
fentanyl_trend = cleaned.groupby("Year")["Fentanyl"].sum()
heroin_trend   = cleaned.groupby("Year")["Heroin"].sum()
cocaine_trend  = cleaned.groupby("Year")["Cocaine"].sum()

fig, ax = plt.subplots(figsize=(11, 5))
fentanyl_trend.plot(ax=ax, marker="o", label="Fentanyl",
                    linewidth=2.5, color="red")
heroin_trend.plot(ax=ax, marker="s", label="Heroin",
                  linewidth=2.5, color="brown")
cocaine_trend.plot(ax=ax, marker="^", label="Cocaine",
                   linewidth=2.5, color="blue")
ax.set_title("Annual Trend — Top Three Substances",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Deaths")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_substance_trend.png")
plt.close()


# ---------------------------------------------------------------
# Research Question 3 — Demographic profile
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("RQ3: What is the demographic profile of victims?")
print("-" * 70)

print(cleaned.groupby("Sex")["Age"].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(cleaned["Age"].dropna(), bins=30,
             color="teal", edgecolor="black")
axes[0].set_title("Age Distribution — All Victims", fontweight="bold")
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Frequency")
axes[0].axvline(cleaned["Age"].mean(), color="red", linestyle="--",
                label=f"Mean = {cleaned['Age'].mean():.1f}")
axes[0].legend()

sns.boxplot(data=cleaned, x="Sex", y="Age", ax=axes[1],
            palette={"Male": "#3498db", "Female": "#e74c3c"})
axes[1].set_title("Age Distribution by Sex", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_age_sex.png")
plt.close()


# ---------------------------------------------------------------
# Research Question 4 — Sex vs Fentanyl involvement
# ---------------------------------------------------------------
print("\n" + "-" * 70)
print("RQ4: Is fentanyl involvement associated with sex?")
print("-" * 70)

contingency_table = pd.crosstab(cleaned["Sex"], cleaned["Fentanyl"])
print(contingency_table)
print("\nProportion of fentanyl-positive deaths by sex (%):")
print((contingency_table[1] / contingency_table.sum(axis=1) * 100).round(2))

fig, ax = plt.subplots(figsize=(8, 5))
contingency_table.plot(kind="bar", ax=ax,
                       color=["lightgray", "darkred"], edgecolor="black")
ax.set_title("Fentanyl Involvement by Sex", fontsize=13, fontweight="bold")
ax.set_xlabel("Sex")
ax.set_ylabel("Number of Deaths")
ax.legend(["No Fentanyl", "Fentanyl Detected"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_fentanyl_by_sex.png")
plt.close()


# ===============================================================
# Phase 5 — Hypothesis testing
# ===============================================================
print("\n" + "=" * 70)
print("Phase 5 — Inferential Statistical Testing")
print("=" * 70)

# ---------------------------------------------------------------
# Hypothesis 1 — Welch's two-sample t-test (mean age by sex)
# ---------------------------------------------------------------
print("\nHypothesis 1 (Welch's two-sample t-test):")
print("H0: There is NO significant difference in mean age between sexes.")
print("H1: There IS a significant difference in mean age between sexes.")

age_male   = cleaned.loc[cleaned["Sex"] == "Male",   "Age"].dropna()
age_female = cleaned.loc[cleaned["Sex"] == "Female", "Age"].dropna()
print(f"Male   : n={len(age_male):,}, "
      f"mean={age_male.mean():.2f}, sd={age_male.std():.2f}")
print(f"Female : n={len(age_female):,}, "
      f"mean={age_female.mean():.2f}, sd={age_female.std():.2f}")

t_stat, p_value = stats.ttest_ind(age_male, age_female, equal_var=False)
print(f"\nt-statistic = {t_stat:.4f}")
print(f"p-value     = {p_value:.6f}")
print(f"Decision    : "
      f"{'REJECT H0' if p_value < 0.05 else 'FAIL to reject H0'}")

# ---------------------------------------------------------------
# Hypothesis 2 — Chi-square test of independence
# ---------------------------------------------------------------
print("\nHypothesis 2 (Chi-square test of independence):")
print("H0: Sex and fentanyl involvement are INDEPENDENT.")
print("H1: Sex and fentanyl involvement are NOT independent.")

chi2_stat, p_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"chi-square = {chi2_stat:.4f}")
print(f"df         = {dof}")
print(f"p-value    = {p_chi:.6f}")

n_total    = contingency_table.values.sum()
cramers_v  = np.sqrt(chi2_stat / (n_total * (min(contingency_table.shape) - 1)))
print(f"Cramer's V (effect size) = {cramers_v:.4f}")
print(f"Decision   : "
      f"{'REJECT H0' if p_chi < 0.05 else 'FAIL to reject H0'}")


# ---------------------------------------------------------------
# Supplementary — correlation matrix & temporal heatmap
# ---------------------------------------------------------------
top_substances    = substance_totals.head(8).index.tolist()
correlation_matrix = cleaned[top_substances].corr()
print("\nCorrelation matrix (top 8 substances):")
print(correlation_matrix.round(3))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, cbar_kws={"label": "Correlation"},
            ax=ax, square=True)
ax.set_title("Correlation Matrix — Top Substances",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_correlation.png")
plt.close()

year_month_pivot = cleaned.pivot_table(index="Month", columns="Year",
                                        values="Age", aggfunc="count")
fig, ax = plt.subplots(figsize=(11, 6))
sns.heatmap(year_month_pivot, annot=True, fmt=".0f", cmap="YlOrRd",
            ax=ax, cbar_kws={"label": "Number of deaths"})
ax.set_title("Deaths by Year and Month", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_year_month.png")
plt.close()


# ---------------------------------------------------------------
# Additional descriptive plots
# ---------------------------------------------------------------
race_counts = cleaned["Race"].value_counts().head(6)
fig, ax = plt.subplots(figsize=(9, 5))
race_counts.plot(kind="bar", ax=ax, color="purple", edgecolor="black")
ax.set_title("Deaths by Race (Top 6)", fontsize=13, fontweight="bold")
ax.set_xlabel("Race")
ax.set_ylabel("Number of Deaths")
plt.xticks(rotation=30, ha="right")
for idx, value in enumerate(race_counts.values):
    ax.text(idx, value + 50, str(value), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_race.png")
plt.close()

age_bins   = [0, 20, 30, 40, 50, 60, 70, 100]
age_labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
cleaned["AgeGroup"] = pd.cut(cleaned["Age"], bins=age_bins,
                              labels=age_labels, right=False)
age_group_counts = cleaned["AgeGroup"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 5))
age_group_counts.plot(kind="bar", ax=ax,
                       color="darkorange", edgecolor="black")
ax.set_title("Deaths by Age Group", fontsize=13, fontweight="bold")
ax.set_xlabel("Age Group")
ax.set_ylabel("Number of Deaths")
plt.xticks(rotation=0)
for idx, value in enumerate(age_group_counts.values):
    ax.text(idx, value + 30, str(value), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_age_groups.png")
plt.close()

print("\n" + "=" * 70)
print("Analysis complete. Figures saved to:", OUTPUT_DIR)
print("=" * 70)
