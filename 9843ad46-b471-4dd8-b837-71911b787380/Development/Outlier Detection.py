
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = seer_df.copy()

# ── Helper: IQR bounds ──────────────────────────────────────────────────────
def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

# ── 1. Age (midpoints already computed) ─────────────────────────────────────
age = age_midpoints.dropna()
age_lo, age_hi = iqr_bounds(age)
age_iqr_outliers = age[(age < age_lo) | (age > age_hi)]
age_hard_outliers = age[(age < 10) | (age > 110)]
print(f"[Age]  IQR bounds: [{age_lo:.1f}, {age_hi:.1f}]  |  IQR outliers: {len(age_iqr_outliers)}  |  Hard outliers (<10 or >110): {len(age_hard_outliers)}")
print(f"        Range: {age.min():.0f}–{age.max():.0f}  |  Median: {age.median():.0f}  |  IQR: {age.quantile(0.25):.0f}–{age.quantile(0.75):.0f}")

# ── 2. Survival months ──────────────────────────────────────────────────────
surv = survival_series.dropna()
surv_lo, surv_hi = iqr_bounds(surv)
surv_iqr_outliers = surv[(surv < surv_lo) | (surv > surv_hi)]
surv_neg = (surv < 0).sum()
surv_over600 = (surv > 600).sum()
print(f"\n[Survival] IQR bounds: [{surv_lo:.1f}, {surv_hi:.1f}]  |  IQR outliers: {len(surv_iqr_outliers)}  |  Negative: {surv_neg}  |  >600: {surv_over600}")
print(f"           Range: {surv.min():.0f}–{surv.max():.0f}  |  Median: {surv.median():.0f}  |  IQR: {surv.quantile(0.25):.0f}–{surv.quantile(0.75):.0f}")

# ── 3. Regional nodes examined ───────────────────────────────────────────────
# Sentinel: 99 = unknown, exclude it
nodes_ex_raw = pd.to_numeric(df["Regional nodes examined (1988+)"], errors="coerce")
nodes_ex = nodes_ex_raw[nodes_ex_raw != 99].dropna()
ne_lo, ne_hi = iqr_bounds(nodes_ex)
ne_iqr_out = nodes_ex[(nodes_ex < ne_lo) | (nodes_ex > ne_hi)]
ne_over100 = (nodes_ex > 100).sum()
print(f"\n[Nodes Examined] IQR bounds: [{ne_lo:.1f}, {ne_hi:.1f}]  |  IQR outliers: {len(ne_iqr_out)}  |  >100: {ne_over100}")
print(f"                 Range: {nodes_ex.min():.0f}–{nodes_ex.max():.0f}  |  Median: {nodes_ex.median():.0f}")
if len(ne_iqr_out) > 0:
    print(f"                 High outlier values (>IQR hi): {sorted(nodes_ex[nodes_ex > ne_hi].unique().tolist())[:15]}")

# ── 4. Regional nodes positive ──────────────────────────────────────────────
nodes_pos_raw = pd.to_numeric(df["Regional nodes positive (1988+)"], errors="coerce")
nodes_pos = nodes_pos_raw[~nodes_pos_raw.isin([97, 98, 99])].dropna()
np_lo, np_hi = iqr_bounds(nodes_pos)
np_iqr_out = nodes_pos[(nodes_pos < np_lo) | (nodes_pos > np_hi)]

# Nodes positive > nodes examined (logical inconsistency)
combined = pd.DataFrame({
    "nodes_ex": nodes_ex_raw,
    "nodes_pos": nodes_pos_raw
}).dropna()
combined_clean = combined[(combined["nodes_ex"] != 99) & (~combined["nodes_pos"].isin([97, 98, 99]))]
pos_gt_ex = (combined_clean["nodes_pos"] > combined_clean["nodes_ex"]).sum()
print(f"\n[Nodes Positive] IQR bounds: [{np_lo:.1f}, {np_hi:.1f}]  |  IQR outliers: {len(np_iqr_out)}")
print(f"                 Nodes Positive > Nodes Examined (logical error): {pos_gt_ex}")

# ── 5. Summary table ─────────────────────────────────────────────────────────
outlier_summary = pd.DataFrame({
    "Column": ["Age", "Survival Months", "Nodes Examined", "Nodes Positive"],
    "N_Valid": [len(age), len(surv), len(nodes_ex), len(nodes_pos)],
    "Median": [age.median(), surv.median(), nodes_ex.median(), nodes_pos.median()],
    "IQR_Lo": [age_lo, surv_lo, ne_lo, np_lo],
    "IQR_Hi": [age_hi, surv_hi, ne_hi, np_hi],
    "IQR_Outliers": [len(age_iqr_outliers), len(surv_iqr_outliers), len(ne_iqr_out), len(np_iqr_out)],
    "Hard_Flag_Count": [len(age_hard_outliers), surv_neg + surv_over600, ne_over100, pos_gt_ex],
    "Hard_Flag_Rule": ["<10 or >110 yrs", "<0 or >600 months", ">100 nodes", "pos>examined"],
})
print(f"\nOutlier Summary Table:")
print(outlier_summary.to_string(index=False))

# ── 6. Visualisation: 4-panel box/strip plots ───────────────────────────────
BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
PAL = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B"]

fig_outliers, axes = plt.subplots(1, 4, figsize=(16, 5))
fig_outliers.patch.set_facecolor(BG)
fig_outliers.suptitle("SEER Esophageal Cancer — Outlier Detection (IQR Method)",
                       color=TXT, fontsize=13, fontweight="bold", y=1.02)

datasets = [
    (age, "Age (years)", age_lo, age_hi, 10, 110),
    (surv, "Survival Months", surv_lo, surv_hi, 0, 600),
    (nodes_ex, "Nodes Examined", ne_lo, ne_hi, 0, 100),
    (nodes_pos, "Nodes Positive", np_lo, np_hi, 0, None),
]

for ax, (data, title, lo, hi, hard_lo, hard_hi), color in zip(axes, datasets, PAL):
    ax.set_facecolor(BG)
    bp = ax.boxplot(data.dropna(), vert=True, patch_artist=True, widths=0.5,
                    medianprops=dict(color="#ffd400", linewidth=2),
                    boxprops=dict(facecolor=color, alpha=0.7, linewidth=0),
                    whiskerprops=dict(color=SUB, linewidth=1.2),
                    capprops=dict(color=SUB, linewidth=1.2),
                    flierprops=dict(marker="o", color="#f04438", alpha=0.4, markersize=4))

    # Hard threshold lines
    if hard_lo is not None:
        ax.axhline(hard_lo, color="#f04438", linestyle="--", linewidth=1, alpha=0.8)
    if hard_hi is not None:
        ax.axhline(hard_hi, color="#f04438", linestyle="--", linewidth=1, alpha=0.8,
                   label=f"Hard limit={hard_hi}")

    n_out = len(data[(data < lo) | (data > hi)])
    ax.set_title(f"{title}\n{n_out} IQR outliers", color=TXT, fontsize=10, fontweight="bold")
    ax.tick_params(colors=TXT, labelsize=9)
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_ylabel("Value", color=SUB, fontsize=9)
    ax.yaxis.set_tick_params(length=0)

plt.tight_layout()
plt.close("all")
