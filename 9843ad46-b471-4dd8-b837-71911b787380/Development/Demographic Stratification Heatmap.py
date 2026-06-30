
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"

df = seer_engineered[seer_engineered["stage"] != "Unknown"].copy()
STAGES = ["I", "II", "III"]

# ═══════════════════════════════════════════════════════════════════════════
# BUILD HEATMAP DATA: % Alive by (Demographic Category × Stage)
# ═══════════════════════════════════════════════════════════════════════════
heatmap_rows = []

# Age groups
for age_grp in ["<60", "60-69", "70-79", "80+"]:
    row = {"Factor": "Age Group", "Category": age_grp}
    for st in STAGES:
        sub = df[(df["age_group"] == age_grp) & (df["stage"] == st)]
        row[f"Stage {st}"] = round(sub["vital_alive"].mean() * 100, 1) if len(sub) > 5 else np.nan
        row[f"N_Stage_{st}"] = len(sub)
    heatmap_rows.append(row)

# Race
for race in ["White", "Black", "Other"]:
    row = {"Factor": "Race", "Category": race}
    for st in STAGES:
        sub = df[(df["race_group"] == race) & (df["stage"] == st)]
        row[f"Stage {st}"] = round(sub["vital_alive"].mean() * 100, 1) if len(sub) > 5 else np.nan
        row[f"N_Stage_{st}"] = len(sub)
    heatmap_rows.append(row)

# Sex
for sex in ["Male", "Female"]:
    row = {"Factor": "Sex", "Category": sex}
    for st in STAGES:
        sub = df[(df["Sex"] == sex) & (df["stage"] == st)]
        row[f"Stage {st}"] = round(sub["vital_alive"].mean() * 100, 1) if len(sub) > 5 else np.nan
        row[f"N_Stage_{st}"] = len(sub)
    heatmap_rows.append(row)

# Histology
for hist in ["Adenocarcinoma", "Squamous Cell Carcinoma"]:
    row = {"Factor": "Histology", "Category": hist.replace("Squamous Cell Carcinoma", "SCC").replace("Adenocarcinoma","Adeno")}
    for st in STAGES:
        sub = df[(df["histology_group"] == hist) & (df["stage"] == st)]
        row[f"Stage {st}"] = round(sub["vital_alive"].mean() * 100, 1) if len(sub) > 5 else np.nan
        row[f"N_Stage_{st}"] = len(sub)
    heatmap_rows.append(row)

# Year Cohort
for cohort_label, cohort_fn in [("2010-2017", lambda d: d["year_dx"] < 2018),
                                  ("2018-2023", lambda d: d["year_dx"] >= 2018)]:
    row = {"Factor": "Year Cohort", "Category": cohort_label}
    for st in STAGES:
        sub = df[cohort_fn(df) & (df["stage"] == st)]
        row[f"Stage {st}"] = round(sub["vital_alive"].mean() * 100, 1) if len(sub) > 5 else np.nan
        row[f"N_Stage_{st}"] = len(sub)
    heatmap_rows.append(row)

demo_heatmap_df = pd.DataFrame(heatmap_rows)

# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZE: Heatmap
# ═══════════════════════════════════════════════════════════════════════════
heat_vals = demo_heatmap_df[["Stage I", "Stage II", "Stage III"]].values.astype(float)
row_labels = demo_heatmap_df["Category"].tolist()
col_labels = ["Stage I", "Stage II", "Stage III"]

# Factor separator positions
factor_bounds = {}
for i, f in enumerate(demo_heatmap_df["Factor"]):
    if f not in factor_bounds:
        factor_bounds[f] = i

fig_demo_heatmap, ax = plt.subplots(figsize=(10, 9))
fig_demo_heatmap.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Custom colormap: red (low survival) → yellow → green (high survival)
cmap = LinearSegmentedColormap.from_list(
    "survival", ["#f04438", "#ffd400", "#17b26a"], N=256
)
im = ax.imshow(heat_vals, cmap=cmap, aspect="auto", vmin=0, vmax=100, interpolation="nearest")

# Cell text
for r in range(len(row_labels)):
    for c in range(len(col_labels)):
        v = heat_vals[r, c]
        if not np.isnan(v):
            n_val = demo_heatmap_df.iloc[r][f"N_Stage_{STAGES[c]}"]
            txt_color = "#1D1D20" if 25 < v < 75 else TXT
            ax.text(c, r, f"{v:.0f}%\n(n={n_val:,})", ha="center", va="center",
                    color=txt_color, fontsize=8.5, fontweight="bold")
        else:
            ax.text(c, r, "N/A", ha="center", va="center", color=SUB, fontsize=8)

# Axes
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, color=TXT, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, color=TXT, fontsize=10)
ax.xaxis.tick_top()
ax.xaxis.set_label_position("top")
ax.tick_params(length=0)

# Factor group dividers
divider_positions = [3.5, 6.5, 8.5, 10.5]  # after Age(4), Race(3), Sex(2), Histo(2)
for pos in divider_positions:
    ax.axhline(pos, color="#444", linewidth=1.5, linestyle="--", alpha=0.6)

# Factor labels on left margin
factor_groups = {
    "AGE\nGROUP": (0, 3.5),
    "RACE": (3.5, 6.5),
    "SEX": (6.5, 8.5),
    "HISTOLOGY": (8.5, 10.5),
    "YEAR\nCOHORT": (10.5, 12)
}
for label, (y0, y1) in factor_groups.items():
    mid = (y0 + y1) / 2
    ax.text(-0.7, mid, label, ha="right", va="center", color="#ffd400",
            fontsize=8.5, fontweight="bold", transform=ax.get_yaxis_transform())

# Colorbar
cbar = fig_demo_heatmap.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
cbar.set_label("% Alive", color=TXT, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=TXT, labelsize=8)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TXT)
cbar.outline.set_visible(False)

ax.set_title(
    "% Patients Alive by Demographic Factor & Proxy Stage\n"
    "⚠️ Stage = Treatment-Pattern Proxy (see caveats)  |  Cells: % Alive (sample n)",
    color=TXT, fontsize=11, fontweight="bold", pad=18
)
ax.spines[:].set_visible(False)

plt.tight_layout()
plt.close("all")

print("=== DEMOGRAPHIC STRATIFICATION: % ALIVE BY STAGE ===")
print(demo_heatmap_df[["Factor","Category","Stage I","Stage II","Stage III"]].to_string(index=False))
print("\n⚠️  Caveats:")
print("  • Stage proxy: no explicit AJCC stage column in this extract")
print("  • 'Stage III' contains both multimodal-surgical and all non-surgical patients")
print("  • Recent cohort (2018-2023) appears lower 5-yr survival due to shorter follow-up time (right-censoring)")
print("  • Race subgroups at Stage I are small (n≈3-73) — interpret with caution")
