
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = seer_df.copy()

# ── 1. Age: parse banded string → numeric midpoint ─────────────────────────
def age_band_to_mid(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if val.startswith("90+") or "90 year" in val.lower(): return 92
    import re
    m = re.match(r"(\d+)-(\d+)", val)
    if m: return (int(m.group(1)) + int(m.group(2))) / 2
    m2 = re.match(r"(\d+)", val)
    if m2: return int(m2.group(1))
    return np.nan

df["_age_mid"] = df["Age recode with <1 year olds and 90+"].apply(age_band_to_mid)
age_min, age_max = df["_age_mid"].min(), df["_age_mid"].max()
age_out_of_range = ((df["_age_mid"] < 10) | (df["_age_mid"] > 110)).sum()

# ── 2. Survival months ─────────────────────────────────────────────────────
surv = pd.to_numeric(df["Survival months"], errors="coerce")
surv_neg = (surv < 0).sum()
surv_over600 = (surv > 600).sum()
surv_max = surv.max()

# ── 3. Stage – use TNM schema col (all 'Esophagus') and Histologic subtype
stage_col = "Esophagus cancer by histologic subtype and diagnostic confirmation"
stage_vals = df[stage_col].value_counts().head(20)

# ── 4. Vital status ─────────────────────────────────────────────────────────
vital_vals = df["Vital status recode (study cutoff used)"].value_counts()

# ── 5. Surgery codes ────────────────────────────────────────────────────────
surg_vals = df["RX Summ--Surg Prim Site (1998-2022)"].value_counts().sort_index()
# 0=none, 99=unknown, others=procedure
surg_labels = {
    0: "No surgery", 19: "Local excision", 20: "Local excision",
    22: "Wedge resection", 30: "Esophagectomy partial",
    40: "Esophagectomy total", 50: "Esophagectomy+other",
    53: "Esophagectomy+other", 99: "Unknown"
}

# ── 6. Radiation ────────────────────────────────────────────────────────────
rad_vals = df["Radiation recode"].value_counts()

# ── 7. Race / Sex / Grade / Histology ──────────────────────────────────────
race_vals = df["Race recode (White, Black, Other)"].value_counts()
sex_vals = df["Sex"].value_counts()
grade_vals = df["Grade Recode (thru 2017)"].value_counts()

print("=== DATA TYPE INTEGRITY REPORT ===")
print(f"\n[Age] Banded strings → midpoints | Range: {age_min:.0f}–{age_max:.0f} yrs | Out of range (<10 or >110): {age_out_of_range}")
print(f"\n[Survival Months] Numeric | Range: {surv.min():.0f}–{surv_max:.0f} | Negative: {surv_neg} | >600: {surv_over600}")
print(f"\n[Vital Status] {dict(vital_vals)}")
print(f"\n[Surgery Code] {dict(surg_vals)}")
print(f"\n[Radiation] {dict(rad_vals)}")
print(f"\n[Race] {dict(race_vals)}")
print(f"\n[Sex] {dict(sex_vals)}")
print(f"\n[Grade] {dict(grade_vals)}")
print(f"\n[Histologic Subtype - top 10]:\n{stage_vals.head(10).to_string()}")

# ── 8. Dashboard: 6-panel bar chart ─────────────────────────────────────────
palette = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4","#9467BD","#8C564B","#C49C94"]
BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"

fig_types, axes = plt.subplots(2, 3, figsize=(16, 9))
fig_types.patch.set_facecolor(BG)

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.spines[:].set_visible(False)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)

# Panel 1: Vital Status
ax = axes[0, 0]
ax.bar(vital_vals.index, vital_vals.values, color=["#17b26a","#f04438"], edgecolor=BG)
for i, (k, v) in enumerate(vital_vals.items()):
    ax.text(i, v + 30, f"{v:,}\n({100*v/len(df):.1f}%)", ha="center", va="bottom", color=TXT, fontsize=9)
style_ax(ax, "Vital Status")
ax.set_ylim(0, vital_vals.max() * 1.2)

# Panel 2: Sex
ax = axes[0, 1]
ax.bar(sex_vals.index, sex_vals.values, color=["#A1C9F4","#FFB482"], edgecolor=BG)
for i, (k, v) in enumerate(sex_vals.items()):
    ax.text(i, v + 30, f"{v:,}\n({100*v/len(df):.1f}%)", ha="center", va="bottom", color=TXT, fontsize=9)
style_ax(ax, "Sex Distribution")
ax.set_ylim(0, sex_vals.max() * 1.2)

# Panel 3: Race
ax = axes[0, 2]
race_colors = [palette[i] for i in range(len(race_vals))]
ax.bar(race_vals.index, race_vals.values, color=race_colors, edgecolor=BG)
for i, (k, v) in enumerate(race_vals.items()):
    ax.text(i, v + 30, f"{v:,}", ha="center", va="bottom", color=TXT, fontsize=9)
ax.set_xticklabels(race_vals.index, rotation=15, ha="right")
style_ax(ax, "Race Distribution")
ax.set_ylim(0, race_vals.max() * 1.2)

# Panel 4: Surgery
ax = axes[1, 0]
surg_x = [str(v) for v in surg_vals.index]
surg_colors = ["#f04438" if v == 99 else "#A1C9F4" for v in surg_vals.index]
ax.bar(surg_x, surg_vals.values, color=surg_colors, edgecolor=BG)
for i, v in enumerate(surg_vals.values):
    ax.text(i, v + 30, f"{v:,}", ha="center", va="bottom", color=TXT, fontsize=8)
ax.set_xlabel("Surg Code (99=Unknown)", color=SUB, fontsize=8)
style_ax(ax, "Surgery Codes")
ax.set_ylim(0, surg_vals.max() * 1.2)

# Panel 5: Radiation
ax = axes[1, 1]
rad_colors = [palette[i % len(palette)] for i in range(len(rad_vals))]
ax.barh(rad_vals.index, rad_vals.values, color=rad_colors, edgecolor=BG)
for i, (k, v) in enumerate(rad_vals.items()):
    ax.text(v + 30, i, f"{v:,}", va="center", color=TXT, fontsize=9)
style_ax(ax, "Radiation Recode")
ax.set_xlim(0, rad_vals.max() * 1.25)

# Panel 6: Grade
ax = axes[1, 2]
grade_clean = grade_vals.copy()
grade_colors = ["#f04438" if "Blank" in str(k) or "Unknown" in str(k) else "#8DE5A1"
                for k in grade_clean.index]
ax.barh(grade_clean.index, grade_clean.values, color=grade_colors, edgecolor=BG)
for i, (k, v) in enumerate(grade_clean.items()):
    ax.text(v + 30, i, f"{v:,}\n({100*v/len(df):.1f}%)", va="center", color=TXT, fontsize=8)
style_ax(ax, "Grade Recode (thru 2017)")
ax.set_xlim(0, grade_clean.max() * 1.35)

fig_types.suptitle("SEER Esophageal Cancer — Data Type Integrity Checks",
                   color=TXT, fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.close("all")

# Export
age_midpoints = df["_age_mid"]
survival_series = surv
