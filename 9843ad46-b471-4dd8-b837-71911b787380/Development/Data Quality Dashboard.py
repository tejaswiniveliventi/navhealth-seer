
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
PAL = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4","#9467BD"]

# ── Main dashboard figure (2×3 layout) ──────────────────────────────────────
fig_dashboard = plt.figure(figsize=(20, 14))
fig_dashboard.patch.set_facecolor(BG)

gs = gridspec.GridSpec(3, 3, figure=fig_dashboard,
                       hspace=0.55, wspace=0.35,
                       left=0.06, right=0.97, top=0.93, bottom=0.07)

fig_dashboard.suptitle(
    "SEER Esophageal Cancer (2010–2023, N=18,101) — Comprehensive Data Quality Audit",
    color=TXT, fontsize=14, fontweight="bold"
)

def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.spines[:].set_visible(False)

# ══════════════════════════════════════════════
# PANEL 1 (row0, col0–1): Missingness heatmap bar
# ══════════════════════════════════════════════
ax1 = fig_dashboard.add_subplot(gs[0, :2])
style(ax1, "1 · Column Missingness (% Missing or Sentinel Values)")

miss_sorted = missingness_pct.sort_values(ascending=True)
miss_labels = {
    "Lymph-vascular Invasion (2004+ varying by schema)": "LVI",
    "Regional nodes positive (1988+)": "Nodes Positive",
    "Grade Recode (thru 2017)": "Grade",
    "Radiation recode": "Radiation",
    "Regional nodes examined (1988+)": "Nodes Examined",
    "Race recode (White, Black, Other)": "Race",
    "Age recode with <1 year olds and 90+": "Age",
    "Sex": "Sex",
    "Year of diagnosis": "Year of Dx",
    "Origin recode NHIA (Hispanic, Non-Hisp)": "Ethnicity",
    "TNM 7/CS v0204+ Schema recode": "TNM Schema",
    "Histologic Type ICD-O-3": "Histologic Type",
    "Survival months": "Survival Months",
    "Vital status recode (study cutoff used)": "Vital Status",
    "RX Summ--Surg Prim Site (1998-2022)": "Surgery Code",
    "SEER cause-specific death classification": "Cause-Specific Death",
    "Diagnostic Confirmation": "Diagnostic Confirm",
    "Esophagus cancer by histologic subtype and diagnostic confirmation": "Histologic Subtype",
}
miss_sorted_labeled = miss_sorted.rename(miss_labels)

bar_colors = ["#f04438" if v > 50 else "#FFB482" if v > 10 else "#ffd400" if v > 0 else "#17b26a"
              for v in miss_sorted_labeled.values]
bars = ax1.barh(miss_sorted_labeled.index, miss_sorted_labeled.values,
                color=bar_colors, height=0.72, edgecolor=BG, linewidth=0.3)
ax1.axvline(50, color="#f04438", linestyle="--", linewidth=1, alpha=0.6)
ax1.axvline(10, color="#ffd400", linestyle=":", linewidth=1, alpha=0.5)
for bar, v in zip(bars, miss_sorted_labeled.values):
    if v > 0:
        ax1.text(v + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{v:.1f}%", va="center", color=TXT, fontsize=7.5)
ax1.set_xlim(0, 115)
ax1.set_xlabel("% Missing / Sentinel", color=SUB, fontsize=8)
legend_e = [
    mpatches.Patch(color="#17b26a", label="✅ 0%"),
    mpatches.Patch(color="#ffd400", label="⚠️ 1–10%"),
    mpatches.Patch(color="#FFB482", label="⚠️ 10–50%"),
    mpatches.Patch(color="#f04438", label="❌ >50%"),
]
ax1.legend(handles=legend_e, loc="lower right", fontsize=7.5,
           labelcolor=TXT, facecolor="#2a2a2e", edgecolor="#444", framealpha=0.4)

# ══════════════════════════════════════════════
# PANEL 2 (row0, col2): KPI cards
# ══════════════════════════════════════════════
ax2 = fig_dashboard.add_subplot(gs[0, 2])
ax2.set_facecolor(BG)
ax2.axis("off")
ax2.set_title("Dataset Overview", color=TXT, fontsize=10, fontweight="bold", pad=8)

kpis = [
    ("Total Records", "18,101"),
    ("Total Columns", "18"),
    ("Year Range", "2010 – 2023"),
    ("Age Range", "22 – 92 yrs"),
    ("Median Age", "67 yrs"),
    ("Male / Female", "78% / 22%"),
    ("Vital Status Dead", "77.5%"),
    ("Median Survival", "10 months"),
    ("Max Survival", "167 months"),
    ("✅ Reliable cols", "10 / 18"),
    ("⚠️ Caution cols", "5 / 18"),
    ("❌ Exclude cols", "3 / 18"),
]
for j, (label, val) in enumerate(kpis):
    y = 0.97 - j * 0.076
    color = "#17b26a" if "✅" in label else "#ffd400" if "⚠️" in label else "#f04438" if "❌" in label else TXT
    ax2.text(0.02, y, label + ":", transform=ax2.transAxes, color=SUB, fontsize=8.5, va="top")
    ax2.text(0.98, y, val, transform=ax2.transAxes, color=color, fontsize=9,
             fontweight="bold", va="top", ha="right")

# ══════════════════════════════════════════════
# PANEL 3 (row1, col0): Vital Status + Sex
# ══════════════════════════════════════════════
ax3 = fig_dashboard.add_subplot(gs[1, 0])
style(ax3, "2 · Vital Status")
vital = {"Dead": 14026, "Alive": 4075}
bars3 = ax3.bar(list(vital.keys()), list(vital.values()),
                color=["#f04438","#17b26a"], edgecolor=BG, width=0.5)
for bar, v in zip(bars3, vital.values()):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 80,
             f"{v:,}\n({100*v/18101:.1f}%)", ha="center", va="bottom", color=TXT, fontsize=8)
ax3.set_ylim(0, 17000)

# ══════════════════════════════════════════════
# PANEL 4 (row1, col1): Radiation breakdown
# ══════════════════════════════════════════════
ax4 = fig_dashboard.add_subplot(gs[1, 1])
style(ax4, "3 · Radiation Recode")
rad = {"Beam\nradiation": 9593, "None/\nUnknown": 7936, "Refused": 320,
       "Recom.\nunknown": 152, "Radiation\nNOS": 71, "Other": 24}
rad_c = ["#A1C9F4","#FFB482","#FF9F9B","#ffd400","#8DE5A1","#D0BBFF"]
bars4 = ax4.bar(list(rad.keys()), list(rad.values()), color=rad_c, edgecolor=BG, width=0.7)
for bar, v in zip(bars4, rad.values()):
    ax4.text(bar.get_x() + bar.get_width()/2, v + 60,
             f"{v:,}", ha="center", va="bottom", color=TXT, fontsize=7.5)
ax4.set_ylim(0, 11000)
ax4.tick_params(axis='x', labelsize=7)

# ══════════════════════════════════════════════
# PANEL 5 (row1, col2): Outlier summary bar
# ══════════════════════════════════════════════
ax5 = fig_dashboard.add_subplot(gs[1, 2])
style(ax5, "4 · Outlier Counts (IQR Method)")
out_cols = ["Age", "Survival\nMonths", "Nodes\nExamined", "Nodes\nPositive"]
out_vals = [177, 1892, 4287, 796]
out_hard = [0, 0, 0, 44]
x = np.arange(len(out_cols))
w = 0.38
b1 = ax5.bar(x - w/2, out_vals, width=w, color="#A1C9F4", label="IQR outliers", edgecolor=BG)
b2 = ax5.bar(x + w/2, out_hard, width=w, color="#f04438", label="Hard-flag", edgecolor=BG)
for bar, v in zip(b1, out_vals):
    ax5.text(bar.get_x() + bar.get_width()/2, v + 30, f"{v:,}",
             ha="center", va="bottom", color=TXT, fontsize=7.5)
for bar, v in zip(b2, out_hard):
    ax5.text(bar.get_x() + bar.get_width()/2, v + 30, f"{v}",
             ha="center", va="bottom", color=TXT, fontsize=7.5)
ax5.set_xticks(x)
ax5.set_xticklabels(out_cols, color=TXT, fontsize=8)
ax5.legend(fontsize=7.5, labelcolor=TXT, facecolor="#2a2a2e", edgecolor="#444", framealpha=0.4)
ax5.set_ylim(0, 5500)

# ══════════════════════════════════════════════
# PANEL 6 (row2, col0–2): Systematic missingness by year
# ══════════════════════════════════════════════
ax6 = fig_dashboard.add_subplot(gs[2, :])
style(ax6, "5 · Systematic Missingness by Year — Grade (structural cutoff 2018) & Radiation 'None/Unknown'")

years = list(range(2010, 2024))
# Grade miss %: 0-2017: ~20%, 2018+: 100%
grade_miss_yr = [20.1, 21.0, 23.7, 22.7, 25.4, 25.0, 23.4, 24.4, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
# Radiation None/Unknown % by year (computed from data)
rad_none_yr  = [None]*14  # will compute from seer_df below

df_tmp = seer_df.copy()
df_tmp["_yr"] = df_tmp["Year of diagnosis"]
rad_by_yr = df_tmp.groupby("_yr")["Radiation recode"].apply(
    lambda x: 100 * (x == "None/Unknown").sum() / len(x)
).reindex(years).fillna(0).tolist()

ax6.plot(years, grade_miss_yr, color="#ffd400", linewidth=2.5, marker="o", markersize=5,
         label="Grade Recode – % Missing")
ax6.fill_between(years, grade_miss_yr, alpha=0.12, color="#ffd400")
ax6.plot(years, rad_by_yr, color="#FFB482", linewidth=2.5, marker="s", markersize=5,
         label="Radiation 'None/Unknown' – %")
ax6.fill_between(years, rad_by_yr, alpha=0.12, color="#FFB482")

ax6.axvline(2017.5, color="#f04438", linestyle="--", linewidth=1.5, alpha=0.8)
ax6.text(2017.6, 95, "Grade cutoff\n2017→2018", color="#f04438", fontsize=8, va="top")
ax6.set_xticks(years)
ax6.set_xticklabels([str(y) for y in years], color=TXT, fontsize=8)
ax6.set_ylim(0, 110)
ax6.set_ylabel("% Missing / None-Unknown", color=SUB, fontsize=8)
ax6.legend(fontsize=8.5, labelcolor=TXT, facecolor="#2a2a2e", edgecolor="#444", framealpha=0.4, loc="upper left")
ax6.yaxis.set_tick_params(labelcolor=TXT)

plt.close("all")
print("Dashboard rendered successfully.")
print("\n=== KEY FINDINGS ===")
print("✅ READY FOR ANALYSIS / RISK SCORE:")
print("   Age, Sex, Race, Ethnicity, Year of Diagnosis, Vital Status,")
print("   Survival Months, Histologic Type, Cause-Specific Death, Diagnostic Confirmation")
print("\n⚠️  USE WITH FLAGS (limited / imputed):")
print("   Surgery Code    — binarise as surgery-yes/no; exclude code 99")
print("   Nodes Examined  — use only for surgical subgroup (n~4,300)")
print("   Radiation       — treat 'None/Unknown' as missing; sensitivity test")
print("   Histologic Subtype & TNM Schema — constant values; drop as features")
print("\n❌ EXCLUDE FROM MODELING:")
print("   Grade           — 58.6% missing; structurally cut off post-2017")
print("   LVI             — 100% blank; unusable")
print("   Nodes Positive  — 76.5% missing; 44 logical errors (pos > examined)")
