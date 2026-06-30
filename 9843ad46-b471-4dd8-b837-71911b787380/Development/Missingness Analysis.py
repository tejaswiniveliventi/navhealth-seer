
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── 1. Define sentinel-based "missing" for each column ─────────────────────
df = seer_df.copy()

SENTINEL_MAP = {
    "Age recode with <1 year olds and 90+":                        [],
    "Race recode (White, Black, Other)":                           ["Unknown"],
    "Sex":                                                         [],
    "Year of diagnosis":                                           [],
    "Origin recode NHIA (Hispanic, Non-Hisp)":                    ["Unknown"],
    "TNM 7/CS v0204+ Schema recode":                               [],
    "Histologic Type ICD-O-3":                                     [9999],
    "Survival months":                                             [],
    "Vital status recode (study cutoff used)":                     [],
    "RX Summ--Surg Prim Site (1998-2022)":                         [99],    # unknown surgery
    "Radiation recode":                                            ["None/Unknown"],
    "Grade Recode (thru 2017)":                                    ["Unknown", "Blank(s)"],
    "Regional nodes examined (1988+)":                             [99],
    "Regional nodes positive (1988+)":                             [97, 98, 99],
    "Lymph-vascular Invasion (2004+ varying by schema)":           ["Blank(s)", "9"],
    "SEER cause-specific death classification":                    ["N/A not first tumor"],
    "Diagnostic Confirmation":                                     [],
    "Esophagus cancer by histologic subtype and diagnostic confirmation": ["Unknown/not stated"],
}

missing_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
for col, sentinels in SENTINEL_MAP.items():
    if col in df.columns:
        null_mask = df[col].isna()
        if sentinels:
            sent_mask = df[col].isin(sentinels)
        else:
            sent_mask = pd.Series(False, index=df.index)
        missing_mask[col] = null_mask | sent_mask

# ── 2. Missingness % per column ─────────────────────────────────────────────
miss_pct = (missing_mask.sum() / len(df) * 100).sort_values(ascending=False)

# Short labels for plotting
short_labels = {
    "Age recode with <1 year olds and 90+":                        "Age",
    "Race recode (White, Black, Other)":                           "Race",
    "Sex":                                                         "Sex",
    "Year of diagnosis":                                           "Year of Diagnosis",
    "Origin recode NHIA (Hispanic, Non-Hisp)":                    "Ethnicity",
    "TNM 7/CS v0204+ Schema recode":                               "TNM Schema",
    "Histologic Type ICD-O-3":                                     "Histologic Type",
    "Survival months":                                             "Survival Months",
    "Vital status recode (study cutoff used)":                     "Vital Status",
    "RX Summ--Surg Prim Site (1998-2022)":                         "Surgery Code",
    "Radiation recode":                                            "Radiation",
    "Grade Recode (thru 2017)":                                    "Grade",
    "Regional nodes examined (1988+)":                             "Nodes Examined",
    "Regional nodes positive (1988+)":                             "Nodes Positive",
    "Lymph-vascular Invasion (2004+ varying by schema)":           "LVI",
    "SEER cause-specific death classification":                    "Cause-Specific Death",
    "Diagnostic Confirmation":                                     "Diagnostic Confirmation",
    "Esophagus cancer by histologic subtype and diagnostic confirmation": "Histologic Subtype",
}
miss_pct_labeled = miss_pct.rename(short_labels)

# ── 3. Heatmap-style bar chart of missingness ────────────────────────────────
fig_miss, ax = plt.subplots(figsize=(12, 7))
fig_miss.patch.set_facecolor("#1D1D20")
ax.set_facecolor("#1D1D20")

colors = []
for v in miss_pct_labeled.values:
    if v > 50:
        colors.append("#f04438")   # red  = data sparse
    elif v > 10:
        colors.append("#FFB482")   # orange = moderate
    elif v > 0:
        colors.append("#ffd400")   # yellow = minor
    else:
        colors.append("#17b26a")   # green = complete

bars = ax.barh(miss_pct_labeled.index[::-1], miss_pct_labeled.values[::-1],
               color=colors[::-1], height=0.7, edgecolor="#1D1D20", linewidth=0.5)

ax.axvline(50, color="#f04438", linestyle="--", linewidth=1.2, alpha=0.7, label=">50% sparse")
ax.axvline(10, color="#ffd400", linestyle=":", linewidth=1.0, alpha=0.7, label=">10% caution")

for bar, val in zip(bars, miss_pct_labeled.values[::-1]):
    label = f"{val:.1f}%"
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", color="#fbfbff", fontsize=9)

ax.set_xlim(0, 105)
ax.set_xlabel("% Missing / Sentinel Values", color="#fbfbff", fontsize=11)
ax.set_title("SEER Esophageal Cancer — Column-Level Missingness Audit",
             color="#fbfbff", fontsize=13, fontweight="bold", pad=14)
ax.tick_params(colors="#fbfbff", labelsize=9)
ax.spines[:].set_visible(False)
ax.xaxis.set_tick_params(length=0)
ax.yaxis.set_tick_params(length=0)
ax.set_xticks([0, 10, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "10%", "25%", "50%", "75%", "100%"], color="#909094")

legend_handles = [
    mpatches.Patch(color="#17b26a", label="✅ Reliable (0% missing)"),
    mpatches.Patch(color="#ffd400", label="⚠️ Minor gaps (1–10%)"),
    mpatches.Patch(color="#FFB482", label="⚠️ Moderate (10–50%)"),
    mpatches.Patch(color="#f04438", label="❌ Data sparse (>50%)"),
]
ax.legend(handles=legend_handles, loc="lower right", framealpha=0.15,
          labelcolor="#fbfbff", fontsize=9, facecolor="#2a2a2e", edgecolor="#444")

plt.tight_layout()
plt.close("all")

# ── 4. Systematic missingness analysis: Grade & LVI by Year ─────────────────
df["_year"] = df["Year of diagnosis"]
df["_grade_miss"] = missing_mask["Grade Recode (thru 2017)"]
df["_lvi_miss"] = missing_mask["Lymph-vascular Invasion (2004+ varying by schema)"]
df["_nodes_miss"] = missing_mask["Regional nodes examined (1988+)"]

by_year = df.groupby("_year").agg(
    grade_miss_pct=("_grade_miss", lambda x: 100 * x.mean()),
    lvi_miss_pct=("_lvi_miss", lambda x: 100 * x.mean()),
    nodes_miss_pct=("_nodes_miss", lambda x: 100 * x.mean()),
    n=("_year", "count"),
).reset_index().rename(columns={"_year": "Year"})

print("Missingness by Year (selected columns):")
print(by_year.to_string(index=False))
print(f"\nMissingness summary:")
for col, pct in miss_pct.items():
    flag = "❌ DATA SPARSE" if pct > 50 else ("⚠️  Moderate" if pct > 10 else ("⚠️  Minor" if pct > 0 else "✅ Reliable"))
    print(f"  {short_labels.get(col, col):<40} {pct:5.1f}%  {flag}")

# Export for downstream use
missingness_pct = miss_pct
missing_mask_df = missing_mask
