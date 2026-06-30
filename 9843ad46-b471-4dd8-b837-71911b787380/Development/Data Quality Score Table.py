
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Build quality score table ─────────────────────────────────────────────────
quality_rows = [
    ("Age",                100.0, "Categorical (bands→numeric)", 177,  "✅",  "Banded to midpoints; no hard outliers; range 22–92"),
    ("Sex",                100.0, "Binary",           0,       "✅",  "Male 78.1%, Female 21.9% – valid"),
    ("Race",               99.7,  "Categorical",       0,       "✅",  "0.3% Unknown – negligible"),
    ("Ethnicity",          100.0, "Categorical",       0,       "✅",  "Hispanic/Non-Hisp — complete"),
    ("Year of Diagnosis",  100.0, "Integer",           0,       "✅",  "2010–2023; no gaps"),
    ("Vital Status",       100.0, "Binary",            0,       "✅",  "Alive/Dead only – clean"),
    ("Survival Months",    100.0, "Numeric",        1892,       "✅",  "Right-skewed; IQR flags long survivors; max 167 mo – plausible"),
    ("Histologic Type",    100.0, "Integer (ICD-O)", 0,       "✅",  "ICD-O codes – complete"),
    ("Diagnostic Confirm", 100.0, "Categorical",       0,       "✅",  "All positive histology"),
    ("Cause-Specific Death",100.0,"Categorical",       0,       "✅",  "Complete for mortality analysis"),
    ("Surgery Code",       92.1,  "Integer (code)",   317,     "⚠️",  "317 unknowns (code 99); 1,430 Blank(s) from 2023 – ~13% uncertain"),
    ("Nodes Examined",     96.7,  "Integer",        4287,       "⚠️",  "IQR skew: median=0 (non-surgical); >0 only in surgical cases (~24%)"),
    ("Radiation",          56.2,  "Categorical",       0,       "⚠️",  "43.8% 'None/Unknown' – can't distinguish no-tx vs missing"),
    ("Histologic Subtype", 100.0, "Categorical",       0,       "⚠️",  "All rows = same value; constant column – not usable as feature"),
    ("TNM Schema",         100.0, "Categorical",       0,       "⚠️",  "All rows = 'Esophagus'; constant column – not discriminating"),
    ("Grade",              41.4,  "Categorical",       0,       "❌",  "58.6% Blank/Unknown; recorded thru 2017 only – structurally missing"),
    ("LVI",                 0.0,  "Categorical",       0,       "❌",  "100% Blank – completely unusable in this extract"),
    ("Nodes Positive",     23.5,  "Integer",          796,     "❌",  "76.5% sentinel-missing; 44 logical errors (pos>examined)"),
]

quality_df = pd.DataFrame(quality_rows, columns=[
    "Column", "% Complete", "Data Type", "Outliers", "Flag", "Notes"
])

flag_order = {"✅": 0, "⚠️": 1, "❌": 2}
quality_df["_sort"] = quality_df["Flag"].map(flag_order)
quality_df = quality_df.sort_values(["_sort", "% Complete"], ascending=[True, False]).drop("_sort", axis=1).reset_index(drop=True)

print("Data Quality Score Table:")
print(quality_df.to_string(index=False))

# ── Styled table visualisation ─────────────────────────────────────────────────
BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
FLAG_COLORS = {"✅": "#17b26a", "⚠️": "#ffd400", "❌": "#f04438"}
ROW_BG = {"✅": "#1a2e22", "⚠️": "#2e2a14", "❌": "#2e1414"}

fig_table, ax = plt.subplots(figsize=(18, 10))
fig_table.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis("off")

fig_table.suptitle("SEER Esophageal Cancer — Data Quality Score by Column",
                   color=TXT, fontsize=14, fontweight="bold", y=0.98)

header_x = [0.01, 0.15, 0.24, 0.36, 0.43, 0.48]
cols_display = ["Column", "% Complete", "Data Type", "Outliers", "Flag", "Notes"]

n_rows = len(quality_df)
row_height = 0.85 / (n_rows + 1)
table_top = 0.93
header_y = table_top - row_height * 0.5

# Header row
for hx, lbl in zip(header_x, cols_display):
    ax.text(hx, header_y, lbl, transform=ax.transAxes,
            color="#A1C9F4", fontsize=9.5, fontweight="bold", va="center")

# Separator line (use plot instead of axhline to avoid transform kwarg issue)
ax.plot([0.01, 0.99], [table_top - row_height, table_top - row_height],
        color="#444", linewidth=0.8, transform=ax.transAxes)

for i, row in quality_df.iterrows():
    y = table_top - row_height * (i + 1.5)
    flag = row["Flag"]

    # Row background tint
    rect = plt.Rectangle((0.0, y - row_height * 0.45), 1.0, row_height * 0.9,
                          transform=ax.transAxes, color=ROW_BG[flag], zorder=0, alpha=0.6)
    ax.add_patch(rect)

    values = [
        str(row["Column"]),
        f"{row['% Complete']:.1f}%",
        str(row["Data Type"]),
        str(row["Outliers"]),
        str(row["Flag"]),
        str(row["Notes"]),
    ]
    for hx, val in zip(header_x, values):
        color = FLAG_COLORS[flag] if val == flag else TXT
        ax.text(hx, y, val, transform=ax.transAxes,
                color=color, fontsize=8.8, va="center")

    if i < n_rows - 1:
        ax.plot([0.01, 0.99], [y - row_height * 0.45, y - row_height * 0.45],
                color="#2a2a2e", linewidth=0.5, transform=ax.transAxes)

# Legend
legend_handles = [
    mpatches.Patch(color="#17b26a", label="✅ Reliable — ready for modeling"),
    mpatches.Patch(color="#ffd400", label="⚠️ Use with caution — flag in analysis"),
    mpatches.Patch(color="#f04438", label="❌ Exclude from modeling"),
]
ax.legend(handles=legend_handles, loc="lower center", ncol=3,
          bbox_to_anchor=(0.5, -0.02), framealpha=0.2,
          labelcolor=TXT, fontsize=9, facecolor="#2a2a2e", edgecolor="#444")

plt.tight_layout()
plt.close("all")
