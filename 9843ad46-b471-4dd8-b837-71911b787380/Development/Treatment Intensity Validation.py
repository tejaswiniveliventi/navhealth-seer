
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
PAL = ["#8DE5A1","#A1C9F4","#FFB482","#FF9F9B","#D0BBFF","#1F77B4","#9467BD"]

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    if title:  ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=SUB, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=SUB, fontsize=8)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.spines[:].set_visible(False)

df = seer_engineered[seer_engineered["stage"] != "Unknown"].copy()

# ═══════════════════════════════════════════════════════════════════════════
# 1. TREATMENT INTENSITY TABLE — ALL STAGES
# ═══════════════════════════════════════════════════════════════════════════
STAGES = ["I","II","III"]
TX_GROUPS = [
    ("Surgery + Radiation (Multimodal)",  lambda d: d["flag_had_both_sx_rad"]),
    ("Surgery Only",                      lambda d: d["flag_had_surgery"] & ~d["flag_had_radiation"]),
    ("Radiation Only",                    lambda d: ~d["flag_had_surgery"] & d["flag_had_radiation"]),
    ("No Treatment (known)",              lambda d: d["flag_no_treatment"]),
    ("Treatment Unknown ⚠️",              lambda d: d["flag_treatment_data_missing"]),
]

tx_rows = []
for st in STAGES:
    sub = df[df["stage"] == st]
    for grp_name, grp_fn in TX_GROUPS:
        g = sub[grp_fn(sub)]
        if len(g) == 0: continue
        tx_rows.append({
            "Stage": st, "Treatment Group": grp_name, "N": len(g),
            "% of Stage": round(len(g)/len(sub)*100, 1),
            "% Alive":         round(g["vital_alive"].mean()*100, 1),
            "Median Surv (mo)":round(g["survival_months"].median(), 1),
            "1-Yr Surv %":     round((g["survival_months"]>=12).mean()*100, 1),
            "3-Yr Surv %":     round((g["survival_months"]>=36).mean()*100, 1),
            "5-Yr Surv %":     round((g["survival_months"]>=60).mean()*100, 1),
        })

treatment_validation_df = pd.DataFrame(tx_rows)

# ═══════════════════════════════════════════════════════════════════════════
# 2. STAGE III DEEP DIVE — YEAR COHORT × TREATMENT
# ═══════════════════════════════════════════════════════════════════════════
s3 = df[df["stage"] == "III"].copy()
era_tx_rows = []
for era_label, era_fn in [("2010-2017", s3["year_dx"] < 2018),
                           ("2018-2023", s3["year_dx"] >= 2018)]:
    era = s3[era_fn]
    for grp_name, grp_fn in TX_GROUPS[:4]:  # skip Unknown
        g = era[grp_fn(era)]
        if len(g) < 10: continue
        era_tx_rows.append({
            "Era": era_label, "Treatment Group": grp_name, "N": len(g),
            "% Alive":         round(g["vital_alive"].mean()*100, 1),
            "Median Surv (mo)":round(g["survival_months"].median(), 1),
            "1-Yr Surv %":     round((g["survival_months"]>=12).mean()*100, 1),
            "3-Yr Surv %":     round((g["survival_months"]>=36).mean()*100, 1),
        })
era_tx_df = pd.DataFrame(era_tx_rows)

# ═══════════════════════════════════════════════════════════════════════════
# 3. VISUALIZE — 4-Panel Treatment Intensity Figure
# ═══════════════════════════════════════════════════════════════════════════
fig_treatment_intensity = plt.figure(figsize=(16, 12))
fig_treatment_intensity.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

# ── Panel 1: Stage III — Treatment group % of cohort (donut) ────────────
ax1 = fig_treatment_intensity.add_subplot(gs[0, 0])
ax1.set_facecolor(BG)

s3_tx_pcts = treatment_validation_df[
    treatment_validation_df["Stage"]=="III"
][["Treatment Group","% of Stage","N"]].copy()

labels_short = {
    "Surgery + Radiation (Multimodal)": "S+R\nMultimodal",
    "Surgery Only": "Surgery\nOnly",
    "Radiation Only": "Radiation\nOnly",
    "No Treatment (known)": "No Tx",
    "Treatment Unknown ⚠️": "Tx\nUnknown⚠️",
}
_labels = [labels_short.get(x, x) for x in s3_tx_pcts["Treatment Group"]]
_sizes  = s3_tx_pcts["% of Stage"].values
_colors = ["#8DE5A1","#A1C9F4","#FFB482","#FF9F9B","#D0BBFF"]

wedges, texts, autotexts = ax1.pie(
    _sizes, labels=_labels, colors=_colors[:len(_sizes)],
    autopct="%1.1f%%", startangle=90, pctdistance=0.78,
    wedgeprops={"edgecolor": BG, "linewidth": 2},
    textprops={"color": TXT, "fontsize": 8}
)
for at in autotexts: at.set_color(BG); at.set_fontsize(8); at.set_fontweight("bold")
centre = plt.Circle((0, 0), 0.55, color=BG)
ax1.add_patch(centre)
ax1.text(0, 0, "Stage III\nN=13,204", ha="center", va="center", color=TXT, fontsize=9, fontweight="bold")
ax1.set_title("Stage III: Treatment Distribution", color=TXT, fontsize=10, fontweight="bold", pad=8)

# ── Panel 2: Median Survival by Treatment Group × Stage ─────────────────
ax2 = fig_treatment_intensity.add_subplot(gs[0, 1])
style_ax(ax2, "Median Survival (mo) by Treatment × Stage", ylabel="Months")

# Reshape for grouped bars
tx_pivot = treatment_validation_df.pivot_table(
    index="Treatment Group", columns="Stage", values="Median Surv (mo)"
).reindex(["Surgery + Radiation (Multimodal)","Surgery Only","Radiation Only","No Treatment (known)"])
tx_pivot = tx_pivot.reindex(columns=STAGES)

x = np.arange(len(tx_pivot))
w = 0.25
stage_bar_colors = {"I":"#8DE5A1","II":"#A1C9F4","III":"#FFB482"}
for j, st in enumerate(STAGES):
    vals = tx_pivot[st].values
    bars = ax2.bar(x + j*w, vals, width=w, color=stage_bar_colors[st],
                   label=f"Stage {st}", edgecolor=BG, linewidth=0.5)
    for b, v in zip(bars, vals):
        if not np.isnan(v):
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f"{v:.0f}", ha="center", va="bottom", color=TXT, fontsize=7)

xt_labels = ["S+R\n(Multimodal)", "Surgery\nOnly", "Radiation\nOnly", "No Tx"]
ax2.set_xticks(x + w)
ax2.set_xticklabels(xt_labels, color=TXT, fontsize=8)
ax2.legend(framealpha=0, labelcolor=TXT, fontsize=8)
ax2.set_ylim(0, 75)
ax2.axhline(49, color="#ffd400", linestyle="--", linewidth=1.2, alpha=0.8)
ax2.text(3.85, 50, "FLOT/CROSS\nbenchmark\n~49mo", color="#ffd400", fontsize=7, ha="right")

# ── Panel 3: 1/3/5-Year Survival — Stage III by Treatment ────────────────
ax3 = fig_treatment_intensity.add_subplot(gs[1, 0])
style_ax(ax3, "Stage III: 1/3/5-Year Survival by Treatment Group", ylabel="% Surviving")

s3_tx = treatment_validation_df[treatment_validation_df["Stage"]=="III"].copy()
s3_tx = s3_tx[s3_tx["Treatment Group"] != "Treatment Unknown ⚠️"]
xt_s3 = [labels_short.get(x, x) for x in s3_tx["Treatment Group"]]
x3 = np.arange(len(s3_tx))
w3 = 0.26

for j, (col, color, lbl) in enumerate([
    ("1-Yr Surv %","#A1C9F4","1-Year"),
    ("3-Yr Surv %","#8DE5A1","3-Year"),
    ("5-Yr Surv %","#FFB482","5-Year"),
]):
    bars = ax3.bar(x3 + j*w3, s3_tx[col].values, width=w3,
                   color=color, label=lbl, edgecolor=BG, linewidth=0.5)
    for b, v in zip(bars, s3_tx[col].values):
        ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                 f"{v:.0f}", ha="center", va="bottom", color=TXT, fontsize=7)

ax3.set_xticks(x3 + w3)
ax3.set_xticklabels(xt_s3, color=TXT, fontsize=8)
ax3.legend(framealpha=0, labelcolor=TXT, fontsize=8)
ax3.set_ylim(0, 110)
ax3.axhline(50, color="#ffd400", linestyle=":", linewidth=1, alpha=0.6)
ax3.text(3.85, 51, "CROSS ~50% pCR\n(not survival)", color="#ffd400", fontsize=6.5, ha="right")

# ── Panel 4: Protocol Era Comparison — Stage III Multimodal ─────────────
ax4 = fig_treatment_intensity.add_subplot(gs[1, 1])
style_ax(ax4, "Stage III Multimodal (S+R): Era Comparison", ylabel="% Surviving")

era_multi = era_tx_df[era_tx_df["Treatment Group"]=="Surgery + Radiation (Multimodal)"].copy()
if len(era_multi) >= 2:
    era_labels = era_multi["Era"].tolist()
    metrics = ["% Alive","1-Yr Surv %","3-Yr Surv %"]
    x4 = np.arange(len(era_labels))
    w4 = 0.28
    era_colors = ["#A1C9F4","#8DE5A1","#FFB482"]
    for j, (met, color) in enumerate(zip(metrics, era_colors)):
        vals = era_multi[met].values
        bars = ax4.bar(x4 + j*w4, vals, width=w4, color=color, label=met, edgecolor=BG, linewidth=0.5)
        for b, v in zip(bars, vals):
            ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                     f"{v:.0f}%", ha="center", va="bottom", color=TXT, fontsize=8)
    ax4.set_xticks(x4 + w4)
    ax4.set_xticklabels(era_labels, color=TXT, fontsize=9)
    ax4.legend(framealpha=0, labelcolor=TXT, fontsize=8)
    ax4.set_ylim(0, 100)

    # annotation
    ax4.annotate("CheckMate 577 era\n(nivolumab adjuvant)", xy=(1.42, era_multi.iloc[1]["% Alive"]+1),
                 xytext=(0.8, 70), color="#ffd400", fontsize=7.5,
                 arrowprops={"arrowstyle":"->","color":"#ffd400","lw":1.2})

    # Median survival diff
    m0 = era_multi.iloc[0]["Median Surv (mo)"]
    m1 = era_multi.iloc[1]["Median Surv (mo)"]
    ax4.text(0.5, 0.05, f"Median OS: {m0:.0f}mo (2010-17) → {m1:.0f}mo (2018-23)",
             transform=ax4.transAxes, color=SUB, fontsize=8, ha="center")

fig_treatment_intensity.suptitle(
    "Treatment Intensity Validation — SEER vs ASCO/CROSS Evidence\n"
    "⚠️ SEER proxy: surgery+radiation codes ≠ trimodal CROSS (neoadj CRT+surgery). "
    "pCR/R0 margins not measurable in SEER.",
    color=TXT, fontsize=11, fontweight="bold", y=1.01
)

plt.tight_layout()
plt.close("all")

print("=== ALL-STAGE TREATMENT INTENSITY TABLE ===")
print(treatment_validation_df.to_string(index=False))
print("\n=== STAGE III ERA × TREATMENT ===")
print(era_tx_df.to_string(index=False))

print("\n=== ASCO/CROSS BENCHMARK COMPARISON ===")
s3_multi = treatment_validation_df[
    (treatment_validation_df["Stage"]=="III") &
    (treatment_validation_df["Treatment Group"]=="Surgery + Radiation (Multimodal)")
]
if len(s3_multi) > 0:
    r = s3_multi.iloc[0]
    print(f"\nCROSS Trial (van Hagen et al., NEJM 2012):")
    print(f"  pCR rate:        ~50% (SEER CANNOT measure this)")
    print(f"  R0 margins:      >90% (SEER CANNOT measure this)")
    print(f"  Median OS:       ~49 months")
    print(f"\nSEER Stage III — Surgery+Radiation (n={r['N']:,}):")
    print(f"  Median OS:       {r['Median Surv (mo)']} months  (vs ~49mo CROSS; proxy stage ≠ CROSS stage)")
    print(f"  1-Year Survival: {r['1-Yr Surv %']}%")
    print(f"  3-Year Survival: {r['3-Yr Surv %']}%")
    print(f"  5-Year Survival: {r['5-Yr Surv %']}%")
    print(f"  % Alive (any time): {r['% Alive']}%")
    print(f"\n  Interpretation: SEER multimodal median OS (23mo) is lower than CROSS trial (49mo)")
    print(f"  Likely reasons: (1) CROSS enrolled fit surgical candidates only;")
    print(f"                  (2) SEER 'Stage III proxy' includes more advanced disease;")
    print(f"                  (3) SEER reflects real-world heterogeneity vs trial population;")
    print(f"                  (4) SEER codes surgery+radiation but cannot confirm neoadjuvant sequence.")

print("\n⚠️  CheckMate 577 (Nivolumab adjuvant, 2021):")
print(f"  Doubled DFS: 22.4mo → 11mo placebo (cannot be validated in SEER — no immunotherapy coding)")
print(f"  SEER 2018-2023 multimodal % Alive is higher BUT this is confounded by shorter follow-up")
