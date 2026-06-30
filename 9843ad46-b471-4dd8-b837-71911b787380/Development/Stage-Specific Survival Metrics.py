
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
PAL = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4","#9467BD","#8C564B"]
STAGE_COLORS = {"I":"#8DE5A1","II":"#A1C9F4","III":"#FFB482","Unknown":"#909094"}

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    if title:   ax.set_title(title, color=TXT, fontsize=10, fontweight="bold", pad=8)
    if xlabel:  ax.set_xlabel(xlabel, color=SUB, fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color=SUB, fontsize=8)
    ax.tick_params(colors=TXT, labelsize=8)
    ax.spines[:].set_visible(False)

df = seer_engineered[seer_engineered["stage"] != "Unknown"].copy()

STAGES = ["I","II","III"]

# ═══════════════════════════════════════════════════════════════════════════
# 1. CORE SURVIVAL METRICS BY STAGE
# ═══════════════════════════════════════════════════════════════════════════
rows = []
for st in STAGES:
    sub = df[df["stage"] == st]
    n   = len(sub)
    if n == 0: continue

    pct_alive   = sub["vital_alive"].mean() * 100
    pct_dead    = 100 - pct_alive
    median_surv = sub["survival_months"].median()
    mean_surv   = sub["survival_months"].mean()
    surv_1yr    = (sub["survival_months"] >= 12).mean() * 100
    surv_3yr    = (sub["survival_months"] >= 36).mean() * 100
    surv_5yr    = (sub["survival_months"] >= 60).mean() * 100

    # Treatment patterns
    pct_sx      = sub["flag_had_surgery"].mean() * 100
    pct_rad     = sub["flag_had_radiation"].mean() * 100
    pct_both    = sub["flag_had_both_sx_rad"].mean() * 100
    pct_none    = sub["flag_no_treatment"].mean() * 100
    pct_tx_miss = sub["flag_treatment_data_missing"].mean() * 100

    # Demographics
    mean_age    = sub["age_mid"].mean()
    pct_female  = (sub["Sex"] == "Female").mean() * 100
    pct_white   = (sub["race_group"] == "White").mean() * 100
    pct_black   = (sub["race_group"] == "Black").mean() * 100
    pct_other   = (sub["race_group"] == "Other").mean() * 100

    # Nodal characteristics (surgical subgroup only)
    surg_sub     = sub[sub["flag_surgical_subgroup"]]
    ne_mean      = surg_sub["nodes_examined"].mean() if len(surg_sub) > 0 else np.nan
    ne_median    = surg_sub["nodes_examined"].median() if len(surg_sub) > 0 else np.nan
    np_mean      = surg_sub["nodes_positive"].mean() if len(surg_sub) > 0 else np.nan
    n_surg_sub   = len(surg_sub)

    rows.append({
        "Stage": st, "N": n, "% Alive": round(pct_alive,1), "% Dead": round(pct_dead,1),
        "Median Survival (mo)": round(median_surv,1), "Mean Survival (mo)": round(mean_surv,1),
        "1-Year Survival %": round(surv_1yr,1), "3-Year Survival %": round(surv_3yr,1),
        "5-Year Survival %": round(surv_5yr,1),
        "% Had Surgery": round(pct_sx,1), "% Had Radiation": round(pct_rad,1),
        "% Had Both (S+R)": round(pct_both,1), "% No Treatment": round(pct_none,1),
        "% Treatment Missing⚠️": round(pct_tx_miss,1),
        "Mean Age": round(mean_age,1), "% Female": round(pct_female,1),
        "% White": round(pct_white,1), "% Black": round(pct_black,1), "% Other Race": round(pct_other,1),
        "Surgical Subgroup N": n_surg_sub,
        "Mean Nodes Examined⚠️": round(ne_mean,1) if pd.notna(ne_mean) else np.nan,
        "Median Nodes Examined⚠️": round(ne_median,1) if pd.notna(ne_median) else np.nan,
        "Mean Nodes Positive⚠️": round(np_mean,1) if pd.notna(np_mean) else np.nan,
    })

stage_metrics_df = pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════
# 2. COHORT COMPARISON: 2010-2017 vs 2018-2023 BY STAGE
# ═══════════════════════════════════════════════════════════════════════════
cohort_rows = []
for st in STAGES:
    sub = df[df["stage"] == st]
    for cohort, mask_fn in [("2010-2017 (Older Protocols)", sub["year_dx"] < 2018),
                             ("2018-2023 (Modern Protocols)", sub["year_dx"] >= 2018)]:
        c = sub[mask_fn]
        if len(c) == 0: continue
        cohort_rows.append({
            "Stage": st, "Year Cohort": cohort, "N": len(c),
            "% Alive":         round(c["vital_alive"].mean()*100, 1),
            "Median Surv (mo)":round(c["survival_months"].median(), 1),
            "1-Yr Survival %": round((c["survival_months"]>=12).mean()*100, 1),
            "3-Yr Survival %": round((c["survival_months"]>=36).mean()*100, 1),
            "5-Yr Survival %": round((c["survival_months"]>=60).mean()*100, 1),
            "% Had S+R":       round(c["flag_had_both_sx_rad"].mean()*100, 1),
        })

cohort_comparison_df = pd.DataFrame(cohort_rows)

# ═══════════════════════════════════════════════════════════════════════════
# 3. STAGE III MULTIMODAL vs SURGERY ONLY (CROSS-trial proxy)
# ═══════════════════════════════════════════════════════════════════════════
s3 = df[df["stage"] == "III"]
cross_rows = []
for grp_name, mask in [
    ("Surgery + Radiation (Multimodal)", s3["flag_had_both_sx_rad"]),
    ("Surgery Only",                     s3["flag_had_surgery"] & ~s3["flag_had_radiation"]),
    ("Radiation Only",                   ~s3["flag_had_surgery"] & s3["flag_had_radiation"]),
    ("No Treatment (known)",             s3["flag_no_treatment"]),
]:
    g = s3[mask]
    if len(g) == 0: continue
    cross_rows.append({
        "Treatment Group": grp_name, "N": len(g),
        "% Alive":         round(g["vital_alive"].mean()*100, 1),
        "Median Surv (mo)":round(g["survival_months"].median(), 1),
        "1-Yr Survival %": round((g["survival_months"]>=12).mean()*100, 1),
        "3-Yr Survival %": round((g["survival_months"]>=36).mean()*100, 1),
        "5-Yr Survival %": round((g["survival_months"]>=60).mean()*100, 1),
    })
cross_trial_df = pd.DataFrame(cross_rows)

# ═══════════════════════════════════════════════════════════════════════════
# PRINT TABLES
# ═══════════════════════════════════════════════════════════════════════════
print("=== STAGE-SPECIFIC SURVIVAL METRICS ===")
print(stage_metrics_df.to_string(index=False))

print("\n=== COHORT COMPARISON: 2010-2017 vs 2018-2023 BY STAGE ===")
print("⚠️  Note: Short max follow-up (167 months) limits 5-year interpretation for recent cases")
print(cohort_comparison_df.to_string(index=False))

print("\n=== STAGE III — TREATMENT INTENSITY (CROSS-Trial Proxy) ===")
print("⚠️  SEER Proxy: surgery+radiation codes ≠ trimodal CROSS protocol (which requires neoadjuvant CRT + surgery)")
print("⚠️  SEER cannot measure pCR rates or R0 margins. Survival here is overall survival, not DFS.")
print(cross_trial_df.to_string(index=False))
print("\nASCO/CROSS benchmark: Trimodal (CRT+Surgery) ~50% pCR; R0 >90%; median OS ~49mo (van Hagen et al.)")
print("SEER Stage III Multimodal median OS:", round(cross_trial_df.iloc[0]["Median Surv (mo)"],1), "months (includes all causes of death)")

print("\n⚠️  Stage IV excluded: No explicit Stage IV in proxy (non-surgical patients mapped to Stage III)")
print("⚠️  Proxy stage limitation: 72.9% coded Stage III (surgical+advanced OR all non-surgical patients)")
print("    For definitive stage-stratified survival analysis, a SEER extract with 'Derived AJCC Stage Group' is required.")
