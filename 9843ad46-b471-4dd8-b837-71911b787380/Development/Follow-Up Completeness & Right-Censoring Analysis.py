import pandas as pd
import numpy as np
import json

# ─── Load engineered dataset ───────────────────────────────────────────────
df = seer_engineered.copy()

# Normalise survival to years (column is 'survival_months' in this pipeline)
_surv_col = "survival_months" if "survival_months" in df.columns else "Survival months"
df["_survival_years"] = df[_surv_col] / 12

# Normalise vital binary
if "vital_binary" not in df.columns:
    if "vital_alive" in df.columns:
        df["vital_binary"] = df["vital_alive"]
    else:
        df["vital_binary"] = (df["vital_status"] == "Alive").astype(int)

yr_col   = "Year of diagnosis"
CUTOFF_YEAR = 2023
total_n  = len(df)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: FOLLOW-UP TIME BY DIAGNOSIS YEAR
# ═══════════════════════════════════════════════════════════════════════════
year_rows = []
for yr in sorted(df[yr_col].dropna().unique()):
    n_yr     = int((df[yr_col] == yr).sum())
    followup = CUTOFF_YEAR - int(yr)
    pct_tot  = round(n_yr / total_n * 100, 1)
    if followup >= 5:
        status = "✅ 5+ yr follow-up"
    elif followup >= 3:
        status = f"⚠️ <5 yr ({followup}-yr max)"
    elif followup >= 1:
        status = f"⚠️ <5 yr ({followup}-yr max)"
    else:
        status = "❌ <1 yr (incomplete)"
    year_rows.append({
        "Year": int(yr),
        "N Diagnosed": n_yr,
        "Follow-up Available (yrs)": followup,
        "% of Total": pct_tot,
        "Status": status,
    })

follow_up_by_year_df = pd.DataFrame(year_rows)

n_lt1yr = follow_up_by_year_df.loc[follow_up_by_year_df["Follow-up Available (yrs)"] < 1, "N Diagnosed"].sum()
n_lt5yr = follow_up_by_year_df.loc[follow_up_by_year_df["Follow-up Available (yrs)"] < 5, "N Diagnosed"].sum()
pct_lt1 = round(n_lt1yr / total_n * 100, 1)
pct_lt5 = round(n_lt5yr / total_n * 100, 1)

print("=" * 70)
print("STEP 1: FOLLOW-UP TIME BY DIAGNOSIS YEAR")
print("=" * 70)
print(follow_up_by_year_df.to_string(index=False))
print(f"\n⚠️  <1 yr follow-up (Year 2023):        n={n_lt1yr:,}  ({pct_lt1}% of cohort)")
print(f"⚠️  <5 yr follow-up (Years 2019–2023):  n={n_lt5yr:,}  ({pct_lt5}% of cohort)")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: FOLLOW-UP COMPLETENESS BY SURVIVAL TIMEPOINT
# ═══════════════════════════════════════════════════════════════════════════
timepoints = [
    {"Timepoint": "1-Year",  "min_followup": 1, "max_dx_year": CUTOFF_YEAR - 1},
    {"Timepoint": "3-Year",  "min_followup": 3, "max_dx_year": CUTOFF_YEAR - 3},
    {"Timepoint": "5-Year",  "min_followup": 5, "max_dx_year": CUTOFF_YEAR - 5},
]

comp_rows = []
for tp in timepoints:
    n_elig   = int((df[yr_col] <= tp["max_dx_year"]).sum())
    pct_elig = round(n_elig / total_n * 100, 1)
    if pct_elig >= 85:
        status, warning = "✅ Complete",      None
    elif pct_elig >= 70:
        status, warning = "✅ Good coverage", None
    else:
        status  = "⚠️ Incomplete"
        warning = (
            f"{tp['Timepoint']} rates exclude patients diagnosed after {tp['max_dx_year']}. "
            "Do not use for modern protocol comparison."
        )
    comp_rows.append({
        "Timepoint":       tp["Timepoint"],
        "N Eligible":      n_elig,
        "% of Cohort":     pct_elig,
        "Max Dx Year":     tp["max_dx_year"],
        "Follow-up Req.":  f"≥{tp['min_followup']} yr",
        "Status":          status,
        "warning":         warning,
    })

follow_up_completeness_df = pd.DataFrame(comp_rows)

print("\n" + "=" * 70)
print("STEP 2: FOLLOW-UP COMPLETENESS BY SURVIVAL TIMEPOINT")
print("=" * 70)
print(follow_up_completeness_df.drop(columns=["warning"]).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: ERA BIAS BY TREATMENT COHORT
# ═══════════════════════════════════════════════════════════════════════════
def _assign_cohort(row):
    sx  = str(row.get("surgery_status",  "Unknown"))
    rad = str(row.get("radiation_status","Unknown"))
    if sx  == "Unknown":             return "E"
    if sx  == "Yes" and rad == "Yes": return "A"
    if sx  == "Yes" and rad != "Yes": return "B"
    if sx  != "Yes" and rad == "Yes": return "C"
    return "D"

if "tx_cohort" not in df.columns:
    df["tx_cohort"] = df.apply(_assign_cohort, axis=1)

COHORT_LABELS = {
    "A": "Surgery + Radiation (Multimodal)",
    "B": "Surgery Only",
    "C": "Radiation Only",
    "D": "No Treatment Recorded",
    "E": "Treatment Unknown (Right-Censored)",
}

ERA_DEFS   = {"2010–2017": (2010, 2017), "2018–2023": (2018, 2023)}
MAX_DX_5YR = CUTOFF_YEAR - 5   # 2018

era_rows = []
for cohort_key in ["A", "B", "C", "D"]:
    c_mask = df["tx_cohort"] == cohort_key
    for era_label, (yr_lo, yr_hi) in ERA_DEFS.items():
        e_mask = (df[yr_col] >= yr_lo) & (df[yr_col] <= yr_hi)
        sub    = df[c_mask & e_mask]
        n      = len(sub)
        if n == 0:
            continue
        pct_alive = round(sub["vital_binary"].mean() * 100, 1)
        med_surv  = round(sub[_surv_col].median(), 1)
        surv_1yr  = round((sub["_survival_years"] >= 1).mean() * 100, 1)
        surv_3yr  = round((sub["_survival_years"] >= 3).mean() * 100, 1)

        sub_5e    = sub[sub[yr_col] <= MAX_DX_5YR]
        if len(sub_5e) > 0:
            surv_5yr      = round((sub_5e["_survival_years"] >= 5).mean() * 100, 1)
            surv_5yr_note = f"{surv_5yr}% (n={len(sub_5e):,} eligible)"
        else:
            surv_5yr      = None
            surv_5yr_note = "N/A — no eligible patients"

        is_modern = era_label == "2018–2023"
        era_rows.append({
            "Cohort":           cohort_key,
            "Cohort Label":     COHORT_LABELS[cohort_key],
            "Era":              era_label,
            "N":                n,
            "Follow-up":        "0–5 yrs" if is_modern else "6–13 yrs",
            "% Alive":          pct_alive,
            "Median Surv (mo)": med_surv,
            "1-Year Surv %":    surv_1yr,
            "3-Year Surv %":    surv_3yr,
            "5-Year Surv %":    surv_5yr,
            "5-Yr Eligible N":  len(sub_5e),
            "5-Yr Note":        surv_5yr_note,
            "Censoring Flag":   (
                "⚠️ Shorter follow-up inflates % Alive" if is_modern
                else "✅ Full follow-up available"
            ),
        })

era_bias_df = pd.DataFrame(era_rows)

print("\n" + "=" * 70)
print("STEP 3: ERA BIAS BY TREATMENT COHORT")
print("=" * 70)
_cols = ["Cohort Label", "Era", "N", "% Alive", "Median Surv (mo)",
         "1-Year Surv %", "3-Year Surv %", "5-Year Surv %", "Censoring Flag"]
print(era_bias_df[_cols].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: RIGHT-CENSORING METADATA DICT
# ═══════════════════════════════════════════════════════════════════════════
_r1 = follow_up_completeness_df[follow_up_completeness_df["Timepoint"] == "1-Year"].iloc[0]
_r3 = follow_up_completeness_df[follow_up_completeness_df["Timepoint"] == "3-Year"].iloc[0]
_r5 = follow_up_completeness_df[follow_up_completeness_df["Timepoint"] == "5-Year"].iloc[0]

_cohA_old  = era_bias_df[(era_bias_df["Cohort"] == "A") & (era_bias_df["Era"] == "2010–2017")]
_cohA_new  = era_bias_df[(era_bias_df["Cohort"] == "A") & (era_bias_df["Era"] == "2018–2023")]
_delta_A1  = None
if len(_cohA_old) and len(_cohA_new):
    _delta_A1 = round(float(_cohA_new["1-Year Surv %"].iloc[0])
                      - float(_cohA_old["1-Year Surv %"].iloc[0]), 1)

_cohE           = df[df["tx_cohort"] == "E"]
_cohE_pct       = round(len(_cohE) / total_n * 100, 1)
_cohE_mean_fu   = round(float(_cohE["_survival_years"].mean()), 2) if len(_cohE) else None

right_censoring_metadata = {
    "follow_up_status": {
        "1_year": {
            "eligible_n":    int(_r1["N Eligible"]),
            "pct_cohort":    float(_r1["% of Cohort"]),
            "max_dx_year":   int(_r1["Max Dx Year"]),
            "status":        "complete",
            "warning":       _r1["warning"],
        },
        "3_year": {
            "eligible_n":    int(_r3["N Eligible"]),
            "pct_cohort":    float(_r3["% of Cohort"]),
            "max_dx_year":   int(_r3["Max Dx Year"]),
            "status":        "good",
            "warning":       _r3["warning"],
        },
        "5_year": {
            "eligible_n":    int(_r5["N Eligible"]),
            "pct_cohort":    float(_r5["% of Cohort"]),
            "max_dx_year":   int(_r5["Max Dx Year"]),
            "status":        "incomplete",
            "warning": (
                _r5["warning"] or
                "5-year rates exclude patients diagnosed after 2018. "
                "Do not use for modern protocol comparison."
            ),
        },
    },
    "era_comparison_bias": {
        "2010_2017_vs_2018_2023": {
            "1yr_delta_cohortA":  f"+{_delta_A1}%" if _delta_A1 else "N/A",
            "note": (
                "Modern cohort (2018–2023) appears to have better survival partly because "
                "patients have had less calendar time to be followed up — not purely treatment "
                "improvement. Use 1-year or 3-year rates only for era comparison. "
                "Median survival DROPPED for multimodal (44 mo → 26 mo) as newer, sicker "
                "patients entered the registry."
            ),
            "safe_timepoints":   ["1-Year", "3-Year"],
            "unsafe_timepoints": ["5-Year"],
            "wait_until":        2028,
        },
    },
    "cohort_e_warning": {
        "label":              "Treatment Status Unknown",
        "n":                  int(len(_cohE)),
        "pct_cohort":         float(_cohE_pct),
        "mean_follow_up_yrs": _cohE_mean_fu,
        "message": (
            "Right-censored 2022–23 records with <1 yr follow-up. "
            "'% Alive' and median survival are artifacts of short follow-up, "
            "not clinical outcomes. EXCLUDE from all survival comparisons."
        ),
        "action": "exclude",
    },
    "global_flags": {
        "stage_iii_proxy": (
            "Stage III = 73% of cohort — proxy coding artifact, not AJCC Stage III. "
            "Use treatment cohort (A–E) as primary stratification."
        ),
        "radiation_ambiguity": (
            "~40–45% of radiation records coded 'None/Unknown' — cannot distinguish "
            "confirmed no-treatment from unreported treatment."
        ),
        "grade_cutoff":   "Grade 100% blank post-2017. Excluded from all models.",
        "lvi_blank":      "LVI 100% blank across all years. Excluded.",
        "nodes_missing":  "Nodes Positive 76.5% missing + 44 logical errors. Surgical subgroup only.",
    },
}

print("\n" + "=" * 70)
print("STEP 4: RIGHT-CENSORING METADATA (selected keys)")
print("=" * 70)
_show = {k: v for k, v in right_censoring_metadata.items()
         if k in ("follow_up_status", "cohort_e_warning")}
print(json.dumps(_show, indent=2, default=str))

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: CAVEATS SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════
_n_2023 = int(follow_up_by_year_df[follow_up_by_year_df["Year"] == 2023]["N Diagnosed"].iloc[0])

caveats_summary_df = pd.DataFrame([
    {
        "Issue":           "5-yr follow-up incomplete for modern era",
        "Affected":        f"2019–2023 ({n_lt5yr:,} patients, {pct_lt5}%)",
        "Impact":          "Cannot assess long-term outcomes of modern protocols yet",
        "Recommendation":  "Use 1-yr or 3-yr only for 2020–2023 comparisons",
    },
    {
        "Issue":           "<1-yr follow-up (2023 cohort)",
        "Affected":        f"Year 2023 only (n={_n_2023:,})",
        "Impact":          "~0 yr data; censoring biases % Alive upward",
        "Recommendation":  "Exclude 2023 cohort from survival rate calculations",
    },
    {
        "Issue":           "Cohort E right-censored",
        "Affected":        f"2022–23, surgery Blank (n={len(_cohE):,}, {_cohE_pct}%)",
        "Impact":          "55% appear alive due to <1 yr follow-up — statistical artifact",
        "Recommendation":  "EXCLUDE from all survival analyses",
    },
    {
        "Issue":           "Modern protocol effect confounded",
        "Affected":        "All 2018–2023 diagnoses",
        "Impact":          "% Alive improvement = treatment + incomplete follow-up",
        "Recommendation":  "Wait until 2028 for true 5-yr data; use 3-yr rates now",
    },
    {
        "Issue":           "Stage III proxy inflation",
        "Affected":        "All cohorts (73% coded Stage III)",
        "Impact":          "Stage distribution ≠ clinical AJCC staging",
        "Recommendation":  "Use treatment cohort A–E as primary stratification",
    },
    {
        "Issue":           "Radiation 'None/Unknown' ambiguity",
        "Affected":        "Cohorts C & D (~40–45% of records)",
        "Impact":          "Confirmed no-treatment + undocumented treatment mixed",
        "Recommendation":  "Sensitivity analysis separating None vs Unknown",
    },
])

print("\n" + "=" * 70)
print("STEP 6: CAVEATS SUMMARY TABLE")
print("=" * 70)
print(caveats_summary_df.to_string(index=False))

print("\n" + "─" * 70)
print("OUTPUT VARIABLES:")
print(f"  follow_up_by_year_df        {len(follow_up_by_year_df)} rows × {len(follow_up_by_year_df.columns)} cols")
print(f"  follow_up_completeness_df   {len(follow_up_completeness_df)} rows × {len(follow_up_completeness_df.columns)} cols")
print(f"  era_bias_df                 {len(era_bias_df)} rows × {len(era_bias_df.columns)} cols")
print(f"  right_censoring_metadata    dict — {len(right_censoring_metadata)} top-level keys")
print(f"  caveats_summary_df          {len(caveats_summary_df)} rows × {len(caveats_summary_df.columns)} cols")
