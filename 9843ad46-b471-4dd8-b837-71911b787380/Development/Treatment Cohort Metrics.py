
import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# STEP 1-2: TREATMENT INTENSITY COHORTS (A–E) + SURVIVAL METRICS
#
# ⚠️ CAVEATS:
#   • Surgery data: Post-2023 sparse (Blank ~9.7%). Unknown cohort (E) flagged.
#   • Radiation: 'None/Unknown' coded as radiation_status='No' in seer_engineered.
#     Cannot distinguish confirmed no-treatment from unknown. Flag in Cohort D/C.
#   • Stage is proxy-derived. Primary stratification here is TREATMENT INTENSITY.
#   • Survival months: Complete, reliable (0% missing). ✅
#   • SEER Limitation: No performance status, comorbidities, treatment regimen
#     detail, adjuvant therapy tracking, surgical margins, pCR status.
# ════════════════════════════════════════════════════════════════════════════

df = seer_engineered.copy()

# ── AUDIT RADIATION_STATUS / SURGERY_STATUS ACTUAL VALUES ───────────────────
_sx_vals  = df["surgery_status"].value_counts(dropna=False)
_rad_vals = df["radiation_status"].value_counts(dropna=False)

# ── COHORT DEFINITIONS ───────────────────────────────────────────────────────
# Cohort A: Surgery + Radiation (multimodal)
# Cohort B: Surgery only (confirmed no/unknown radiation)
# Cohort C: Radiation only (no surgery)
# Cohort D: No treatment recorded — both surgery=No AND radiation=No or None/Unknown
# Cohort E: Treatment status unknown — surgery Blank/Unknown
#
# NOTE: In seer_engineered (65-col), radiation_status values are:
#   "Yes" = active radiation
#   "No"  = confirmed None (includes 'None/Unknown' raw values — ambiguous ⚠️)
#   "Unknown" / "Refused" = truly unknown or refused

def _assign_cohort(row):
    sx  = row["surgery_status"]   # "Yes" / "No" / "Unknown"
    rad = row["radiation_status"] # "Yes" / "No" / "Unknown" / "Refused"
    has_s = (sx  == "Yes")
    has_r = (rad == "Yes")
    unk_s = (sx  == "Unknown")
    no_s  = (sx  == "No")
    no_r  = (rad in ("No", "Unknown", "Refused"))  # treat unknown rad as non-positive

    if unk_s:               return "E"   # surgery unknown → unclassifiable
    if has_s and has_r:     return "A"   # Surgery + Radiation
    if has_s and not has_r: return "B"   # Surgery only
    if has_r and no_s:      return "C"   # Radiation only
    if no_s  and not has_r: return "D"   # No treatment (incl ambiguous rad)
    return "E"

df["tx_cohort"] = df.apply(_assign_cohort, axis=1)

COHORT_LABELS = {
    "A": "Surgery + Radiation (Multimodal)",
    "B": "Surgery Only",
    "C": "Radiation Only",
    "D": "No Treatment Recorded ⚠️",
    "E": "Treatment Unknown / Missing ⚠️",
}
COHORT_ORDER = ["A", "B", "C", "D", "E"]

# ── HELPERS ──────────────────────────────────────────────────────────────────
def _surv_metrics(sub):
    n = len(sub)
    if n == 0:
        return dict(N=0, pct_alive=np.nan, pct_dead=np.nan,
                    median_surv_mo=np.nan, mean_surv_mo=np.nan,
                    surv_1yr=np.nan, surv_3yr=np.nan, surv_5yr=np.nan)
    sm = sub["survival_months"].dropna()
    va = sub["vital_alive"]
    return {
        "N":              n,
        "pct_alive":      round(100 * va.mean(), 1),
        "pct_dead":       round(100 * (1 - va.mean()), 1),
        "median_surv_mo": round(sm.median(), 1),
        "mean_surv_mo":   round(sm.mean(), 1),
        "surv_1yr":       round(100 * (sm >= 12).mean(), 1),
        "surv_3yr":       round(100 * (sm >= 36).mean(), 1),
        "surv_5yr":       round(100 * (sm >= 60).mean(), 1),
    }

def _demo_metrics(sub):
    n = len(sub)
    if n == 0:
        return {k: np.nan for k in [
            "mean_age","pct_age_under60","pct_age_60_69","pct_age_70_79","pct_age_80plus",
            "pct_female","pct_white","pct_black","pct_other","pct_adenocarcinoma","pct_squamous"]}
    age = sub["age_mid"]
    return {
        "mean_age":          round(age.mean(), 1),
        "pct_age_under60":   round(100 * (age < 60).mean(), 1),
        "pct_age_60_69":     round(100 * ((age >= 60) & (age < 70)).mean(), 1),
        "pct_age_70_79":     round(100 * ((age >= 70) & (age < 80)).mean(), 1),
        "pct_age_80plus":    round(100 * (age >= 80).mean(), 1),
        "pct_female":        round(100 * (sub["Sex"] == "Female").mean(), 1),
        "pct_white":         round(100 * (sub["race_group"] == "White").mean(), 1),
        "pct_black":         round(100 * (sub["race_group"] == "Black").mean(), 1),
        "pct_other":         round(100 * (sub["race_group"] == "Other").mean(), 1),
        "pct_adenocarcinoma":round(100 * sub["flag_adenocarcinoma"].mean(), 1),
        "pct_squamous":      round(100 * sub["flag_squamous"].mean(), 1),
    }

# ── BUILD METRICS TABLE ──────────────────────────────────────────────────────
rows = []
for cohort in COHORT_ORDER:
    sub = df[df["tx_cohort"] == cohort]
    row = {"cohort_code": cohort, "cohort_label": COHORT_LABELS[cohort]}
    row.update(_surv_metrics(sub))
    row.update(_demo_metrics(sub))
    rows.append(row)

treatment_cohort_metrics_df = pd.DataFrame(rows).set_index("cohort_code")

# ── PRINT SUMMARY ────────────────────────────────────────────────────────────
print("=" * 100)
print("TREATMENT INTENSITY COHORTS — SURVIVAL METRICS  (N=18,101)")
print("⚠️  Cohort D 'No Treatment': includes radiation='None/Unknown' (ambiguous); n~5,700")
print("⚠️  Cohort E 'Unknown': Surgery Blank/Unknown records (mainly post-2022), n~7,455")
print("=" * 100)

print(f"\n{'Cohort':<6} {'Treatment Group':<40} {'N':>6} {'%Alive':>7} {'Med(mo)':>8} "
      f"{'Mean(mo)':>9} {'1yr%':>6} {'3yr%':>6} {'5yr%':>6} {'MnAge':>7} {'%Fem':>6}")
print("-" * 110)
for code in COHORT_ORDER:
    r = treatment_cohort_metrics_df.loc[code]
    lbl = r["cohort_label"][:39]
    print(f"  {code}    {lbl:<40} {int(r['N']):>6} {r['pct_alive']:>7.1f} "
          f"{r['median_surv_mo']:>8.1f} {r['mean_surv_mo']:>9.1f} "
          f"{r['surv_1yr']:>6.1f} {r['surv_3yr']:>6.1f} {r['surv_5yr']:>6.1f} "
          f"{r['mean_age']:>7.1f} {r['pct_female']:>6.1f}")

print(f"\n{'Cohort':<6} {'Treatment Group':<40} {'N':>6} {'<60':>6} {'60-69':>6} "
      f"{'70-79':>6} {'80+':>5} {'%Wht':>6} {'%Blk':>6} {'%Oth':>6} {'%Adn':>6} {'%SCC':>6}")
print("-" * 110)
for code in COHORT_ORDER:
    r = treatment_cohort_metrics_df.loc[code]
    lbl = r["cohort_label"][:39]
    print(f"  {code}    {lbl:<40} {int(r['N']):>6} {r['pct_age_under60']:>6.1f} "
          f"{r['pct_age_60_69']:>6.1f} {r['pct_age_70_79']:>6.1f} {r['pct_age_80plus']:>5.1f} "
          f"{r['pct_white']:>6.1f} {r['pct_black']:>6.1f} {r['pct_other']:>6.1f} "
          f"{r['pct_adenocarcinoma']:>6.1f} {r['pct_squamous']:>6.1f}")

print(f"\n✅ treatment_cohort_metrics_df: {treatment_cohort_metrics_df.shape[0]} cohorts × "
      f"{treatment_cohort_metrics_df.shape[1]} metrics")
total = int(treatment_cohort_metrics_df['N'].sum())
print(f"   Total patients: {total:,} / 18,101")
print(f"\n── Radiation / Surgery status value counts (for audit) ─────────────────────")
print("radiation_status:", dict(_rad_vals))
print("surgery_status:  ", dict(_sx_vals))
print("tx_cohort counts:", dict(df["tx_cohort"].value_counts()))
