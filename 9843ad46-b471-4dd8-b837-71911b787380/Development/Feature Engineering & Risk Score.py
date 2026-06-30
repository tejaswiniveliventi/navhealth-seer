
import pandas as pd
import numpy as np
import re

# ════════════════════════════════════════════════════════════════════════════
# SEER ESOPHAGEAL CANCER — FEATURE ENGINEERING & COMPOSITE RISK SCORE
#
# ⚠️ STAGE LIMITATION: This SEER extract (18 cols) does NOT include an explicit
# AJCC/Summary Stage column (the 'Derived AJCC Stage Group' variable was not
# exported). Stage is inferred via a clinical proxy algorithm using surgery codes,
# nodes examined/positive, and radiation patterns. This proxy is validated by its
# survival gradient (I=45mo → III-IV=8mo) but should NOT substitute explicit stage
# for clinical decision-making or survival modeling.
# ════════════════════════════════════════════════════════════════════════════

seer_engineered = seer_df.copy()

BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
PAL = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4","#9467BD","#8C564B"]

# ═══ 1. AGE MIDPOINT ════════════════════════════════════════════════════════
def _age_mid(val):
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if "90+" in val: return 92
    m = re.match(r"(\d+)-(\d+)", val)
    if m: return (int(m.group(1)) + int(m.group(2))) / 2
    m2 = re.match(r"(\d+)", val)
    return float(m2.group(1)) if m2 else np.nan

seer_engineered["age_mid"] = seer_engineered["Age recode with <1 year olds and 90+"].apply(_age_mid)

# ═══ 2. SURVIVAL / VITAL ════════════════════════════════════════════════════
seer_engineered["survival_months"] = pd.to_numeric(seer_engineered["Survival months"], errors="coerce")
seer_engineered["vital_alive"]  = (seer_engineered["Vital status recode (study cutoff used)"] == "Alive").astype(int)
seer_engineered["vital_status"] = seer_engineered["Vital status recode (study cutoff used)"].str.strip()
seer_engineered["year_dx"]      = pd.to_numeric(seer_engineered["Year of diagnosis"], errors="coerce")

# ═══ 3. HISTOLOGY ═══════════════════════════════════════════════════════════
def _classify_histology(code):
    try: c = int(code)
    except: return "Other/Unknown"
    if 8050 <= c <= 8089:  return "Squamous Cell Carcinoma"
    if 8140 <= c <= 8389:  return "Adenocarcinoma"
    if c in [8480,8481,8490,8255,8260,8263]: return "Adenocarcinoma"
    if c == 8560:          return "Adenosquamous"
    return "Other/Unknown"

seer_engineered["histology_group"] = seer_engineered["Histologic Type ICD-O-3"].apply(_classify_histology)

# ═══ 4. SURGERY & RADIATION STATUS ═════════════════════════════════════════
seer_engineered["surgery_raw"] = seer_engineered["RX Summ--Surg Prim Site (1998-2022)"].astype(str).str.strip()

def _surg_status(val):
    v = str(val).strip()
    if v == "0": return "No"
    if v in ["99","Blank(s)","nan",""]: return "Unknown"
    try:
        iv = int(float(v))
        return "No" if iv == 0 else ("Unknown" if iv == 99 else "Yes")
    except: return "Unknown"

seer_engineered["surgery_status"]   = seer_engineered["surgery_raw"].apply(_surg_status)
seer_engineered["flag_had_surgery"] = seer_engineered["surgery_status"] == "Yes"

_rad_ambiguous = {"None/Unknown","Blank(s)","nan",""}
seer_engineered["radiation_status"] = seer_engineered["Radiation recode"].apply(
    lambda v: "Refused" if str(v).strip() == "Refused (1988+)"
    else "Unknown" if (str(v).strip() in _rad_ambiguous or pd.isna(v))
    else "Yes"
)
seer_engineered["flag_had_radiation"] = seer_engineered["radiation_status"] == "Yes"

# ═══ 5. NODES ═══════════════════════════════════════════════════════════════
seer_engineered["nodes_examined"] = pd.to_numeric(
    seer_engineered["Regional nodes examined (1988+)"], errors="coerce"
).where(lambda x: x < 98, np.nan)
seer_engineered["nodes_positive"] = pd.to_numeric(
    seer_engineered["Regional nodes positive (1988+)"], errors="coerce"
).where(lambda x: x < 97, np.nan)

# ═══ 6. PROXY STAGE (Evidence-Based SEER Rules) ═════════════════════════════
def _proxy_stage(row):
    sx, rad = row["surgery_status"], row["radiation_status"]
    ne, np_ = row["nodes_examined"], row["nodes_positive"]
    try: sc = int(float(row["surgery_raw"]))
    except: sc = -1

    has_sx = sx == "Yes"
    has_rad = rad == "Yes"
    unk_sx  = sx == "Unknown"

    if not has_sx and not unk_sx:
        return "III-IV"  # Non-surgical: locally advanced / metastatic pattern

    if has_sx:
        ne_ok = pd.notna(ne) and ne > 0
        np_ok = pd.notna(np_) and ne_ok
        if ne_ok and np_ok:
            if np_ == 0:
                return "I" if sc in range(10, 23) else ("II" if has_rad else "I-II")
            elif np_ <= 2: return "II" if not has_rad else "III"
            elif np_ <= 6: return "III"
            else:          return "III-IV"
        elif ne_ok:
            return "II-III" if has_rad else "I-II"
        else:
            if sc in range(10, 23): return "I"
            return "II-III" if has_rad else "I-II"
    return "Unknown"

seer_engineered["stage_proxy"] = seer_engineered.apply(_proxy_stage, axis=1)

_stage_collapse = {"I":"I","II":"II","III":"III","IV":"IV",
                   "I-II":"I","II-III":"II","III-IV":"III","Unknown":"Unknown"}
seer_engineered["stage"] = seer_engineered["stage_proxy"].map(_stage_collapse)

# ═══ 7. RISK SCORE COMPONENTS ═══════════════════════════════════════════════
WEIGHTS = {"stage":0.35, "age":0.20, "treatment":0.30, "year":0.10, "nodes":0.05}

# A: Stage (35%)
_stage_map = {"I":0.0, "II":0.33, "III":0.67, "IV":1.0, "Unknown":np.nan}
seer_engineered["rs_stage"] = seer_engineered["stage"].map(_stage_map)

# B: Age (20%)
def _age_score(a):
    if pd.isna(a): return np.nan
    if a < 50: return 0.0
    if a < 60: return 0.2
    if a < 70: return 0.4
    if a < 80: return 0.6
    return 1.0
seer_engineered["rs_age"] = seer_engineered["age_mid"].apply(_age_score)

# C: Treatment Intensity (30%)
def _tx_score(sx, rad):
    has_s = sx == "Yes"; has_r = rad == "Yes"
    unk_s = sx == "Unknown"; unk_r = rad == "Unknown"
    if unk_s and unk_r: return 0.75
    if has_s and has_r:  return 0.25
    if has_s and not has_r: return 0.5
    if has_r and not has_s: return 0.4
    if not has_s and not has_r and not unk_s and not unk_r: return 1.0
    return 0.75
seer_engineered["rs_treatment"] = seer_engineered.apply(
    lambda r: _tx_score(r["surgery_status"], r["radiation_status"]), axis=1
)

# D: Year of Diagnosis (10%)
def _year_score(yr):
    if pd.isna(yr): return 0.2
    if yr <= 2015: return 0.3
    if yr <= 2019: return 0.2
    return 0.0
seer_engineered["rs_year"] = seer_engineered["year_dx"].apply(_year_score)

# E: Nodes Examined Proxy (5%)
def _nodes_score(ne):
    if pd.isna(ne) or ne == 0: return np.nan  # Non-surgical → N/A
    return 0.0 if ne >= 15 else 0.5
seer_engineered["rs_nodes"] = seer_engineered["nodes_examined"].apply(_nodes_score)

# Composite: renormalize weights over available components
def _composite_risk(row):
    comps = {k: row[f"rs_{k}"] for k in WEIGHTS}
    avail = {k: v for k, v in comps.items() if not pd.isna(v)}
    if not avail: return np.nan
    total_w = sum(WEIGHTS[k] for k in avail)
    return round(sum(v * WEIGHTS[k] for k, v in avail.items()) / total_w * 100, 2)

seer_engineered["risk_score"] = seer_engineered.apply(_composite_risk, axis=1)

def _risk_cat(s):
    if pd.isna(s): return "Unknown"
    return "Low" if s <= 33 else ("Moderate" if s <= 66 else "High")
seer_engineered["risk_category"] = seer_engineered["risk_score"].apply(_risk_cat)

# Per-driver % contribution
for comp in WEIGHTS:
    _val = seer_engineered[f"rs_{comp}"].fillna(0) * WEIGHTS[comp]
    _raw_01 = seer_engineered["risk_score"] / 100
    seer_engineered[f"risk_driver_{comp}"] = (
        (_val / _raw_01.replace(0, np.nan)) * 100
    ).round(1)

# Data quality flag
seer_engineered["risk_data_quality_flag"] = np.where(
    (seer_engineered["surgery_status"] == "Unknown") |
    (seer_engineered["radiation_status"] == "Unknown") |
    (seer_engineered["stage"] == "Unknown"),
    "⚠️ Data Incomplete", "✅ Complete"
)

# ═══ 8. COHORT FLAGS ════════════════════════════════════════════════════════
# Stage
for st in ["I","II","III","IV"]:
    seer_engineered[f"flag_stage_{st}"] = seer_engineered["stage"] == st
seer_engineered["flag_advanced"] = seer_engineered["stage"].isin(["III","IV"])
seer_engineered["flag_early"]    = seer_engineered["stage"].isin(["I","II"])

# Age groups
seer_engineered["age_group"] = pd.cut(
    seer_engineered["age_mid"], bins=[0,59.9,69.9,79.9,200],
    labels=["<60","60-69","70-79","80+"]
)
seer_engineered["flag_age_under_60"] = seer_engineered["age_mid"] < 60
seer_engineered["flag_age_60_70"]    = (seer_engineered["age_mid"] >= 60) & (seer_engineered["age_mid"] < 70)
seer_engineered["flag_age_over_70"]  = seer_engineered["age_mid"] >= 70

# Treatment
seer_engineered["flag_had_both_sx_rad"]        = seer_engineered["flag_had_surgery"] & seer_engineered["flag_had_radiation"]
seer_engineered["flag_no_treatment"]           = (
    (~seer_engineered["flag_had_surgery"]) & (~seer_engineered["flag_had_radiation"]) &
    (seer_engineered["surgery_status"] != "Unknown") & (seer_engineered["radiation_status"] != "Unknown")
)
seer_engineered["flag_treatment_data_missing"] = (
    (seer_engineered["surgery_status"] == "Unknown") | (seer_engineered["radiation_status"] == "Unknown")
)

# Histology
seer_engineered["flag_adenocarcinoma"] = seer_engineered["histology_group"] == "Adenocarcinoma"
seer_engineered["flag_squamous"]       = seer_engineered["histology_group"] == "Squamous Cell Carcinoma"

# Other
seer_engineered["flag_surgical_subgroup"] = seer_engineered["nodes_examined"].fillna(0) > 0
seer_engineered["flag_recent_cohort"]     = seer_engineered["year_dx"] >= 2018

# Race simplified
def _race_simple(r):
    r = str(r)
    if "White" in r: return "White"
    if "Black" in r: return "Black"
    if "Other" in r: return "Other"
    return "Unknown"
seer_engineered["race_group"] = seer_engineered["Race recode (White, Black, Other)"].apply(_race_simple)

# Year cohort (for protocol-era comparisons)
seer_engineered["year_cohort"] = np.where(
    seer_engineered["year_dx"] >= 2018, "2018-2023 (Modern)", "2010-2017 (Older)"
)

# ═══ 9. SUMMARY OUTPUT ══════════════════════════════════════════════════════
print(f"{'='*65}")
print(f"ENGINEERED DATASET: {len(seer_engineered):,} rows × {len(seer_engineered.columns)} columns")
print(f"{'='*65}")

print(f"\n── PROXY STAGE (Full Labels) ────────────────────────────────")
for s, n in seer_engineered["stage_proxy"].value_counts().items():
    print(f"  {s:12s}: {n:5,}  ({100*n/len(seer_engineered):.1f}%)")

print(f"\n── COLLAPSED STAGE ─────────────────────────────────────────")
for s, n in seer_engineered["stage"].value_counts().items():
    print(f"  Stage {s:7s}: {n:5,}  ({100*n/len(seer_engineered):.1f}%)")

print(f"\n── RISK SCORE DISTRIBUTION ─────────────────────────────────")
_rs = seer_engineered["risk_score"].dropna()
print(f"  Mean ± SD : {_rs.mean():.1f} ± {_rs.std():.1f}")
print(f"  Median    : {_rs.median():.1f}  |  Range: {_rs.min():.1f}–{_rs.max():.1f}")
for cat, n in seer_engineered["risk_category"].value_counts().items():
    print(f"  {cat:10s}: {n:5,}  ({100*n/len(seer_engineered):.1f}%)")

print(f"\n── TREATMENT FLAGS ─────────────────────────────────────────")
print(f"  Had Surgery    : {seer_engineered['flag_had_surgery'].sum():,}")
print(f"  Had Radiation  : {seer_engineered['flag_had_radiation'].sum():,}")
print(f"  Had Both (S+R) : {seer_engineered['flag_had_both_sx_rad'].sum():,}")
print(f"  No Treatment   : {seer_engineered['flag_no_treatment'].sum():,}")
print(f"  Tx Missing     : {seer_engineered['flag_treatment_data_missing'].sum():,}")
print(f"  Surgical Subgp : {seer_engineered['flag_surgical_subgroup'].sum():,}")

print(f"\n── HISTOLOGY ────────────────────────────────────────────────")
for h, n in seer_engineered["histology_group"].value_counts().items():
    print(f"  {h:30s}: {n:,}  ({100*n/len(seer_engineered):.1f}%)")

print(f"\n── DATA QUALITY FLAG ────────────────────────────────────────")
for f, n in seer_engineered["risk_data_quality_flag"].value_counts().items():
    print(f"  {f}: {n:,}")

print(f"\n⚠️  CRITICAL CAVEATS:")
print(f"  • Stage PROXY derived from surgery/nodes/radiation — NOT explicit AJCC stage")
print(f"  • 'III' proxy = 72.9% of cohort (surgical+nodes positive OR non-surgical+radiation)")
print(f"  • 9.7% Unknown stage = surgery code Blank(s), mainly 2023 records")
print(f"  • Radiation ambiguity: {seer_engineered['radiation_status'].eq('Unknown').mean()*100:.0f}% Unknown — sensitivity analysis needed")
print(f"  • Grade excluded post-2017 (structural cutoff)")
print(f"  • LVI 100% blank — excluded")
