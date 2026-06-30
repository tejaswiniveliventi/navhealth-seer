
import pandas as pd
import numpy as np
import re

# ════════════════════════════════════════════════════════════════════════════
# SEER ESOPHAGEAL CANCER — CLEAN FEATURE ENGINEERING & COMPOSITE RISK SCORE
#
# ⚠️ STAGE NOTE: This SEER extract lacks an explicit AJCC/Summary Stage column.
# Stage is inferred via proxy algorithm (surgery codes, nodes, radiation).
# ════════════════════════════════════════════════════════════════════════════

BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
WEIGHTS = {"stage": 0.35, "age": 0.20, "treatment": 0.30, "year": 0.10}

# ─── STEP 1: PARSE & STANDARDIZE ───────────────────────────────────────────

def _age_mid(val):
    """Convert age band string to numeric midpoint."""
    if pd.isna(val): return np.nan
    val = str(val).strip()
    if "90+" in val: return 92
    m = re.match(r"(\d+)-(\d+)", val)
    if m: return (int(m.group(1)) + int(m.group(2))) / 2
    m2 = re.match(r"(\d+)", val)
    return float(m2.group(1)) if m2 else np.nan

def _surg_binary(val):
    """Surgery: 1=Yes (codes 1-80), 0=No (code 0), NaN=Unknown (99/Blank)."""
    v = str(val).strip()
    if v in ["99", "Blank(s)", "nan", ""]: return np.nan
    try:
        iv = int(float(v))
        if iv == 0:  return 0
        if iv == 99: return np.nan
        return 1
    except: return np.nan

def _rad_binary(val):
    """Radiation: 1=Yes (any treatment), 0=No (None/Unknown → treated as no), NaN=Blank."""
    v = str(val).strip()
    if v in ["Blank(s)", "nan", ""]: return np.nan
    if v in ["None/Unknown", "Refused (1988+)"]: return 0
    return 1

def _stage_label(row):
    """Proxy stage from surgery/nodes/radiation — not explicit AJCC."""
    sx, rad = row["Surgery_binary"], row["Radiation_binary"]
    ne, np_ = row["nodes_examined"], row["nodes_positive"]
    try: sc = int(float(str(row["surgery_raw"]).strip()))
    except: sc = -1

    has_sx = sx == 1
    has_rad = rad == 1
    unk_sx  = pd.isna(sx)

    if not has_sx and not unk_sx:
        return "III"   # Non-surgical → locally advanced

    if has_sx:
        ne_ok = pd.notna(ne) and ne > 0
        np_ok = pd.notna(np_) and ne_ok
        if ne_ok and np_ok:
            if np_ == 0:   return "I" if sc in range(10, 23) else ("II" if has_rad else "I")
            elif np_ <= 2: return "II" if not has_rad else "III"
            elif np_ <= 6: return "III"
            else:          return "III"
        elif ne_ok:
            return "III" if has_rad else "II"
        else:
            if sc in range(10, 23): return "I"
            return "II" if has_rad else "I"
    return "Unknown"

def _histology(code):
    try: c = int(code)
    except: return "Other/Unknown"
    if 8050 <= c <= 8089: return "Squamous Cell Carcinoma"
    if 8140 <= c <= 8389: return "Adenocarcinoma"
    if c in [8480, 8481, 8490, 8255, 8260, 8263]: return "Adenocarcinoma"
    if c == 8560: return "Adenosquamous"
    return "Other/Unknown"

def _race_simple(r):
    r = str(r)
    if "White" in r: return "White"
    if "Black" in r: return "Black"
    if "Other" in r: return "Other"
    return "Unknown"

# Build base frame from raw SEER
df = seer_df.copy()

# Core numeric / date columns
df["Age_numeric"]      = df["Age recode with <1 year olds and 90+"].apply(_age_mid)
df["survival_months"]  = pd.to_numeric(df["Survival months"], errors="coerce")
df["survival_years"]   = (df["survival_months"] / 12).round(3)
df["year_dx"]          = pd.to_numeric(df["Year of diagnosis"], errors="coerce")
df["Vital_binary"]     = (df["Vital status recode (study cutoff used)"] == "Alive").astype(int)
df["vital_status"]     = df["Vital status recode (study cutoff used)"].str.strip()

# Histology
df["histology_group"] = df["Histologic Type ICD-O-3"].apply(_histology)

# Surgery & Radiation binary (Step 1 requirement)
df["surgery_raw"]      = df["RX Summ--Surg Prim Site (1998-2022)"].astype(str).str.strip()
df["Surgery_binary"]   = df["surgery_raw"].apply(_surg_binary)   # 1/0/NaN
df["Radiation_binary"] = df["Radiation recode"].apply(_rad_binary) # 1/0/NaN

# Surgery & Radiation status labels (for treatment intensity scoring)
df["surgery_status"]   = df["Surgery_binary"].map({1: "Yes", 0: "No"}).fillna("Unknown")
df["radiation_status"] = df["Radiation_binary"].map({1: "Yes", 0: "No"}).fillna("Unknown")

# Nodes
df["nodes_examined"] = pd.to_numeric(
    df["Regional nodes examined (1988+)"], errors="coerce"
).where(lambda x: x < 98, np.nan)
df["nodes_positive"] = pd.to_numeric(
    df["Regional nodes positive (1988+)"], errors="coerce"
).where(lambda x: x < 97, np.nan)

# Stage proxy (uses Surgery_binary + Radiation_binary)
df["Stage"] = df.apply(_stage_label, axis=1)
_stage_numeric_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "Unknown": np.nan}
df["Stage_Numeric"] = df["Stage"].map(_stage_numeric_map)

# Race / Sex simplified
df["race_group"] = df["Race recode (White, Black, Other)"].apply(_race_simple)
df["sex"]        = df["Sex"].str.strip()

# Age group
df["age_group"] = pd.cut(
    df["Age_numeric"], bins=[0, 49.9, 59.9, 69.9, 79.9, 200],
    labels=["<50", "50-59", "60-69", "70-79", "80+"]
)

# Year cohort
df["year_cohort"] = np.where(df["year_dx"] >= 2018, "2018-2023 (Modern)", "2010-2017 (Older)")

# ─── STEP 2: COHORT FLAGS ──────────────────────────────────────────────────

# Stage flags
for st in ["I", "II", "III", "IV"]:
    df[f"flag_stage_{st}"] = df["Stage"] == st
df["flag_advanced"] = df["Stage"].isin(["III", "IV"])
df["flag_early"]    = df["Stage"].isin(["I", "II"])

# Age flags
df["flag_age_under_60"] = df["Age_numeric"] < 60
df["flag_age_60_70"]    = (df["Age_numeric"] >= 60) & (df["Age_numeric"] < 70)
df["flag_age_over_70"]  = df["Age_numeric"] >= 70

# Treatment flags  ⚠️ Surgery post-2023 sparse; Radiation ~40-45% None/Unknown
df["flag_had_surgery"]         = df["Surgery_binary"] == 1
df["flag_had_radiation"]       = df["Radiation_binary"] == 1
df["flag_both_sx_rad"]         = df["flag_had_surgery"] & df["flag_had_radiation"]
df["flag_no_treatment"]        = (df["Surgery_binary"] == 0) & (df["Radiation_binary"] == 0)
df["flag_treatment_missing"]   = df["Surgery_binary"].isna() | df["Radiation_binary"].isna()

# Histology flags
df["flag_adenocarcinoma"] = df["histology_group"] == "Adenocarcinoma"
df["flag_squamous"]       = df["histology_group"] == "Squamous Cell Carcinoma"

# Auxiliary flags
df["flag_surgical_subgroup"] = df["nodes_examined"].fillna(0) > 0
df["flag_recent_cohort"]     = df["year_dx"] >= 2018

# ─── STEP 3: COMPOSITE RISK SCORE (0-100) ─────────────────────────────────

# A: Stage component (35%)
_stage_score = {"I": 0.0, "II": 0.33, "III": 0.67, "IV": 1.0, "Unknown": np.nan}
df["rs_stage"] = df["Stage"].map(_stage_score)

# B: Age component (20%)
def _age_score(a):
    if pd.isna(a): return np.nan
    if a < 50: return 0.0
    if a < 60: return 0.2
    if a < 70: return 0.4
    if a < 80: return 0.6
    return 1.0
df["rs_age"] = df["Age_numeric"].apply(_age_score)

# C: Treatment Intensity component (30%)
def _tx_score(sx_bin, rad_bin):
    has_s = sx_bin == 1;  has_r = rad_bin == 1
    unk_s = pd.isna(sx_bin); unk_r = pd.isna(rad_bin)
    if has_s and has_r:               return 0.25   # Surgery + Radiation (best)
    if has_s and not has_r:           return 0.50   # Surgery only
    if has_r and not has_s:           return 0.40   # Radiation only (partial)
    if not has_s and not has_r and not unk_s and not unk_r: return 1.0  # No treatment
    return 0.75                                      # Unknown → assumed under-treated ⚠️
df["rs_treatment"] = df.apply(lambda r: _tx_score(r["Surgery_binary"], r["Radiation_binary"]), axis=1)

# D: Year component (10%)
def _year_score(yr):
    if pd.isna(yr): return 0.2
    if yr <= 2015: return 0.3
    if yr <= 2019: return 0.2
    return 0.0
df["rs_year"] = df["year_dx"].apply(_year_score)

# Composite (renormalize over available components — no nodes in primary score per spec)
def _composite(row):
    comps = {k: row[f"rs_{k}"] for k in WEIGHTS}
    avail = {k: v for k, v in comps.items() if not pd.isna(v)}
    if not avail: return np.nan
    total_w = sum(WEIGHTS[k] for k in avail)
    return round(sum(v * WEIGHTS[k] for k, v in avail.items()) / total_w * 100, 2)

df["Risk_Score"] = df.apply(_composite, axis=1)

def _risk_cat(s):
    if pd.isna(s): return "Unknown"
    return "Low" if s <= 33 else ("Moderate" if s <= 66 else "High")
df["Risk_Category"] = df["Risk_Score"].apply(_risk_cat)

# Data quality flag
df["risk_data_quality_flag"] = np.where(
    df["Surgery_binary"].isna() | df["Radiation_binary"].isna() | (df["Stage"] == "Unknown"),
    "⚠️ Data Incomplete", "✅ Complete"
)

# ─── FINAL DATASET ─────────────────────────────────────────────────────────
# Select and order clean output columns
_keep = [
    # Original identifiers & demographics
    "Age_numeric", "age_group", "sex", "race_group", "year_dx", "year_cohort",
    "histology_group",
    # Clinical outcomes
    "vital_status", "Vital_binary", "survival_months", "survival_years",
    # Treatment (⚠️ caution)
    "Surgery_binary", "surgery_status", "Radiation_binary", "radiation_status",
    "nodes_examined", "nodes_positive",
    # Engineered stage & score
    "Stage", "Stage_Numeric",
    "Risk_Score", "Risk_Category",
    "rs_stage", "rs_age", "rs_treatment", "rs_year",
    "risk_data_quality_flag",
    # Cohort flags — stage
    "flag_stage_I", "flag_stage_II", "flag_stage_III", "flag_stage_IV",
    "flag_advanced", "flag_early",
    # Cohort flags — age
    "flag_age_under_60", "flag_age_60_70", "flag_age_over_70",
    # Cohort flags — treatment
    "flag_had_surgery", "flag_had_radiation", "flag_both_sx_rad",
    "flag_no_treatment", "flag_treatment_missing",
    # Cohort flags — histology / subgroup
    "flag_adenocarcinoma", "flag_squamous",
    "flag_surgical_subgroup", "flag_recent_cohort",
]
seer_engineered = df[_keep].reset_index(drop=True)

# ─── STEP 4: SUMMARY OUTPUT ────────────────────────────────────────────────
_rs   = seer_engineered["Risk_Score"].dropna()
_n    = len(seer_engineered)
_miss_sx  = seer_engineered["Surgery_binary"].isna().sum()
_miss_rad = seer_engineered["Radiation_binary"].isna().sum()

print("=" * 60)
print(f" SEER ENGINEERED DATASET — {_n:,} rows × {len(seer_engineered.columns)} columns")
print("=" * 60)

print(f"\n── ROW / COLUMN COUNT {'─'*38}")
print(f"  Total rows    : {_n:,}")
print(f"  Total columns : {len(seer_engineered.columns)}")

print(f"\n── MISSING DATA ⚠️  {'─'*40}")
print(f"  Surgery missing (Blank/99)     : {_miss_sx:,}  ({100*_miss_sx/_n:.1f}%)")
print(f"  Radiation missing (Blank)      : {_miss_rad:,}  ({100*_miss_rad/_n:.1f}%)")
print(f"  Stage Unknown                  : {(seer_engineered['Stage']=='Unknown').sum():,}  ({100*(seer_engineered['Stage']=='Unknown').sum()/_n:.1f}%)")

print(f"\n── STAGE DISTRIBUTION (Proxy ⚠️ Not AJCC) {'─'*19}")
for st, n in seer_engineered["Stage"].value_counts().items():
    bar = "█" * int(30 * n / _n)
    print(f"  Stage {st:7s}: {n:6,}  ({100*n/_n:5.1f}%)  {bar}")

print(f"\n── AGE_NUMERIC ─────────────────────────────────────────")
_age = seer_engineered["Age_numeric"]
print(f"  Range   : {_age.min():.0f} – {_age.max():.0f} yrs")
print(f"  Median  : {_age.median():.0f} yrs  |  Mean: {_age.mean():.1f} yrs")
print(f"  Outside 0-120 : {(((_age<0)|(_age>120)).sum())}")

print(f"\n── SURVIVAL MONTHS ─────────────────────────────────────")
_sm = seer_engineered["survival_months"]
print(f"  Range   : {_sm.min():.0f} – {_sm.max():.0f} months")
print(f"  Median  : {_sm.median():.0f}  |  Mean: {_sm.mean():.1f}")
print(f"  Negative: {(_sm<0).sum()}  |  >600 months: {(_sm>600).sum()}")

print(f"\n── VITAL STATUS ─────────────────────────────────────────")
for k, v in seer_engineered["vital_status"].value_counts().items():
    print(f"  {k:8s}: {v:,}  ({100*v/_n:.1f}%)")

print(f"\n── SURGERY_BINARY ⚠️  {'─'*38}")
_sx_vc = seer_engineered["Surgery_binary"].value_counts(dropna=False)
print(f"  1  (Yes)   : {int(_sx_vc.get(1.0, 0)):,}")
print(f"  0  (No)    : {int(_sx_vc.get(0.0, 0)):,}")
print(f"  NaN (Unk)  : {seer_engineered['Surgery_binary'].isna().sum():,}")

print(f"\n── RADIATION_BINARY ⚠️  {'─'*37}")
_rd_vc = seer_engineered["Radiation_binary"].value_counts(dropna=False)
print(f"  1  (Yes)   : {int(_rd_vc.get(1.0, 0)):,}")
print(f"  0  (No)    : {int(_rd_vc.get(0.0, 0)):,}")
print(f"  NaN (Unk)  : {seer_engineered['Radiation_binary'].isna().sum():,}")

print(f"\n── RISK SCORE DISTRIBUTION ─────────────────────────────")
print(f"  Min    : {_rs.min():.1f}   |  Max: {_rs.max():.1f}")
print(f"  Mean   : {_rs.mean():.1f}  |  Median: {_rs.median():.1f}")
print(f"  Std    : {_rs.std():.1f}")

print(f"\n── RISK CATEGORY BREAKDOWN ─────────────────────────────")
for cat, n in seer_engineered["Risk_Category"].value_counts().items():
    bar = "█" * int(30 * n / _n)
    print(f"  {cat:12s}: {n:6,}  ({100*n/_n:5.1f}%)  {bar}")

print(f"\n── KEY FLAGS SUMMARY ───────────────────────────────────")
flag_cols = [c for c in seer_engineered.columns if c.startswith("flag_")]
for fc in flag_cols:
    ct = seer_engineered[fc].sum()
    print(f"  {fc:35s}: {int(ct):,}  ({100*ct/_n:.1f}%)")

print(f"\n── FIRST 10 ROWS (key columns) ─────────────────────────")
_preview_cols = ["Age_numeric","age_group","sex","Stage","Risk_Score","Risk_Category",
                 "Surgery_binary","Radiation_binary","vital_status","survival_months",
                 "survival_years","risk_data_quality_flag"]
print(seer_engineered[_preview_cols].head(10).to_string(index=True))

print(f"\n── COLUMN LIST ({len(seer_engineered.columns)} columns) ─────────────────────")
for i, c in enumerate(seer_engineered.columns, 1):
    print(f"  {i:2d}. {c}")
