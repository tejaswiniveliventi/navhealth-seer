# 🩺 Esophageal Cancer Survival Analysis — SEER Population Study

> **Population-level survival analysis and patient decision support tool built on 18,101 SEER records (2010–2023).**
> For educational and research purposes only. Not a medical diagnosis tool.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Data Quality & Limitations](#data-quality--limitations)
- [Streamlit App — Patient Journey Wizard](#streamlit-app--patient-journey-wizard)
- [Pipeline Architecture](#pipeline-architecture)
- [Clinical Context](#clinical-context)
- [Disclaimer](#disclaimer)

---

## Project Overview

This project performs a comprehensive, end-to-end survival analysis of esophageal cancer patients using data from the **Surveillance, Epidemiology, and End Results (SEER)** program. It combines rigorous data quality auditing, feature engineering, treatment cohort stratification, and an interactive patient-facing decision guide.

**Core questions this project answers:**

- How does treatment intensity (surgery, radiation, multimodal) affect survival outcomes at 1, 3, and 5 years?
- How do age, histology (adenocarcinoma vs. squamous cell), and diagnosis year modulate those outcomes?
- Has survival improved in the modern protocol era (2018–2023, post-CheckMate 577)?
- How reliable are SEER-derived survival estimates, and where is right-censoring a problem?

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [SEER Program](https://seer.cancer.gov/), National Cancer Institute |
| **Cancer site** | Esophageal cancer (all stages) |
| **Records** | 18,101 patients |
| **Diagnosis years** | 2010–2023 |
| **Columns (raw)** | 18 |
| **Columns (engineered)** | 44–65 (after feature engineering) |

### Key Variables

| Variable | Type | Completeness | Notes |
|---|---|---|---|
| Age | Numeric (midpoint) | ✅ 100% | Banded strings → midpoints |
| Sex | Categorical | ✅ 100% | Male / Female |
| Race | Categorical | ✅ 99.7% | White / Black / Other |
| Vital Status | Binary | ✅ 100% | Alive / Dead |
| Survival Months | Numeric | ✅ 100% | Range 0–167 months |
| Histologic Type | Categorical | ✅ 100% | Adenocarcinoma (64%), SCC (29%) |
| Year of Diagnosis | Numeric | ✅ 100% | 2010–2023 |
| Surgery Code | Binary (engineered) | ⚠️ 90.3% | Blank in 9.7% (2022–23 sparse) |
| Radiation | Binary (engineered) | ⚠️ ambiguous | 46% "None/Unknown" — underdetermined |
| Grade | Categorical | ❌ 41.4% | Structurally blank post-2017 |
| LVI | Categorical | ❌ 0% | 100% blank — excluded entirely |
| Nodes Positive | Numeric | ❌ 23.5% | 76.5% missing + 44 logical errors |

---

## Project Structure

```
.
├── analysis/
│   ├── 01_load_seer_data.py              # Load & profile raw SEER CSV
│   ├── 02_missingness_analysis.py        # Sentinel-value detection, heatmap
│   ├── 03_data_type_integrity.py         # Type validation, distribution checks
│   ├── 04_outlier_detection.py           # IQR-based outlier flagging
│   ├── 05_data_quality_score_table.py    # Per-column quality scorecard
│   ├── 06_data_quality_dashboard.py      # 6-panel audit dashboard (Matplotlib)
│   ├── 07_feature_engineering.py         # Risk score + seer_engineered dataframe
│   ├── 08_stage_survival_metrics.py      # Stage-specific survival tables
│   ├── 09_demographic_heatmap.py         # Age/race/sex/histology × stage
│   ├── 10_treatment_intensity.py         # CROSS trial proxy validation
│   ├── 11_treatment_cohort_metrics.py    # 5-cohort survival stratification
│   ├── 12_stage_proxy_stratification.py  # Stage proxy + age/histology heatmaps
│   ├── 13_cohort_evolution.py            # 2010–17 vs 2018–23 era comparison
│   ├── 14_risk_score_validation.py       # Risk score stratification chart
│   ├── 15_era_comparison_chart.py        # Modern vs historical Plotly chart
│   └── 16_followup_censoring_analysis.py # Right-censoring metadata & flags
│
├── app/
│   └── main.py                           # Streamlit Patient Journey Wizard
│
├── outputs/
│   ├── fig_treatment_outcomes.html       # Dual-axis cohort chart
│   ├── fig_risk_stratification.html      # Risk tier grouped bar chart
│   ├── fig_age_heatmap.html              # Age × treatment heatmap
│   ├── fig_histology_heatmap.html        # Histology × treatment heatmap
│   └── fig_era_comparison.html           # Era comparison (2-panel)
│
└── README.md
```

---

## Key Findings

### 1. Treatment Intensity Drives Survival More Than Any Other Factor

| Treatment Cohort | N | % Alive | Median Survival | 1-Year | 3-Year | 5-Year |
|---|---|---|---|---|---|---|
| 🟢 Surgery + Radiation (Multimodal) | 2,618 | 40.6% | 31 mo | 84.7% | 45.9% | 28.3% |
| 🔵 Surgery Only | 1,519 | 55.6% | 49 mo | 90.1% | 62.1% | 41.9% |
| 🟠 Radiation Only | 6,509 | 12.8% | 11 mo | 49.1% | 15.4% | 7.5% |
| 🔴 No Treatment Recorded | 5,708 | 6.4% | 4 mo | 24.4% | 5.4% | 2.7% |
| ⚫ Treatment Unknown ⚠️ | 1,747 | 55.1%* | 4 mo | 2.0% | 0.9% | 0.6% |

> ⚠️ *Cohort E's 55% "alive" figure is a **right-censoring artifact** — recent 2022–23 records with <1 year follow-up, not genuine survival benefit. Exclude from all comparisons.

> ⚠️ Surgery Only > Multimodal (raw %) reflects **selection bias**: multimodal patients were more advanced at baseline. Use 1/3/5-year rates for fairer comparison.

---

### 2. Risk Score Stratifies Strongly (73pp Gap at 1 Year)

The composite risk score (Stage 35% · Treatment 30% · Age 20% · Year 10%) separates outcomes clearly:

| Risk Category | N | % Alive | 1-Year Survival | 5-Year Survival |
|---|---|---|---|---|
| 🟢 Low (0–33) | 2,719 | 58.0% | 84.7% | 36.5% |
| 🟡 Moderate (34–66) | 13,628 | 17.8% | 42.3% | 7.6% |
| 🔴 High (67–100) | 1,754 | 4.4% | 11.8% | 1.0% |

---

### 3. SCC vs Adenocarcinoma — Histology Matters in Multimodal Therapy

- SCC shows a **+5pp survival advantage** over adenocarcinoma in multimodal treatment (33.6% vs 28.6%)
- Consistent with CROSS trial biology: SCC is more radiosensitive (CROSS cohort was 75% ESCC-enriched)
- Adenocarcinoma dominates numerically in SEER (64%), reflecting Western Barrett's-driven disease geography

---

### 4. Modern Era Improvement — Partially Confounded by Follow-Up

| Cohort | 2010–17 % Alive | 2018–23 % Alive | Δ | Interpretation |
|---|---|---|---|---|
| Multimodal | 29.7% | 56.9% | +27.2pp | ⚠️ Inflated by right-censoring |
| Surgery Only | 42.7% | 74.4% | +31.7pp | ⚠️ Inflated by right-censoring |
| Radiation Only | 6.3% | 22.5% | +16.2pp | ⚠️ Partially confounded |
| No Treatment | 3.8% | 10.0% | +6.2pp | ⚠️ Short follow-up |

Median survival **dropped** for multimodal (44 mo → 26 mo) in the modern era — confirming that the apparent % Alive improvement is substantially a follow-up artifact. True 5-year data for 2019–2023 patients won't be available until **approximately 2028**.

---

## Data Quality & Limitations

### Follow-Up Completeness

| Timepoint | N Eligible | % of Cohort | Status |
|---|---|---|---|
| 1-Year Survival | 16,671 | 92.1% | ✅ Complete |
| 3-Year Survival | 13,743 | 75.9% | ✅ Good coverage |
| 5-Year Survival | 11,070 | 61.2% | ⚠️ Pre-2018 patients only |

### Known Caveats

| Issue | Impact | Recommendation |
|---|---|---|
| **Stage is proxy-derived** (no AJCC column) | Stage III = 73% of cohort (coding artifact). Cannot distinguish true TNM staging. | Use treatment intensity as primary stratification. Stage shown as descriptive reference only. |
| **Radiation ambiguity** | ~46% coded "None/Unknown" — cannot distinguish confirmed no-treatment from undocumented treatment | Treat Radiation=0 as ambiguous. Sensitivity analysis recommended. |
| **5-year follow-up incomplete** | 38.8% of cohort (2019–2023) has <5 years follow-up | Use 1-year or 3-year survival for modern era comparisons |
| **Cohort E right-censored** | 1,747 surgery-blank records (9.6%) — mean follow-up 0.33 years | **Exclude from all survival analyses** |
| **Grade missing post-2017** | 58.6% overall, 100% blank 2018–2023 (registry policy change) | Excluded from risk score and all modeling |
| **LVI 100% blank** | Never populated in this extract | Excluded from all analysis |
| **No performance status** | SEER has no ECOG/Karnofsky data | Cannot adjust for patient fitness; treatment selection bias is uncontrolled |
| **No treatment detail** | No chemotherapy regimen, dose, cycles, or adjuvant immunotherapy tracking | Cannot validate specific CROSS/CheckMate 577 protocols |
| **No surgical margins** | R0/R1/R2 status not recorded in SEER | Cannot assess resection quality effect on outcomes |

---

## Streamlit App — Patient Journey Wizard

The Streamlit app (`app/main.py`) is a **5-step Patient Journey Wizard** designed for patients, families, and clinicians to explore population-level outcomes for a specific patient profile.

### Steps

| Step | Title | What It Shows |
|---|---|---|
| **Step 0** | 🧑 Profile Input | Central wizard: Age, Stage, Histology, Surgery, Radiation, Year Dx |
| **Step 1** | 🧮 Risk Assessment | Composite risk score gauge (0–100) + driver breakdown bars |
| **Step 2** | 📊 Clinical Benchmarking | Survival histogram (your cohort vs. all 18,101) + follow-up completeness panel |
| **Step 3** | 💊 Treatment Pathway | KPI cards + cohort comparison chart + era bias expander |
| **Step 4** | 📚 Evidence & Next Steps | CROSS / CheckMate 577 / FLOT evidence cards + oncology team checklist |

### Design System

```python
BG      = "#1D1D20"   # Dark background
BG2     = "#26262B"   # Card surface
TXT     = "#fbfbff"   # Primary text
SUB     = "#909094"   # Secondary text
GREEN   = "#17b26a"   # Low risk / positive
YELLOW  = "#ffd400"   # Moderate risk / caution
RED     = "#f04438"   # High risk / warning
```

### Running Locally

```bash
pip install streamlit pandas numpy plotly
streamlit run app/main.py
```

### Key App Features

- **`compute_risk_score()`** — Core engine. Weighted composite score (Stage · Age · Treatment · Year). Preserved across all refactors.
- **`filter_cohort()`** — Filters `seer_engineered` by age ±10 years, same stage, same histology for histogram benchmarking.
- **"About This Data" expander** on every step — plain-language explanation of SEER limitations, written for non-technical patients and families (no jargon).
- **Right-censoring metadata** — `right_censoring_metadata` JSON dict auto-applies ⚠️ footnotes to any 5-year survival card showing pre-2018 data only.

---

## Pipeline Architecture

```
Raw SEER CSV (18,101 rows × 18 columns)
        │
        ▼
[01] Load & Profile ──────────────────────────────► seer_df
        │
        ▼
[02–06] Data Quality Audit ───────────────────────► missingness_pct, quality_df
        │                                            figs: fig_miss, fig_dashboard
        ▼
[07] Feature Engineering & Risk Score ────────────► seer_engineered (44–65 cols)
        │
        ├──► [08] Stage Survival Metrics ──────────► stage_metrics_df
        │
        ├──► [09] Demographic Heatmap ─────────────► demo_heatmap_df
        │
        ├──► [10] Treatment Validation ────────────► treatment_validation_df
        │
        └──► [11] Treatment Cohort Metrics ────────► treatment_cohort_metrics_df
                    │
                    ├──► [12] Stage Proxy & Heatmaps ──► stage_proxy_metrics_df
                    │                                     age_histology_stratification_df
                    │
                    ├──► [13] Cohort Evolution ────────► cohort_evolution_df
                    │
                    ├──► [14] Risk Score Chart ────────► fig_risk_stratification.html
                    │
                    ├──► [15] Era Comparison Chart ────► fig_era_comparison.html
                    │
                    └──► [16] Follow-Up Censoring ─────► follow_up_completeness_df
                                                          era_bias_df
                                                          right_censoring_metadata
                                                                │
                                                                ▼
                                                     app/main.py — Streamlit Wizard
```

---

## Clinical Context

This project is grounded in published clinical trial evidence. Key references:

| Trial | Finding | SEER Proxy |
|---|---|---|
| **CROSS Trial** (van Hagen et al., *NEJM* 2012) | Trimodal CRT+Surgery → ~50% pCR in ESCC, >90% R0 margins | Surgery+Radiation cohort (n=2,618); SCC subgroup shows +5pp advantage consistent with ESCC radiosensitivity |
| **CheckMate 577** (Kelly et al., *NEJM* 2021) | Adjuvant nivolumab → doubled DFS (22.4 mo vs 11 mo) post-resection | Era 2018–2023 comparison (confounded; SEER cannot track adjuvant immunotherapy) |
| **FLOT4 Trial** (Al-Batran et al., *Lancet* 2019) | Perioperative FLOT → median OS 50 months in EAC | Surgery cohort 5-year rates (selection bias caveat applies) |
| **MDT Benefit** (Barabrah et al., 2026) | Multidisciplinary care → 27% mortality reduction even in advanced disease | Treatment intensity gradient across Cohorts A–D |

> **Note:** SEER does not track chemotherapy regimens, adjuvant immunotherapy, pCR status, or surgical margins (R0/R1/R2). Clinical trial benchmarks are referenced for context only — SEER population-level data cannot replicate RCT efficacy estimates.

---

## Disclaimer

> This project uses **SEER population registry data** for research and educational purposes only.
>
> - Results are **population-level statistics**, not individual predictions.
> - This tool is **not a medical diagnosis**, prognosis calculator, or treatment recommendation.
> - Survival statistics describe historical outcomes for groups of patients — they cannot predict what will happen to any individual.
> - Always consult a qualified oncologist and multidisciplinary care team for diagnosis and treatment decisions.
>
> **Data source:** National Cancer Institute, SEER Program, [seer.cancer.gov](https://seer.cancer.gov)
> **Built with:** Python · Pandas · NumPy · Plotly · Matplotlib · Streamlit · Zerve
