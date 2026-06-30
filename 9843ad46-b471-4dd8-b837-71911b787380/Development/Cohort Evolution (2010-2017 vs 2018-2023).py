
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
ERA_COLORS = {"2010-2017 (Older)": "#FFB482", "2018-2023 (Modern)": "#A1C9F4"}

df = seer_engineered.copy()

# Re-derive tx_cohort
def _assign_cohort(row):
    sx  = row["surgery_status"]; rad = row["radiation_status"]
    has_s = (sx == "Yes"); has_r = (rad == "Yes")
    unk_s = (sx == "Unknown"); no_s = (sx == "No")
    no_r  = rad in ("No", "Unknown", "Refused")
    if unk_s:               return "E"
    if has_s and has_r:     return "A"
    if has_s and not has_r: return "B"
    if has_r and no_s:      return "C"
    if no_s  and not has_r: return "D"
    return "E"

df["tx_cohort"] = df.apply(_assign_cohort, axis=1)

COHORT_SHORT = {"A":"Multimodal (Sx+Rad)","B":"Surgery Only",
                "C":"Radiation Only","D":"No Treatment","E":"Tx Unknown"}
ERAS = {"2010-2017 (Older)": df["year_dx"] < 2018,
        "2018-2023 (Modern)": df["year_dx"] >= 2018}

# ════════════════════════════════════════════════════════════════════════════
# COHORT EVOLUTION — ALL TREATMENT GROUPS (full dataset + Stage III proxy)
# ⚠️ Modern cohort (2018-2023) has shorter follow-up — % Alive partially
#    reflects truncation, not only treatment improvement. Interpret carefully.
# ════════════════════════════════════════════════════════════════════════════

def _era_metrics(sub):
    n = len(sub)
    if n < 5:
        return dict(N=n, pct_alive=np.nan, median_surv=np.nan,
                    surv_1yr=np.nan, surv_3yr=np.nan, surv_5yr=np.nan)
    sm = sub["survival_months"].dropna(); va = sub["vital_alive"]
    return {
        "N":          n,
        "pct_alive":  round(100*va.mean(), 1),
        "median_surv":round(sm.median(), 1),
        "surv_1yr":   round(100*(sm>=12).mean(), 1),
        "surv_3yr":   round(100*(sm>=36).mean(), 1),
        "surv_5yr":   round(100*(sm>=60).mean(), 1),
    }

rows_all, rows_s3 = [], []

for cohort in ["A","B","C","D"]:  # exclude E for clarity
    for era_label, era_mask in ERAS.items():
        # All stages
        sub_all = df[era_mask & (df["tx_cohort"] == cohort)]
        met_all = _era_metrics(sub_all)
        rows_all.append({"cohort":cohort,"cohort_label":COHORT_SHORT[cohort],
                         "era":era_label, **met_all})
        # Stage III proxy only
        sub_s3 = df[era_mask & (df["tx_cohort"] == cohort) & (df["stage"]=="III")]
        met_s3 = _era_metrics(sub_s3)
        rows_s3.append({"cohort":cohort,"cohort_label":COHORT_SHORT[cohort],
                        "era":era_label, **met_s3})

cohort_evolution_df = pd.DataFrame(rows_all)
_s3_evo_df          = pd.DataFrame(rows_s3)

# ── PRINT ALL STAGES ─────────────────────────────────────────────────────────
print("=" * 110)
print("COHORT EVOLUTION — ALL TREATMENT GROUPS (All Stages Combined)")
print("CAVEAT: 2018-2023 has shorter follow-up. % Alive includes right-censored survivors.")
print("=" * 110)
hdr = ("  " + "Cohort".ljust(22) + "Era".ljust(24) +
       "N".rjust(6) + "%Alive".rjust(8) + "Median(mo)".rjust(11) +
       "1yr%".rjust(7) + "3yr%".rjust(7) + "5yr%".rjust(7))
print(hdr); print("-"*90)
for _, r in cohort_evolution_df.iterrows():
    pcts = [r['pct_alive'], r['surv_1yr'], r['surv_3yr'], r['surv_5yr']]
    pct_strs = [f"{p:>7.1f}" if not np.isnan(p) else "    n/a" for p in pcts]
    med_str = f"{r['median_surv']:>11.1f}" if not np.isnan(r['median_surv']) else "        n/a"
    print("  " + r['cohort_label'][:21].ljust(22) + r['era'].ljust(24) +
          str(int(r['N'])).rjust(6) + pct_strs[0] + med_str +
          pct_strs[1] + pct_strs[2] + pct_strs[3])

print("\n" + "=" * 110)
print("COHORT EVOLUTION — STAGE III PROXY ONLY (n=13,208)")
print("KEY INSIGHT: Modern era (2018-2023) shows higher % Alive but shorter follow-up window.")
print("CheckMate 577 (nivolumab adjuvant) adopted 2021+; SEER cannot confirm immunotherapy.")
print("=" * 110)
print(hdr); print("-"*90)
for _, r in _s3_evo_df.iterrows():
    pcts = [r['pct_alive'], r['surv_1yr'], r['surv_3yr'], r['surv_5yr']]
    pct_strs = [f"{p:>7.1f}" if not np.isnan(p) else "    n/a" for p in pcts]
    med_str = f"{r['median_surv']:>11.1f}" if not np.isnan(r['median_surv']) else "        n/a"
    print("  " + r['cohort_label'][:21].ljust(22) + r['era'].ljust(24) +
          str(int(r['N'])).rjust(6) + pct_strs[0] + med_str +
          pct_strs[1] + pct_strs[2] + pct_strs[3])

# ── FIGURE: Era × Cohort Comparison (Stage III proxy) ─────────────────────
fig_cohort_evolution, axes = plt.subplots(1, 3, figsize=(15, 6))
fig_cohort_evolution.patch.set_facecolor(BG)

_cohorts = ["A","B","C","D"]
_clabels = ["Multimodal\n(Sx+Rad)","Surgery\nOnly","Radiation\nOnly","No\nTreatment"]
_era_list = ["2010-2017 (Older)","2018-2023 (Modern)"]
_pal = ["#FFB482","#A1C9F4"]

metrics_plot = [
    ("pct_alive",   "% Alive (any time)", _s3_evo_df),
    ("surv_1yr",    "1-Year Survival %",  _s3_evo_df),
    ("surv_3yr",    "3-Year Survival %",  _s3_evo_df),
]

x = np.arange(len(_cohorts)); w = 0.38

for ax_idx, (metric, ylabel, src_df) in enumerate(metrics_plot):
    ax = axes[ax_idx]; ax.set_facecolor(BG)
    for ei, era in enumerate(_era_list):
        sub = src_df[src_df["era"]==era].set_index("cohort")
        vals = [sub.loc[c,metric] if c in sub.index else np.nan for c in _cohorts]
        offset = (ei - 0.5) * w
        bars = ax.bar(x + offset, vals, w, color=_pal[ei],
                      label="Older (2010-17)" if ei==0 else "Modern (2018-23)",
                      alpha=0.9, zorder=3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                        str(round(v,1)), ha="center", va="bottom",
                        color=TXT, fontsize=7.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(_clabels, color=TXT, fontsize=8.5)
    ax.tick_params(colors=TXT, labelsize=9)
    ax.set_ylabel(ylabel, color=TXT, fontsize=9.5)
    ax.set_title(ylabel, color=TXT, fontsize=10, pad=8)
    ax.set_facecolor(BG); ax.grid(axis="y", color=SUB, alpha=0.25, zorder=0)
    for sp in ax.spines.values(): sp.set_edgecolor(SUB)
    ax.yaxis.label.set_color(TXT)
    _yticks = ax.get_yticks()
    ax.set_yticklabels([str(int(t)) for t in _yticks], color=TXT)
    if ax_idx == 0: ax.legend(fontsize=8.5, labelcolor=TXT,
                               facecolor=BG, edgecolor=SUB, loc="upper right")

fig_cohort_evolution.suptitle(
    "Stage III Proxy: Protocol Era Comparison (2010-2017 vs 2018-2023)\n"
    "Caution: Modern era shows shorter follow-up; % Alive comparison is not pure outcome improvement",
    color=TXT, fontsize=11, y=1.01)
fig_cohort_evolution.tight_layout()
plt.close("all")

print("\ncohort_evolution_df: " + str(cohort_evolution_df.shape[0]) + " rows x " +
      str(cohort_evolution_df.shape[1]) + " cols")
print("fig_cohort_evolution: rendered")
