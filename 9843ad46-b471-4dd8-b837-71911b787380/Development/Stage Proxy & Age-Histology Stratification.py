
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

BG = "#1D1D20"; TXT = "#fbfbff"; SUB = "#909094"
PAL = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#1F77B4","#9467BD"]

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
COHORT_SHORT = {"A":"Multimodal","B":"Surgery Only","C":"Radiation Only",
                "D":"No Treatment","E":"Tx Unknown"}
COHORT_ORDER = ["A","B","C","D","E"]
STAGE_ORDER  = ["I","II","III","Unknown"]

# ── STAGE PROXY METRICS TABLE ────────────────────────────────────────────────
def _stage_metrics(sub):
    n = len(sub); sm = sub["survival_months"].dropna(); va = sub["vital_alive"]
    sx = sub["surgery_status"]; rad = sub["radiation_status"]
    return {
        "N":               n,
        "pct_alive":       round(100*va.mean(), 1),
        "pct_dead":        round(100*(1-va.mean()), 1),
        "median_surv_mo":  round(sm.median(), 1),
        "mean_surv_mo":    round(sm.mean(), 1),
        "surv_1yr":        round(100*(sm>=12).mean(), 1),
        "surv_3yr":        round(100*(sm>=36).mean(), 1),
        "surv_5yr":        round(100*(sm>=60).mean(), 1),
        "mean_age":        round(sub["age_mid"].mean(), 1),
        "pct_female":      round(100*(sub["Sex"]=="Female").mean(), 1),
        "pct_white":       round(100*(sub["race_group"]=="White").mean(), 1),
        "pct_black":       round(100*(sub["race_group"]=="Black").mean(), 1),
        "pct_had_surgery": round(100*(sx=="Yes").mean(), 1),
        "pct_had_rad":     round(100*(rad=="Yes").mean(), 1),
        "pct_both":        round(100*(df.loc[sub.index,"tx_cohort"]=="A").mean(), 1),
        "pct_no_tx":       round(100*(df.loc[sub.index,"tx_cohort"]=="D").mean(), 1),
    }

stage_rows = []
for st in STAGE_ORDER:
    sub = df[df["stage"] == st]
    lbl = ("Stage " + st + " (proxy)") if st != "Unknown" else "Stage Unknown"
    row = {"stage": lbl}
    row.update(_stage_metrics(sub))
    stage_rows.append(row)

stage_proxy_metrics_df = pd.DataFrame(stage_rows).set_index("stage")

print("=" * 100)
print("STAGE PROXY METRICS")
print("CAVEAT: Stage is proxy-derived, not AJCC. Stage III = 73% (coding artifact).")
print("=" * 100)
hdr = ("  " + "Stage".ljust(26) + "N".rjust(6) + "%Alive".rjust(8) +
       "Med(mo)".rjust(9) + "Mean(mo)".rjust(9) +
       "1yr%".rjust(7) + "3yr%".rjust(7) + "5yr%".rjust(7) +
       "%Sx".rjust(7) + "%Rad".rjust(7) + "%Both".rjust(8) + "%NoTx".rjust(8))
print(hdr); print("-" * 110)
for idx, r in stage_proxy_metrics_df.iterrows():
    print("  " + idx.ljust(26) + str(int(r["N"])).rjust(6) +
          f"{r['pct_alive']:>8.1f}" + f"{r['median_surv_mo']:>9.1f}" +
          f"{r['mean_surv_mo']:>9.1f}" + f"{r['surv_1yr']:>7.1f}" +
          f"{r['surv_3yr']:>7.1f}" + f"{r['surv_5yr']:>7.1f}" +
          f"{r['pct_had_surgery']:>7.1f}" + f"{r['pct_had_rad']:>7.1f}" +
          f"{r['pct_both']:>8.1f}" + f"{r['pct_no_tx']:>8.1f}")

# ── AGE × TREATMENT HEATMAP DATA (Stage III proxy) ──────────────────────────
s3 = df[df["stage"] == "III"].copy()
AGE_GROUPS     = ["<60","60-69","70-79","80+"]
HIST_GROUPS    = ["Adenocarcinoma","Squamous Cell Carcinoma","Other/Unknown"]
COHORT_DISPLAY = ["A","B","C","D"]

age_heat_data, age_heat_n = [], []
for ag in AGE_GROUPS:
    row_vals, row_n = [], []
    for c in COHORT_DISPLAY:
        sub = s3[(s3["age_group"]==ag) & (s3["tx_cohort"]==c)]
        row_vals.append(round(100*sub["vital_alive"].mean(),1) if len(sub)>=5 else np.nan)
        row_n.append(len(sub))
    age_heat_data.append(row_vals); age_heat_n.append(row_n)
age_heat_arr = np.array(age_heat_data, dtype=float)

hist_heat_data, hist_heat_n = [], []
for hg in HIST_GROUPS:
    row_vals, row_n = [], []
    for c in COHORT_DISPLAY:
        sub = s3[(s3["histology_group"]==hg) & (s3["tx_cohort"]==c)]
        row_vals.append(round(100*sub["vital_alive"].mean(),1) if len(sub)>=5 else np.nan)
        row_n.append(len(sub))
    hist_heat_data.append(row_vals); hist_heat_n.append(row_n)
hist_heat_arr = np.array(hist_heat_data, dtype=float)

# Underlying DataFrame
_strat_rows = []
for i, ag in enumerate(AGE_GROUPS):
    for j, c in enumerate(COHORT_DISPLAY):
        _strat_rows.append({"factor_type":"Age Group","factor_value":ag,
                             "tx_cohort":COHORT_SHORT[c],
                             "pct_alive":age_heat_data[i][j],"N":age_heat_n[i][j]})
for i, hg in enumerate(HIST_GROUPS):
    for j, c in enumerate(COHORT_DISPLAY):
        _strat_rows.append({"factor_type":"Histology","factor_value":hg,
                             "tx_cohort":COHORT_SHORT[c],
                             "pct_alive":hist_heat_data[i][j],"N":hist_heat_n[i][j]})
age_histology_stratification_df = pd.DataFrame(_strat_rows)
print("\nage_histology_stratification_df: " + str(age_histology_stratification_df.shape))

# Print age table
print("\n-- Stage III: % Alive by Age Group x Treatment (Stage III proxy) --")
print("  Age Group      Multimodal     Surgery Only   Radiation Only  No Treatment")
print("  " + "-"*80)
for i, ag in enumerate(AGE_GROUPS):
    row_str = "  " + ag.ljust(14)
    for j in range(4):
        v = age_heat_data[i][j]; n = age_heat_n[i][j]
        cell = (str(round(v,1)) + "% n=" + str(n)) if not np.isnan(v) else "n/a"
        row_str += cell.rjust(15)
    print(row_str)

# Print histology table
print("\n-- Stage III: % Alive by Histology x Treatment --")
print("  Histology                    Multimodal     Surgery Only   Radiation Only  No Treatment")
print("  " + "-"*95)
for i, hg in enumerate(HIST_GROUPS):
    row_str = "  " + hg[:27].ljust(28)
    for j in range(4):
        v = hist_heat_data[i][j]; n = hist_heat_n[i][j]
        cell = (str(round(v,1)) + "% n=" + str(n)) if not np.isnan(v) else "n/a"
        row_str += cell.rjust(15)
    print(row_str)

# ── HEATMAP 1: Age x Treatment ───────────────────────────────────────────────
_cmap = LinearSegmentedColormap.from_list("surv", ["#f04438","#ffd400","#17b26a"])
_col_lbl = ["Multimodal\n(Sx+Rad)","Surgery\nOnly","Radiation\nOnly","No\nTreatment"]

fig_age_heatmap, ax = plt.subplots(figsize=(9, 5))
fig_age_heatmap.patch.set_facecolor(BG); ax.set_facecolor(BG)
im = ax.imshow(age_heat_arr, aspect="auto", cmap=_cmap, vmin=0, vmax=65)
for i in range(len(AGE_GROUPS)):
    for j in range(len(COHORT_DISPLAY)):
        v = age_heat_arr[i, j]; nc = age_heat_n[i][j]
        if np.isnan(v): continue
        tc = TXT if (v < 32 or v > 52) else BG
        ax.text(j, i, str(round(v)) + "%\nn=" + str(nc),
                ha="center", va="center", fontsize=9.5, fontweight="bold", color=tc)
ax.set_xticks(range(4)); ax.set_xticklabels(_col_lbl, color=TXT, fontsize=10)
ax.set_yticks(range(len(AGE_GROUPS))); ax.set_yticklabels(AGE_GROUPS, color=TXT, fontsize=10)
ax.tick_params(colors=TXT)
for sp in ax.spines.values(): sp.set_edgecolor(SUB)
cb = fig_age_heatmap.colorbar(im, ax=ax, pad=0.02)
cb.ax.yaxis.set_tick_params(color=TXT); cb.outline.set_edgecolor(SUB)
for tl in cb.ax.get_yticklabels(): tl.set_color(TXT)
cb.set_label("% Alive", color=TXT, fontsize=9)
ax.set_title("Stage III Proxy: % Alive by Age Group x Treatment Intensity\n"
             "(n=13,208 — proxy stage only, not AJCC confirmed)",
             color=TXT, fontsize=11, pad=12)
ax.set_ylabel("Age Group", color=TXT, fontsize=10)
ax.set_xlabel("Treatment Cohort", color=TXT, fontsize=10)
fig_age_heatmap.tight_layout()
plt.close("all")

# ── HEATMAP 2: Histology x Treatment ────────────────────────────────────────
_hist_lbl = ["Adenocarcinoma","Squamous Cell\nCarcinoma","Other /\nUnknown"]
fig_histology_heatmap, ax2 = plt.subplots(figsize=(9, 4))
fig_histology_heatmap.patch.set_facecolor(BG); ax2.set_facecolor(BG)
im2 = ax2.imshow(hist_heat_arr, aspect="auto", cmap=_cmap, vmin=0, vmax=65)
for i in range(len(HIST_GROUPS)):
    for j in range(len(COHORT_DISPLAY)):
        v = hist_heat_arr[i, j]; nc = hist_heat_n[i][j]
        if np.isnan(v): continue
        tc = TXT if (v < 32 or v > 52) else BG
        ax2.text(j, i, str(round(v)) + "%\nn=" + str(nc),
                 ha="center", va="center", fontsize=9.5, fontweight="bold", color=tc)
ax2.set_xticks(range(4)); ax2.set_xticklabels(_col_lbl, color=TXT, fontsize=10)
ax2.set_yticks(range(len(HIST_GROUPS))); ax2.set_yticklabels(_hist_lbl, color=TXT, fontsize=10)
ax2.tick_params(colors=TXT)
for sp in ax2.spines.values(): sp.set_edgecolor(SUB)
cb2 = fig_histology_heatmap.colorbar(im2, ax=ax2, pad=0.02)
cb2.ax.yaxis.set_tick_params(color=TXT); cb2.outline.set_edgecolor(SUB)
for tl in cb2.ax.get_yticklabels(): tl.set_color(TXT)
cb2.set_label("% Alive", color=TXT, fontsize=9)
ax2.set_title("Stage III Proxy: % Alive by Histology x Treatment Intensity\n"
              "(n=13,208 — proxy stage only, not AJCC confirmed)",
              color=TXT, fontsize=11, pad=12)
ax2.set_ylabel("Histologic Type", color=TXT, fontsize=10)
ax2.set_xlabel("Treatment Cohort", color=TXT, fontsize=10)
fig_histology_heatmap.tight_layout()
plt.close("all")

print("\nstage_proxy_metrics_df: " + str(stage_proxy_metrics_df.shape[0]) + " stages x " +
      str(stage_proxy_metrics_df.shape[1]) + " metrics")
print("fig_age_heatmap: rendered")
print("fig_histology_heatmap: rendered")
