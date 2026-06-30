
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Palette & constants ────────────────────────────────────────────────────────
BG        = "#1D1D20"
TXT       = "#fbfbff"
SUB       = "#909094"
GRID      = "rgba(255,255,255,0.07)"
ERA_OLD   = "#A1C9F4"   # light blue  → 2010-2017
ERA_NEW   = "#8DE5A1"   # green       → 2018-2023
DELTA_CLR = "#ffd400"   # gold        → Δ labels

# ── Pull data directly from cohort_evolution_df ───────────────────────────────
# columns: cohort, cohort_label, era, N, pct_alive, median_surv, surv_1yr, surv_3yr, surv_5yr

COHORT_ORDER = ["A", "B", "C", "D"]
COHORT_DISPLAY = {
    "A": "S+R\nMultimodal",
    "B": "Surgery\nOnly",
    "C": "Radiation\nOnly",
    "D": "No\nTreatment",
}
COHORT_FULL = {
    "A": "Surgery + Radiation",
    "B": "Surgery Only",
    "C": "Radiation Only",
    "D": "No Treatment",
}

old_rows = {
    r["cohort"]: r
    for _, r in cohort_evolution_df[cohort_evolution_df["era"].str.contains("Older")].iterrows()
}
new_rows = {
    r["cohort"]: r
    for _, r in cohort_evolution_df[cohort_evolution_df["era"].str.contains("Modern")].iterrows()
}

old_pct  = [round(old_rows[c]["pct_alive"],  1) for c in COHORT_ORDER]
new_pct  = [round(new_rows[c]["pct_alive"],  1) for c in COHORT_ORDER]
old_1yr  = [round(old_rows[c]["surv_1yr"],   1) for c in COHORT_ORDER]
new_1yr  = [round(new_rows[c]["surv_1yr"],   1) for c in COHORT_ORDER]
old_3yr  = [round(old_rows[c]["surv_3yr"],   1) for c in COHORT_ORDER]
new_3yr  = [round(new_rows[c]["surv_3yr"],   1) for c in COHORT_ORDER]
old_med  = [int(old_rows[c]["median_surv"])  for c in COHORT_ORDER]
new_med  = [int(new_rows[c]["median_surv"])  for c in COHORT_ORDER]
old_n    = [int(old_rows[c]["N"])            for c in COHORT_ORDER]
new_n    = [int(new_rows[c]["N"])            for c in COHORT_ORDER]

delta_pct = [round(n - o, 1) for o, n in zip(old_pct, new_pct)]
delta_1yr = [round(n - o, 1) for o, n in zip(old_1yr, new_1yr)]
delta_3yr = [round(n - o, 1) for o, n in zip(old_3yr, new_3yr)]

x_pos  = list(range(len(COHORT_ORDER)))
bar_w  = 0.34
x_old  = [x - bar_w / 2 for x in x_pos]
x_new  = [x + bar_w / 2 for x in x_pos]

# ── Hover text builders ───────────────────────────────────────────────────────
def make_hover(era_label, ns, pcts, yr1, yr3, meds):
    return [
        f"<b>{era_label}</b><br>"
        f"N = {n:,}<br>"
        f"% Alive: <b>{p}%</b><br>"
        f"1-Yr Survival: {y1}%<br>"
        f"3-Yr Survival: {y3}%<br>"
        f"Median Survival: {m} mo"
        for n, p, y1, y3, m in zip(ns, pcts, yr1, yr3, meds)
    ]

hover_old = make_hover("2010–2017 (Older Protocols)", old_n, old_pct, old_1yr, old_3yr, old_med)
hover_new = make_hover("2018–2023 (Modern Protocols)", new_n, new_pct, new_1yr, new_3yr, new_med)

# ── Figure: 2 rows ─────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.68, 0.32],
    vertical_spacing=0.13,
    subplot_titles=[
        "% Alive by Treatment Cohort & Protocol Era",
        "Δ Change (Modern − Historical, percentage points)",
    ],
)

# ── Row 1: grouped bars ────────────────────────────────────────────────────────
fig.add_trace(
    go.Bar(
        x=x_old, y=old_pct,
        width=bar_w,
        name="2010–2017 (Historical)",
        marker=dict(color=ERA_OLD, opacity=0.85, line=dict(color=ERA_OLD, width=0.8)),
        text=[f"{v}%" for v in old_pct],
        textposition="outside",
        textfont=dict(color=ERA_OLD, size=12, family="Arial"),
        hovertext=hover_old, hoverinfo="text",
    ),
    row=1, col=1,
)
fig.add_trace(
    go.Bar(
        x=x_new, y=new_pct,
        width=bar_w,
        name="2018–2023 (Modern)",
        marker=dict(color=ERA_NEW, opacity=0.90, line=dict(color=ERA_NEW, width=0.8)),
        text=[f"{v}%" for v in new_pct],
        textposition="outside",
        textfont=dict(color=ERA_NEW, size=12, family="Arial"),
        hovertext=hover_new, hoverinfo="text",
    ),
    row=1, col=1,
)

# N= sub-labels (below bars)
for xi, no, nn in zip(x_old, old_n, new_n):
    fig.add_annotation(
        x=xi, y=2, text=f"n={no:,}",
        showarrow=False, font=dict(size=8, color=SUB),
        xanchor="center", yanchor="bottom", row=1, col=1,
    )
for xi, nn in zip(x_new, new_n):
    fig.add_annotation(
        x=xi, y=2, text=f"n={nn:,}",
        showarrow=False, font=dict(size=8, color=SUB),
        xanchor="center", yanchor="bottom", row=1, col=1,
    )

# ── Row 2: delta bars ──────────────────────────────────────────────────────────
delta_colors = [DELTA_CLR if d >= 0 else "#f04438" for d in delta_pct]
delta_hover  = [
    f"<b>Δ % Alive: +{d}pp</b><br>Δ 1-Yr: +{d1}pp<br>Δ 3-Yr: {d3:+.1f}pp"
    for d, d1, d3 in zip(delta_pct, delta_1yr, delta_3yr)
]

fig.add_trace(
    go.Bar(
        x=x_pos, y=delta_pct,
        width=0.48,
        name="Δ % Alive (pp)",
        marker=dict(color=delta_colors, opacity=0.85),
        text=[f"+{d}pp" for d in delta_pct],
        textposition="outside",
        textfont=dict(color=DELTA_CLR, size=11, family="Arial"),
        hovertext=delta_hover, hoverinfo="text",
    ),
    row=2, col=1,
)
fig.add_hline(y=0, line=dict(color=SUB, width=1, dash="dot"), row=2, col=1)

# ── ⚠️ Critical caveat annotation ─────────────────────────────────────────────
caveat = (
    "⚠️  CRITICAL CAVEAT: Modern cohort (2018–23) has shorter follow-up than historical (2010–17).  "
    "Higher % Alive reflects <b>incomplete follow-up, NOT pure treatment improvement</b>.  "
    "Median survival DROPPED for Multimodal: 44 mo → 26 mo.  Use 3–5 yr rates for true comparison."
)
fig.add_annotation(
    x=0.5, y=1.015,
    xref="paper", yref="paper",
    text=caveat,
    showarrow=False,
    bgcolor="rgba(240,68,56,0.13)",
    bordercolor="#f04438",
    borderwidth=1.5,
    borderpad=9,
    font=dict(color="#FF9F9B", size=10.5, family="Arial"),
    align="center",
    xanchor="center",
    yanchor="bottom",
)

# ── Layout ─────────────────────────────────────────────────────────────────────
tick_labels = [COHORT_DISPLAY[c].replace("\n", "<br>") for c in COHORT_ORDER]

fig.update_layout(
    title=dict(
        text=(
            "Protocol Era Comparison: 2010–2017 vs 2018–2023<br>"
            "<sup>Esophageal Cancer — SEER Registry (N=18,101) | All Treatment Cohorts | All Stages Combined</sup>"
        ),
        font=dict(size=16, color=TXT),
        x=0.5, xanchor="center", y=0.97,
    ),
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=TXT, family="Arial"),
    legend=dict(
        orientation="h",
        x=0.5, xanchor="center", y=-0.05,
        font=dict(size=12),
        bgcolor="rgba(0,0,0,0)",
    ),
    barmode="group",
    bargap=0.25,
    bargroupgap=0.04,
    height=680,
    margin=dict(t=155, b=90, l=65, r=55),
)

# Row 1 axes
fig.update_xaxes(
    tickvals=x_pos, ticktext=tick_labels,
    tickfont=dict(size=12, color=TXT),
    showgrid=False, zeroline=False, row=1, col=1,
)
fig.update_yaxes(
    title=dict(text="% Alive", font=dict(size=12, color=SUB)),
    range=[-8, 92],
    ticksuffix="%",
    tickfont=dict(size=11, color=SUB),
    gridcolor=GRID, zeroline=False, row=1, col=1,
)

# Row 2 axes
fig.update_xaxes(
    tickvals=x_pos, ticktext=tick_labels,
    tickfont=dict(size=12, color=TXT),
    showgrid=False, zeroline=False, row=2, col=1,
)
fig.update_yaxes(
    title=dict(text="Δ pp", font=dict(size=11, color=SUB)),
    range=[-4, 46],
    tickfont=dict(size=10, color=SUB),
    gridcolor=GRID, zeroline=False, row=2, col=1,
)

# Subplot title colors
for ann in fig.layout.annotations:
    if ann.text in [
        "% Alive by Treatment Cohort & Protocol Era",
        "Δ Change (Modern − Historical, percentage points)",
    ]:
        ann.font.update(size=13, color=SUB)

# Footer
fig.add_annotation(
    x=0.5, y=-0.12,
    xref="paper", yref="paper",
    text=(
        "⚠️ Surgery Blank: 1,747 post-2023 records excluded  │  "
        "⚠️ Radiation=0 includes 'None/Unknown' (~6k ambiguous records)  │  "
        "⚠️ Stage III = 73% is proxy artifact  │  "
        "✅ Survival months: 0% missing"
    ),
    showarrow=False,
    font=dict(size=9, color=SUB),
    align="center", xanchor="center",
)

# ── Save & display ─────────────────────────────────────────────────────────────
fig.write_html("fig_era_comparison.html", include_plotlyjs="cdn")
print("✅  fig_era_comparison.html saved.")

print("\nERA COMPARISON DATA SUMMARY")
print("─" * 78)
print(f"{'Cohort':<26} {'Old N':>7} {'Old %':>7} {'New N':>7} {'New %':>7} {'Δ pp':>7}")
print("─" * 78)
for i, c in enumerate(COHORT_ORDER):
    print(f"{COHORT_FULL[c]:<26} {old_n[i]:>7,} {old_pct[i]:>6.1f}% {new_n[i]:>7,} {new_pct[i]:>6.1f}% {delta_pct[i]:>+6.1f}")
print("─" * 78)

print("\nINTERPRETATION FLAGS:")
print("  ⚠️  % Alive is confounded by follow-up length — not a pure treatment improvement")
print("  ⚠️  Median survival DROPPED for Multimodal: 44 mo (2010-17) → 26 mo (2018-23)")
print("  ⚠️  3-yr and 5-yr survival rates are the reliable era comparison metrics")
print("  ✅  Delta bars show apparent Δ; red caveat box explains the follow-up bias")

fig_era_comparison = fig
