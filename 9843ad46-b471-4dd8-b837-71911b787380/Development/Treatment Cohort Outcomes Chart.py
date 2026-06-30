
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Data ─────────────────────────────────────────────────────────────────────
cohorts_short = ["S+R (Multimodal)", "Surgery Only", "Radiation Only", "No Treatment", "Unknown ⚠️"]
n_vals       = [2618,  1519,  6509,  5708,  1747]
pct_alive    = [40.6,  55.6,  12.8,   6.4,  55.1]
median_surv  = [31,    49,    11,     4,     4   ]
surv_1yr     = [84.7,  90.1,  49.1,  24.4,   2.0]
surv_3yr     = [45.9,  62.1,  15.4,   5.4,   0.9]
surv_5yr     = [28.3,  41.9,   7.5,   2.7,   0.6]

bar_colors = ["#8DE5A1", "#A1C9F4", "#FFB482", "#FF9F9B", "#909094"]

BG   = "#1D1D20"
TXT  = "#fbfbff"
SUB  = "#909094"
GRID = "#2E2E34"

# ── Hover text ───────────────────────────────────────────────────────────────
hover_bars = []
hover_line = []
for i in range(5):
    extra = ""
    if i == 4:
        extra = "<br><b>⚠️ Right-censored:</b> 55% alive with 4-mo median<br>= incomplete follow-up, not a survival benefit"
    hover_bars.append(
        f"<b>{cohorts_short[i]}</b><br>"
        f"N = {n_vals[i]:,}<br>"
        f"% Alive: <b>{pct_alive[i]}%</b><br>"
        f"Median Survival: {median_surv[i]} mo<br>"
        f"1-yr: {surv_1yr[i]}%  |  3-yr: {surv_3yr[i]}%  |  5-yr: {surv_5yr[i]}%"
        + extra
    )
    hover_line.append(
        f"<b>{cohorts_short[i]}</b><br>"
        f"Median Survival: <b>{median_surv[i]} mo</b><br>"
        f"N = {n_vals[i]:,}"
        + extra
    )

# ── Figure ───────────────────────────────────────────────────────────────────
fig_treatment_outcomes = go.Figure()

# Bars — % Alive (left y-axis)
fig_treatment_outcomes.add_trace(go.Bar(
    x=cohorts_short,
    y=pct_alive,
    name="% Alive",
    marker=dict(
        color=bar_colors,
        line=dict(color=bar_colors, width=1.5),
        opacity=0.88,
    ),
    yaxis="y1",
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_bars,
    text=[f"n={n:,}" for n in n_vals],
    textposition="inside",
    textfont=dict(color=BG, size=11, family="Arial Black"),
))

# % Alive value labels above bars
fig_treatment_outcomes.add_trace(go.Scatter(
    x=cohorts_short,
    y=[v + 2.8 for v in pct_alive],
    mode="text",
    text=[f"<b>{v}%</b>" for v in pct_alive],
    textfont=dict(color=TXT, size=13),
    yaxis="y1",
    hoverinfo="skip",
    showlegend=False,
))

# Line — Median Survival (right y-axis)
fig_treatment_outcomes.add_trace(go.Scatter(
    x=cohorts_short,
    y=median_surv,
    name="Median Survival (mo)",
    mode="lines+markers+text",
    line=dict(color="#ffd400", width=2.8),
    marker=dict(symbol="circle", size=12, color="#ffd400",
                line=dict(color=BG, width=2)),
    text=[f"<b>{v} mo</b>" for v in median_surv],
    textposition=["top center", "top center", "top center", "bottom center", "bottom center"],
    textfont=dict(color="#ffd400", size=12),
    yaxis="y2",
    hovertemplate="%{customdata}<extra></extra>",
    customdata=hover_line,
))

# ── Annotation: Unknown cohort callout ───────────────────────────────────────
fig_treatment_outcomes.add_annotation(
    x="Unknown ⚠️",
    y=62,
    yref="y1",
    text=(
        "⚠️ Right-censored<br>"
        "recent records (2022–23)<br>"
        "55% alive, 4-mo median<br>"
        "= incomplete follow-up,<br>"
        "not survival benefit"
    ),
    showarrow=True,
    arrowhead=2,
    arrowcolor=SUB,
    arrowwidth=1.5,
    ax=-120,
    ay=-55,
    font=dict(size=10.5, color=SUB),
    bgcolor="#2A2A2E",
    bordercolor=SUB,
    borderwidth=1,
    borderpad=5,
    align="left",
)

# ── Layout ───────────────────────────────────────────────────────────────────
fig_treatment_outcomes.update_layout(
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    height=570,
    margin=dict(t=110, b=110, l=75, r=95),
    title=dict(
        text=(
            "<b>Esophageal Cancer — Outcomes by Treatment Intensity</b><br>"
            "<span style='font-size:12px;color:#909094'>"
            "SEER 2010–2023 · N=18,101 · "
            "⚠️ Stage proxy-derived · Radiation 'None/Unknown' ~44% ambiguous"
            "</span>"
        ),
        font=dict(size=17, color=TXT),
        x=0.5, xanchor="center",
        y=0.97,
    ),
    legend=dict(
        orientation="h",
        x=0.5, xanchor="center",
        y=-0.18,
        font=dict(color=TXT, size=12),
        bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(
        title=dict(text="Treatment Cohort", font=dict(color=SUB, size=12)),
        tickfont=dict(color=TXT, size=12),
        gridcolor=GRID,
        linecolor=GRID,
    ),
    yaxis=dict(
        title=dict(text="% Alive (current status)", font=dict(color=TXT, size=12)),
        tickfont=dict(color=TXT, size=11),
        range=[0, 80],
        gridcolor=GRID,
        gridwidth=0.5,
        zeroline=False,
        ticksuffix="%",
    ),
    yaxis2=dict(
        title=dict(text="Median Survival (months)", font=dict(color="#ffd400", size=12)),
        tickfont=dict(color="#ffd400", size=11),
        overlaying="y",
        side="right",
        range=[0, 72],
        showgrid=False,
        zeroline=False,
        ticksuffix=" mo",
    ),
    hoverlabel=dict(
        bgcolor="#2A2A2E",
        font_size=12,
        font_color=TXT,
        bordercolor=SUB,
    ),
)

# ── Data-quality footer annotation ───────────────────────────────────────────
flags_text = (
    "✅ Survival Months: 0% missing — reliable  |  "
    "⚠️ Radiation: ~44% None/Unknown — not purely 'no treatment'  |  "
    "⚠️ Surgery: 1,747 Blank records (post-2022) → Unknown cohort  |  "
    "SEER: No pCR, R0 margins, chemo regimen, or adjuvant therapy"
)
fig_treatment_outcomes.add_annotation(
    xref="paper", yref="paper",
    x=0.0, y=-0.34,
    text=flags_text,
    showarrow=False,
    font=dict(size=9, color=SUB),
    align="left",
    xanchor="left",
)

# ── Save HTML ─────────────────────────────────────────────────────────────────
fig_treatment_outcomes.write_html("fig_treatment_outcomes.html", include_plotlyjs="cdn")
print("✅  fig_treatment_outcomes.html saved.")
print()
print("CHART DATA SUMMARY")
print("─" * 76)
print(f"{'Cohort':<26} {'N':>6}  {'%Alive':>7}  {'Med(mo)':>8}  {'1yr%':>6}  {'3yr%':>6}  {'5yr%':>6}")
print("─" * 76)
for i in range(5):
    print(
        f"{cohorts_short[i]:<26} {n_vals[i]:>6,}  "
        f"{pct_alive[i]:>7.1f}%  {median_surv[i]:>7}mo  "
        f"{surv_1yr[i]:>5.1f}%  {surv_3yr[i]:>5.1f}%  {surv_5yr[i]:>5.1f}%"
    )
print("─" * 76)
print()
print("⚠️  Unknown (E): 55% alive with 4-mo median = right-censored 2022-23")
print("    records with incomplete follow-up — NOT a survival benefit signal.")
print("⚠️  Surgery Only raw % Alive > Multimodal = selection bias.")
print("    Multimodal patients had more advanced disease at baseline.")
