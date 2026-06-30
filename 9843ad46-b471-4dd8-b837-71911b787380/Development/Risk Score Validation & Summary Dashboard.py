
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Data ──────────────────────────────────────────────────────────────────────
risk_cats   = ["Low (0–33)",   "Moderate (34–66)", "High (67–100)"]
n_vals      = [2719,            13628,               1754]
pct_alive   = [58.0,            17.8,                4.4]
surv_1yr    = [84.7,            42.3,                11.8]
surv_5yr    = [36.5,             7.6,                 1.0]
mean_rs     = [27.5,            51.5,                72.0]

bar_colors  = ["#17b26a", "#ffd400", "#f04438"]   # green / yellow / red
BG, TXT, SUB, GRID = "#1D1D20", "#fbfbff", "#909094", "rgba(255,255,255,0.07)"

# ── Hover templates ───────────────────────────────────────────────────────────
def hover(cat, n, alive, yr1, yr5, rs):
    return (
        f"<b>{cat}</b><br>"
        f"N = {n:,}<br>"
        f"Mean Risk Score: {rs}<br>"
        f"━━━━━━━━━━━━━━━━━━━<br>"
        f"% Alive (current): <b>{alive}%</b><br>"
        f"1-Year Survival:   <b>{yr1}%</b><br>"
        f"5-Year Survival:   <b>{yr5}%</b><br>"
        "<extra></extra>"
    )

hover_texts = [hover(*args) for args in zip(risk_cats, n_vals, pct_alive, surv_1yr, surv_5yr, mean_rs)]

# ── Figure ────────────────────────────────────────────────────────────────────
fig_risk_stratification = go.Figure()

x_pos = [0, 1, 2]

# ── Grouped bars: % Alive, 1-yr, 5-yr ────────────────────────────────────────
bar_width  = 0.22
offsets    = [-0.24, 0.0, 0.24]
metrics    = [pct_alive, surv_1yr, surv_5yr]
m_labels   = ["% Alive", "1-Year Survival", "5-Year Survival"]
m_opacity  = [1.0, 0.72, 0.46]
m_patterns = ["", "/", "x"]

for idx, (vals, lbl, opac, pat) in enumerate(zip(metrics, m_labels, m_opacity, m_patterns)):
    fig_risk_stratification.add_trace(go.Bar(
        x=[p + offsets[idx] for p in x_pos],
        y=vals,
        name=lbl,
        width=bar_width,
        marker=dict(
            color=bar_colors,
            opacity=opac,
            pattern_shape=pat,
            pattern_fgcolor="rgba(255,255,255,0.25)",
            line=dict(color="rgba(255,255,255,0.15)", width=0.8),
        ),
        text=[f"{v}%" for v in vals],
        textposition="outside",
        textfont=dict(size=11, color=TXT),
        hovertemplate=hover_texts,
        showlegend=True,
    ))

# ── Risk score mean line (secondary axis) ────────────────────────────────────
fig_risk_stratification.add_trace(go.Scatter(
    x=x_pos,
    y=mean_rs,
    name="Mean Risk Score",
    yaxis="y2",
    mode="lines+markers+text",
    line=dict(color="#D0BBFF", width=2.5, dash="dot"),
    marker=dict(size=10, color="#D0BBFF", symbol="diamond",
                line=dict(color=TXT, width=1.5)),
    text=[f"RS {v}" for v in mean_rs],
    textposition=["top center", "top center", "bottom center"],
    textfont=dict(size=10, color="#D0BBFF"),
    hovertemplate="<b>Mean Risk Score</b><br>%{y:.1f}<extra></extra>",
))

# ── N annotations inside bars ────────────────────────────────────────────────
for i, (xi, n, col) in enumerate(zip(x_pos, n_vals, bar_colors)):
    fig_risk_stratification.add_annotation(
        x=xi, y=2,
        text=f"n={n:,}",
        showarrow=False,
        font=dict(size=10, color="rgba(255,255,255,0.75)"),
        xanchor="center", yanchor="bottom",
    )

# ── Insight annotation ────────────────────────────────────────────────────────
fig_risk_stratification.add_annotation(
    x=0.5, y=1.09,
    xref="paper", yref="paper",
    text=(
        "📊 <b>73-point gap</b> in 1-year survival between Low and High risk. "
        "Risk score stratifies meaningfully. (Δ1yr = 84.7% → 11.8%)"
    ),
    showarrow=False,
    font=dict(size=12, color="#A1C9F4"),
    bgcolor="rgba(161,201,244,0.10)",
    bordercolor="#A1C9F4",
    borderwidth=1,
    borderpad=6,
    xanchor="center",
)

# ── SEER limitation footnote ─────────────────────────────────────────────────
flags_text = (
    "⚠️ Risk score uses proxy stage (73% Stage III coding artifact) & treatment flags (not regimen/dose).  "
    "⚠️ Radiation=0 includes 'None/Unknown' — ambiguous.  "
    "⚠️ No performance status, comorbidities, or margins. Population-level stratification only."
)
fig_risk_stratification.add_annotation(
    x=0.5, y=-0.18,
    xref="paper", yref="paper",
    text=flags_text,
    showarrow=False,
    font=dict(size=9.5, color=SUB),
    xanchor="center",
    align="center",
)

# ── Layout ────────────────────────────────────────────────────────────────────
fig_risk_stratification.update_layout(
    title=dict(
        text="<b>Composite Risk Score Stratification</b>"
             "<br><sup>SEER Esophageal Cancer · N=18,101 · Risk Score 0–100 (Stage 35% · Treatment 30% · Age 20% · Year 10%)</sup>",
        font=dict(size=18, color=TXT),
        x=0.5, xanchor="center", y=0.97,
    ),
    barmode="group",
    bargap=0.28,
    plot_bgcolor=BG,
    paper_bgcolor=BG,
    font=dict(family="Inter, Arial, sans-serif", color=TXT),
    height=560,
    margin=dict(t=120, b=100, l=70, r=90),

    xaxis=dict(
        tickvals=x_pos,
        ticktext=[
            f"<b>🟢 Low</b><br>Score 0–33",
            f"<b>🟡 Moderate</b><br>Score 34–66",
            f"<b>🔴 High</b><br>Score 67–100",
        ],
        tickfont=dict(size=12),
        showgrid=False,
        zeroline=False,
    ),
    yaxis=dict(
        title="Survival Rate (%)",
        range=[0, 105],
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        ticksuffix="%",
        title_font=dict(size=12),
    ),
    yaxis2=dict(
        title="Mean Risk Score",
        overlaying="y",
        side="right",
        range=[0, 105],
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#D0BBFF"),
        title_font=dict(size=12, color="#D0BBFF"),
    ),
    legend=dict(
        bgcolor="rgba(255,255,255,0.06)",
        bordercolor="rgba(255,255,255,0.12)",
        borderwidth=1,
        font=dict(size=11),
        x=0.01, y=0.98,
        orientation="v",
    ),
)

fig_risk_stratification.write_html("fig_risk_stratification.html")
print("✅  fig_risk_stratification.html saved.")
print()
print("CHART DATA SUMMARY")
print("─" * 72)
print(f"{'Category':<22} {'N':>7}  {'%Alive':>7}  {'1yr%':>6}  {'5yr%':>6}  {'MeanRS':>7}")
print("─" * 72)
for c, n, a, y1, y5, rs in zip(risk_cats, n_vals, pct_alive, surv_1yr, surv_5yr, mean_rs):
    print(f"{c:<22} {n:>7,}  {a:>6.1f}%  {y1:>5.1f}%  {y5:>5.1f}%  {rs:>7.1f}")
print("─" * 72)
print()
print(f"  Δ 1-year survival  Low → High : {surv_1yr[0]:.1f}% → {surv_1yr[2]:.1f}%  (delta = {surv_1yr[0]-surv_1yr[2]:.1f}pp)")
print(f"  Δ 5-year survival  Low → High : {surv_5yr[0]:.1f}% → {surv_5yr[2]:.1f}%  (delta = {surv_5yr[0]-surv_5yr[2]:.1f}pp)")
print(f"  Δ % Alive          Low → High : {pct_alive[0]:.1f}% → {pct_alive[2]:.1f}%  (delta = {pct_alive[0]-pct_alive[2]:.1f}pp)")
