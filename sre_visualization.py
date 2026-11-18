"""Plotly-Figuren (alle Achsen fest, kein Zoomen: Touch-Scrollen bleibt nicht am Diagramm hängen).
Annotationen auf Kategorieachsen stehen an Positionsindizes (0, 1, ...), nicht an den Beschriftungen."""

import numpy as np
import plotly.graph_objects as go

import sre_constants as C

DECADES = dict(dtick=1, exponentformat="power", showexponent="all")  # Zehnerpotenzen als Achsenmarken auf logarithmischen Achsen
LABELS = {"SS": "anfällig (SS)", "RS": "resistent gegen A (RS)", "SR": "resistent gegen B (SR)", "RR": "resistent gegen beide (RR)"}


def _fixed(fig, height=340):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.25))
    return fig


def fig_season_density(trace_n, spray_weeks, title):
    """Dichte am Wochenende je Genotyp (logarithmische Achse, fest), gespritzte Wochen als senkrechte Linien. trace_n: Form (W, 4)."""
    W = trace_n.shape[0]
    x = list(range(1, W + 1))
    fig = go.Figure()
    total = trace_n.sum(axis=1)
    fig.add_trace(go.Scatter(x=x, y=total, mode="lines+markers", name="alle zusammen", line=dict(color=C.COLORS[7], width=3)))
    for j, (g, col) in enumerate(zip(("SS", "RS", "SR", "RR"), (C.COLORS[0], C.COLORS[1], C.COLORS[2], C.COLORS[3]))):
        if np.any(trace_n[:, j] > 0):
            fig.add_trace(go.Scatter(x=x, y=np.maximum(trace_n[:, j], 1e-12), mode="lines", name=LABELS[g], line=dict(color=col)))
    for w in spray_weeks:
        fig.add_vline(x=w + 1, line_dash="dot", line_color="rgba(200,50,50,0.7)")
    fig.update_xaxes(title="Woche (rote Linien: gespritzt)", dtick=2, range=[0.5, W + 0.5])
    fig.update_yaxes(title="Dichte (Kapazität = 1), logarithmisch", type="log", range=[-7, 0.4], **DECADES)
    fig.update_layout(title=title)
    return _fixed(fig, 340)


def fig_season_loss(loss, title):
    W = len(loss)
    fig = go.Figure(go.Bar(x=list(range(1, W + 1)), y=loss, marker_color=C.COLORS[3], name="Schaden"))
    fig.update_xaxes(title="Woche", dtick=2, range=[0.5, W + 0.5])
    fig.update_yaxes(title="Wochenschaden (Ertragspunkte)", range=[0, 100.0 / W * 1.05])
    fig.update_layout(title=title)
    return _fixed(fig, 240)


def fig_years(series, band=None, band_name=None, title="Ertrag je Jahr (Mittel über die Pfade)"):
    """Ertrag je Jahr: series {Regel: Array(Y)}; band = (unten, oben) für die Regel `band_name` (10- und 90-Prozent-Quantil über die Pfade)."""
    fig = go.Figure()
    names = list(series)
    for i, name in enumerate(names):
        col = C.COLORS[i % len(C.COLORS)]
        vals = series[name]
        x = list(range(1, len(vals) + 1))
        if band is not None and name == band_name:
            lo, hi = band
            fig.add_trace(go.Scatter(x=x + x[::-1], y=list(hi) + list(lo)[::-1], fill="toself", fillcolor="rgba(120,120,120,0.18)", line=dict(width=0),
                                     name=f"{name}: 10 bis 90 % der Pfade", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=name, line=dict(color=col)))
    lows = [float(np.min(v)) for v in series.values()] + ([float(np.min(band[0]))] if band is not None else [])
    lo = min(0.0, 10.0 * np.floor(min(lows) / 10.0)) if lows else 0.0
    fig.update_xaxes(title="Jahr", dtick=2)
    fig.update_yaxes(title="Ertragspunkte im Jahr", range=[lo, 100])
    fig.update_layout(title=title)
    return _fixed(fig, 380)


def fig_resistance(series_a, series_b=None, title="Anteil Resistenter am Saisonende (Mittel über die Pfade)"):
    """Resistenzanteil gegen A (durchgezogen) und, falls vorhanden, gegen B (gestrichelt) je Jahr, logarithmische Achse."""
    fig = go.Figure()
    for i, name in enumerate(series_a):
        col = C.COLORS[i % len(C.COLORS)]
        vals = np.maximum(series_a[name], 1e-12)
        x = list(range(1, len(vals) + 1))
        fig.add_trace(go.Scatter(x=x, y=vals, mode="lines+markers", name=name + (" (A)" if series_b and name in series_b else ""), line=dict(color=col)))
        if series_b and name in series_b:
            fig.add_trace(go.Scatter(x=x, y=np.maximum(series_b[name], 1e-12), mode="lines", name=name + " (B)", line=dict(color=col, dash="dash")))
    fig.update_xaxes(title="Jahr", dtick=2)
    fig.update_yaxes(title="Resistenzanteil, logarithmisch", type="log", range=[-6, 0.1], **DECADES)
    fig.update_layout(title=title)
    return _fixed(fig, 340)


def fig_horizon(rows, nie_value):
    """Summe über 20 Jahre der auf H Jahre abgestimmten Schwelle und des Kalenders; Linie: nie spritzen."""
    xs = [str(int(r["H"])) for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["sum_thr"] for r in rows], name="Schwelle", marker_color=C.COLORS[0],
                         text=[f"S = {r['S']:.3g}" for r in rows], textposition="auto"))
    fig.add_trace(go.Bar(x=xs, y=[r["sum_cal"] for r in rows], name="Kalender", marker_color=C.COLORS[2],
                         text=[f"k = {int(r['k'])}" for r in rows], textposition="auto"))
    fig.add_hline(y=nie_value, line_dash="dash", line_color=C.COLORS[1], annotation_text=f"nie spritzen {nie_value:.0f}", annotation_position="top left")
    fig.update_xaxes(title="Abgestimmt auf so viele Jahre (Planungshorizont)", type="category")
    fig.update_yaxes(title="Summe über 20 Jahre (Ertragspunkte)", rangemode="tozero")
    fig.update_layout(barmode="group")
    return _fixed(fig, 340)


def fig_profile(S, tot, nie_value):
    """Summe über die Jahre in Abhängigkeit von der Schwelle S (logarithmische Achse); gestrichelt: nie spritzen."""
    fig = go.Figure(go.Scatter(x=S, y=tot, mode="lines+markers", name="Schwelle", line=dict(color=C.COLORS[0])))
    i = int(np.argmax(tot))
    fig.add_trace(go.Scatter(x=[S[i]], y=[tot[i]], mode="markers", name="beste Schwelle", marker=dict(size=13, color=C.COLORS[2], symbol="star")))
    fig.add_hline(y=nie_value, line_dash="dash", line_color=C.COLORS[1], annotation_text=f"nie spritzen {nie_value:.0f}", annotation_position="bottom right")
    fig.update_xaxes(title="Schwelle S (Dichte, Kapazität = 1), logarithmisch", type="log", range=[np.log10(C.S_GRID_LO) - 0.05, np.log10(C.S_GRID_HI) + 0.05], **DECADES)
    fig.update_yaxes(title="Summe über 20 Jahre (Ertragspunkte)", rangemode="tozero")
    return _fixed(fig, 340)


def fig_dp(policy):
    """Spritzungen je Jahr der Rückwärtsrechnung gegen die jahresweise beste Antwort (Balken) und Resistenzanteil am Saisonende (Linien, rechts)."""
    years = [p["year"] for p in policy]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=[p["dp_sprays"] for p in policy], name="Rückwärtsrechnung: Spritzungen", marker_color=C.COLORS[0]))
    fig.add_trace(go.Bar(x=years, y=[p["my_sprays"] for p in policy], name="jahresweise beste Antwort: Spritzungen", marker_color=C.COLORS[3]))
    fig.add_trace(go.Scatter(x=years, y=[max(p["dp_res"], 1e-12) for p in policy], name="Resistenzanteil, Rückwärtsrechnung", yaxis="y2", line=dict(color=C.COLORS[0], dash="dot")))
    fig.add_trace(go.Scatter(x=years, y=[max(p["my_res"], 1e-12) for p in policy], name="Resistenzanteil, jahresweise beste Antwort", yaxis="y2", line=dict(color=C.COLORS[3], dash="dot")))
    fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", type="log", range=[-6, 0.1], title="Resistenzanteil", fixedrange=True, **DECADES))
    fig.update_xaxes(title="Jahr", dtick=1)
    fig.update_yaxes(title="Spritzungen je Jahr", rangemode="tozero")
    return _fixed(fig, 360)


def fig_dp2(policy):
    """Zwei Wirkstoffe: Spritzungen mit A und B je Jahr (gestapelt) und Resistenzanteile am Saisonende."""
    years = [p["year"] for p in policy]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=[p["sprays_A"] for p in policy], name="Spritzungen mit A", marker_color=C.COLORS[0]))
    fig.add_trace(go.Bar(x=years, y=[p["sprays_B"] for p in policy], name="Spritzungen mit B", marker_color=C.COLORS[2]))
    fig.add_trace(go.Scatter(x=years, y=[max(p["res_A"], 1e-12) for p in policy], name="Resistenzanteil gegen A", yaxis="y2", line=dict(color=C.COLORS[0], dash="dot")))
    fig.add_trace(go.Scatter(x=years, y=[max(p["res_B"], 1e-12) for p in policy], name="Resistenzanteil gegen B", yaxis="y2", line=dict(color=C.COLORS[2], dash="dot")))
    fig.update_layout(barmode="stack", yaxis2=dict(overlaying="y", side="right", type="log", range=[-6, 0.1], title="Resistenzanteil", fixedrange=True, **DECADES))
    fig.update_xaxes(title="Jahr", dtick=1)
    fig.update_yaxes(title="Spritzungen je Jahr", rangemode="tozero")
    return _fixed(fig, 360)


def fig_raster(grid, metric, current=None, title=""):
    """Heatmap Resistenzgrad (Zeilen) x Fitnesskosten (Spalten). metric: Funktion Zelle -> (Zahl, Text); current = (φ, c) der nächstliegenden Zelle."""
    phis = sorted({r["phi"] for r in grid})
    costs = sorted({r["cost"] for r in grid})
    z, text = [], []
    for p in phis:
        zr, tr = [], []
        for c in costs:
            cell = next((r for r in grid if abs(r["phi"] - p) < 1e-9 and abs(r["cost"] - c) < 1e-9), None)
            val, label = metric(cell) if cell else (None, "")
            zr.append(val)
            tr.append(label)
        z.append(zr)
        text.append(tr)
    fig = go.Figure(go.Heatmap(z=z, x=[f"{c:g}" for c in costs], y=[f"{p:g}" for p in phis], text=text, texttemplate="%{text}", colorscale="RdYlGn_r",
                               showscale=False, hovertemplate="φ %{y}, c %{x}: %{text}<extra></extra>"))
    if current is not None:
        ci = next((i for i, c in enumerate(costs) if abs(c - current[1]) < 1e-9), None)
        pi = next((i for i, p in enumerate(phis) if abs(p - current[0]) < 1e-9), None)
        if ci is not None and pi is not None:
            fig.add_annotation(x=ci, y=pi, text="◎", showarrow=False, font=dict(size=26, color="black"))
    fig.update_xaxes(title="Fitnesskosten c", type="category")
    fig.update_yaxes(title="Resistenzgrad φ", type="category")
    fig.update_layout(title=title)
    return _fixed(fig, 360)
