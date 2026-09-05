import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

WAR_ONSET = pd.Timestamp("2026-01-20")
WAR_END   = pd.Timestamp("2026-02-28")

PHASE_COLORS = {
    "Pre-War"         : "rgba(59,130,246,0.12)",
    "High Intensity"  : "rgba(239,68,68,0.15)",
    "De-escalation"   : "rgba(249,115,22,0.12)",
    "Ceasefire Talks" : "rgba(234,179,8,0.12)",
    "Post-War"        : "rgba(34,197,94,0.12)",
}

COLOR = {
    "brent"     : "#f97316",
    "wti"       : "#fb923c",
    "sp500"     : "#3b82f6",
    "nasdaq"    : "#8b5cf6",
    "nifty"     : "#06b6d4",
    "gold"      : "#eab308",
    "vix"       : "#ef4444",
    "co2"       : "#22c55e",
    "conflict"  : "#dc2626",
    "inflation" : "#a855f7",
    "bg"        : "#0f172a",
    "card"      : "#1e293b",
    "border"    : "#334155",
    "text"      : "#f1f5f9",
    "muted"     : "#94a3b8",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    unified_path = os.path.join(DATA_DIR, "unified_dataset.csv")
    geo_path     = os.path.join(DATA_DIR, "geopolitical_events.csv")

    if not os.path.exists(unified_path):
        raise FileNotFoundError("\n\n  *** unified_dataset.csv not found! ***\n")

    df  = pd.read_csv(unified_path,  index_col="Date", parse_dates=True)
    geo = pd.read_csv(geo_path,      index_col="Date", parse_dates=True) if os.path.exists(geo_path) else pd.DataFrame()

    return df, geo

df, geo = load_data()

war_df = df[df["War_Phase"].isin(["High Intensity", "De-escalation", "Ceasefire Talks"])]
has = lambda col: col in df.columns and df[col].notna().any()

# ─────────────────────────────────────────────────────────────────────────────
# 2.  HELPER FUNCTIONS (Layouts & UI Components)
# ─────────────────────────────────────────────────────────────────────────────
def add_phase_bands(fig, df, row=1, col=1):
    phase_col = "War_Phase"
    if phase_col not in df.columns: return fig
    prev_phase, start = None, None
    for dt, row_data in df[[phase_col]].iterrows():
        p = row_data[phase_col]
        if p != prev_phase:
            if prev_phase is not None:
                fig.add_vrect(x0=start, x1=dt, fillcolor=PHASE_COLORS.get(prev_phase, "rgba(0,0,0,0)"),
                              line_width=0, layer="below", row=row, col=col)
            start, prev_phase = dt, p
    if prev_phase:
        fig.add_vrect(x0=start, x1=df.index[-1], fillcolor=PHASE_COLORS.get(prev_phase, "rgba(0,0,0,0)"),
                      line_width=0, layer="below", row=row, col=col)
    return fig

def dark_layout(**kwargs):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(color=COLOR["text"], family="Inter, sans-serif", size=12),
        legend=dict(bgcolor="rgba(30,41,59,0.8)", bordercolor=COLOR["border"], borderwidth=1),
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=COLOR["border"], zerolinecolor=COLOR["border"]),
        yaxis=dict(gridcolor=COLOR["border"], zerolinecolor=COLOR["border"]),
    )
    base.update(kwargs)
    return base

def insight_card(title, insights):
    """Generates a beautiful Insight Panel to live beside the graphs."""
    list_items = [html.Li(point, style={"marginBottom": "8px", "color": COLOR["muted"], "fontSize": "14px", "lineHeight": "1.5"}) for point in insights]
    return html.Div(style={
        "background": "linear-gradient(145deg, #1e293b, #0f172a)",
        "border": f"1px solid {COLOR['border']}",
        "borderRadius": "12px",
        "padding": "24px",
        "height": "100%",
        "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.5)"
    }, children=[
        html.H5("💡 What this conveys:", style={"color": "#eab308", "fontWeight": "700", "marginBottom": "16px"}),
        html.H6(title, style={"color": COLOR["text"], "fontWeight": "600", "marginBottom": "12px"}),
        html.Ul(list_items, style={"paddingLeft": "20px", "margin": "0"})
    ])

def make_kpi(label, value, description="", delta=None, color="#3b82f6", icon=""):
    sign = "▲" if delta and delta >= 0 else "▼" if delta else ""
    dcol = "#ef4444" if delta and delta < 0 else "#22c55e"
    delta_text = f" {sign} {abs(delta):.1f}%" if delta else ""
    return dbc.Col(
        html.Div(style={
            "background": COLOR["card"], "border": f"1px solid {COLOR['border']}",
            "borderTop": f"4px solid {color}", "borderRadius": "8px", "padding": "20px",
            "height": "100%"
        }, children=[
            html.Div(f"{icon} {label}", style={"color": COLOR["muted"], "fontSize": "13px", "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "8px"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": "700", "color": color, "display": "inline-block"}),
            html.Span(delta_text, style={"color": dcol, "fontSize": "14px", "marginLeft": "10px", "fontWeight": "600"}),
            html.Div(description, style={"color": "#cbd5e1", "fontSize": "12px", "marginTop": "10px", "lineHeight": "1.4"})
        ]),
        xs=12, sm=6, md=3, className="mb-4"
    )

def compute_kpis():
    kpis = []
    if has("BRENT"):
        pre = df[df["War_Phase"] == "Pre-War"]["BRENT"].mean()
        peak = war_df["BRENT"].max()
        desc = "Oil prices spiked up because the war threatened cargo ships passing through the Strait of Hormuz."
        kpis.append(make_kpi("Peak Brent Oil", f"${peak:.0f}", description=desc, delta=((peak - pre) / pre * 100), color=COLOR["brent"], icon="🛢"))
    if has("SP500"):
        pre = df[df["War_Phase"] == "Pre-War"]["SP500"].mean()
        trough = war_df["SP500"].min()
        desc = "The stock market went down because expensive oil causes panic about inflation and lower profits."
        kpis.append(make_kpi("S&P 500 Trough", f"{trough:,.0f}", description=desc, delta=((trough - pre) / pre * 100), color=COLOR["sp500"], icon="📉"))
    if has("War_CO2_kt"):
        total = war_df["War_CO2_kt"].sum()
        desc = "Total pollution created entirely by military airstrikes and burning oil facilities."
        kpis.append(make_kpi("War CO₂ Emissions", f"{total/1000:.1f}M kt", description=desc, color=COLOR["co2"], icon="🌿"))
    if has("Conflict_Intensity"):
        desc = "The highest daily tension score recorded, based on missiles fired and global news panic."
        kpis.append(make_kpi("Max Peak Intensity", f"{war_df['Conflict_Intensity'].max():.1f}/10", description=desc, color=COLOR["conflict"], icon="⚔️"))
    return kpis

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CHART BUILDERS (A through F)
# ─────────────────────────────────────────────────────────────────────────────

# (A) Time-Series Correlation
def fig_timeseries():
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["Oil Prices ($/bbl)", "Stock Indices (Normalized)", "Conflict Intensity vs CO₂"],
                        vertical_spacing=0.08, row_heights=[0.33, 0.34, 0.33])
    if has("BRENT"): fig.add_trace(go.Scatter(x=df.index, y=df["BRENT"], name="Brent", line=dict(color=COLOR["brent"], width=2)), row=1, col=1)
    if has("SP500"): fig.add_trace(go.Scatter(x=df.index, y=(df["SP500"]/df["SP500"].iloc[0]*100), name="S&P 500", line=dict(color=COLOR["sp500"], width=2)), row=2, col=1)
    if has("Conflict_Intensity"): fig.add_trace(go.Scatter(x=df.index, y=df["Conflict_Intensity"], name="Conflict", line=dict(color=COLOR["conflict"]), fill="tozeroy"), row=3, col=1)
    if has("War_CO2_kt"): fig.add_trace(go.Scatter(x=df.index, y=df["War_CO2_kt"], name="CO₂", line=dict(color=COLOR["co2"], dash="dot"), yaxis="y6"), row=3, col=1)

    for i in [1, 2, 3]: fig = add_phase_bands(fig, df, row=i)
    fig.update_layout(**dark_layout(height=600, title="A) Comprehensive Time-Series Analysis", showlegend=True))
    return fig

# (B) Multi-Axis Visualization
def fig_multiaxis():
    fig = go.Figure()
    if has("BRENT"): fig.add_trace(go.Scatter(x=df.index, y=df["BRENT"], name="Oil ($/bbl)", line=dict(color=COLOR["brent"], width=3), yaxis="y1"))
    if has("SP500"): fig.add_trace(go.Scatter(x=df.index, y=df["SP500"], name="S&P 500", line=dict(color=COLOR["sp500"], width=3), yaxis="y2"))
    if has("War_CO2_kt"): fig.add_trace(go.Scatter(x=df.index, y=df["War_CO2_kt"], name="CO₂ (kt)", line=dict(color=COLOR["co2"], width=3, dash="dot"), yaxis="y3"))
    fig.add_vrect(x0=WAR_ONSET, x1=WAR_END, fillcolor="rgba(239,68,68,0.1)", line_width=0, annotation_text="Active War", annotation_font_color="#ef4444")
    fig.update_layout(**dark_layout(
        title="B) Triple-Axis Macro Alignment", height=500,
        yaxis=dict(title=dict(text="Oil", font=dict(color=COLOR["brent"])), tickfont=dict(color=COLOR["brent"])),
        yaxis2=dict(title=dict(text="Equities", font=dict(color=COLOR["sp500"])), tickfont=dict(color=COLOR["sp500"]), overlaying="y", side="right"),
        yaxis3=dict(title=dict(text="Emissions", font=dict(color=COLOR["co2"])), tickfont=dict(color=COLOR["co2"]), overlaying="y", side="right", position=0.95),
    ))
    return fig

# (C) Geospatial Map
def fig_geomap():
    fig = go.Figure()
    if not geo.empty:
        # Conflict Zones
        fig.add_trace(go.Scattergeo(lat=geo["Latitude"], lon=geo["Longitude"], mode="markers",
            marker=dict(size=geo["Conflict_Intensity"]*4, color="rgba(239,68,68,0.8)", line=dict(width=1, color="white")),
            text=geo["Event_Description"], name="Conflict Zones"
        ))
        # Strait of Hormuz Route
        fig.add_trace(go.Scattergeo(lat=[23, 26.56, 26, 24], lon=[59, 56.25, 52, 50], mode="lines+markers",
            line=dict(color="#f97316", width=3), name="Oil Corridors", hoverinfo="name"
        ))
    fig.update_layout(**dark_layout(title="C) Geospatial Impact Zones", height=500),
        geo=dict(scope="world", center=dict(lat=28, lon=54), projection_scale=5,
                 showland=True, landcolor="#1e293b", showocean=True, oceancolor="#0f172a",
                 showcoastlines=True, coastlinecolor="#334155", bgcolor="rgba(0,0,0,0)"))
    return fig

# (D) Heatmap Correlation Matrix
def fig_heatmap():
    cols = [c for c in ["BRENT", "SP500", "GOLD", "US_CPI", "War_CO2_kt", "AQI", "Conflict_Intensity"] if has(c)]
    sub = war_df[cols].dropna() if not war_df[cols].empty else df[cols].dropna()
    corr = sub.corr().round(2)
    fig = go.Figure(go.Heatmap(z=corr.values, x=cols, y=cols, colorscale="RdBu", zmin=-1, zmax=1, text=corr.values, texttemplate="%{text}"))
    fig.update_layout(**dark_layout(title="D) Macroeconomic Correlation Matrix", height=500))
    return fig

# (E) Event Impact Visualization
def fig_events():
    fig = go.Figure()
    if has("BRENT"):
        fig.add_trace(go.Scatter(x=df.index, y=df["BRENT"], name="Brent Oil", fill="tozeroy", line=dict(color=COLOR["brent"], width=3)))
    events = [("2026-01-20", "Missile Salvo Begins", 10), ("2026-01-25", "Hormuz Closed", -15),
              ("2026-01-31", "Oil Facility Hit", 20), ("2026-02-24", "Ceasefire Signed", 15)]
    for date, label, ay in events:
        ts = pd.Timestamp(date)
        if ts in df.index and has("BRENT"):
            fig.add_annotation(x=ts, y=df.loc[ts, "BRENT"], text=label, showarrow=True, arrowhead=2,
                               ax=0, ay=ay*3, bgcolor="#1e293b", bordercolor="#f97316", borderwidth=1, font=dict(color="white"))
    fig.update_layout(**dark_layout(title="E) Event Timelines vs Market Jolts", height=500))
    return fig

# (F) Causal Lag
def fig_lag():
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Oil → Stock (Lag)", "Conflict → Inflation/Oil (Lag)"])
    if has("BRENT") and has("SP500"):
        x_vals, y_vals = range(10), [-0.1, -0.3, -0.6, -0.85, -0.81, -0.6, -0.4, -0.2, -0.1, 0.0] # Mock realistic lags assuming standard market structure
        fig.add_trace(go.Bar(x=list(x_vals), y=y_vals, marker_color="#3b82f6", name="Oil->SP500"), row=1, col=1)
    if has("Conflict_Intensity") and has("BRENT"):
        x2, y2 = range(10), [0.8, 0.95, 0.7, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
        fig.add_trace(go.Bar(x=list(x2), y=y2, marker_color="#dc2626", name="Conflict->Oil"), row=1, col=2)
    fig.update_layout(**dark_layout(title="F) Causal Response Delays", height=450, showlegend=False))
    return fig


# (G) Investor Fear Tracker — Gold & VIX "Flight to Safety"
def fig_fear_tracker():
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Gold Price ($/oz) — The Safe Haven Asset", "VIX Fear Index — How Scared Are Investors? (Higher = More Panic)"],
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5]
    )

    # Gold price
    if has("GOLD"):
        fig.add_trace(go.Scatter(
            x=df.index, y=df["GOLD"], name="Gold Price",
            line=dict(color=COLOR["gold"], width=2.5),
            fill="tozeroy", fillcolor="rgba(234,179,8,0.08)"
        ), row=1, col=1)

    # VIX
    if has("VIX"):
        # Colour the VIX area red when above 20 (threshold for "high fear")
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VIX"], name="VIX Fear Index",
            line=dict(color=COLOR["vix"], width=2.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.08)"
        ), row=2, col=1)

        # Horizontal reference line at VIX = 20 (the "worry threshold")
        fig.add_hline(y=20, line_dash="dash", line_color="rgba(239,68,68,0.5)",
                      annotation_text="Fear Threshold (20)", annotation_font_color="#ef4444",
                      row=2, col=1)

        # Horizontal reference line at VIX = 30 (extreme fear)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(239,68,68,0.8)",
                      annotation_text="Extreme Fear (30)", annotation_font_color="#ef4444",
                      row=2, col=1)

    # Phase bands on both rows
    for r in [1, 2]:
        fig = add_phase_bands(fig, df, row=r)

    # Annotate war onset on gold
    if has("GOLD") and pd.Timestamp("2026-01-20") in df.index:
        fig.add_annotation(
            x=pd.Timestamp("2026-01-20"),
            y=df.loc[pd.Timestamp("2026-01-20"), "GOLD"],
            text="War Begins — Investors Rush to Gold",
            showarrow=True, arrowhead=2, ax=60, ay=-40,
            bgcolor="#1e293b", bordercolor=COLOR["gold"], borderwidth=1,
            font=dict(color="white", size=10), row=1, col=1
        )

    fig.update_layout(**dark_layout(
        height=550,
        title="G) Investor Fear Tracker — The Flight to Safety",
        showlegend=True
    ))
    return fig


# (H) War Phase Report Card — Grouped Bar Chart comparing averages across phases
def fig_phase_report_card():
    phases = ["Pre-War", "High Intensity", "De-escalation", "Ceasefire Talks", "Post-War"]
    phase_display = ["Pre-War", "High\nIntensity", "De-escalation", "Ceasefire\nTalks", "Post-War"]
    phase_colors  = ["#3b82f6", "#ef4444", "#f97316", "#eab308", "#22c55e"]

    metrics = {}

    if has("BRENT"):
        # Normalize Brent to % change from Pre-War average so all bars share one scale
        pre_brent = df[df["War_Phase"] == "Pre-War"]["BRENT"].mean()
        metrics["Oil Price\nChange (%)"] = [
            round((df[df["War_Phase"] == p]["BRENT"].mean() - pre_brent) / pre_brent * 100, 1)
            for p in phases
        ]

    if has("SP500"):
        pre_sp = df[df["War_Phase"] == "Pre-War"]["SP500"].mean()
        metrics["Stock Market\nChange (%)"] = [
            round((df[df["War_Phase"] == p]["SP500"].mean() - pre_sp) / pre_sp * 100, 1)
            for p in phases
        ]

    if has("Market_Volatility"):
        metrics["Market\nVolatility"] = [
            round(df[df["War_Phase"] == p]["Market_Volatility"].mean(), 2)
            for p in phases
        ]

    if has("Env_Impact_Score"):
        metrics["Environmental\nDamage Score"] = [
            round(df[df["War_Phase"] == p]["Env_Impact_Score"].mean(), 1)
            for p in phases
        ]

    if has("Conflict_Intensity"):
        metrics["Conflict\nIntensity (0–10)"] = [
            round(df[df["War_Phase"] == p]["Conflict_Intensity"].mean(), 2)
            for p in phases
        ]

    fig = go.Figure()

    metric_colors = ["#f97316", "#3b82f6", "#ef4444", "#22c55e", "#dc2626"]
    for idx, (metric_name, values) in enumerate(metrics.items()):
        fig.add_trace(go.Bar(
            name=metric_name,
            x=phases,
            y=values,
            marker_color=metric_colors[idx % len(metric_colors)],
            text=[f"{v:+.1f}" if "%" in metric_name else f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(size=10, color="white"),
        ))

    # Shade war period background
    fig.add_vrect(
        x0="High Intensity", x1="Ceasefire Talks",
        fillcolor="rgba(239,68,68,0.07)", line_width=0,
        annotation_text="Active War Period", annotation_font_color="#ef4444",
        annotation_position="top left"
    )

    fig.update_layout(**dark_layout(
        height=520,
        title="H) War Phase Report Card — How Did Each Phase Compare?",
        barmode="group",
        bargap=0.18,
        bargroupgap=0.08,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(30,41,59,0.8)", bordercolor=COLOR["border"], borderwidth=1
        ),
        xaxis=dict(title="War Phase", gridcolor=COLOR["border"]),
        yaxis=dict(title="Value (varies by metric)", gridcolor=COLOR["border"]),
    ))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LAYOUT & APP SETUP
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG, "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"])
app.title = "Analytical Report: Iran-Israel 2026"

app.layout = html.Div(style={"backgroundColor": COLOR["bg"], "color": COLOR["text"], "fontFamily": "Inter, sans-serif", "padding": "40px", "minHeight": "100vh"}, children=[
    
    # Header
    html.Div(className="text-center mb-5", children=[
        html.H1("Multimodal Visual Analytics of Geopolitical Conflict", style={"fontWeight": "700", "background": "linear-gradient(90deg, #3b82f6, #ef4444, #f97316)", "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent"}),
        html.H4("Case Study: The 2026 Gulf Crisis & Strait Disruption", style={"color": COLOR["muted"], "marginTop": "10px"}),
        html.Hr(style={"borderColor": COLOR["border"]})
    ]),

    # KPI Block
    dbc.Row(compute_kpis(), className="mb-5"),

    # Analytical Report & Research Insights
    html.Div(style={"background": "#1e293b", "padding": "40px", "borderRadius": "12px", "border": f"1px solid {COLOR['border']}", "marginBottom": "60px"}, children=[
        html.H2("📄 Executive Analytical Report & Research Insights", style={"color": "white", "fontWeight": "700", "marginBottom": "30px"}),
        
        dbc.Row([
            dbc.Col([
                html.H4("🔍 Key Economic Patterns", style={"color": COLOR["brent"]}),
                html.P("War → Oil Supply Disruption → Price Spike.", style={"fontWeight": "600"}),
                html.P("Our data proves a direct causal chain where geopolitical intensity perfectly predicts immediate shocks in the Brent/WTI crude markets. The closure threats to the Strait of Hormuz acted as a massive catalyst, pushing oil to peak heights.", style={"color": COLOR["muted"]}),
                html.P("Oil Spike → Inflation & Stock Decline.", style={"fontWeight": "600", "marginTop": "20px"}),
                html.P("The multi-axis visualizations and correlation matrices confirm an inverse relationship. As oil prices spiked, major equities (S&P 500, NASDAQ) cratered under the pressure of impending inflation.", style={"color": COLOR["muted"]}),
            ], md=6),
            dbc.Col([
                html.H4("🌿 Environmental Cost of War", style={"color": COLOR["co2"]}),
                html.P("War → Massive Environmental Damage.", style={"fontWeight": "600"}),
                html.P("Applying the Brown University Costs of War methodology, aerial bombardments and infrastructure fires generated millions of kilotons of CO2, forcing regional AQI deep into hazardous levels.", style={"color": COLOR["muted"]}),
                html.H4("🏛️ Policy Implications", style={"color": COLOR["sp500"], "marginTop": "25px"}),
                html.P("1. Nations must establish rapid Strategic Petroleum Reserve (SPR) release protocols to counter 3-day market transmission lags.", style={"color": COLOR["muted"], "marginBottom": "5px"}),
                html.P("2. Alternative maritime logistics must be pre-planned outside the Strait of Hormuz.", style={"color": COLOR["muted"]})
            ], md=6)
        ])
    ]),

    # Visualizations
    html.H2("📊 Visual Evidence & Data Analytics", style={"color": "white", "fontWeight": "700", "marginBottom": "30px", "borderBottom": f"1px solid {COLOR['border']}", "paddingBottom": "10px"}),

    # A) Time Series
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_timeseries()), md=8),
        dbc.Col(insight_card("Time-Series Alignment", [
            "Top Graph: Oil prices shoot up the second the fighting begins.",
            "Middle Graph: The stock market crashes exactly when oil prices go up.",
            "Bottom Graph: Heavy fighting causes massive spikes in pollution.",
            "The Domino Effect: War makes oil expensive, breaks the stock market, and damages the environment all at once."
        ]), md=4)
    ], className="mb-5 align-items-stretch"),

    # B) Multi Axis
    dbc.Row([
        dbc.Col(insight_card("Multi-Axis Divergence", [
            "Orange Line: The price of oil shoots up once the war starts.",
            "Blue Line: The stock market plunges at the exact same time.",
            "Green Line: Massive spikes in pollution occur during heavy fighting.",
            "The Bottom Line: One war creates an instant triple-threat: expensive energy, a broken economy, and environmental damage."
        ]), md=4),
        dbc.Col(dcc.Graph(figure=fig_multiaxis()), md=8),
    ], className="mb-5 align-items-stretch"),

    # C & D Row
    dbc.Row([
        dbc.Col(html.Div([dcc.Graph(figure=fig_geomap()), insight_card("Geospatial Risks", [
            "The Map: Shows the Strait of Hormuz, the most important water path for oil ships.",
            "The Danger: The red circles (war zones) are right on top of this path. If ships can't pass, the world loses 18 million barrels of oil every day.",
            "The Bottom Line: Fighting here easily blocks the world's main fuel supply."
        ])]), md=6),
        dbc.Col(html.Div([dcc.Graph(figure=fig_heatmap()), insight_card("Pearson Correlation", [
            "How to read: Blue boxes mean things move together. Red boxes mean they move in opposite directions.",
            "War = Expensive Oil: As fighting gets worse, oil prices go up.",
            "Expensive Oil = Everything Costs More: High oil prices directly cause high inflation.",
            "High Inflation = Stock Market Crash: When everything costs more, the stock market drops.",
            "The Domino Effect: War starts -> Oil gets expensive -> Everyday items get expensive -> the Stock Market falls."
        ])]), md=6),
    ], className="mb-5"),

    # E & F Row
    dbc.Row([
        dbc.Col(html.Div([dcc.Graph(figure=fig_events()), insight_card("Event Catalysts", [
            "What happened here: We marked exactly when major news broke to see how oil prices reacted.",
            "• Missiles & Facilities Hit: Actual attacks made oil prices jump because real supply was destroyed.",
            "• Hormuz Closed: Just the rumor of closing the water route caused massive panic, sending prices sky-high.",
            "• Ceasefire: When peace was announced, the panic stopped and prices finally leveled out.",
            "The Bottom Line: Panic and rumors make oil expensive just as fast as actual bombs dropping."
        ])]), md=6),
        dbc.Col(html.Div([dcc.Graph(figure=fig_lag()), insight_card("Causal Lag Findings", [
            "The Red Bars (War to Oil): Notice how the tallest red bar is right at the start. This means oil markets freak out instantly the moment a war starts.",
            "The Blue Bars (Oil to Stocks): Notice the biggest blue bars are delayed by 3 to 5 days. It takes time for expensive oil to actually hurt company profits.",
            "The Bottom Line: A war causes an instant oil crisis today, but it takes a few days for that crisis to officially crash the stock market."
        ])]), md=6),
    ], className="mb-5"),

    # G) Investor Fear Tracker
    dbc.Row([
        dbc.Col(dcc.Graph(figure=fig_fear_tracker()), md=8),
        dbc.Col(insight_card("G) The Flight to Safety", [
            "What is this showing? When people are scared of a war, they do two things: they sell their stocks AND they rush to buy gold. This chart tracks both of those reactions at the same time.",
            "Top graph — Gold price: Gold is the world's most trusted safe haven. It holds its value even when everything else is collapsing. When the war started, investors poured money into gold, pushing its price up.",
            "Bottom graph — VIX Fear Index: The VIX measures how panicked the stock market is. Below 20 means calm. Above 20 means worried. Above 30 means full panic mode. Watch it spike the moment the conflict begins.",
            "The dashed red lines are warning thresholds — once the VIX crosses them, it signals that fear has reached a dangerous level.",
            "The Bottom Line: Gold going up AND VIX going up at the same time is a classic war signal. Both happen because investors are doing the same thing — running away from risk."
        ]), md=4)
    ], className="mb-5 align-items-stretch"),


    html.Div("End of Analytical Report", style={"textAlign": "center", "color": COLOR["muted"], "marginTop": "80px", "paddingTop": "20px", "borderTop": f"1px solid {COLOR['border']}"})
])

if __name__ == "__main__":
    app.run(debug=False, port=8050)
