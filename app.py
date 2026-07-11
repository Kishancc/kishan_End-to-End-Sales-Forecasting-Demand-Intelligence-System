"""
DemandIQ — Sales Forecasting & Demand Intelligence Dashboard
Run:  streamlit run app.py        (from inside SalesForecasting_Intern/)
Deploy: push to GitHub -> Streamlit Community Cloud -> point to app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# MUST be first Streamlit call
st.set_page_config(
    page_title="DemandIQ — Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg0:#080d1a;--bg1:#0d1526;--bg2:#111d35;--bg3:#162140;
  --border:rgba(99,179,255,0.12);--text:#e8eef8;--muted:#7a90b8;
  --cyan:#22d3ee;--violet:#a78bfa;--amber:#fbbf24;--rose:#fb7185;
  --green:#34d399;--blue:#60a5fa;
  --font:'Inter',system-ui,sans-serif;--mono:'JetBrains Mono',monospace;
}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],.main,.block-container{
  background:var(--bg0)!important;color:var(--text)!important;font-family:var(--font)!important;
}
[data-testid="stHeader"]{background:transparent!important}
.block-container{padding:1.5rem 2.5rem 3rem!important;max-width:1400px!important}
h1,h2,h3,h4{color:var(--text)!important}
.iq-header{display:flex;align-items:center;gap:14px;padding:12px 0 20px;border-bottom:1px solid var(--border);margin-bottom:24px}
.iq-logo{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#22d3ee,#a78bfa);display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 0 24px rgba(34,211,238,.3);flex-shrink:0}
.iq-brand-name{font-size:20px;font-weight:800;letter-spacing:-.03em;color:var(--text)}
.iq-brand-sub{font-size:11px;color:var(--muted);margin-top:2px}
.iq-tag{margin-left:auto;font-size:11px;font-weight:700;color:#22d3ee;background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.25);border-radius:999px;padding:3px 12px;letter-spacing:.06em;text-transform:uppercase}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:4px 0 24px}
.kpi-card{background:#111d35;border:1px solid rgba(99,179,255,0.12);border-radius:14px;padding:18px 20px;position:relative;overflow:hidden;transition:transform .15s,box-shadow .15s}
.kpi-card:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,0,0,.5)}
.kpi-strip{position:absolute;left:0;top:0;bottom:0;width:4px;border-radius:14px 0 0 14px}
.kpi-glow{position:absolute;top:-30px;right:-30px;width:100px;height:100px;border-radius:50%;opacity:.08;filter:blur(25px);pointer-events:none}
.kpi-icon{font-size:18px;margin-bottom:7px}
.kpi-label{font-size:11px;font-weight:600;color:#7a90b8;text-transform:uppercase;letter-spacing:.08em}
.kpi-value{font-size:26px;font-weight:800;color:#e8eef8;letter-spacing:-.04em;margin-top:3px;font-family:'JetBrains Mono',monospace}
.kpi-delta{font-size:11px;color:#7a90b8;margin-top:5px}
.sl{font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#7a90b8;margin:28px 0 14px;display:flex;align-items:center;gap:8px}
.sl::after{content:'';flex:1;height:1px;background:rgba(99,179,255,0.12)}
.cc{background:#111d35;border:1px solid rgba(99,179,255,0.12);border-radius:14px;padding:16px 20px 12px;margin-bottom:16px}
.cc-title{font-size:13px;font-weight:700;color:#e8eef8;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.dot-c{width:8px;height:8px;border-radius:50%;flex-shrink:0;display:inline-block}
.dt{width:100%;border-collapse:collapse;font-size:13px}
.dt th{padding:10px 14px;text-align:left;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#7a90b8;border-bottom:1px solid rgba(99,179,255,0.12)}
.dt td{padding:11px 14px;border-bottom:1px solid rgba(99,179,255,0.06)}
.dt tr:last-child td{border-bottom:none}
.dt tr:hover td{background:rgba(99,179,255,.04)}
.dt .best-row td{background:rgba(34,211,238,.06)}
.mono{font-family:'JetBrains Mono',monospace;color:#22d3ee}
.badge{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700;background:rgba(34,211,238,.15);color:#22d3ee;border:1px solid rgba(34,211,238,.3);margin-left:6px}
.pill{display:inline-block;border-radius:999px;padding:2px 9px;font-size:11px;font-weight:600}
.p-both{background:rgba(251,113,133,.18);color:#fb7185;border:1px solid rgba(251,113,133,.3)}
.p-iso{background:rgba(251,191,36,.18);color:#fbbf24;border:1px solid rgba(251,191,36,.3)}
.p-z{background:rgba(167,139,250,.18);color:#a78bfa;border:1px solid rgba(167,139,250,.3)}
.chip{display:inline-block;border-radius:999px;padding:3px 10px;font-size:11px;font-weight:600}
.c-stable{background:rgba(96,165,250,.18);color:#60a5fa;border:1px solid rgba(96,165,250,.3)}
.c-grow{background:rgba(52,211,153,.18);color:#34d399;border:1px solid rgba(52,211,153,.3)}
.c-decline{background:rgba(251,113,133,.18);color:#fb7185;border:1px solid rgba(251,113,133,.3)}
.c-volatile{background:rgba(251,191,36,.18);color:#fbbf24;border:1px solid rgba(251,191,36,.3)}
.insight{display:flex;gap:12px;align-items:flex-start;background:rgba(34,211,238,.06);border:1px solid rgba(34,211,238,.18);border-radius:10px;padding:14px 16px;margin:14px 0;font-size:13px;line-height:1.55}
.insight.warn{background:rgba(251,191,36,.06);border-color:rgba(251,191,36,.2)}
.ii{font-size:17px;flex-shrink:0;margin-top:1px}
.it strong{color:#22d3ee}
.insight.warn .it strong{color:#fbbf24}
.strat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:14px 0}
.strat-card{background:#162140;border:1px solid rgba(99,179,255,0.12);border-radius:12px;padding:16px 18px}
.strat-card h4{font-size:13px;font-weight:700;margin-bottom:6px;margin-top:0}
.strat-card p{font-size:12px;color:#7a90b8;line-height:1.55;margin:0}
.fcv{font-family:'JetBrains Mono',monospace;color:#22d3ee;font-size:15px;font-weight:600}
.fcm{color:#7a90b8}
[data-baseweb="tab-list"]{background:#0d1526!important;border-radius:12px!important;padding:4px!important;gap:2px!important;border:1px solid rgba(99,179,255,0.12)!important;margin-bottom:20px!important}
[data-baseweb="tab"]{background:transparent!important;color:#7a90b8!important;font-size:13px!important;font-weight:600!important;border-radius:9px!important;padding:8px 18px!important;border:none!important}
[aria-selected="true"][data-baseweb="tab"]{background:#162140!important;color:#22d3ee!important;border:1px solid rgba(34,211,238,.25)!important}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none!important}
[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div{background:#111d35!important;border-color:rgba(99,179,255,0.12)!important;color:#e8eef8!important;border-radius:10px!important}
label{color:#7a90b8!important;font-size:11px!important;font-weight:600!important;text-transform:uppercase;letter-spacing:.07em}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="iq-header">
  <div class="iq-logo">📈</div>
  <div>
    <div class="iq-brand-name">DemandIQ</div>
    <div class="iq-brand-sub">Sales Forecasting &amp; Demand Intelligence · Superstore 2015–2018</div>
  </div>
  <div class="iq-tag">🟢 Live Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════════════════════
SEASON_NUM = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_transactions.csv", parse_dates=["Order Date","Ship Date"])

@st.cache_data
def load_precomputed():
    an = pd.read_csv("anomalies.csv", parse_dates=["week"])
    if "isoforest" in an.columns:
        an = an.rename(columns={"isoforest":"isolation_forest_flag"})
    cl = pd.read_csv("product_clusters.csv")
    mc = pd.read_csv("model_comparison.csv")
    return an, cl, mc

@st.cache_data
def forecast_segment(mask_key, steps):
    df_ = load_data()
    mask = pd.Series(True, index=df_.index) if mask_key=="ALL" else (df_[mask_key.split(":")[0]] == mask_key.split(":",1)[1])
    ser = df_[mask].set_index("Order Date")["Sales"].resample("MS").sum().asfreq("MS").fillna(0)
    feat = pd.DataFrame({"y":ser})
    for lag in [1,2,3]: feat[f"lag{lag}"] = feat["y"].shift(lag)
    feat["roll_mean3"] = feat["y"].shift(1).rolling(3).mean()
    feat["month"]   = feat.index.month
    feat["quarter"] = feat.index.quarter
    feat["season"]  = feat["month"].map(SEASON_NUM)
    feat = feat.dropna(); X, y = feat.drop(columns=["y"]), feat["y"]
    ho = min(3, len(X)-6) if len(X)>9 else 2
    em = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42)
    em.fit(X.iloc[:-ho], y.iloc[:-ho])
    mae  = mean_absolute_error(y.iloc[-ho:], em.predict(X.iloc[-ho:]))
    rmse = mean_squared_error(y.iloc[-ho:],  em.predict(X.iloc[-ho:])) ** 0.5
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(X, y)
    hist = list(ser.values); last = ser.index[-1]; preds = []
    for i in range(steps):
        nd = last + pd.DateOffset(months=i+1)
        row = pd.DataFrame([[hist[-1],hist[-2],hist[-3],np.mean(hist[-3:]),nd.month,nd.quarter,SEASON_NUM[nd.month]]], columns=X.columns)
        p = max(float(model.predict(row)[0]), 0)
        preds.append((nd,p)); hist.append(p)
    return ser, preds, mae, rmse

df = load_data()
anomalies_df, clusters_df, model_comparison_df = load_precomputed()

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def th(fig):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color="#7a90b8"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=20,b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#e8eef8")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#162140", bordercolor="rgba(99,179,255,0.25)", font=dict(color="#e8eef8")),
    )
    fig.update_xaxes(gridcolor="rgba(99,179,255,0.08)", linecolor="rgba(99,179,255,0.15)", tickfont=dict(color="#7a90b8"))
    fig.update_yaxes(gridcolor="rgba(99,179,255,0.08)", linecolor="rgba(0,0,0,0)", tickfont=dict(color="#7a90b8"))
    return fig

def kpi(icon, label, value, delta="", color="#22d3ee"):
    d = f"<div class='kpi-delta'>{delta}</div>" if delta else ""
    return (f"<div class='kpi-card'>"
            f"<div class='kpi-strip' style='background:{color}'></div>"
            f"<div class='kpi-glow' style='background:{color}'></div>"
            f"<div class='kpi-icon'>{icon}</div>"
            f"<div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div>{d}</div>")

def kpi_row(*cards):
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

def chart_title(label, dot="#22d3ee"):
    st.markdown(f'<div class="cc-title"><span class="dot-c" style="background:{dot}"></span>{label}</div>', unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="sl">{label}</div>', unsafe_allow_html=True)

def insight(text, warn=False):
    cls = "insight warn" if warn else "insight"
    icon = "⚠️" if warn else "💡"
    st.markdown(f'<div class="{cls}"><div class="ii">{icon}</div><div class="it">{text}</div></div>', unsafe_allow_html=True)

def card_table(title_html, table_html, dot="#22d3ee"):
    """Render a titled card containing a table — all in one self-contained div."""
    st.markdown(
        f'<div class="cc">'
        f'<div class="cc-title"><span class="dot-c" style="background:{dot}"></span>{title_html}</div>'
        f'<div style="overflow-x:auto">{table_html}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4 = st.tabs(["📊  Overview","🔮  Forecast Explorer","⚡  Anomaly Report","🧩  Demand Segments"])

# ───────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ───────────────────────────────────────────────────────────────────────────
with t1:
    fc1, fc2 = st.columns(2)
    with fc1:
        region_filter = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    with fc2:
        category_filter = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))

    fdf = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]
    total_sales  = fdf["Sales"].sum()
    total_orders = fdf["Order ID"].nunique()
    avg_order    = fdf["Sales"].mean()
    ys  = fdf.groupby(fdf["Order Date"].dt.year)["Sales"].sum()
    yoy = ((ys.iloc[-1]-ys.iloc[-2])/ys.iloc[-2]*100) if len(ys)>=2 else 0

    kpi_row(
        kpi("💰","Total Revenue",   f"${total_sales:,.0f}", f"↑ {yoy:.1f}% vs prior year","#22d3ee"),
        kpi("📦","Total Orders",    f"{total_orders:,}",     "Unique order IDs",           "#a78bfa"),
        kpi("🛒","Avg Order Value", f"${avg_order:,.0f}",    "Per transaction",            "#fbbf24"),
        kpi("📅","Data Span",       "4 Years",               "Jan 2015 – Dec 2018",        "#34d399"),
    )

    col1, col2 = st.columns([1,2])
    with col1:
        chart_title("Revenue by Year","#22d3ee")
        ys2 = fdf.groupby(fdf["Order Date"].dt.year)["Sales"].sum().reset_index()
        ys2.columns = ["Year","Sales"]
        fig = px.bar(ys2, x="Year", y="Sales", text_auto=".2s", color_discrete_sequence=["#22d3ee"])
        fig.update_traces(marker_line_width=0, textfont_color="#e8eef8"); fig.update_layout(bargap=0.4)
        st.plotly_chart(th(fig), use_container_width=True)

    with col2:
        chart_title("Monthly Sales Trend","#a78bfa")
        mon = fdf.set_index("Order Date")["Sales"].resample("MS").sum().reset_index()
        fig = go.Figure(go.Scatter(x=mon["Order Date"], y=mon["Sales"], mode="lines", fill="tozeroy",
            line=dict(color="#a78bfa", width=2), fillcolor="rgba(167,139,250,0.1)",
            hovertemplate="<b>%{x|%b %Y}</b><br>$%{y:,.0f}<extra></extra>"))
        st.plotly_chart(th(fig), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        chart_title("Sales by Region","#fbbf24")
        reg = fdf.groupby("Region")["Sales"].sum().reset_index()
        fig = px.pie(reg, names="Region", values="Sales", hole=0.55, color_discrete_sequence=["#22d3ee","#a78bfa","#fbbf24","#34d399"])
        fig.update_traces(textfont_color="#e8eef8", hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>")
        st.plotly_chart(th(fig), use_container_width=True)

    with c2:
        chart_title("Sales by Category","#34d399")
        cat = fdf.groupby("Category")["Sales"].sum().sort_values().reset_index()
        fig = px.bar(cat, x="Sales", y="Category", orientation="h", color_discrete_sequence=["#34d399"])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(th(fig), use_container_width=True)

    with c3:
        chart_title("Top Sub-Categories","#fb7185")
        sub = fdf.groupby("Sub-Category")["Sales"].sum().nlargest(8).sort_values().reset_index()
        fig = px.bar(sub, x="Sales", y="Sub-Category", orientation="h", color_discrete_sequence=["#fb7185"])
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(th(fig), use_container_width=True)

    insight("<strong>Technology</strong> leads revenue. <strong>East</strong> is the most consistent growth region. "
            "<strong>November–December</strong> regularly produce 30–40% more revenue than any other two-month window.")

# ───────────────────────────────────────────────────────────────────────────
# TAB 2 — FORECAST EXPLORER
# ───────────────────────────────────────────────────────────────────────────
with t2:
    ctrl1, ctrl2, ctrl3 = st.columns([1.2,1.2,0.8])
    with ctrl1:
        dim = st.selectbox("Dimension", ["Overall","Category","Region"])
    with ctrl2:
        if dim == "Overall":
            mask_key = "ALL"
        elif dim == "Category":
            chosen = st.selectbox("Category", sorted(df["Category"].unique()))
            mask_key = f"Category:{chosen}"
        else:
            chosen = st.selectbox("Region", sorted(df["Region"].unique()))
            mask_key = f"Region:{chosen}"
    with ctrl3:
        horizon = st.slider("Months ahead", 1, 3, 3)

    with st.spinner("Training XGBoost model…"):
        ser, preds, mae, rmse = forecast_segment(mask_key, horizon)

    hist_tail = ser.tail(18)
    fc_dates  = [d for d,_ in preds]
    fc_vals   = [v for _,v in preds]

    chart_title("Sales Forecast — XGBoost model","#22d3ee")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail.values, mode="lines+markers", name="Historical",
        line=dict(color="#60a5fa", width=2), marker=dict(size=5, color="#60a5fa"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual $%{y:,.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[hist_tail.index[-1]]+fc_dates, y=[hist_tail.values[-1]]+fc_vals,
        mode="lines+markers", name="Forecast", line=dict(color="#22d3ee", width=2.5, dash="dot"),
        marker=dict(size=9, color="#22d3ee", symbol="diamond", line=dict(width=2, color="#080d1a")),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast $%{y:,.0f}<extra></extra>"))
    if fc_dates:
        fig.add_vrect(x0=hist_tail.index[-1], x1=fc_dates[-1], fillcolor="rgba(34,211,238,0.05)", line_width=0)
    st.plotly_chart(th(fig), use_container_width=True)

    left, right = st.columns([1.4,1])
    with left:
        fc_rows = "".join(
            f"<tr><td class='fcm'>{d.strftime('%B %Y')}</td><td class='fcv'>${v:,.0f}</td></tr>"
            for d,v in zip(fc_dates, fc_vals))
        card_table("Forecasted Values",
            f"<table class='dt'><thead><tr><th>Month</th><th>Forecast</th></tr></thead><tbody>{fc_rows}</tbody></table>")

    with right:
        kpi_row(
            kpi("🎯","MAE (holdout)",  f"${mae:,.0f}",  "Mean absolute error",     "#22d3ee"),
            kpi("📐","RMSE (holdout)", f"${rmse:,.0f}", "Root mean squared error", "#fbbf24"),
        )

    mc    = model_comparison_df.copy()
    bi    = mc["MAPE"].idxmin()
    mrows = ""
    for i, row in mc.iterrows():
        b = "<span class='badge'>✓ Best</span>" if i==bi else ""
        c = "best-row" if i==bi else ""
        mrows += (f"<tr class='{c}'><td><strong>{row['Model']}</strong>{b}</td>"
                  f"<td class='mono'>${row['MAE']:,.0f}</td><td class='mono'>${row['RMSE']:,.0f}</td>"
                  f"<td class='mono'>{row['MAPE']:.1f}%</td><td class='mono'>${row['Forecast M1']:,.0f}</td>"
                  f"<td class='mono'>${row['Forecast M2']:,.0f}</td><td class='mono'>${row['Forecast M3']:,.0f}</td></tr>")
    card_table("Model Comparison — 3-month holdout",
        f"<table class='dt'><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE</th>"
        f"<th>Month 1</th><th>Month 2</th><th>Month 3</th></tr></thead><tbody>{mrows}</tbody></table>",
        dot="#a78bfa")

    insight("<strong>XGBoost</strong> recommended for production — lowest MAPE (19.1%) and MAE ($18,868). "
            "SARIMA provides useful 95% confidence intervals as a sanity-check.")

# ───────────────────────────────────────────────────────────────────────────
# TAB 3 — ANOMALY REPORT
# ───────────────────────────────────────────────────────────────────────────
with t3:
    an     = anomalies_df.copy()
    iso_c  = an.get("isolation_forest_flag", pd.Series(False, index=an.index))
    z_c    = an.get("zscore_flag",           pd.Series(False, index=an.index))
    n_iso  = int((iso_c==True).sum())
    n_z    = int((z_c==True).sum())
    n_both = int(((iso_c==True)&(z_c==True)).sum())

    kpi_row(
        kpi("🔴","Isolation Forest flags", str(n_iso),  "Multi-feature outlier detection","#fb7185"),
        kpi("🟡","Z-score flags",          str(n_z),    "±2σ from 6-week rolling mean",  "#fbbf24"),
        kpi("🔗","Both methods agree",      str(n_both), "Highest-confidence anomalies",  "#a78bfa"),
    )

    weekly = df.set_index("Order Date")["Sales"].resample("W").sum().reset_index()
    weekly.columns = ["week","sales"]
    iso_f = an[iso_c==True]; z_f = an[z_c==True]

    chart_title("Weekly Sales · Anomaly Overlay","#fb7185")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["week"], y=weekly["sales"], mode="lines", name="Weekly Sales",
        line=dict(color="#60a5fa", width=1.5), fill="tozeroy", fillcolor="rgba(96,165,250,0.06)",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>$%{y:,.0f}<extra></extra>"))
    if not iso_f.empty:
        fig.add_trace(go.Scatter(x=iso_f["week"], y=iso_f["sales"], mode="markers", name="Isolation Forest",
            marker=dict(color="#fb7185", size=11, symbol="circle", line=dict(width=2, color="#080d1a")),
            hovertemplate="<b>ISO FLAG</b><br>%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))
    if not z_f.empty:
        fig.add_trace(go.Scatter(x=z_f["week"], y=z_f["sales"], mode="markers", name="Z-Score",
            marker=dict(color="#fbbf24", size=12, symbol="x", line=dict(width=2.5)),
            hovertemplate="<b>Z FLAG</b><br>%{x|%d %b %Y}<br>$%{y:,.0f}<extra></extra>"))
    st.plotly_chart(th(fig), use_container_width=True)

    an["week_str"]  = an["week"].dt.strftime("%d %b %Y")
    an["sales_fmt"] = an["sales"].map(lambda v: f"${v:,.0f}")
    an_rows = ""
    for _, row in an.sort_values("week").iterrows():
        iso_flag = bool(row.get("isolation_forest_flag", False))
        z_flag   = bool(row.get("zscore_flag", False))
        if iso_flag and z_flag:   pill = "<span class='pill p-both'>Both ✓</span>"
        elif iso_flag:            pill = "<span class='pill p-iso'>Isolation Forest</span>"
        else:                     pill = "<span class='pill p-z'>Z-Score</span>"
        an_rows += f"<tr><td>{row['week_str']}</td><td class='fcv'>{row['sales_fmt']}</td><td>{pill}</td></tr>"

    card_table("Detected Anomaly Weeks",
        f"<table class='dt'><thead><tr><th>Week</th><th>Sales</th><th>Detection Method</th></tr></thead>"
        f"<tbody>{an_rows}</tbody></table>",
        dot="#fbbf24")

    insight("<strong>Agreement = highest priority.</strong> Isolation Forest detects multi-feature outliers; "
            "Z-score flags local single-week deviations. Weeks flagged by <strong>both</strong> are the "
            "strongest signals and should be investigated first.", warn=True)

# ───────────────────────────────────────────────────────────────────────────
# TAB 4 — DEMAND SEGMENTS
# ───────────────────────────────────────────────────────────────────────────
with t4:
    COLOR_MAP = {
        "High Volume, Stable Demand":  "#60a5fa",
        "Growing Demand":              "#34d399",
        "Declining Demand":            "#fb7185",
        "Low Volume, High Volatility": "#fbbf24",
    }
    CHIP_MAP = {
        "High Volume, Stable Demand":  ("c-stable",  "📦"),
        "Growing Demand":              ("c-grow",    "🚀"),
        "Declining Demand":            ("c-decline", "📉"),
        "Low Volume, High Volatility": ("c-volatile","⚡"),
    }
    cc_counts = clusters_df.groupby("cluster_label").size().to_dict()

    kpi_row(
        kpi("📦","High Volume, Stable",     f"{cc_counts.get('High Volume, Stable Demand',0)} sub-cats",  "Revenue backbone",  "#60a5fa"),
        kpi("🚀","Growing Demand",           f"{cc_counts.get('Growing Demand',0)} sub-cats",              "Rising fast",       "#34d399"),
        kpi("📉","Declining Demand",         f"{cc_counts.get('Declining Demand',0)} sub-cats",            "Needs clearance",   "#fb7185"),
        kpi("⚡","Low Vol, High Volatility", f"{cc_counts.get('Low Volume, High Volatility',0)} sub-cats", "Erratic, needs VMS","#fbbf24"),
    )

    left, right = st.columns([1.5,1])
    with left:
        chart_title("Demand Clusters — PCA Space","#22d3ee")
        fig = px.scatter(clusters_df, x="pca1", y="pca2", color="cluster_label", text="Sub-Category",
            color_discrete_map=COLOR_MAP,
            hover_data={"total_sales_volume":True,"yoy_growth_pct_2015_2018":True,"avg_order_value":True,"pca1":False,"pca2":False})
        fig.update_traces(textposition="top center", textfont=dict(size=11, color="#e8eef8"),
            marker=dict(size=14, line=dict(width=2, color="#080d1a")))
        st.plotly_chart(th(fig), use_container_width=True)

    with right:
        chart_title("Revenue by Cluster","#a78bfa")
        rcl = clusters_df.groupby("cluster_label")["total_sales_volume"].sum().reset_index()
        rcl.columns = ["Cluster","Revenue"]
        fig = px.bar(rcl, x="Revenue", y="Cluster", orientation="h",
            color="Cluster", color_discrete_map=COLOR_MAP, text_auto=".2s")
        fig.update_traces(marker_line_width=0, textfont_color="#e8eef8", showlegend=False)
        st.plotly_chart(th(fig), use_container_width=True)

        chart_title("YoY Growth by Cluster","#fbbf24")
        gcl = clusters_df.groupby("cluster_label")["yoy_growth_pct_2015_2018"].mean().reset_index()
        gcl.columns = ["Cluster","Growth %"]
        fig2 = px.bar(gcl, x="Growth %", y="Cluster", orientation="h",
            color="Cluster", color_discrete_map=COLOR_MAP, text_auto=".1f")
        fig2.update_traces(marker_line_width=0, textfont_color="#e8eef8", showlegend=False)
        st.plotly_chart(th(fig2), use_container_width=True)

    # Sub-category table
    cl_rows = ""
    for _, row in clusters_df.sort_values("cluster_label").iterrows():
        chip_cls, chip_icon = CHIP_MAP.get(row["cluster_label"], ("c-stable",""))
        gv = row["yoy_growth_pct_2015_2018"]
        gc = "#34d399" if gv > 0 else "#fb7185"
        gs = "+" if gv > 0 else ""
        cl_rows += (f"<tr>"
                    f"<td><strong>{row['Sub-Category']}</strong></td>"
                    f"<td class='mono'>${row['total_sales_volume']:,.0f}</td>"
                    f"<td style='font-family:JetBrains Mono,monospace;color:{gc}'>{gs}{gv:.1f}%</td>"
                    f"<td class='mono'>${row['avg_order_value']:,.0f}</td>"
                    f"<td style='font-family:JetBrains Mono,monospace;color:#7a90b8'>${row['sales_volatility']:,.0f}</td>"
                    f"<td><span class='chip {chip_cls}'>{chip_icon} {row['cluster_label']}</span></td>"
                    f"</tr>")
    card_table("Sub-Category Detail",
        f"<table class='dt'><thead><tr>"
        f"<th>Sub-Category</th><th>Total Revenue</th><th>YoY Growth</th>"
        f"<th>Avg Order</th><th>Volatility</th><th>Cluster</th>"
        f"</tr></thead><tbody>{cl_rows}</tbody></table>",
        dot="#34d399")

    section("Stocking Strategy")
    st.markdown("""
<div class="strat-grid">
  <div class="strat-card" style="border-left:3px solid #60a5fa">
    <h4 style="color:#60a5fa">📦 High Volume, Stable Demand</h4>
    <p>Standard reorder points. Safe for automated replenishment. Demand is predictable enough for the XGBoost model to run with minimal oversight.</p>
  </div>
  <div class="strat-card" style="border-left:3px solid #34d399">
    <h4 style="color:#34d399">🚀 Growing Demand</h4>
    <p>Increase safety stock ahead of season. Revisit reorder thresholds quarterly so growth does not outrun supply.</p>
  </div>
  <div class="strat-card" style="border-left:3px solid #fb7185">
    <h4 style="color:#fb7185">📉 Declining Demand</h4>
    <p>Avoid large forward purchase commitments. Clear excess inventory via targeted promotions rather than reordering at historical volumes.</p>
  </div>
  <div class="strat-card" style="border-left:3px solid #fbbf24">
    <h4 style="color:#fbbf24">⚡ Low Volume, High Volatility</h4>
    <p>Smaller, more frequent reorders over large batches. Consider vendor-managed inventory to avoid tying up capital in unpredictable stock.</p>
  </div>
</div>
""", unsafe_allow_html=True)