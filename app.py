import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_curve, auc,
                              mean_absolute_error, r2_score, confusion_matrix,
                              silhouette_score)
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Customer Intelligence",
    page_icon="🇦🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --brand-red:    #E8341C;
    --brand-orange: #F97316;
    --brand-amber:  #FBBF24;
    --dark-bg:      #0D0D0D;
    --card-bg:      #161616;
    --card-border:  #2A2A2A;
    --text-primary: #F5F5F5;
    --text-muted:   #888888;
    --gradient:     linear-gradient(135deg, #E8341C 0%, #F97316 50%, #FBBF24 100%);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark-bg) !important;
    color: var(--text-primary) !important;
}
.stApp { background-color: var(--dark-bg) !important; }

[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid var(--card-border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.hero {
    background: var(--gradient);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "🇦🇪";
    position: absolute;
    right: 48px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.25;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    color: #fff !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -1px;
}
.hero p {
    color: rgba(255,255,255,0.85) !important;
    font-size: 1.05rem !important;
    margin: 0 !important;
    font-weight: 300;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--card-border);
    margin-left: 8px;
}

.kpi-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gradient);
}
.kpi-card:hover { border-color: var(--brand-orange); }
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 6px;
}

.upload-zone {
    background: var(--card-bg);
    border: 2px dashed var(--card-border);
    border-radius: 16px;
    padding: 48px;
    text-align: center;
}

.chart-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
}

[data-testid="stFileUploader"] {
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
}

[data-testid="stMetricDelta"] { color: var(--brand-orange) !important; }
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

.stButton>button {
    background: var(--gradient) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton>button:hover { opacity: 0.88 !important; }

[data-baseweb="tab-list"] { background: var(--card-bg) !important; border-radius: 10px !important; padding: 4px !important; }
[data-baseweb="tab"] { color: var(--text-muted) !important; font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; }
[aria-selected="true"] { background: var(--brand-red) !important; color: white !important; border-radius: 8px !important; }

[data-baseweb="select"] > div { background: var(--card-bg) !important; border-color: var(--card-border) !important; }
.stSlider [data-baseweb="slider"] { color: var(--brand-orange) !important; }

hr { border-color: var(--card-border) !important; }

.pred-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    font-weight: 700;
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
}
.pred-yes { background: rgba(232,52,28,0.2); color: #F97316; border: 1px solid var(--brand-orange); }
.pred-no  { background: rgba(100,100,100,0.2); color: #888; border: 1px solid #444; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--dark-bg); }
::-webkit-scrollbar-thumb { background: var(--card-border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#F5F5F5"),
    margin=dict(l=0, r=0, t=36, b=0),
    colorway=["#E8341C", "#F97316", "#FBBF24", "#FB923C", "#FCA5A5"],
    xaxis=dict(gridcolor="#2A2A2A", showline=False),
    yaxis=dict(gridcolor="#2A2A2A", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def kpi(label, value, sub=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""

def section(icon, title):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def card_chart(fig, height=380):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def is_string_col(series):
    """Detect string/object columns — works on both pandas 2.x and 3.x."""
    return pd.api.types.is_string_dtype(series) or series.dtype == "object"

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>UAE Customer Intelligence</h1>
    <p>ML-powered segmentation · purchase prediction · spend forecasting</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
        <span style='font-family:Syne;font-size:1.1rem;font-weight:700;
                     background:linear-gradient(135deg,#E8341C,#FBBF24);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        ⚙ Controls
        </span>
    </div>""", unsafe_allow_html=True)

    n_clusters = st.slider("Number of Segments", 2, 8, 4)
    test_size  = st.slider("Test Split %", 10, 40, 20)
    n_trees    = st.selectbox("Random Forest Trees", [50, 100, 200, 500], index=1)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem;color:#666;line-height:1.6'>
    <b style='color:#F97316'>Required columns:</b><br>
    • <code>Will_Buy</code> — classification target<br>
    • <code>Max_Spend</code> — regression target<br>
    • Any other numeric / categorical features
    </div>""", unsafe_allow_html=True)

# ── File upload ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📂  Upload your customer dataset (CSV)", type=["csv"])

if not uploaded_file:
    st.markdown("""
    <div class="upload-zone">
        <div style='font-size:3rem;margin-bottom:12px'>📊</div>
        <div style='font-family:Syne;font-size:1.2rem;font-weight:700;color:#F5F5F5;margin-bottom:8px'>
            Drop your CSV to begin
        </div>
        <div style='color:#666;font-size:0.9rem'>
            Needs <code>Will_Buy</code> and <code>Max_Spend</code> columns
        </div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load & encode ─────────────────────────────────────────────────────────────
df = pd.read_csv(uploaded_file)

# ── FIX: use pd.api.types.is_string_dtype() to catch both pandas 2.x "object"
#         and pandas 3.x "str" dtype — the old `dtype == "object"` check silently
#         skipped string columns in pandas 3.x, leaving raw strings in X_train
#         and causing RandomForest to crash with a NumPy conversion error.
df_encoded = df.copy()
le_dict = {}
for col in df_encoded.columns:
    if is_string_col(df_encoded[col]):
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le

# Drop any non-numeric columns that slipped through (safety net)
non_numeric_remaining = df_encoded.select_dtypes(exclude="number").columns.tolist()
if non_numeric_remaining:
    df_encoded = df_encoded.drop(columns=non_numeric_remaining)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab_eda, tab_diag, tab_pred, tab_presc, tab2, tab3, tab4, tab5, tab_arima, tab_assoc, tab_whatif = st.tabs([
    "📋 Overview",
    "🔍 EDA",
    "🩺 Diagnostic",
    "📈 Predictive",
    "🧭 Prescriptive",
    "🎯 Classification",
    "👥 Segmentation",
    "💰 Spend Forecast",
    "🔮 Predict New",
    "📉 ARIMA Forecast",
    "🔗 Association Rules",
    "🎛 What-If Simulator",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    section("📋", "Dataset Overview")
    total     = len(df)
    buyers    = int(df["Will_Buy"].sum()) if "Will_Buy" in df.columns else "—"
    buy_rate  = f"{buyers/total*100:.1f}%" if isinstance(buyers, int) else "—"
    avg_spend = f"AED {df['Max_Spend'].mean():,.0f}" if "Max_Spend" in df.columns else "—"
    n_feat    = df.shape[1]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi("Total Customers", f"{total:,}", "in dataset"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Likely Buyers", f"{buyers:,}" if isinstance(buyers,int) else buyers, f"{buy_rate} conversion rate"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Avg Max Spend", avg_spend, "per customer"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Features", f"{n_feat}", "columns detected"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])
    with col_l:
        section("📊", "Feature Distributions")
        num_cols_ov = df.select_dtypes(include=np.number).columns.tolist()
        sel = st.selectbox("Select feature", num_cols_ov, label_visibility="collapsed")
        fig = px.histogram(df, x=sel, nbins=30, color_discrete_sequence=["#F97316"])
        fig.update_traces(marker_line_width=0)
        card_chart(fig, 320)
    with col_r:
        section("🥧", "Buy Intent Split")
        if "Will_Buy" in df.columns:
            vc = df["Will_Buy"].value_counts().reset_index()
            vc.columns = ["Will_Buy", "Count"]
            vc["Label"] = vc["Will_Buy"].map({1: "Will Buy", 0: "Won't Buy"})
            fig2 = px.pie(vc, names="Label", values="Count",
                          color_discrete_sequence=["#E8341C", "#2A2A2A"], hole=0.55)
            fig2.update_traces(textfont_color="white", pull=[0.04, 0])
            card_chart(fig2, 320)
    section("🗂", "Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB EDA — EXPLORATORY DATA ANALYSIS (with Drilled-Down Analysis)
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    section("🔍", "Exploratory Data Analysis")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if is_string_col(df[c])]

    total_cells  = df.shape[0] * df.shape[1]
    missing_vals = df.isnull().sum().sum()
    dup_rows     = df.duplicated().sum()
    completeness = (1 - missing_vals / total_cells) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi("Rows", f"{df.shape[0]:,}", "records"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Columns", f"{df.shape[1]}", f"{len(num_cols)} numeric · {len(cat_cols)} categorical"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Missing Values", f"{missing_vals:,}", f"{missing_vals/total_cells*100:.1f}% of cells"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Completeness", f"{completeness:.1f}%", f"{dup_rows} duplicate rows"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    missing_per_col = df.isnull().sum()
    missing_per_col = missing_per_col[missing_per_col > 0]
    if len(missing_per_col) > 0:
        section("🕳", "Missing Values by Column")
        miss_df = missing_per_col.reset_index()
        miss_df.columns = ["Column", "Missing"]
        miss_df["Pct"] = (miss_df["Missing"] / len(df) * 100).round(2)
        fig_miss = px.bar(miss_df, x="Column", y="Pct", color="Pct",
                          color_continuous_scale=["#F97316", "#E8341C"], labels={"Pct": "Missing (%)"})
        fig_miss.update_layout(coloraxis_showscale=False)
        card_chart(fig_miss, 280)
    else:
        st.markdown("""
        <div style='background:#161616;border:1px solid #2A2A2A;border-radius:12px;
                    padding:20px 24px;color:#F97316;font-weight:600;text-align:center;margin-bottom:24px;'>
            ✅ No missing values detected — dataset is complete!
        </div>""", unsafe_allow_html=True)

    section("📐", "Descriptive Statistics")
    desc = df[num_cols].describe().T.round(3)
    desc.index.name = "Feature"
    st.dataframe(desc.reset_index(), use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    section("📊", "Distribution Explorer")
    col_l, col_r = st.columns(2)
    with col_l:
        sel_num = st.selectbox("Numeric feature", num_cols, key="eda_num")
        fig_hist = px.histogram(df, x=sel_num, nbins=35, marginal="box",
                                color_discrete_sequence=["#F97316"])
        fig_hist.update_traces(marker_line_width=0)
        card_chart(fig_hist, 360)
    with col_r:
        if cat_cols:
            sel_cat = st.selectbox("Categorical feature", cat_cols, key="eda_cat")
            vc = df[sel_cat].value_counts().head(15).reset_index()
            vc.columns = [sel_cat, "Count"]
            fig_cat = px.bar(vc, x=sel_cat, y="Count", color="Count",
                             color_continuous_scale=["#2A2A2A", "#E8341C", "#FBBF24"])
            fig_cat.update_layout(coloraxis_showscale=False)
            card_chart(fig_cat, 360)
        else:
            st.info("No categorical columns found.")

    section("🌡", "Correlation Heatmap")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().round(2)
        fig_corr = px.imshow(corr, color_continuous_scale=["#0D0D0D", "#E8341C", "#FBBF24"],
                             zmin=-1, zmax=1, text_auto=True, aspect="auto")
        fig_corr.update_traces(textfont_size=11)
        card_chart(fig_corr, max(350, len(num_cols) * 45))

    section("🔗", "Pairwise Scatter Explorer")
    col_l2, col_r2 = st.columns(2)
    with col_l2:
        x_feat = st.selectbox("X axis", num_cols, index=0, key="scat_x")
    with col_r2:
        y_feat = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="scat_y")
    color_feat = df["Will_Buy"].astype(str) if "Will_Buy" in df.columns else None
    fig_scat = px.scatter(df, x=x_feat, y=y_feat, color=color_feat,
                          color_discrete_sequence=["#E8341C", "#FBBF24"], opacity=0.7,
                          labels={"color": "Will Buy"})
    x_vals = pd.to_numeric(df[x_feat], errors="coerce").dropna()
    y_vals = pd.to_numeric(df[y_feat], errors="coerce").loc[x_vals.index]
    if len(x_vals) > 1:
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        fig_scat.add_trace(go.Scatter(x=x_line, y=m*x_line+b, mode="lines", name="Trend",
                                      line=dict(color="#F97316", width=2, dash="dash")))
    fig_scat.update_traces(marker=dict(size=7))
    card_chart(fig_scat, 400)

    if "Will_Buy" in df.columns and len(num_cols) > 0:
        section("📦", "Feature Distribution by Purchase Intent")
        sel_box = st.selectbox("Feature to compare", num_cols, key="eda_box")
        df_box = df.copy()
        df_box["Will_Buy_Label"] = df_box["Will_Buy"].map({1: "Will Buy ✓", 0: "Won't Buy ✗"})
        fig_box = px.box(df_box, x="Will_Buy_Label", y=sel_box, color="Will_Buy_Label",
                         color_discrete_sequence=["#E8341C", "#2A2A2A"], points="outliers")
        fig_box.update_layout(showlegend=False)
        card_chart(fig_box, 380)

    if "Will_Buy" in df.columns and cat_cols:
        section("🏷", "Categorical Breakdown vs Purchase Intent")
        sel_cat2 = st.selectbox("Categorical feature", cat_cols, key="eda_cat2")
        grp = df.groupby([sel_cat2, "Will_Buy"]).size().reset_index(name="Count")
        grp["Will_Buy_Label"] = grp["Will_Buy"].map({1: "Will Buy", 0: "Won't Buy"})
        fig_grp = px.bar(grp, x=sel_cat2, y="Count", color="Will_Buy_Label", barmode="group",
                         color_discrete_sequence=["#E8341C", "#2A2A2A"], labels={"Will_Buy_Label": ""})
        card_chart(fig_grp, 380)

    section("⚠️", "Outlier Detection (IQR Method)")
    outlier_summary = []
    for col in num_cols:
        Q1 = df[col].quantile(0.25); Q3 = df[col].quantile(0.75); IQR = Q3 - Q1
        n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
        outlier_summary.append({"Feature": col, "Outliers": int(n_out),
                                 "Outlier %": round(n_out/len(df)*100, 2),
                                 "Q1": round(Q1,2), "Q3": round(Q3,2), "IQR": round(IQR,2)})
    out_df = pd.DataFrame(outlier_summary).sort_values("Outliers", ascending=False)
    col_l3, col_r3 = st.columns([2, 3])
    with col_l3:
        st.dataframe(out_df, use_container_width=True)
    with col_r3:
        fig_out = px.bar(out_df[out_df["Outliers"]>0], x="Feature", y="Outlier %",
                         color="Outlier %", color_continuous_scale=["#F97316","#E8341C"])
        fig_out.update_layout(coloraxis_showscale=False)
        card_chart(fig_out, 320)

    # ── DRILLED-DOWN ANALYSIS ────────────────────────────────────────────────
    section("🔬", "Drilled-Down Cross-Segment Analysis")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #FBBF24;border-radius:8px;
                padding:14px 18px;margin-bottom:22px;color:#aaa;font-size:0.88rem;'>
    Select two categorical dimensions and a numeric metric to explore cross-segment patterns as a heatmap.
    </div>""", unsafe_allow_html=True)

    if len(cat_cols) >= 2:
        dd_c1, dd_c2, dd_c3 = st.columns(3)
        with dd_c1: drill_cat1 = st.selectbox("Primary Dimension", cat_cols, index=0, key="drill1")
        with dd_c2: drill_cat2 = st.selectbox("Secondary Dimension", cat_cols, index=1, key="drill2")
        with dd_c3: drill_metric = st.selectbox("Metric", num_cols, key="drill_m")
        drill_grp = df.groupby([drill_cat1, drill_cat2])[drill_metric].mean().reset_index()
        drill_pivot = drill_grp.pivot(index=drill_cat1, columns=drill_cat2, values=drill_metric).fillna(0).round(2)
        fig_drill = px.imshow(drill_pivot, color_continuous_scale=["#0D0D0D","#E8341C","#FBBF24"],
                              text_auto=".1f", aspect="auto", labels={"color": drill_metric})
        fig_drill.update_traces(textfont_size=11)
        card_chart(fig_drill, max(350, len(drill_pivot)*40))

        section("🌞", "Sunburst Drill-Down")
        sb_c1, sb_c2, sb_c3 = st.columns(3)
        with sb_c1: sb_p1 = st.selectbox("Level 1", cat_cols, index=0, key="sb1")
        with sb_c2: sb_p2 = st.selectbox("Level 2", cat_cols, index=1, key="sb2")
        with sb_c3: sb_v  = st.selectbox("Value", num_cols, key="sb_v")
        fig_sun = px.sunburst(df, path=[sb_p1, sb_p2], values=sb_v,
                              color=sb_v, color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"])
        card_chart(fig_sun, 480)

        section("📊", "Cross-Segment Summary Statistics")
        cs_c1, cs_c2 = st.columns(2)
        with cs_c1: cs_cat = st.selectbox("Group by", cat_cols, key="cs_cat")
        with cs_c2: cs_m   = st.multiselect("Metrics", num_cols, default=num_cols[:min(4,len(num_cols))], key="cs_m")
        if cs_m:
            cs_tbl = df.groupby(cs_cat)[cs_m].agg(["mean","median","std"]).round(2)
            cs_tbl.columns = [f"{c[0]}_{c[1]}" for c in cs_tbl.columns]
            st.dataframe(cs_tbl.reset_index(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
with tab_diag:
    section("🩺", "Diagnostic Analysis — Why Is It Happening?")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #E8341C;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>What is Diagnostic Analysis?</b><br>
    Investigates <em>why</em> patterns happen — drilling into root causes of purchase intent
    and spending behaviour across demographics, channels, and cultural events.
    </div>""", unsafe_allow_html=True)

    num_cols_d = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols_d = [c for c in df.columns if is_string_col(df[c])]

    if "Will_Buy" in df.columns and "Max_Spend" in df.columns:
        buyers_df    = df[df["Will_Buy"]==1]
        non_buyers   = df[df["Will_Buy"]==0]
        avg_spend_b  = buyers_df["Max_Spend"].mean()
        avg_spend_nb = non_buyers["Max_Spend"].mean()
        spend_lift   = ((avg_spend_b - avg_spend_nb)/avg_spend_nb*100)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi("Buyer Avg Spend",     f"AED {avg_spend_b:,.0f}", "who will buy"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Non-Buyer Avg Spend", f"AED {avg_spend_nb:,.0f}","who won't buy"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Spend Lift",          f"+{spend_lift:.1f}%",     "buyers vs non-buyers"), unsafe_allow_html=True)
        with c4: st.markdown(kpi("Buy Rate",            f"{len(buyers_df)/len(df)*100:.1f}%","overall"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    section("📊", "Purchase Rate by Key Segment")
    if "Will_Buy" in df.columns and cat_cols_d:
        diag_cat = st.selectbox("Select segment to diagnose", cat_cols_d, key="diag_cat")
        grp = df.groupby(diag_cat)["Will_Buy"].agg(["mean","count"]).reset_index()
        grp.columns = [diag_cat,"Buy_Rate","Count"]
        grp["Buy_Rate_Pct"] = (grp["Buy_Rate"]*100).round(1)
        grp = grp.sort_values("Buy_Rate_Pct", ascending=False)
        col_l,col_r = st.columns(2)
        with col_l:
            fig_dr = px.bar(grp, x=diag_cat, y="Buy_Rate_Pct", color="Buy_Rate_Pct",
                            color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                            labels={"Buy_Rate_Pct":"Buy Rate (%)"})
            fig_dr.update_layout(coloraxis_showscale=False)
            card_chart(fig_dr, 360)
        with col_r:
            fig_cnt = px.bar(grp, x=diag_cat, y="Count", color="Count",
                             color_continuous_scale=["#2A2A2A","#F97316","#FBBF24"],
                             labels={"Count":"Number of Customers"})
            fig_cnt.update_layout(coloraxis_showscale=False)
            card_chart(fig_cnt, 360)

    if "Preferred_Channel" in df.columns and "Max_Spend" in df.columns:
        section("🛒", "Average Spend by Shopping Channel")
        ch = df.groupby("Preferred_Channel")["Max_Spend"].mean().reset_index()
        ch.columns = ["Channel","Avg_Spend"]
        fig_ch = px.bar(ch, x="Channel", y="Avg_Spend", color="Avg_Spend",
                        color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                        labels={"Avg_Spend":"Avg Max Spend (AED)"})
        fig_ch.update_layout(coloraxis_showscale=False)
        card_chart(fig_ch, 300)

    if "Monthly_Income_AED" in df.columns and "Max_Spend" in df.columns:
        section("💰", "Income vs Spending Capacity")
        color_diag = df["Will_Buy"].astype(str) if "Will_Buy" in df.columns else None
        fig_inc = px.scatter(df, x="Monthly_Income_AED", y="Max_Spend", color=color_diag,
                             color_discrete_sequence=["#E8341C","#FBBF24"], opacity=0.5,
                             labels={"Monthly_Income_AED":"Monthly Income (AED)",
                                     "Max_Spend":"Max Spend (AED)","color":"Will Buy"})
        fig_inc.update_traces(marker=dict(size=5))
        card_chart(fig_inc, 400)


# ══════════════════════════════════════════════════════════════════════════════
# TAB PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    section("📈", "Predictive Analysis — What Will Happen?")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #F97316;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>What is Predictive Analysis?</b><br>
    Uses ML to forecast which customers will buy, how much they will spend, and surface high-value targets.
    </div>""", unsafe_allow_html=True)

    if "Will_Buy" in df_encoded.columns:
        X_p = df_encoded.drop(columns=[c for c in ["Will_Buy","Max_Spend"] if c in df_encoded.columns])
        y_p = df_encoded["Will_Buy"]
        Xp_tr,Xp_te,yp_tr,yp_te = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
        clf_p = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_p.fit(Xp_tr, yp_tr)
        yp_pred  = clf_p.predict(Xp_te)
        yp_proba = clf_p.predict_proba(Xp_te)[:,1]

        acc_p  = accuracy_score(yp_te, yp_pred)
        prec_p = precision_score(yp_te, yp_pred, zero_division=0)
        rec_p  = recall_score(yp_te, yp_pred, zero_division=0)
        f1_p   = f1_score(yp_te, yp_pred, zero_division=0)

        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi("Model Accuracy", f"{acc_p*100:.1f}%", "correctly predicted"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Precision", f"{prec_p*100:.1f}%","predicted buyers correct"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Recall", f"{rec_p*100:.1f}%","actual buyers caught"), unsafe_allow_html=True)
        with c4: st.markdown(kpi("F1 Score", f"{f1_p:.3f}","harmonic mean"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_l,col_r = st.columns(2)
        with col_l:
            section("📉", "ROC Curve")
            fpr_p,tpr_p,_ = roc_curve(yp_te, yp_proba)
            roc_p = auc(fpr_p, tpr_p)
            fig_rp = go.Figure()
            fig_rp.add_trace(go.Scatter(x=fpr_p, y=tpr_p, mode="lines",
                name=f"AUC = {roc_p:.3f}", line=dict(color="#F97316",width=3),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.15)"))
            fig_rp.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                name="Random",line=dict(color="#444",width=1.5,dash="dash")))
            fig_rp.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
            card_chart(fig_rp, 360)
        with col_r:
            section("🟥", "Confusion Matrix")
            cm_vals = confusion_matrix(yp_te, yp_pred)
            fig_cm = px.imshow(cm_vals, labels=dict(x="Predicted",y="Actual",color="Count"),
                x=["Predicted No","Predicted Yes"], y=["Actual No","Actual Yes"],
                color_continuous_scale=["#0D0D0D","#E8341C","#FBBF24"], text_auto=True)
            fig_cm.update_traces(textfont_size=18)
            card_chart(fig_cm, 360)

        section("🏆", "Top Predictive Features")
        fi_p = pd.DataFrame({"Feature":X_p.columns,"Importance":clf_p.feature_importances_})
        fi_p = fi_p.sort_values("Importance",ascending=True).tail(15)
        fig_fi_p = px.bar(fi_p, x="Importance", y="Feature", orientation="h",
                          color="Importance", color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"])
        fig_fi_p.update_layout(coloraxis_showscale=False, yaxis_title="")
        card_chart(fig_fi_p, 420)

        section("⭐", "Top 20 High-Intent, High-Value Customers")
        pred_all  = clf_p.predict_proba(X_p)[:,1]
        df_ranked = df.copy()
        df_ranked["Buy_Probability_%"] = (pred_all*100).round(1)
        if "Max_Spend" in df_ranked.columns:
            df_ranked["Revenue_Potential_AED"] = (df_ranked["Buy_Probability_%"]/100 * df_ranked["Max_Spend"]).round(0)
            show_cols = [c for c in ["Customer_ID","Age","Gender","City","Preferred_Category",
                                      "Monthly_Income_AED","Buy_Probability_%","Max_Spend",
                                      "Revenue_Potential_AED"] if c in df_ranked.columns]
            st.dataframe(df_ranked.nlargest(20,"Revenue_Potential_AED")[show_cols], use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB PRESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_presc:
    section("🧭", "Prescriptive Analysis — What Should We Do?")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #FBBF24;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>What is Prescriptive Analysis?</b><br>
    Recommends specific actions based on diagnostic and predictive findings —
    concrete, GEO-informed marketing strategies for each UAE customer segment.
    </div>""", unsafe_allow_html=True)

    if "Will_Buy" in df_encoded.columns:
        X_pr = df_encoded.drop(columns=[c for c in ["Will_Buy","Max_Spend"] if c in df_encoded.columns])
        clf_pr = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_pr.fit(X_pr, df_encoded["Will_Buy"])
        all_proba = clf_pr.predict_proba(X_pr)[:,1]
        df_pr = df.copy()
        df_pr["Buy_Probability"] = all_proba
        med_prob  = df_pr["Buy_Probability"].median()
        med_spend = df_pr["Max_Spend"].median() if "Max_Spend" in df_pr.columns else 5000

        def assign_quadrant(row):
            hp = row["Buy_Probability"] >= med_prob
            hs = row.get("Max_Spend",0) >= med_spend
            if hp and hs:     return "High Intent · High Value"
            if hp and not hs: return "High Intent · Budget"
            if not hp and hs: return "Low Intent · High Value"
            return "Low Intent · Low Value"

        df_pr["Strategy_Segment"] = df_pr.apply(assign_quadrant, axis=1)
        seg_counts = df_pr["Strategy_Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]

        section("🗺", "Customer Strategy Matrix")
        col_l,col_r = st.columns([2,3])
        icons_map = {"High Intent · High Value":"🏆","High Intent · Budget":"🎯",
                     "Low Intent · High Value":"💎","Low Intent · Low Value":"💤"}
        with col_l:
            for _,row in seg_counts.iterrows():
                pct = row["Count"]/len(df_pr)*100
                ic = icons_map.get(row["Segment"],"•")
                st.markdown(f"""
                <div class='kpi-card' style='margin-bottom:12px;text-align:left;padding:16px 18px;'>
                    <div style='font-size:1rem;font-weight:700;color:#F5F5F5;margin-bottom:4px'>{ic} {row['Segment']}</div>
                    <div style='color:#F97316;font-family:Syne;font-size:1.5rem;font-weight:800'>{row['Count']:,}</div>
                    <div style='color:#666;font-size:0.78rem'>{pct:.1f}% of customers</div>
                </div>""", unsafe_allow_html=True)
        with col_r:
            fig_quad = px.scatter(df_pr, x="Buy_Probability", y="Max_Spend",
                color="Strategy_Segment",
                color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#444"],
                opacity=0.65, labels={"Buy_Probability":"Buy Probability","Max_Spend":"Max Spend (AED)"})
            fig_quad.add_vline(x=med_prob,line_color="#555",line_dash="dash",annotation_text="Median Probability")
            fig_quad.add_hline(y=med_spend,line_color="#555",line_dash="dash",annotation_text="Median Spend")
            fig_quad.update_traces(marker=dict(size=5))
            card_chart(fig_quad, 420)

        section("💰", "Budget Allocation")
        col_l2,col_r2 = st.columns(2)
        with col_l2:
            budget_data = pd.DataFrame({
                "Segment":  ["High Intent High Value","High Intent Budget","Low Intent High Value","Low Intent Low Value"],
                "Budget_%": [45, 30, 20, 5]})
            fig_bud = px.pie(budget_data, names="Segment", values="Budget_%",
                             color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#333"], hole=0.52)
            fig_bud.update_traces(textfont_color="white", pull=[0.05,0.03,0.02,0])
            card_chart(fig_bud, 380)
        with col_r2:
            section("📡", "Channel Priority (1=Low, 5=High)")
            channel_df = pd.DataFrame({
                "Channel": ["Influencer/TikTok","Email Campaigns","BNPL Promotions","DSF/Ramadan Events","Retargeting Ads","GEO/AI Content"],
                "High Intent High Value": [5,4,2,5,3,5],
                "High Intent Budget":     [3,5,5,4,4,3],
                "Low Intent High Value":  [3,4,2,5,5,4],
                "Low Intent Low Value":   [1,2,1,2,1,3]})
            fig_ch2 = px.bar(channel_df, x="Channel",
                             y=["High Intent High Value","High Intent Budget","Low Intent High Value","Low Intent Low Value"],
                             barmode="group",
                             color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#444"],
                             labels={"value":"Priority (1-5)","variable":"Segment"})
            card_chart(fig_ch2, 380)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLASSIFICATION (All algorithms + comparison)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    section("🎯", "Classification — All Algorithms Compared")

    if "Will_Buy" not in df_encoded.columns:
        st.warning("Column `Will_Buy` not found.")
    else:
        X = df_encoded.drop(columns=["Will_Buy"])
        if "Max_Spend" in X.columns: X = X.drop(columns=["Max_Spend"])
        y = df_encoded["Will_Buy"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # Scale for distance-based models
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        # ── Define all classifiers ──
        classifiers = {
            "Random Forest":       RandomForestClassifier(n_estimators=n_trees, random_state=42),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
            "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree":       DecisionTreeClassifier(random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
            "Naive Bayes":         GaussianNB(),
            "SVM (RBF)":           SVC(probability=True, random_state=42),
        }

        results_list = []
        roc_traces   = []
        trained_clfs = {}

        with st.spinner("Training all classifiers… this may take ~20 seconds"):
            for name, model in classifiers.items():
                use_scaled = name in ["Logistic Regression","K-Nearest Neighbors","SVM (RBF)"]
                Xtr = X_tr_sc if use_scaled else X_train
                Xte = X_te_sc if use_scaled else X_test
                model.fit(Xtr, y_train)
                yp  = model.predict(Xte)
                ypr = model.predict_proba(Xte)[:,1]
                trained_clfs[name] = (model, use_scaled)
                results_list.append({
                    "Model":     name,
                    "Accuracy":  round(accuracy_score(y_test, yp)*100, 2),
                    "Precision": round(precision_score(y_test, yp, zero_division=0)*100, 2),
                    "Recall":    round(recall_score(y_test, yp, zero_division=0)*100, 2),
                    "F1 Score":  round(f1_score(y_test, yp, zero_division=0)*100, 2),
                })
                fpr_m, tpr_m, _ = roc_curve(y_test, ypr)
                roc_auc_m = auc(fpr_m, tpr_m)
                roc_traces.append((name, fpr_m, tpr_m, roc_auc_m))

        res_df = pd.DataFrame(results_list).sort_values("F1 Score", ascending=False)
        best_model_name = res_df.iloc[0]["Model"]

        # ── KPIs for best model ──
        best_row = res_df.iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.markdown(kpi("Best Model", best_model_name.split()[0], "by F1 Score"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Accuracy",  f"{best_row['Accuracy']:.1f}%","top model"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("F1 Score",  f"{best_row['F1 Score']:.1f}%","harmonic mean"), unsafe_allow_html=True)
        with c4: st.markdown(kpi("Models Run", f"{len(classifiers)}","algorithms"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Comparison Table ──
        section("📊", "Algorithm Comparison Table")
        res_styled = res_df.copy()
        st.dataframe(res_styled.set_index("Model"), use_container_width=True)

        # ── Grouped bar comparison chart ──
        section("📈", "Accuracy · Precision · Recall · F1 Comparison")
        comp_melt = res_df.melt(id_vars="Model", var_name="Metric", value_name="Score (%)")
        fig_comp = px.bar(comp_melt, x="Model", y="Score (%)", color="Metric", barmode="group",
                          color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#FB923C"])
        fig_comp.update_layout(xaxis_tickangle=-30)
        card_chart(fig_comp, 400)

        # ── ROC curves for all models ──
        section("📉", "ROC Curves — All Models")
        fig_roc_all = go.Figure()
        colors_roc = ["#E8341C","#F97316","#FBBF24","#FB923C","#FCA5A5","#FDE68A","#4ADE80","#60A5FA"]
        for i,(name,fpr_m,tpr_m,roc_auc_m) in enumerate(roc_traces):
            fig_roc_all.add_trace(go.Scatter(x=fpr_m, y=tpr_m, mode="lines",
                name=f"{name} (AUC={roc_auc_m:.3f})",
                line=dict(color=colors_roc[i % len(colors_roc)], width=2)))
        fig_roc_all.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",
            line=dict(color="#444",width=1.5,dash="dash")))
        fig_roc_all.update_layout(xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
        card_chart(fig_roc_all, 460)

        # ── Confusion matrix for selected model ──
        section("🟥", "Confusion Matrix — Select Model")
        sel_model_name = st.selectbox("Choose model", list(classifiers.keys()), key="clf_cm_sel")
        sel_model, sel_scaled = trained_clfs[sel_model_name]
        Xte_sel = X_te_sc if sel_scaled else X_test
        yp_sel  = sel_model.predict(Xte_sel)
        cm_sel  = confusion_matrix(y_test, yp_sel)
        fig_cm_sel = px.imshow(cm_sel, labels=dict(x="Predicted",y="Actual",color="Count"),
            x=["No","Yes"], y=["No","Yes"],
            color_continuous_scale=["#0D0D0D","#E8341C","#FBBF24"], text_auto=True)
        fig_cm_sel.update_traces(textfont_size=20)
        col_cm, col_fi = st.columns(2)
        with col_cm:
            card_chart(fig_cm_sel, 360)
        with col_fi:
            section("🏆", f"Feature Importance — {sel_model_name}")
            if hasattr(sel_model, "feature_importances_"):
                fi = pd.DataFrame({"Feature":X.columns,"Importance":sel_model.feature_importances_})
                fi = fi.sort_values("Importance",ascending=True).tail(12)
                fig_fi_sel = px.bar(fi, x="Importance", y="Feature", orientation="h",
                                    color="Importance",
                                    color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"])
                fig_fi_sel.update_layout(coloraxis_showscale=False, yaxis_title="")
                card_chart(fig_fi_sel, 360)
            else:
                st.info(f"{sel_model_name} does not expose feature importances.")

        # Store best RF clf for Predict New tab
        clf = trained_clfs["Random Forest"][0]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENTATION (Elbow, Silhouette, Hierarchical + Dendrogram)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    section("👥", "Customer Segmentation")

    X_clust = df_encoded.drop(columns=[c for c in ["Will_Buy","Max_Spend"] if c in df_encoded.columns])

    # ── Elbow Method ──────────────────────────────────────────────────────────
    section("📐", "Elbow Method — Optimal k")
    max_k = 10
    inertias, sil_scores_k = [], []
    k_range = range(2, max_k+1)
    with st.spinner("Computing Elbow & Silhouette scores…"):
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbls = km.fit_predict(X_clust)
            inertias.append(km.inertia_)
            sil_scores_k.append(silhouette_score(X_clust, lbls))

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
            line=dict(color="#F97316",width=3), marker=dict(size=9,color="#FBBF24"),
            name="Inertia"))
        fig_elbow.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia (WCSS)")
        card_chart(fig_elbow, 340)
    with col_e2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=list(k_range), y=sil_scores_k, mode="lines+markers",
            line=dict(color="#E8341C",width=3), marker=dict(size=9,color="#FBBF24"),
            name="Silhouette Score"))
        best_k = list(k_range)[sil_scores_k.index(max(sil_scores_k))]
        fig_sil.add_vline(x=best_k, line_color="#FBBF24", line_dash="dash",
                          annotation_text=f"Best k={best_k}")
        fig_sil.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Silhouette Score")
        card_chart(fig_sil, 340)

    st.markdown(f"""
    <div style='background:var(--card-bg);border:1px solid #2A2A2A;border-radius:10px;
                padding:14px 20px;margin-bottom:20px;color:#aaa;font-size:0.88rem;'>
    📌 <b style='color:#FBBF24'>Recommendation:</b> Silhouette score peaks at
    <b style='color:#F97316'>k = {best_k}</b> — this is the statistically optimal number of clusters.
    Your sidebar slider is currently set to <b style='color:#F97316'>k = {n_clusters}</b>.
    </div>""", unsafe_allow_html=True)

    # ── K-Means with chosen k ─────────────────────────────────────────────────
    section("🗺", "K-Means Clustering")
    kmeans   = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_clust)
    df_seg   = df.copy()
    df_seg["Segment"] = [f"Segment {i+1}" for i in clusters]

    sil_val = silhouette_score(X_clust, clusters)
    pca     = PCA(n_components=2, random_state=42)
    comps   = pca.fit_transform(X_clust)

    st.markdown(f"""
    <div style='background:var(--card-bg);border:1px solid #2A2A2A;border-radius:10px;
                padding:12px 18px;margin-bottom:18px;color:#aaa;font-size:0.88rem;'>
    Current k={n_clusters} → <b style='color:#F97316'>Silhouette Score: {sil_val:.4f}</b>
    (range −1 to +1; higher = better-defined clusters)
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])
    with col_l:
        section("🗺", "PCA Cluster Map")
        fig_pca = px.scatter(x=comps[:,0], y=comps[:,1], color=df_seg["Segment"],
            color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#FB923C",
                                     "#FCA5A5","#FDE68A","#FED7AA","#FDBA74"],
            labels={"x":"PCA 1","y":"PCA 2"}, opacity=0.8)
        fig_pca.update_traces(marker=dict(size=8))
        card_chart(fig_pca, 400)
    with col_r:
        section("📊", "Segment Sizes")
        seg_counts = df_seg["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Count"]
        fig_bar = px.bar(seg_counts, x="Segment", y="Count", color="Segment",
            color_discrete_sequence=["#E8341C","#F97316","#FBBF24","#FB923C",
                                     "#FCA5A5","#FDE68A","#FED7AA","#FDBA74"])
        fig_bar.update_layout(showlegend=False)
        card_chart(fig_bar, 400)

    section("📋", "Segment Profiles")
    profile = df_seg.groupby("Segment").mean(numeric_only=True).round(2).reset_index()
    st.dataframe(profile, use_container_width=True)

    if "Max_Spend" in df_seg.columns:
        section("💰", "Avg Spend by Segment")
        spend_seg = df_seg.groupby("Segment")["Max_Spend"].mean().reset_index()
        fig_sp = px.bar(spend_seg, x="Segment", y="Max_Spend", color="Max_Spend",
                        color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                        labels={"Max_Spend":"Avg Max Spend (AED)"})
        fig_sp.update_layout(coloraxis_showscale=False)
        card_chart(fig_sp, 300)

    # ── Hierarchical Clustering + Dendrogram ──────────────────────────────────
    section("🌳", "Hierarchical Clustering — Dendrogram Validation")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #F97316;border-radius:8px;
                padding:14px 18px;margin-bottom:20px;color:#aaa;font-size:0.88rem;'>
    The dendrogram shows how customers merge into clusters step-by-step.
    Long vertical lines indicate well-separated, natural cluster boundaries.
    </div>""", unsafe_allow_html=True)

    hc_col1, hc_col2 = st.columns(2)
    with hc_col1: hc_method  = st.selectbox("Linkage Method", ["ward","complete","average","single"], key="hc_method")
    with hc_col2: hc_sample  = st.slider("Sample size for dendrogram", 50, 300, 100, key="hc_sample")

    X_hc_sample = X_clust.sample(hc_sample, random_state=42)
    Z = linkage(X_hc_sample.values, method=hc_method)

    # Build dendrogram using plotly
    from scipy.cluster.hierarchy import dendrogram as sp_dendrogram
    ddata = sp_dendrogram(Z, no_plot=True, truncate_mode="level", p=5)

    fig_dend = go.Figure()
    for i, d in zip(ddata["icoord"], ddata["dcoord"]):
        fig_dend.add_trace(go.Scatter(x=i, y=d, mode="lines",
            line=dict(color="#F97316",width=1.5), showlegend=False))
    fig_dend.update_layout(
        xaxis=dict(showticklabels=False, title="Customers"),
        yaxis=dict(title="Distance"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F5F5F5"), margin=dict(l=0,r=0,t=36,b=0), height=420)
    st.plotly_chart(fig_dend, use_container_width=True, config={"displayModeBar": False})

    # Agglomerative vs KMeans comparison
    section("🔀", "Agglomerative vs K-Means Silhouette Comparison")
    agg_sil_scores = []
    km_sil_scores  = []
    k_vals = list(range(2, 8))
    with st.spinner("Comparing clustering methods…"):
        for k in k_vals:
            km_l = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_clust)
            ag_l = AgglomerativeClustering(n_clusters=k).fit_predict(X_clust)
            km_sil_scores.append(round(silhouette_score(X_clust, km_l),4))
            agg_sil_scores.append(round(silhouette_score(X_clust, ag_l),4))

    fig_hc_comp = go.Figure()
    fig_hc_comp.add_trace(go.Scatter(x=k_vals, y=km_sil_scores, mode="lines+markers",
        name="K-Means", line=dict(color="#F97316",width=3), marker=dict(size=8)))
    fig_hc_comp.add_trace(go.Scatter(x=k_vals, y=agg_sil_scores, mode="lines+markers",
        name="Agglomerative", line=dict(color="#E8341C",width=3), marker=dict(size=8,symbol="square")))
    fig_hc_comp.update_layout(xaxis_title="k", yaxis_title="Silhouette Score",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#F5F5F5"), height=340,
                               xaxis=dict(gridcolor="#2A2A2A"), yaxis=dict(gridcolor="#2A2A2A"),
                               legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_hc_comp, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SPEND FORECAST (Random Forest regression)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    section("💰", "Spending Forecast Model (Random Forest Regressor)")

    if "Max_Spend" not in df_encoded.columns:
        st.warning("Column `Max_Spend` not found.")
    else:
        y_reg = df_encoded["Max_Spend"]
        X_reg = df_encoded.drop(columns=[c for c in ["Max_Spend","Will_Buy"] if c in df_encoded.columns])
        Xr_tr,Xr_te,yr_tr,yr_te = train_test_split(X_reg, y_reg, test_size=test_size/100, random_state=42)
        reg = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        reg.fit(Xr_tr, yr_tr)
        preds = reg.predict(Xr_te)
        mae  = mean_absolute_error(yr_te, preds)
        r2   = r2_score(yr_te, preds)
        rmse = np.sqrt(((preds - yr_te)**2).mean())

        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(kpi("R² Score", f"{r2:.3f}","variance explained"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("MAE", f"AED {mae:,.0f}","mean abs error"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("RMSE", f"AED {rmse:,.0f}","root mean sq error"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_l,col_r = st.columns(2)
        with col_l:
            section("🎯", "Actual vs Predicted")
            fig_av = go.Figure()
            fig_av.add_trace(go.Scatter(x=yr_te.values, y=preds, mode="markers",
                marker=dict(color="#F97316",size=7,opacity=0.7), name="Predictions"))
            mn,mx = float(yr_te.min()),float(yr_te.max())
            fig_av.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode="lines",
                line=dict(color="#FBBF24",width=2,dash="dash"), name="Perfect fit"))
            fig_av.update_layout(xaxis_title="Actual Spend (AED)",yaxis_title="Predicted Spend (AED)")
            card_chart(fig_av, 380)
        with col_r:
            section("📉", "Residual Distribution")
            residuals = preds - yr_te.values
            fig_res = px.histogram(x=residuals, nbins=35,
                                   color_discrete_sequence=["#E8341C"], labels={"x":"Residual (AED)"})
            fig_res.add_vline(x=0, line_color="#FBBF24", line_dash="dash")
            card_chart(fig_res, 380)

        section("🏆", "Feature Importance — Spend")
        fi_r = pd.DataFrame({"Feature":X_reg.columns,"Importance":reg.feature_importances_})
        fi_r = fi_r.sort_values("Importance",ascending=True).tail(12)
        fig_fir = px.bar(fi_r, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"])
        fig_fir.update_layout(coloraxis_showscale=False, yaxis_title="")
        card_chart(fig_fir, 340)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT NEW
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    section("🔮", "Predict New Customers")
    st.markdown("""
    <div style='background:var(--card-bg);border:1px solid var(--card-border);
                border-radius:12px;padding:20px 24px;margin-bottom:24px;
                font-size:0.88rem;color:#888;'>
    Upload a CSV with the same columns as your training data
    (excluding <code>Will_Buy</code> and <code>Max_Spend</code>).
    </div>""", unsafe_allow_html=True)

    new_file = st.file_uploader("📂  Upload new customer data", key="new_pred", type=["csv"])
    if new_file and "Will_Buy" in df_encoded.columns:
        new_df = pd.read_csv(new_file)
        for col in new_df.columns:
            if col in le_dict:
                known = set(le_dict[col].classes_)
                new_df[col] = new_df[col].astype(str).apply(
                    lambda x: x if x in known else le_dict[col].classes_[0])
                new_df[col] = le_dict[col].transform(new_df[col].astype(str))
        X_new = new_df.reindex(columns=clf.feature_names_in_, fill_value=0)
        buy_pred  = clf.predict(X_new)
        buy_proba = clf.predict_proba(X_new)[:,1]
        results = new_df.copy()
        results["Will_Buy_Prediction"] = buy_pred
        results["Buy_Probability"]     = (buy_proba*100).round(1)
        n_new = len(results); n_buyers = int(buy_pred.sum()); avg_prob = buy_proba.mean()*100
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(kpi("Customers Scored",f"{n_new:,}","records"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("Predicted Buyers",f"{n_buyers:,}",f"{n_buyers/n_new*100:.1f}%"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Avg Buy Prob",f"{avg_prob:.1f}%","confidence"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        section("📊", "Buy Probability Distribution")
        fig_np = px.histogram(x=buy_proba*100, nbins=30,
            color_discrete_sequence=["#F97316"], labels={"x":"Buy Probability (%)"})
        fig_np.add_vline(x=50, line_color="#FBBF24", line_dash="dash", annotation_text="50% threshold")
        card_chart(fig_np, 280)
        section("📋", "Prediction Results")
        st.dataframe(results, use_container_width=True)
        st.download_button("⬇  Download Predictions CSV", results.to_csv(index=False).encode(),
                           "predictions.csv", "text/csv")
    elif new_file:
        st.warning("Train the classification model first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB ARIMA — TIME SERIES FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab_arima:
    section("📉", "ARIMA Time-Series Forecasting")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #E8341C;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>ARIMA (Auto-Regressive Integrated Moving Average)</b> forecasts future values
    of a numeric metric by modelling its temporal patterns. Here we simulate a monthly time series
    from the customer dataset and project it forward.
    </div>""", unsafe_allow_html=True)

    num_cols_a = df.select_dtypes(include=np.number).columns.tolist()

    ar_c1, ar_c2, ar_c3 = st.columns(3)
    with ar_c1: arima_metric = st.selectbox("Metric to forecast", num_cols_a, key="arima_m",
                                             index=num_cols_a.index("Max_Spend") if "Max_Spend" in num_cols_a else 0)
    with ar_c2: arima_periods = st.slider("Forecast periods (months)", 3, 24, 12, key="arima_p")
    with ar_c3:
        ar_order_p = st.selectbox("AR order (p)", [1,2,3,4,5], index=1, key="arima_op")
        ar_order_q = st.selectbox("MA order (q)", [0,1,2,3], index=1, key="arima_oq")

    # Build synthetic monthly series from data — group by row index bins
    n_months = min(48, len(df) // 20)
    bins = pd.cut(df.index, bins=n_months, labels=False)
    ts_data = df.groupby(bins)[arima_metric].mean().dropna()
    ts_data.index = pd.date_range(start="2021-01-01", periods=len(ts_data), freq="ME")

    # Fit ARIMA
    try:
        with st.spinner("Fitting ARIMA model…"):
            model_arima = ARIMA(ts_data, order=(ar_order_p, 1, ar_order_q))
            res_arima   = model_arima.fit()
            forecast    = res_arima.get_forecast(steps=arima_periods)
            fc_mean     = forecast.predicted_mean
            fc_ci       = forecast.conf_int()

        # KPIs
        c1,c2,c3 = st.columns(3)
        with c1: st.markdown(kpi("AIC",  f"{res_arima.aic:.1f}","lower = better fit"), unsafe_allow_html=True)
        with c2: st.markdown(kpi("BIC",  f"{res_arima.bic:.1f}","penalises complexity"), unsafe_allow_html=True)
        with c3: st.markdown(kpi("Forecast Horizon", f"{arima_periods} months","ahead"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Plot
        fig_arima = go.Figure()
        fig_arima.add_trace(go.Scatter(
            x=ts_data.index, y=ts_data.values,
            mode="lines+markers", name="Historical",
            line=dict(color="#F97316", width=2.5),
            marker=dict(size=6)))
        fig_arima.add_trace(go.Scatter(
            x=fc_mean.index, y=fc_mean.values,
            mode="lines+markers", name="Forecast",
            line=dict(color="#FBBF24", width=2.5, dash="dash"),
            marker=dict(size=7, symbol="diamond")))
        fig_arima.add_trace(go.Scatter(
            x=list(fc_ci.index) + list(fc_ci.index[::-1]),
            y=list(fc_ci.iloc[:,0]) + list(fc_ci.iloc[:,1][::-1]),
            fill="toself", fillcolor="rgba(251,191,36,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
        fig_arima.add_vline(x=str(ts_data.index[-1]), line_color="#555", line_dash="dash",
                            annotation_text="Forecast start")
        fig_arima.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F5F5F5"), height=440,
            xaxis=dict(gridcolor="#2A2A2A", title="Month"),
            yaxis=dict(gridcolor="#2A2A2A", title=arima_metric),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=36,b=0))
        st.plotly_chart(fig_arima, use_container_width=True, config={"displayModeBar": False})

        # Forecast table
        section("📋", "Forecast Values")
        fc_df = pd.DataFrame({
            "Month":    fc_mean.index.strftime("%b %Y"),
            "Forecast": fc_mean.values.round(2),
            "Lower CI": fc_ci.iloc[:,0].values.round(2),
            "Upper CI": fc_ci.iloc[:,1].values.round(2)})
        st.dataframe(fc_df, use_container_width=True)

        section("📉", "ARIMA Residuals")
        resid = res_arima.resid
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_resid = px.line(x=resid.index, y=resid.values,
                                color_discrete_sequence=["#F97316"],
                                labels={"x":"Month","y":"Residual"})
            fig_resid.add_hline(y=0, line_color="#FBBF24", line_dash="dash")
            card_chart(fig_resid, 300)
        with col_r2:
            fig_resid_hist = px.histogram(x=resid.values, nbins=20,
                color_discrete_sequence=["#E8341C"], labels={"x":"Residual Value"})
            card_chart(fig_resid_hist, 300)

    except Exception as e:
        st.error(f"ARIMA fitting failed: {e}. Try adjusting the p/q parameters.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB ASSOCIATION RULES (with scatter plot)
# ══════════════════════════════════════════════════════════════════════════════
with tab_assoc:
    section("🔗", "Association Rule Mining")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #F97316;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>Association Rule Mining</b> discovers co-occurrence patterns —
    e.g. which customer traits or preferences tend to appear together.
    Uses <b>Apriori</b> on binarised categorical features.
    </div>""", unsafe_allow_html=True)

    cat_cols_a = [c for c in df.columns if is_string_col(df[c])]
    num_cols_a2 = df.select_dtypes(include=np.number).columns.tolist()

    if len(cat_cols_a) < 2:
        st.warning("Need at least 2 categorical columns for association mining.")
    else:
        ar_c1, ar_c2, ar_c3 = st.columns(3)
        with ar_c1:
            ar_cats = st.multiselect("Columns to mine", cat_cols_a,
                                     default=cat_cols_a[:min(5,len(cat_cols_a))], key="ar_cats")
        with ar_c2:
            min_sup  = st.slider("Min Support", 0.05, 0.5, 0.15, 0.01, key="ar_sup")
        with ar_c3:
            min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05, key="ar_conf")

        if ar_cats:
            # Build basket (one-hot encoded transactions)
            basket_df = pd.get_dummies(df[ar_cats].astype(str), prefix_sep="=")
            basket_df = basket_df.astype(bool)

            try:
                with st.spinner("Running Apriori…"):
                    freq_items = apriori(basket_df, min_support=min_sup, use_colnames=True)
                    if len(freq_items) == 0:
                        st.warning("No frequent itemsets found — try lowering minimum support.")
                    else:
                        rules = association_rules(freq_items, metric="confidence",
                                                  min_threshold=min_conf, num_itemsets=len(freq_items))
                        rules = rules.sort_values("lift", ascending=False).head(50)
                        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

                        # KPIs
                        c1,c2,c3 = st.columns(3)
                        with c1: st.markdown(kpi("Frequent Itemsets", f"{len(freq_items):,}","found"), unsafe_allow_html=True)
                        with c2: st.markdown(kpi("Rules Generated", f"{len(rules):,}","after filtering"), unsafe_allow_html=True)
                        with c3: st.markdown(kpi("Max Lift", f"{rules['lift'].max():.2f}","top rule"), unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

                        # Rules table
                        section("📋", "Top Association Rules")
                        show_rules = rules[["antecedents_str","consequents_str",
                                            "support","confidence","lift"]].copy()
                        show_rules.columns = ["Antecedents","Consequents","Support","Confidence","Lift"]
                        show_rules = show_rules.round(4)
                        st.dataframe(show_rules, use_container_width=True)

                        # ── Scatter plot: Support vs Confidence (size=Lift) ──
                        section("📍", "Support vs Confidence Scatter (bubble size = Lift)")
                        fig_ar_scat = px.scatter(
                            rules,
                            x="support", y="confidence", size="lift",
                            color="lift",
                            color_continuous_scale=["#2A2A2A","#E8341C","#FBBF24"],
                            hover_data={"antecedents_str": True, "consequents_str": True,
                                        "lift": ":.3f", "support": ":.3f", "confidence": ":.3f"},
                            labels={"support":"Support","confidence":"Confidence",
                                    "lift":"Lift","antecedents_str":"IF","consequents_str":"THEN"},
                            size_max=40)
                        fig_ar_scat.update_traces(marker=dict(opacity=0.8, line=dict(width=1,color="#111")))
                        fig_ar_scat.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#F5F5F5"), height=460,
                            xaxis=dict(gridcolor="#2A2A2A"),
                            yaxis=dict(gridcolor="#2A2A2A"),
                            coloraxis_colorbar=dict(title="Lift"),
                            margin=dict(l=0,r=0,t=36,b=0))
                        st.plotly_chart(fig_ar_scat, use_container_width=True,
                                        config={"displayModeBar": False})

                        # ── Lift heatmap (top rules) ──
                        section("🌡", "Lift Heatmap — Top Antecedent × Consequent Pairs")
                        top_rules = rules.head(20)
                        pivot_lift = top_rules.pivot_table(
                            index="antecedents_str", columns="consequents_str",
                            values="lift", aggfunc="max").fillna(0).round(3)
                        if not pivot_lift.empty:
                            fig_lift_heat = px.imshow(
                                pivot_lift,
                                color_continuous_scale=["#0D0D0D","#E8341C","#FBBF24"],
                                text_auto=".2f", aspect="auto",
                                labels={"color":"Lift"})
                            fig_lift_heat.update_traces(textfont_size=10)
                            card_chart(fig_lift_heat, max(300, len(pivot_lift)*45))

            except Exception as e:
                st.error(f"Association mining error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_whatif:
    section("🎛", "What-If Simulator — Predictive Outcomes")
    st.markdown("""
    <div style='background:var(--card-bg);border-left:4px solid #FBBF24;border-radius:8px;
                padding:16px 20px;margin-bottom:28px;color:#aaa;font-size:0.9rem;line-height:1.7'>
    <b style='color:#F5F5F5'>What-If Simulator</b> — Adjust customer profile attributes using
    dropdowns and sliders, then instantly see the predicted <b>purchase probability</b> and
    <b>expected spend</b> from the trained models.
    </div>""", unsafe_allow_html=True)

    if "Will_Buy" not in df_encoded.columns:
        st.warning("Train the models by uploading a dataset first.")
    else:
        # Build models for simulator
        X_sim = df_encoded.drop(columns=[c for c in ["Will_Buy","Max_Spend"] if c in df_encoded.columns])
        y_sim_clf = df_encoded["Will_Buy"]

        @st.cache_resource
        def get_sim_models(_X, _y_clf, _y_reg):
            clf_s  = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_s.fit(_X, _y_clf)
            reg_s = None
            if _y_reg is not None:
                reg_s = RandomForestRegressor(n_estimators=100, random_state=42)
                reg_s.fit(_X, _y_reg)
            return clf_s, reg_s

        y_sim_reg = df_encoded["Max_Spend"] if "Max_Spend" in df_encoded.columns else None
        clf_sim, reg_sim = get_sim_models(X_sim, y_sim_clf, y_sim_reg)

        st.markdown("### 🧑 Build a Customer Profile")
        sim_input = {}

        # ── Categorical fields → dropdowns ──
        cat_sim_cols = [c for c in df.columns if is_string_col(df[c]) and c in X_sim.columns]
        num_sim_cols = [c for c in df.select_dtypes(include=np.number).columns
                        if c not in ["Will_Buy","Max_Spend"] and c in X_sim.columns]

        # Layout: 3 columns for dropdowns
        if cat_sim_cols:
            st.markdown("**Categorical Attributes**")
            cat_chunks = [cat_sim_cols[i:i+3] for i in range(0, len(cat_sim_cols), 3)]
            for chunk in cat_chunks:
                cols_row = st.columns(len(chunk))
                for ci, col_name in enumerate(chunk):
                    with cols_row[ci]:
                        opts = sorted(df[col_name].dropna().unique().tolist())
                        default_idx = 0
                        if col_name in le_dict:
                            sim_input[col_name] = st.selectbox(
                                col_name, opts, index=default_idx, key=f"wi_{col_name}")
                        else:
                            sim_input[col_name] = st.selectbox(
                                col_name, opts, index=default_idx, key=f"wi_{col_name}")

        # ── Numeric fields → sliders ──
        if num_sim_cols:
            st.markdown("**Numeric Attributes**")
            num_chunks = [num_sim_cols[i:i+3] for i in range(0, len(num_sim_cols), 3)]
            for chunk in num_chunks:
                cols_row = st.columns(len(chunk))
                for ci, col_name in enumerate(chunk):
                    with cols_row[ci]:
                        col_min = float(df[col_name].min())
                        col_max = float(df[col_name].max())
                        col_med = float(df[col_name].median())
                        if df[col_name].dtype in [np.float64, float]:
                            sim_input[col_name] = st.slider(
                                col_name, col_min, col_max, col_med,
                                step=round((col_max-col_min)/100, 2),
                                key=f"wi_{col_name}")
                        else:
                            sim_input[col_name] = st.slider(
                                col_name, int(col_min), int(col_max), int(col_med),
                                key=f"wi_{col_name}")

        st.markdown("---")

        if st.button("🔮  Run Prediction", key="wi_run"):
            # Encode categorical inputs
            row_dict = {}
            for col_name, val in sim_input.items():
                if col_name in le_dict:
                    known = set(le_dict[col_name].classes_)
                    v = str(val) if str(val) in known else le_dict[col_name].classes_[0]
                    row_dict[col_name] = le_dict[col_name].transform([v])[0]
                else:
                    row_dict[col_name] = val

            input_df = pd.DataFrame([row_dict]).reindex(columns=X_sim.columns, fill_value=0)

            buy_prob   = clf_sim.predict_proba(input_df)[0,1] * 100
            buy_label  = "Will Buy ✅" if buy_prob >= 50 else "Won't Buy ❌"
            spend_pred = reg_sim.predict(input_df)[0] if reg_sim is not None else None

            # Results
            st.markdown("<br>", unsafe_allow_html=True)
            section("📊", "Simulation Results")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(kpi("Purchase Probability", f"{buy_prob:.1f}%",
                                "above 50% = likely buyer"), unsafe_allow_html=True)
            with r2:
                badge_cls = "pred-yes" if buy_prob >= 50 else "pred-no"
                st.markdown(f"""
                <div class='kpi-card' style='padding-top:32px;'>
                    <div class='kpi-label'>Prediction</div>
                    <span class='pred-badge {badge_cls}'>{buy_label}</span>
                </div>""", unsafe_allow_html=True)
            with r3:
                if spend_pred is not None:
                    st.markdown(kpi("Expected Spend", f"AED {spend_pred:,.0f}",
                                    "model estimate"), unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=buy_prob,
                delta={"reference": 50, "valueformat": ".1f"},
                title={"text": "Purchase Probability (%)", "font": {"color": "#F5F5F5"}},
                gauge={
                    "axis": {"range":[0,100], "tickcolor":"#888"},
                    "bar":  {"color": "#F97316"},
                    "bgcolor": "#161616",
                    "steps": [
                        {"range":[0,50],  "color":"#1A1A1A"},
                        {"range":[50,100],"color":"#2A1A0A"}],
                    "threshold": {
                        "line": {"color":"#FBBF24","width":3},
                        "thickness":0.75, "value":50}}))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5F5F5"), height=320,
                margin=dict(l=40,r=40,t=40,b=20))
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

            # Scenario comparison: vary one feature
            section("🔄", "Sensitivity Analysis — Vary a Feature")
            vary_num = [c for c in num_sim_cols if c in input_df.columns]
            if vary_num:
                vary_feat = st.selectbox("Feature to vary", vary_num, key="wi_vary")
                vary_range = np.linspace(float(df[vary_feat].min()),
                                         float(df[vary_feat].max()), 30)
                vary_probs = []
                for v in vary_range:
                    tmp = input_df.copy()
                    tmp[vary_feat] = v
                    vary_probs.append(clf_sim.predict_proba(tmp)[0,1]*100)

                fig_vary = go.Figure()
                fig_vary.add_trace(go.Scatter(x=vary_range, y=vary_probs, mode="lines+markers",
                    line=dict(color="#F97316",width=2.5), marker=dict(size=6,color="#FBBF24"),
                    name="Buy Probability"))
                fig_vary.add_hline(y=50, line_color="#E8341C", line_dash="dash",
                                   annotation_text="50% decision threshold")
                fig_vary.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#F5F5F5"), height=320,
                    xaxis=dict(gridcolor="#2A2A2A", title=vary_feat),
                    yaxis=dict(gridcolor="#2A2A2A", title="Buy Probability (%)"),
                    margin=dict(l=0,r=0,t=36,b=0))
                st.plotly_chart(fig_vary, use_container_width=True,
                                config={"displayModeBar": False})
        else:
            st.info("👆 Configure the customer profile above and click **Run Prediction**.")

