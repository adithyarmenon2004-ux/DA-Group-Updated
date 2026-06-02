import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="UAE Customer Analytics", layout="wide", page_icon="🏙️")

st.markdown("""
<style>
/* ── General ── */
.block-container { padding-top: 1rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color: #1e3a5f; }
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] { background-color: #2563eb; }
[data-testid="stSidebar"] label { color: #ffffff !important; font-weight: 600; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #cbd5e1 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #e2e8f0;
    padding: 5px 6px;
    border-radius: 10px;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    padding: 8px 14px !important;
    border-radius: 7px !important;
    background-color: #ffffff !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    border: 1px solid #cbd5e1 !important;
    white-space: nowrap;
}
.stTabs [aria-selected="true"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-color: #2563eb !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #dbeafe !important;
    color: #1e40af !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Metric cards ── */
.metric-card {
    background: #f0f7ff;
    border-radius: 10px;
    padding: 14px 18px;
    border-left: 4px solid #2563eb;
    margin-bottom: 8px;
}
.metric-card h3 { color: #64748b; font-size: 0.85rem; margin:0; }
.metric-card h1 { color: #1e293b; font-size: 1.8rem; margin:4px 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def generate_sample_data(n=2000):
    np.random.seed(42)
    emirates   = ["Dubai","Abu Dhabi","Sharjah","Ajman","Ras Al Khaimah","Fujairah","Umm Al Quwain"]
    categories = ["Electronics","Fashion","Home & Garden","Sports","Groceries","Beauty","Toys"]
    payment    = ["Credit Card","Debit Card","Cash","Digital Wallet"]
    segments   = ["Premium","Standard","Economy"]
    genders    = ["Male","Female"]

    df = pd.DataFrame({
        "customer_id":      range(1, n+1),
        "age":              np.random.randint(18, 70, n),
        "gender":           np.random.choice(genders, n),
        "emirate":          np.random.choice(emirates, n, p=[0.35,0.30,0.15,0.07,0.06,0.04,0.03]),
        "annual_income":    np.random.lognormal(10.8, 0.6, n).astype(int),
        "purchase_amount":  np.random.exponential(500, n).round(2),
        "num_purchases":    np.random.poisson(8, n),
        "product_category": np.random.choice(categories, n),
        "payment_method":   np.random.choice(payment, n),
        "customer_segment": np.random.choice(segments, n, p=[0.25,0.50,0.25]),
        "satisfaction_score": np.random.randint(1, 6, n),
        "churn":            np.random.choice([0,1], n, p=[0.75,0.25]),
        "tenure_months":    np.random.randint(1, 60, n),
        "date":             pd.date_range("2022-01-01", periods=n, freq="8h"),
        "loyalty_points":   np.random.randint(0, 5000, n),
        "discount_used":    np.random.choice([0,1], n, p=[0.60,0.40]),
        "return_customer":  np.random.choice([0,1], n, p=[0.50,0.50]),
    })
    mask = df["customer_segment"] == "Premium"
    df.loc[mask, "annual_income"]   = (df.loc[mask, "annual_income"]   * 1.5).astype(int)
    df.loc[mask, "purchase_amount"] = (df.loc[mask, "purchase_amount"] * 1.8).round(2)
    return df


@st.cache_data
def load_data():
    import glob, os
    for pattern in ["uae_customers*.csv", "*.csv"]:
        files = glob.glob(pattern)
        if files:
            try:
                df = pd.read_csv(files[0])
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                return df
            except Exception:
                pass
    return generate_sample_data()


df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏙️ UAE Customer Analytics")
    st.markdown("---")
    st.markdown("### 🔍 Global Filters")

    emirate_col = next((c for c in df.columns if c in ("emirate","city","region")), None)
    segment_col = next((c for c in df.columns if "segment" in c), None)
    gender_col  = next((c for c in df.columns if "gender"  in c), None)

    fdf = df.copy()

    if emirate_col:
        opts = sorted(df[emirate_col].dropna().unique().tolist())
        sel  = st.multiselect("Emirate", opts, default=opts, key="f_emirate")
        if sel:
            fdf = fdf[fdf[emirate_col].isin(sel)]

    if segment_col:
        opts = sorted(df[segment_col].dropna().unique().tolist())
        sel  = st.multiselect("Segment", opts, default=opts, key="f_segment")
        if sel:
            fdf = fdf[fdf[segment_col].isin(sel)]

    if gender_col:
        opts = sorted(df[gender_col].dropna().unique().tolist())
        sel  = st.multiselect("Gender", opts, default=opts, key="f_gender")
        if sel:
            fdf = fdf[fdf[gender_col].isin(sel)]

    st.markdown("---")
    st.markdown(f"**Showing:** {len(fdf):,} / {len(df):,} records")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — safe column pickers
# ─────────────────────────────────────────────────────────────────────────────
def num_cols(data):
    return data.select_dtypes(include=np.number).columns.tolist()

def cat_cols(data):
    return data.select_dtypes(include="object").columns.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_eda, tab_clf, tab_arima, tab_clust, tab_arm, tab_sim = st.tabs([
    "📊 EDA",
    "🤖 Classification",
    "📈 ARIMA Forecast",
    "🔗 Clustering",
    "🛒 Association Rules",
    "🎛️ What-If Simulator",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.header("📊 Exploratory Data Analysis")

    nc = num_cols(fdf)
    cc = cat_cols(fdf)

    if not nc:
        st.warning("No numeric columns found.")
        st.stop()

    # KPIs
    amount_col = next((c for c in nc if any(k in c for k in ("amount","purchase","spend","revenue"))), nc[0])
    churn_col  = next((c for c in fdf.columns if "churn" in c), None)
    age_col    = next((c for c in nc if "age" in c), None)
    income_col = next((c for c in nc if "income" in c), None)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",  f"{len(fdf):,}")
    k2.metric("Avg Purchase",     f"AED {fdf[amount_col].mean():,.0f}" if amount_col else "—")
    k3.metric("Churn Rate",       f"{fdf[churn_col].mean()*100:.1f}%" if churn_col else "—")
    k4.metric("Avg Income",       f"AED {fdf[income_col].mean():,.0f}" if income_col else "—")

    st.markdown("---")
    st.subheader("🔎 Drill-Down Analysis")

    if cc:
        da, db, dc = st.columns(3)
        with da:
            drill_primary = st.selectbox("Primary Dimension", cc, key="dp")
        with db:
            drill_secondary = st.selectbox("Secondary Breakdown", ["None"] + cc, key="ds")
        with dc:
            drill_metric = st.selectbox("Metric", nc, key="dm")

        agg_fn = st.radio("Aggregation", ["Mean","Sum","Count","Median"], horizontal=True, key="agg")
        agg_map = {"Mean":"mean","Sum":"sum","Count":"count","Median":"median"}
        ag = agg_map[agg_fn]

        try:
            if drill_secondary == "None":
                grp = fdf.groupby(drill_primary)[drill_metric].agg(ag).reset_index().sort_values(drill_metric, ascending=False)
                fig_drill = px.bar(grp, x=drill_primary, y=drill_metric, color=drill_primary,
                                   template="plotly_white", title=f"{agg_fn} of {drill_metric} by {drill_primary}")
            else:
                grp = fdf.groupby([drill_primary, drill_secondary])[drill_metric].agg(ag).reset_index()
                fig_drill = px.bar(grp, x=drill_primary, y=drill_metric, color=drill_secondary,
                                   barmode="group", template="plotly_white",
                                   title=f"{agg_fn} of {drill_metric} by {drill_primary} & {drill_secondary}")
            st.plotly_chart(fig_drill, use_container_width=True)
        except Exception as e:
            st.error(f"Drill-down error: {e}")

    st.markdown("---")

    # Distribution & Heatmap
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📉 Distribution")
        h_col   = st.selectbox("Column", nc, key="hcol")
        h_color = st.selectbox("Colour by", ["None"] + cc, key="hcolor")
        try:
            fig_hist = px.histogram(fdf, x=h_col,
                                    color=(None if h_color == "None" else h_color),
                                    nbins=40, marginal="box", template="plotly_white",
                                    title=f"Distribution: {h_col}")
            st.plotly_chart(fig_hist, use_container_width=True)
        except Exception as e:
            st.error(f"Histogram error: {e}")

    with c2:
        st.subheader("🌡️ Correlation Heatmap")
        corr_sel = st.multiselect("Columns", nc, default=nc[:min(8, len(nc))], key="corrsel")
        if len(corr_sel) >= 2:
            try:
                corr = fdf[corr_sel].corr()
                fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                     template="plotly_white", aspect="auto", title="Correlation Matrix")
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(f"Heatmap error: {e}")
        else:
            st.info("Select at least 2 columns.")

    # Scatter
    st.subheader("🔵 Scatter Plot")
    s1, s2, s3, s4 = st.columns(4)
    with s1: sx = st.selectbox("X", nc, key="scx")
    with s2: sy = st.selectbox("Y", nc, index=min(1,len(nc)-1), key="scy")
    with s3: sc_color = st.selectbox("Colour", ["None"]+cc, key="scc")
    with s4: sc_size  = st.selectbox("Size",   ["None"]+nc,  key="scs")
    try:
        fig_sc = px.scatter(fdf, x=sx, y=sy,
                            color=(None if sc_color=="None" else sc_color),
                            size=(None  if sc_size =="None" else sc_size),
                            opacity=0.6, template="plotly_white", title=f"{sx} vs {sy}")
        st.plotly_chart(fig_sc, use_container_width=True)
    except Exception as e:
        st.error(f"Scatter error: {e}")

    # Box plot
    st.subheader("📦 Box Plot")
    if cc:
        b1, b2 = st.columns(2)
        with b1: bn = st.selectbox("Numeric", nc, key="bnum")
        with b2: bc = st.selectbox("Category", cc, key="bcat")
        try:
            fig_box = px.box(fdf, x=bc, y=bn, color=bc, points="outliers",
                             template="plotly_white", title=f"{bn} by {bc}")
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception as e:
            st.error(f"Box plot error: {e}")

    # Pie chart
    st.subheader("🥧 Segment Distribution")
    if cc:
        pie_col = st.selectbox("Column", cc, key="pie")
        try:
            pie_data = fdf[pie_col].value_counts().reset_index()
            pie_data.columns = [pie_col, "count"]
            fig_pie = px.pie(pie_data, names=pie_col, values="count",
                             template="plotly_white", title=f"{pie_col} Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.error(f"Pie error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_clf:
    st.header("🤖 Classification — All Algorithms")

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, roc_auc_score)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                   AdaBoostClassifier, ExtraTreesClassifier)
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier

    nc_c = num_cols(fdf)
    cc_c = cat_cols(fdf)
    bin_c = [c for c in nc_c if fdf[c].nunique() == 2]
    target_opts = bin_c + cc_c

    if not target_opts:
        st.warning("No suitable target columns found in the dataset.")
    else:
        cl1, cl2 = st.columns(2)
        with cl1:
            clf_target = st.selectbox("Target variable", target_opts, key="clf_target")
        feat_pool = [c for c in nc_c if c != clf_target]
        with cl2:
            clf_features = st.multiselect("Feature columns", feat_pool,
                                          default=feat_pool[:min(6,len(feat_pool))],
                                          key="clf_feats")

        clf_test = st.slider("Test split %", 10, 40, 20, key="clf_test") / 100

        if st.button("🚀 Run All Classifiers", key="clf_run"):
            if not clf_features:
                st.warning("Please select at least one feature column.")
            else:
                with st.spinner("Training 10 classifiers — this may take ~30 seconds…"):
                    try:
                        tmp = fdf[clf_features + [clf_target]].dropna()
                        X   = tmp[clf_features].values
                        y   = tmp[clf_target].values
                        if tmp[clf_target].dtype == object:
                            le_y = LabelEncoder()
                            y    = le_y.fit_transform(y)
                        else:
                            le_y = None

                        strat = y if len(np.unique(y)) < 20 else None
                        X_tr, X_te, y_tr, y_te = train_test_split(
                            X, y, test_size=clf_test, random_state=42, stratify=strat)

                        sc = StandardScaler()
                        X_tr_s = sc.fit_transform(X_tr)
                        X_te_s = sc.transform(X_te)

                        multi = len(np.unique(y)) > 2
                        avg   = "weighted" if multi else "binary"

                        MODELS = {
                            "Logistic Regression":  (LogisticRegression(max_iter=1000, random_state=42), True),
                            "Decision Tree":        (DecisionTreeClassifier(random_state=42), False),
                            "Random Forest":        (RandomForestClassifier(100, random_state=42, n_jobs=-1), False),
                            "Gradient Boosting":    (GradientBoostingClassifier(random_state=42), False),
                            "AdaBoost":             (AdaBoostClassifier(random_state=42), False),
                            "Extra Trees":          (ExtraTreesClassifier(100, random_state=42, n_jobs=-1), False),
                            "SVM":                  (SVC(probability=True, random_state=42), True),
                            "KNN":                  (KNeighborsClassifier(5), True),
                            "Naive Bayes":          (GaussianNB(), False),
                            "Neural Network (MLP)": (MLPClassifier(max_iter=500, random_state=42), True),
                        }

                        results, cms = [], {}
                        trained = {}
                        for name, (mdl, scaled) in MODELS.items():
                            Xt = X_tr_s if scaled else X_tr
                            Xv = X_te_s if scaled else X_te
                            mdl.fit(Xt, y_tr)
                            pred = mdl.predict(Xv)
                            trained[name] = mdl
                            try:
                                prob = mdl.predict_proba(Xv)
                                auc  = roc_auc_score(y_te, prob if multi else prob[:,1],
                                                      multi_class="ovr" if multi else "raise")
                            except Exception:
                                auc = np.nan
                            results.append({
                                "Model":     name,
                                "Accuracy":  round(accuracy_score(y_te, pred), 4),
                                "Precision": round(precision_score(y_te, pred, average=avg, zero_division=0), 4),
                                "Recall":    round(recall_score(y_te, pred, average=avg, zero_division=0), 4),
                                "F1 Score":  round(f1_score(y_te, pred, average=avg, zero_division=0), 4),
                                "AUC-ROC":   round(auc, 4) if not np.isnan(auc) else "N/A",
                            })
                            cms[name] = confusion_matrix(y_te, pred)

                        res_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
                        res_df.index += 1

                        st.subheader("📋 Results Table")
                        st.dataframe(
                            res_df.style.background_gradient(
                                subset=["Accuracy","Precision","Recall","F1 Score"], cmap="Blues"),
                            use_container_width=True)

                        # Grouped bar
                        melt = res_df.melt(id_vars="Model",
                                           value_vars=["Accuracy","Precision","Recall","F1 Score"],
                                           var_name="Metric", value_name="Score")
                        fig_bar = px.bar(melt, x="Model", y="Score", color="Metric", barmode="group",
                                         template="plotly_white",
                                         title="Model Comparison — Accuracy / Precision / Recall / F1",
                                         height=450)
                        fig_bar.update_layout(xaxis_tickangle=-30)
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # Radar
                        metrics_r = ["Accuracy","Precision","Recall","F1 Score"]
                        fig_rad = go.Figure()
                        for _, row in res_df.iterrows():
                            vals = [row[m] for m in metrics_r] + [row[metrics_r[0]]]
                            fig_rad.add_trace(go.Scatterpolar(
                                r=vals, theta=metrics_r+[metrics_r[0]],
                                fill="toself", name=row["Model"]))
                        fig_rad.update_layout(polar=dict(radialaxis=dict(range=[0,1])),
                                              title="Radar — All Models", template="plotly_white")
                        st.plotly_chart(fig_rad, use_container_width=True)

                        # Confusion matrices (top 4)
                        st.subheader("🟥 Confusion Matrices (Top 4 Models)")
                        top4 = res_df["Model"].head(4).tolist()
                        cm_cols = st.columns(2)
                        for i, m in enumerate(top4):
                            with cm_cols[i % 2]:
                                fig_cm = px.imshow(cms[m], text_auto=True,
                                                   color_continuous_scale="Blues",
                                                   title=m, template="plotly_white", aspect="auto")
                                st.plotly_chart(fig_cm, use_container_width=True)

                        # Feature importance
                        tree_names = [n for n in trained if hasattr(trained[n], "feature_importances_")]
                        if tree_names:
                            best_tree = res_df[res_df["Model"].isin(tree_names)].iloc[0]["Model"]
                            fi = pd.Series(trained[best_tree].feature_importances_,
                                           index=clf_features).sort_values(ascending=True)
                            fig_fi = px.bar(fi, orientation="h", template="plotly_white",
                                            title=f"Feature Importance — {best_tree}",
                                            labels={"value":"Importance","index":"Feature"})
                            st.plotly_chart(fig_fi, use_container_width=True)

                    except Exception as e:
                        st.error(f"Classification error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARIMA
# ══════════════════════════════════════════════════════════════════════════════
with tab_arima:
    st.header("📈 ARIMA Time-Series Forecasting")

    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose

    nc_a  = num_cols(fdf)
    date_col = next((c for c in fdf.columns if any(k in c for k in ("date","time","month","year"))), None)
    default_metric = next((c for c in nc_a if any(k in c for k in ("amount","purchase","revenue","spend"))), nc_a[0] if nc_a else None)

    if not nc_a:
        st.warning("No numeric columns found.")
    else:
        a1, a2, a3 = st.columns(3)
        with a1:
            ts_metric = st.selectbox("Metric to forecast", nc_a,
                                     index=nc_a.index(default_metric) if default_metric in nc_a else 0,
                                     key="ts_metric")
        with a2:
            ts_freq = st.selectbox("Frequency", ["Daily","Weekly","Monthly"], index=2, key="ts_freq")
        with a3:
            ts_periods = st.slider("Forecast periods", 3, 24, 6, key="ts_periods")

        p1, p2, p3 = st.columns(3)
        with p1: p_val = st.slider("p  (AR order)",      0, 5, 2, key="arima_p")
        with p2: d_val = st.slider("d  (differencing)",  0, 2, 1, key="arima_d")
        with p3: q_val = st.slider("q  (MA order)",      0, 5, 2, key="arima_q")

        if st.button("🔮 Run ARIMA Forecast", key="arima_run"):
            with st.spinner("Fitting ARIMA…"):
                try:
                    freq_map = {"Daily":"D","Weekly":"W","Monthly":"MS"}
                    resample_freq = freq_map[ts_freq]

                    tmp_a = fdf.copy()
                    if date_col:
                        tmp_a[date_col] = pd.to_datetime(tmp_a[date_col], errors="coerce")
                        tmp_a = tmp_a.dropna(subset=[date_col]).set_index(date_col)
                        ts = tmp_a[ts_metric].resample(resample_freq).mean().dropna()
                    else:
                        idx = pd.date_range("2022-01-01", periods=len(fdf), freq="D")
                        ts  = pd.Series(fdf[ts_metric].values, index=idx).resample("MS").mean().dropna()

                    if len(ts) < 8:
                        st.warning("Not enough data points (need ≥8). Try a different frequency or metric.")
                    else:
                        # ADF test
                        adf_stat, adf_p = adfuller(ts)[:2]
                        st.info(f"**ADF Statistic:** {adf_stat:.4f}  |  **p-value:** {adf_p:.4f}  →  "
                                f"{'✅ Stationary' if adf_p < 0.05 else '⚠️ Non-stationary — consider d ≥ 1'}")

                        # Decomposition
                        if len(ts) >= 24:
                            try:
                                decomp = seasonal_decompose(ts, model="additive", period=12)
                                fig_dec = make_subplots(rows=4, cols=1,
                                                        subplot_titles=["Observed","Trend","Seasonal","Residual"],
                                                        shared_xaxes=True)
                                for i, (lbl, comp) in enumerate(zip(
                                        ["Observed","Trend","Seasonal","Residual"],
                                        [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]), 1):
                                    fig_dec.add_trace(go.Scatter(x=comp.index, y=comp.values,
                                                                  name=lbl, mode="lines"), row=i, col=1)
                                fig_dec.update_layout(height=650, title="Seasonal Decomposition",
                                                      template="plotly_white", showlegend=False)
                                st.plotly_chart(fig_dec, use_container_width=True)
                            except Exception:
                                pass

                        # Fit ARIMA
                        model_a = ARIMA(ts, order=(p_val, d_val, q_val))
                        fitted_a = model_a.fit()
                        fc       = fitted_a.get_forecast(steps=ts_periods)
                        fc_mean  = fc.predicted_mean
                        fc_ci    = fc.conf_int()

                        fig_fc = go.Figure()
                        fig_fc.add_trace(go.Scatter(x=ts.index, y=ts.values,
                                                     name="Historical", mode="lines",
                                                     line=dict(color="#2563eb", width=2)))
                        fig_fc.add_trace(go.Scatter(x=fitted_a.fittedvalues.index,
                                                     y=fitted_a.fittedvalues.values,
                                                     name="Fitted", mode="lines",
                                                     line=dict(color="#10b981", dash="dash")))
                        fig_fc.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values,
                                                     name="Forecast", mode="lines+markers",
                                                     line=dict(color="#f59e0b", width=2)))
                        ci_x = list(fc_ci.index) + list(fc_ci.index[::-1])
                        ci_y = list(fc_ci.iloc[:,1]) + list(fc_ci.iloc[:,0][::-1])
                        fig_fc.add_trace(go.Scatter(x=ci_x, y=ci_y, fill="toself",
                                                     fillcolor="rgba(245,158,11,0.15)",
                                                     line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
                        fig_fc.update_layout(title=f"ARIMA({p_val},{d_val},{q_val}) — {ts_metric}",
                                             template="plotly_white", height=450)
                        st.plotly_chart(fig_fc, use_container_width=True)

                        # Metrics
                        from sklearn.metrics import mean_absolute_error, mean_squared_error
                        fv = fitted_a.fittedvalues.dropna()
                        av = ts[fv.index]
                        mae  = mean_absolute_error(av, fv)
                        rmse = np.sqrt(mean_squared_error(av, fv))
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("AIC",  f"{fitted_a.aic:.2f}")
                        m2.metric("BIC",  f"{fitted_a.bic:.2f}")
                        m3.metric("MAE",  f"{mae:.2f}")
                        m4.metric("RMSE", f"{rmse:.2f}")

                        # Forecast table
                        fc_df = pd.DataFrame({
                            "Period":   fc_mean.index.astype(str),
                            "Forecast": fc_mean.values.round(2),
                            "Lower CI": fc_ci.iloc[:,0].values.round(2),
                            "Upper CI": fc_ci.iloc[:,1].values.round(2),
                        })
                        st.dataframe(fc_df, use_container_width=True)

                except Exception as e:
                    st.error(f"ARIMA error: {e}. Try different p/d/q values or frequency.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab_clust:
    st.header("🔗 Clustering — K-Means + Hierarchical")

    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler as SS
    from sklearn.metrics import silhouette_score
    import plotly.figure_factory as ff_lib

    nc_cl = num_cols(fdf)

    if len(nc_cl) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
    else:
        cl_feat = st.multiselect("Features for clustering", nc_cl,
                                  default=nc_cl[:min(4,len(nc_cl))], key="cl_feat")
        c1, c2, c3 = st.columns(3)
        with c1: max_k    = st.slider("Max K to test", 2, 15, 10, key="maxk")
        with c2: chosen_k = st.slider("Final K", 2, 15, 4, key="chosenk")
        with c3: linkage_m = st.selectbox("Linkage method", ["ward","complete","average","single"], key="lm")

        if st.button("🔍 Run Clustering", key="cl_run") and len(cl_feat) >= 2:
            with st.spinner("Running clustering…"):
                try:
                    tmp_cl = fdf[cl_feat].dropna()
                    scaler_cl = SS()
                    Xcl = scaler_cl.fit_transform(tmp_cl)

                    # Elbow + Silhouette
                    inertias, sils = [], []
                    k_range = range(2, max_k+1)
                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        lbl = km.fit_predict(Xcl)
                        inertias.append(km.inertia_)
                        sils.append(silhouette_score(Xcl, lbl))

                    fig_elbow = make_subplots(rows=1, cols=2,
                                              subplot_titles=["Elbow — Inertia","Silhouette Score"])
                    fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias,
                                                   mode="lines+markers", name="Inertia",
                                                   line=dict(color="#2563eb")), row=1, col=1)
                    fig_elbow.add_trace(go.Scatter(x=list(k_range), y=sils,
                                                   mode="lines+markers", name="Silhouette",
                                                   line=dict(color="#10b981")), row=1, col=2)
                    fig_elbow.update_layout(title="K Selection", template="plotly_white", height=350)
                    st.plotly_chart(fig_elbow, use_container_width=True)

                    best_k = list(k_range)[int(np.argmax(sils))]
                    st.success(f"Best K by Silhouette: **{best_k}**  (score = {max(sils):.4f})")

                    # K-Means
                    km_f = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
                    tmp_cl = tmp_cl.copy()
                    tmp_cl["KMeans_Cluster"] = km_f.fit_predict(Xcl).astype(str)

                    cx_col, cy_col = st.columns(2)
                    with cx_col: cx = st.selectbox("X axis", cl_feat, key="cx")
                    with cy_col: cy = st.selectbox("Y axis", cl_feat, index=min(1,len(cl_feat)-1), key="cy")

                    fig_km = px.scatter(tmp_cl, x=cx, y=cy, color="KMeans_Cluster",
                                        template="plotly_white",
                                        title=f"K-Means (K={chosen_k})", opacity=0.7)
                    st.plotly_chart(fig_km, use_container_width=True)

                    # Cluster profiles
                    st.subheader("📋 Cluster Mean Profiles")
                    profile = tmp_cl.groupby("KMeans_Cluster")[cl_feat].mean().round(2)
                    st.dataframe(profile.style.background_gradient(cmap="Blues"), use_container_width=True)

                    # Hierarchical + Dendrogram
                    st.subheader("🌳 Hierarchical Clustering — Dendrogram")
                    n_samp = min(300, len(Xcl))
                    Xsamp  = Xcl[:n_samp]
                    try:
                        from scipy.cluster.hierarchy import linkage as sp_linkage
                        fig_dend = ff_lib.create_dendrogram(
                            Xsamp,
                            linkagefun=lambda x: sp_linkage(x, method=linkage_m),
                            color_threshold=0.6)
                        fig_dend.update_layout(
                            title=f"Dendrogram (linkage='{linkage_m}', n={n_samp})",
                            template="plotly_white", height=500,
                            xaxis=dict(showticklabels=False))
                        st.plotly_chart(fig_dend, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Dendrogram: {e}")

                    # Agglomerative
                    hc = AgglomerativeClustering(n_clusters=chosen_k, linkage=linkage_m)
                    tmp_cl["HC_Cluster"] = hc.fit_predict(Xcl).astype(str)
                    fig_hc = px.scatter(tmp_cl, x=cx, y=cy, color="HC_Cluster",
                                        template="plotly_white",
                                        title=f"Agglomerative Clustering (K={chosen_k})", opacity=0.7)
                    st.plotly_chart(fig_hc, use_container_width=True)

                    # ARI
                    from sklearn.metrics import adjusted_rand_score
                    ari = adjusted_rand_score(tmp_cl["KMeans_Cluster"].astype(int),
                                              tmp_cl["HC_Cluster"].astype(int))
                    st.info(f"**Adjusted Rand Index (KMeans vs HC):** {ari:.4f}  (1.0 = identical, 0 = random)")

                except Exception as e:
                    st.error(f"Clustering error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
with tab_arm:
    st.header("🛒 Association Rule Mining")

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        cc_ar = cat_cols(fdf)
        if len(cc_ar) < 2:
            st.warning("Need at least 2 categorical columns.")
        else:
            arm1, arm2 = st.columns(2)
            with arm1:
                basket_cols = st.multiselect("Basket columns", cc_ar,
                                              default=cc_ar[:min(3,len(cc_ar))], key="arm_bc")
            with arm2:
                min_sup = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01, key="arm_sup")

            min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05, key="arm_conf")
            min_lift = st.slider("Min Lift",       1.0, 10.0, 1.2, 0.1,  key="arm_lift")

            if st.button("⛏️ Mine Association Rules", key="arm_run"):
                if len(basket_cols) < 2:
                    st.warning("Select at least 2 basket columns.")
                else:
                    with st.spinner("Mining…"):
                        try:
                            tmp_ar = fdf[basket_cols].dropna().astype(str)
                            txns   = tmp_ar.apply(
                                lambda r: [f"{c}={v}" for c,v in r.items()], axis=1).tolist()
                            te     = TransactionEncoder()
                            basket_df = pd.DataFrame(te.fit_transform(txns), columns=te.columns_)

                            freq = apriori(basket_df, min_support=min_sup, use_colnames=True)
                            if len(freq) == 0:
                                st.warning("No frequent itemsets — lower min support.")
                            else:
                                rules = association_rules(freq, metric="lift", min_threshold=min_lift)
                                rules = rules[rules["confidence"] >= min_conf].sort_values("lift", ascending=False)

                                st.success(f"Found **{len(rules)}** rules from **{len(freq)}** itemsets.")

                                # Display table
                                rd = rules.copy()
                                rd["antecedents"] = rd["antecedents"].apply(lambda x: ", ".join(list(x)))
                                rd["consequents"]  = rd["consequents"].apply(lambda x: ", ".join(list(x)))
                                show_cols = [c for c in ["antecedents","consequents","support",
                                                          "confidence","lift","leverage"] if c in rd.columns]
                                st.dataframe(
                                    rd[show_cols].head(30)
                                    .style.background_gradient(subset=["lift","confidence"], cmap="Greens"),
                                    use_container_width=True)

                                # Scatter
                                st.subheader("🔵 Support × Confidence × Lift Scatter")
                                sc1c, sc2c = st.columns(2)
                                with sc1c: arm_sx = st.selectbox("X", ["support","confidence","lift"], index=0, key="arm_sx")
                                with sc2c: arm_sy = st.selectbox("Y", ["support","confidence","lift"], index=1, key="arm_sy")
                                rd["rule"] = rd["antecedents"] + " → " + rd["consequents"]
                                fig_arm_sc = px.scatter(rd.head(100), x=arm_sx, y=arm_sy,
                                                        size="lift", color="lift",
                                                        hover_data=["rule","support","confidence","lift"],
                                                        color_continuous_scale="Viridis",
                                                        title=f"Rules Scatter — {arm_sx} vs {arm_sy}",
                                                        template="plotly_white", height=500)
                                st.plotly_chart(fig_arm_sc, use_container_width=True)

                                # Lift histogram
                                st.subheader("📊 Lift Distribution")
                                fig_lift = px.histogram(rd, x="lift", nbins=30,
                                                        color_discrete_sequence=["#2563eb"],
                                                        template="plotly_white", title="Lift Distribution")
                                st.plotly_chart(fig_lift, use_container_width=True)

                                # Top rules bar
                                st.subheader("🏆 Top 15 Rules by Lift")
                                fig_top = px.bar(rd.head(15), x="lift", y="rule",
                                                 orientation="h", color="confidence",
                                                 color_continuous_scale="Blues",
                                                 template="plotly_white", height=500,
                                                 title="Top 15 Rules")
                                st.plotly_chart(fig_top, use_container_width=True)

                        except Exception as e:
                            st.error(f"Association rules error: {e}")

    except ImportError:
        st.error("mlxtend not installed. Run: pip install mlxtend")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.header("🎛️ What-If Simulator")
    st.markdown("Build a customer profile using the controls below and predict an outcome.")

    from sklearn.ensemble import RandomForestClassifier as RFC, RandomForestRegressor as RFR
    from sklearn.preprocessing import LabelEncoder as LE

    nc_s  = num_cols(fdf)
    cc_s  = cat_cols(fdf)
    bin_s = [c for c in nc_s if fdf[c].nunique() == 2]
    tgt_opts = bin_s + cc_s

    if not tgt_opts:
        st.warning("No suitable target columns found.")
    else:
        with st.expander("⚙️ Model Settings", expanded=True):
            sw1, sw2 = st.columns(2)
            with sw1:
                sim_tgt = st.selectbox("Prediction target", tgt_opts, key="sim_tgt")
            feat_num_pool = [c for c in nc_s if c not in bin_s]
            with sw2:
                sim_num_feats = st.multiselect("Numeric features", feat_num_pool,
                                               default=feat_num_pool[:min(5,len(feat_num_pool))],
                                               key="sim_nf")
            sim_cat_feats = st.multiselect("Categorical features", cc_s,
                                            default=cc_s[:min(3,len(cc_s))], key="sim_cf")

        all_feats = sim_num_feats + sim_cat_feats
        if not all_feats:
            st.info("Select at least one feature above.")
        else:
            # Train model
            @st.cache_resource
            def train_sim(target, num_f, cat_f, _df_hash):
                data = st.session_state._sim_df
                cols = num_f + cat_f + [target]
                tmp  = data[cols].dropna()
                X    = tmp[num_f + cat_f].copy()
                y    = tmp[target]
                lem  = {}
                for c in cat_f:
                    le = LE(); X[c] = le.fit_transform(X[c].astype(str)); lem[c] = le
                is_cls = (y.dtype == object) or (y.nunique() <= 10)
                le_y = None
                if y.dtype == object:
                    le_y = LE(); y = le_y.fit_transform(y)
                mdl = RFC(100, random_state=42, n_jobs=-1) if is_cls else RFR(100, random_state=42, n_jobs=-1)
                mdl.fit(X, y)
                return mdl, lem, le_y, is_cls

            # Store fdf in session state so cache function can access it without being part of hash
            st.session_state._sim_df = fdf
            df_hash = f"{len(fdf)}_{sim_tgt}_{'_'.join(all_feats)}"

            try:
                sim_mdl, sim_lem, sim_le_y, sim_is_cls = train_sim(sim_tgt, sim_num_feats, sim_cat_feats, df_hash)
                st.success("✅ Model trained. Adjust the profile below and click Predict.")
            except Exception as e:
                st.error(f"Model training failed: {e}")
                sim_mdl = None

            if sim_mdl is not None:
                st.markdown("---")
                st.subheader("🎚️ Customer Profile")

                input_vals = {}

                # Numeric sliders
                if sim_num_feats:
                    n_per_row = 3
                    rows = [sim_num_feats[i:i+n_per_row] for i in range(0, len(sim_num_feats), n_per_row)]
                    for row_feats in rows:
                        cols_row = st.columns(len(row_feats))
                        for col, feat in zip(cols_row, row_feats):
                            mn  = float(fdf[feat].min())
                            mx  = float(fdf[feat].max())
                            med = float(fdf[feat].median())
                            with col:
                                input_vals[feat] = st.slider(
                                    feat.replace("_"," ").title(), mn, mx, med,
                                    key=f"sim_sl_{feat}")

                # Categorical dropdowns
                if sim_cat_feats:
                    n_per_row = 3
                    rows = [sim_cat_feats[i:i+n_per_row] for i in range(0, len(sim_cat_feats), n_per_row)]
                    for row_feats in rows:
                        cols_row = st.columns(len(row_feats))
                        for col, feat in zip(cols_row, row_feats):
                            opts = sorted(fdf[feat].dropna().unique().tolist())
                            with col:
                                input_vals[feat] = st.selectbox(
                                    feat.replace("_"," ").title(), opts,
                                    key=f"sim_se_{feat}")

                if st.button("🔮 Predict Outcome", key="sim_pred"):
                    try:
                        row_in = {}
                        for feat in sim_num_feats:
                            row_in[feat] = input_vals[feat]
                        for feat in sim_cat_feats:
                            le = sim_lem[feat]
                            val = str(input_vals[feat])
                            row_in[feat] = le.transform([val])[0] if val in le.classes_ else 0

                        X_in = pd.DataFrame([row_in])[sim_num_feats + sim_cat_feats]

                        st.markdown("---")
                        r1, r2, r3 = st.columns(3)

                        if sim_is_cls:
                            pred_cls  = sim_mdl.predict(X_in)[0]
                            pred_prob = sim_mdl.predict_proba(X_in)[0]
                            classes   = sim_mdl.classes_
                            label     = str(sim_le_y.inverse_transform([pred_cls])[0]) if sim_le_y else str(pred_cls)
                            conf      = pred_prob.max()

                            with r1:
                                st.markdown(f"""<div class="metric-card">
                                    <h3>Predicted Class</h3><h1 style="color:#2563eb">{label}</h1>
                                </div>""", unsafe_allow_html=True)
                            with r2:
                                st.markdown(f"""<div class="metric-card">
                                    <h3>Confidence</h3><h1 style="color:#10b981">{conf*100:.1f}%</h1>
                                </div>""", unsafe_allow_html=True)

                            cls_lbls = ([str(sim_le_y.inverse_transform([c])[0]) for c in classes]
                                        if sim_le_y else [str(c) for c in classes])
                            fig_pb = px.bar(x=cls_lbls, y=pred_prob, color=cls_lbls,
                                            title="Class Probability", template="plotly_white",
                                            labels={"x":"Class","y":"Probability"})
                            st.plotly_chart(fig_pb, use_container_width=True)

                        else:
                            pred_val = sim_mdl.predict(X_in)[0]
                            actual   = fdf[sim_tgt].dropna()
                            pct      = float((actual < pred_val).mean() * 100)

                            with r1:
                                st.markdown(f"""<div class="metric-card">
                                    <h3>Predicted {sim_tgt.replace("_"," ").title()}</h3>
                                    <h1 style="color:#2563eb">{pred_val:,.2f}</h1>
                                </div>""", unsafe_allow_html=True)
                            with r2:
                                st.markdown(f"""<div class="metric-card">
                                    <h3>Percentile in Dataset</h3>
                                    <h1 style="color:#10b981">{pct:.0f}th</h1>
                                </div>""", unsafe_allow_html=True)

                            fig_dist = px.histogram(actual, nbins=40, template="plotly_white",
                                                     title="Prediction vs Dataset Distribution")
                            fig_dist.add_vline(x=pred_val, line_color="red", line_width=2,
                                               annotation_text=f"Prediction: {pred_val:,.2f}",
                                               annotation_position="top right")
                            st.plotly_chart(fig_dist, use_container_width=True)

                        # Feature importance
                        st.subheader("📊 Feature Importance")
                        fi_s = pd.Series(sim_mdl.feature_importances_,
                                         index=sim_num_feats + sim_cat_feats).sort_values(ascending=True)
                        fig_fi_s = px.bar(fi_s, orientation="h", template="plotly_white",
                                          title="Feature Importance",
                                          labels={"value":"Importance","index":"Feature"})
                        st.plotly_chart(fig_fi_s, use_container_width=True)

                        # Sensitivity analysis
                        if sim_num_feats:
                            st.subheader("📐 Sensitivity Analysis")
                            sens_f = st.selectbox("Vary this feature", sim_num_feats, key="sens_f")
                            sens_r = np.linspace(float(fdf[sens_f].min()), float(fdf[sens_f].max()), 30)
                            sens_p = []
                            for v in sens_r:
                                ri = row_in.copy(); ri[sens_f] = v
                                Xi = pd.DataFrame([ri])[sim_num_feats + sim_cat_feats]
                                if sim_is_cls:
                                    sens_p.append(sim_mdl.predict_proba(Xi)[0].max())
                                else:
                                    sens_p.append(sim_mdl.predict(Xi)[0])
                            fig_sens = px.line(x=sens_r, y=sens_p, template="plotly_white",
                                               title=f"Sensitivity: {sens_f} → {sim_tgt}",
                                               labels={"x":sens_f,"y":"Prediction"})
                            fig_sens.add_vline(x=input_vals[sens_f], line_dash="dash",
                                               line_color="red", annotation_text="Current value")
                            st.plotly_chart(fig_sens, use_container_width=True)

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
