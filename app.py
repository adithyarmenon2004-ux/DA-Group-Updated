import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="UAE Customer Analytics", layout="wide", page_icon="🏙️")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; border-radius: 6px; }
    h1 { color: #1a3c5e; }
    h2 { color: #2563eb; }
    .metric-card { background: #f0f7ff; border-radius: 10px; padding: 16px; border-left: 4px solid #2563eb; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("uae_customers_2000_improved.csv")
    except FileNotFoundError:
        try:
            import glob
            files = glob.glob("*.csv")
            if files:
                df = pd.read_csv(files[0])
            else:
                df = generate_sample_data()
        except Exception:
            df = generate_sample_data()

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def generate_sample_data(n=2000):
    np.random.seed(42)
    emirates = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"]
    categories = ["Electronics", "Fashion", "Home & Garden", "Sports", "Groceries", "Beauty", "Toys"]
    payment = ["Credit Card", "Debit Card", "Cash", "Digital Wallet"]
    gender = ["Male", "Female"]
    segments = ["Premium", "Standard", "Economy"]

    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.randint(18, 70, n),
        "gender": np.random.choice(gender, n),
        "emirate": np.random.choice(emirates, n, p=[0.35, 0.30, 0.15, 0.07, 0.06, 0.04, 0.03]),
        "annual_income": np.random.lognormal(10.8, 0.6, n).astype(int),
        "purchase_amount": np.random.exponential(500, n).round(2),
        "num_purchases": np.random.poisson(8, n),
        "product_category": np.random.choice(categories, n),
        "payment_method": np.random.choice(payment, n),
        "customer_segment": np.random.choice(segments, n, p=[0.25, 0.50, 0.25]),
        "satisfaction_score": np.random.randint(1, 6, n),
        "churn": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "tenure_months": np.random.randint(1, 60, n),
        "date": pd.date_range("2022-01-01", periods=n, freq="8H"),
        "loyalty_points": np.random.randint(0, 5000, n),
        "discount_used": np.random.choice([0, 1], n, p=[0.60, 0.40]),
        "return_customer": np.random.choice([0, 1], n, p=[0.50, 0.50]),
    })
    # Add some correlation
    df.loc[df["customer_segment"] == "Premium", "annual_income"] = (
        df.loc[df["customer_segment"] == "Premium", "annual_income"] * 1.5
    ).astype(int)
    df.loc[df["customer_segment"] == "Premium", "purchase_amount"] = (
        df.loc[df["customer_segment"] == "Premium", "purchase_amount"] * 1.8
    )
    return df


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Flag_of_the_United_Arab_Emirates.svg/320px-Flag_of_the_United_Arab_Emirates.svg.png", use_container_width=True)
    st.title("🏙️ UAE Customer Analytics")
    st.markdown("---")

    # Global filters
    st.subheader("🔍 Global Filters")
    emirate_col = next((c for c in df.columns if "emirate" in c or "city" in c or "region" in c), None)
    segment_col = next((c for c in df.columns if "segment" in c), None)
    gender_col  = next((c for c in df.columns if "gender" in c), None)

    sel_emirate = (
        st.multiselect("Emirates", sorted(df[emirate_col].dropna().unique()), default=list(df[emirate_col].dropna().unique()))
        if emirate_col else None
    )
    sel_segment = (
        st.multiselect("Segment", sorted(df[segment_col].dropna().unique()), default=list(df[segment_col].dropna().unique()))
        if segment_col else None
    )
    sel_gender = (
        st.multiselect("Gender", sorted(df[gender_col].dropna().unique()), default=list(df[gender_col].dropna().unique()))
        if gender_col else None
    )

    fdf = df.copy()
    if emirate_col and sel_emirate:
        fdf = fdf[fdf[emirate_col].isin(sel_emirate)]
    if segment_col and sel_segment:
        fdf = fdf[fdf[segment_col].isin(sel_segment)]
    if gender_col and sel_gender:
        fdf = fdf[fdf[gender_col].isin(sel_gender)]

    st.markdown(f"**Records:** {len(fdf):,} / {len(df):,}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 EDA",
    "🤖 Classification",
    "📈 ARIMA Forecast",
    "🔗 Clustering",
    "🛒 Association Rules",
    "🎛️ What-If Simulator",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – EDA (Drilled-down)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("📊 Exploratory Data Analysis — Drilled-Down")

    num_cols = fdf.select_dtypes(include=np.number).columns.tolist()
    cat_cols = fdf.select_dtypes(include="object").columns.tolist()

    # ── KPI row ──
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    amount_col = next((c for c in num_cols if "amount" in c or "purchase" in c or "spend" in c), num_cols[0] if num_cols else None)
    churn_col  = next((c for c in df.columns if "churn" in c), None)
    age_col    = next((c for c in num_cols if "age" in c), None)

    with kpi1:
        st.metric("Total Customers", f"{len(fdf):,}")
    with kpi2:
        if amount_col:
            st.metric("Avg Purchase", f"AED {fdf[amount_col].mean():,.0f}")
    with kpi3:
        if churn_col:
            st.metric("Churn Rate", f"{fdf[churn_col].mean()*100:.1f}%")
    with kpi4:
        if age_col:
            st.metric("Avg Age", f"{fdf[age_col].mean():.1f}")

    st.markdown("---")

    # ── Drill-down controls ──
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        drill_primary = st.selectbox("Primary Dimension", cat_cols, key="drill_primary")
    with col_b:
        drill_secondary = st.selectbox("Secondary Dimension (breakdown)", ["None"] + cat_cols, key="drill_secondary")
    with col_c:
        drill_metric = st.selectbox("Metric", num_cols, key="drill_metric")

    agg_func = st.radio("Aggregation", ["Mean", "Sum", "Count", "Median"], horizontal=True)

    agg_map = {"Mean": "mean", "Sum": "sum", "Count": "count", "Median": "median"}
    ag = agg_map[agg_func]

    if drill_secondary == "None":
        grouped = fdf.groupby(drill_primary)[drill_metric].agg(ag).reset_index().sort_values(drill_metric, ascending=False)
        fig = px.bar(grouped, x=drill_primary, y=drill_metric, color=drill_primary,
                     title=f"{agg_func} of {drill_metric} by {drill_primary}",
                     template="plotly_white")
    else:
        grouped = fdf.groupby([drill_primary, drill_secondary])[drill_metric].agg(ag).reset_index()
        fig = px.bar(grouped, x=drill_primary, y=drill_metric, color=drill_secondary,
                     barmode="group", title=f"{agg_func} of {drill_metric} by {drill_primary} & {drill_secondary}",
                     template="plotly_white")

    st.plotly_chart(fig, use_container_width=True)

    # ── Distribution + Correlation ──
    col1, col2 = st.columns(2)
    with col1:
        hist_col = st.selectbox("Distribution of", num_cols, key="hist_col")
        color_by  = st.selectbox("Colour by", ["None"] + cat_cols, key="hist_color")
        fig2 = px.histogram(fdf, x=hist_col, color=(None if color_by == "None" else color_by),
                            nbins=40, marginal="box", template="plotly_white",
                            title=f"Distribution: {hist_col}")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Correlation Heatmap")
        corr_cols = st.multiselect("Select numeric columns", num_cols, default=num_cols[:8], key="corr_cols")
        if len(corr_cols) >= 2:
            corr = fdf[corr_cols].corr()
            fig3 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             title="Correlation Matrix", template="plotly_white", aspect="auto")
            st.plotly_chart(fig3, use_container_width=True)

    # ── Scatter drill-down ──
    st.subheader("🔎 Scatter Drill-Down")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        sx = st.selectbox("X axis", num_cols, key="sx")
    with sc2:
        sy = st.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sy")
    with sc3:
        sc_color = st.selectbox("Colour", ["None"] + cat_cols, key="sc_color")
    with sc4:
        sc_size = st.selectbox("Size", ["None"] + num_cols, key="sc_size")

    fig4 = px.scatter(fdf, x=sx, y=sy,
                      color=(None if sc_color == "None" else sc_color),
                      size=(None if sc_size == "None" else sc_size),
                      opacity=0.6, template="plotly_white",
                      title=f"{sx} vs {sy}", trendline="ols")
    st.plotly_chart(fig4, use_container_width=True)

    # ── Box plots ──
    st.subheader("📦 Box Plot by Category")
    bp1, bp2 = st.columns(2)
    with bp1:
        box_num = st.selectbox("Numeric", num_cols, key="box_num")
    with bp2:
        box_cat = st.selectbox("Category", cat_cols, key="box_cat")
    fig5 = px.box(fdf, x=box_cat, y=box_num, color=box_cat, points="outliers",
                  template="plotly_white", title=f"{box_num} by {box_cat}")
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("🤖 Classification — All Algorithms Comparison")

    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier

    num_cols_c = fdf.select_dtypes(include=np.number).columns.tolist()
    cat_cols_c = fdf.select_dtypes(include="object").columns.tolist()
    bin_cols    = [c for c in num_cols_c if fdf[c].nunique() == 2]

    target_options = bin_cols + cat_cols_c
    if not target_options:
        st.warning("No suitable target column found.")
        st.stop()

    cl1, cl2 = st.columns(2)
    with cl1:
        target_col = st.selectbox("Target variable", target_options, key="clf_target")
    with cl2:
        feature_cols = st.multiselect("Feature columns", [c for c in num_cols_c if c != target_col], default=[c for c in num_cols_c if c != target_col][:6], key="clf_features")

    test_size = st.slider("Test size %", 10, 40, 20, key="clf_test") / 100

    if st.button("🚀 Run All Classifiers", key="clf_run"):
        with st.spinner("Training 10 classifiers…"):
            tmp = fdf[feature_cols + [target_col]].dropna()
            X = tmp[feature_cols]
            y = tmp[target_col]
            if y.dtype == object:
                le = LabelEncoder()
                y = le.fit_transform(y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) < 20 else None)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s  = scaler.transform(X_test)

            multi = len(np.unique(y)) > 2
            avg   = "weighted" if multi else "binary"

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                "SVM": SVC(probability=True, random_state=42),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Naive Bayes": GaussianNB(),
                "Neural Network (MLP)": MLPClassifier(max_iter=500, random_state=42),
            }

            results = []
            cms = {}
            for name, model in models.items():
                use_scaled = name in ["Logistic Regression", "SVM", "KNN", "Neural Network (MLP)"]
                Xt = X_train_s if use_scaled else X_train
                Xv = X_test_s  if use_scaled else X_test
                model.fit(Xt, y_train)
                pred = model.predict(Xv)
                try:
                    prob = model.predict_proba(Xv)
                    auc  = roc_auc_score(y_test, prob if multi else prob[:, 1], multi_class="ovr" if multi else "raise")
                except Exception:
                    auc = np.nan
                results.append({
                    "Model": name,
                    "Accuracy":  round(accuracy_score(y_test, pred), 4),
                    "Precision": round(precision_score(y_test, pred, average=avg, zero_division=0), 4),
                    "Recall":    round(recall_score(y_test, pred, average=avg, zero_division=0), 4),
                    "F1 Score":  round(f1_score(y_test, pred, average=avg, zero_division=0), 4),
                    "AUC-ROC":   round(auc, 4) if not np.isnan(auc) else "N/A",
                })
                cms[name] = confusion_matrix(y_test, pred)

            res_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
            res_df.index += 1
            st.subheader("📋 Metrics Comparison Table")
            st.dataframe(res_df.style.background_gradient(subset=["Accuracy","Precision","Recall","F1 Score"], cmap="Blues"), use_container_width=True)

            # ── Grouped bar chart ──
            melt = res_df.melt(id_vars="Model", value_vars=["Accuracy","Precision","Recall","F1 Score"], var_name="Metric", value_name="Score")
            fig = px.bar(melt, x="Model", y="Score", color="Metric", barmode="group",
                         template="plotly_white", title="All Models — Accuracy / Precision / Recall / F1",
                         height=450)
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

            # ── Radar chart ──
            st.subheader("🕸️ Radar Chart")
            metrics_radar = ["Accuracy","Precision","Recall","F1 Score"]
            fig_r = go.Figure()
            for _, row in res_df.iterrows():
                vals = [row[m] for m in metrics_radar]
                vals += [vals[0]]
                fig_r.add_trace(go.Scatterpolar(r=vals, theta=metrics_radar+[metrics_radar[0]], fill="toself", name=row["Model"]))
            fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,1])), title="Model Radar Comparison", template="plotly_white")
            st.plotly_chart(fig_r, use_container_width=True)

            # ── Confusion matrices ──
            st.subheader("🟥 Confusion Matrices")
            best_models = res_df["Model"].head(4).tolist()
            cols_cm = st.columns(2)
            for i, m in enumerate(best_models):
                with cols_cm[i % 2]:
                    cm = cms[m]
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                       title=f"{m}", template="plotly_white", aspect="auto")
                    st.plotly_chart(fig_cm, use_container_width=True)

            # ── Feature importance (best tree model) ──
            tree_models = {k: v for k, v in models.items() if hasattr(v, "feature_importances_")}
            if tree_models:
                best_tree_name = res_df[res_df["Model"].isin(tree_models)].iloc[0]["Model"]
                fi = pd.Series(tree_models[best_tree_name].feature_importances_, index=feature_cols).sort_values(ascending=True)
                fig_fi = px.bar(fi, orientation="h", template="plotly_white",
                                title=f"Feature Importance — {best_tree_name}", labels={"value":"Importance","index":"Feature"})
                st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – ARIMA FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("📈 ARIMA Time-Series Forecasting")

    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose

    date_col = next((c for c in df.columns if "date" in c or "time" in c or "month" in c or "year" in c), None)
    amount_col_ts = next((c for c in num_cols if "amount" in c or "purchase" in c or "revenue" in c or "spend" in c), num_cols[0] if num_cols else None)

    ar1, ar2, ar3 = st.columns(3)
    with ar1:
        ts_metric = st.selectbox("Metric to forecast", num_cols, index=num_cols.index(amount_col_ts) if amount_col_ts in num_cols else 0, key="ts_metric")
    with ar2:
        ts_freq = st.selectbox("Aggregation frequency", ["Daily", "Weekly", "Monthly"], index=2, key="ts_freq")
    with ar3:
        ts_periods = st.slider("Forecast periods", 3, 24, 6, key="ts_periods")

    col_p, col_d, col_q = st.columns(3)
    with col_p:
        p = st.slider("ARIMA p (AR order)", 0, 5, 2, key="arima_p")
    with col_d:
        d = st.slider("ARIMA d (differencing)", 0, 2, 1, key="arima_d")
    with col_q:
        q = st.slider("ARIMA q (MA order)", 0, 5, 2, key="arima_q")

    if st.button("🔮 Run ARIMA Forecast", key="arima_run"):
        with st.spinner("Running ARIMA…"):
            tmp = fdf.copy()
            if date_col:
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col])
                freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
                ts = tmp.set_index(date_col)[ts_metric].resample(freq_map[ts_freq]).mean().dropna()
            else:
                ts = pd.Series(fdf[ts_metric].values, index=pd.date_range("2022-01-01", periods=len(fdf), freq="D"))
                ts = ts.resample("ME").mean().dropna()

            if len(ts) < 12:
                st.warning("Not enough data points for reliable forecasting. Try a finer frequency.")
            else:
                # ── Stationarity test ──
                adf_result = adfuller(ts.dropna())
                st.info(f"**ADF Statistic:** {adf_result[0]:.4f} | **p-value:** {adf_result[1]:.4f} — {'✅ Stationary' if adf_result[1] < 0.05 else '⚠️ Non-stationary (consider d≥1)'}")

                # ── Decomposition ──
                if len(ts) >= 24:
                    try:
                        decomp = seasonal_decompose(ts, model="additive", period=12)
                        fig_d = make_subplots(rows=4, cols=1, subplot_titles=["Observed","Trend","Seasonal","Residual"], shared_xaxes=True)
                        for i, (name, comp) in enumerate(zip(["Observed","Trend","Seasonal","Residual"],
                                                              [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]), 1):
                            fig_d.add_trace(go.Scatter(x=comp.index, y=comp.values, name=name, mode="lines"), row=i, col=1)
                        fig_d.update_layout(height=700, title="Seasonal Decomposition", template="plotly_white", showlegend=False)
                        st.plotly_chart(fig_d, use_container_width=True)
                    except Exception:
                        pass

                # ── ARIMA fit ──
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted = model.fit()
                    forecast = fitted.get_forecast(steps=ts_periods)
                    fc_mean  = forecast.predicted_mean
                    fc_ci    = forecast.conf_int()

                    fig_f = go.Figure()
                    fig_f.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Historical", mode="lines", line=dict(color="#2563eb")))
                    fig_f.add_trace(go.Scatter(x=fitted.fittedvalues.index, y=fitted.fittedvalues, name="Fitted", mode="lines", line=dict(color="#10b981", dash="dash")))
                    fig_f.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, name="Forecast", mode="lines+markers", line=dict(color="#f59e0b", width=2)))
                    fig_f.add_trace(go.Scatter(
                        x=list(fc_ci.index) + list(fc_ci.index[::-1]),
                        y=list(fc_ci.iloc[:, 1]) + list(fc_ci.iloc[:, 0][::-1]),
                        fill="toself", fillcolor="rgba(245,158,11,0.15)", line=dict(color="rgba(0,0,0,0)"),
                        name="95% CI"
                    ))
                    fig_f.update_layout(title=f"ARIMA({p},{d},{q}) Forecast — {ts_metric}", template="plotly_white", height=450)
                    st.plotly_chart(fig_f, use_container_width=True)

                    # ── Summary metrics ──
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    fitted_vals = fitted.fittedvalues.dropna()
                    actual_vals = ts[fitted_vals.index]
                    mae  = mean_absolute_error(actual_vals, fitted_vals)
                    rmse = np.sqrt(mean_squared_error(actual_vals, fitted_vals))
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("AIC",  f"{fitted.aic:.2f}")
                    c2.metric("BIC",  f"{fitted.bic:.2f}")
                    c3.metric("MAE",  f"{mae:.2f}")
                    c4.metric("RMSE", f"{rmse:.2f}")

                    # ── Forecast table ──
                    fc_df = pd.DataFrame({"Period": fc_mean.index, "Forecast": fc_mean.values.round(2),
                                          "Lower CI": fc_ci.iloc[:, 0].values.round(2),
                                          "Upper CI": fc_ci.iloc[:, 1].values.round(2)})
                    st.dataframe(fc_df, use_container_width=True)
                except Exception as e:
                    st.error(f"ARIMA failed: {e}. Try different p/d/q values.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("🔗 Clustering — K-Means + Hierarchical")

    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import dendrogram, linkage
    import plotly.figure_factory as ff

    num_cols_cl = fdf.select_dtypes(include=np.number).columns.tolist()

    cl_feat = st.multiselect("Features for clustering", num_cols_cl, default=num_cols_cl[:4], key="cl_feat")
    cl1, cl2 = st.columns(2)
    with cl1:
        max_k = st.slider("Max K (Elbow)", 2, 15, 10, key="max_k")
    with cl2:
        chosen_k = st.slider("Final K", 2, 15, 4, key="chosen_k")

    linkage_method = st.selectbox("Hierarchical Linkage Method", ["ward", "complete", "average", "single"], key="linkage")

    if st.button("🔍 Run Clustering Analysis", key="cl_run") and len(cl_feat) >= 2:
        with st.spinner("Running clustering…"):
            tmp_cl = fdf[cl_feat].dropna()
            sc = StandardScaler()
            X_cl = sc.fit_transform(tmp_cl)

            # ── Elbow method ──
            inertias = []
            sil_scores = []
            k_range = range(2, max_k + 1)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_cl)
                inertias.append(km.inertia_)
                sil_scores.append(silhouette_score(X_cl, labels))

            fig_elbow = make_subplots(rows=1, cols=2, subplot_titles=["Elbow Method (Inertia)", "Silhouette Scores"])
            fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers", name="Inertia", line=dict(color="#2563eb")), row=1, col=1)
            fig_elbow.add_trace(go.Scatter(x=list(k_range), y=sil_scores, mode="lines+markers", name="Silhouette", line=dict(color="#10b981")), row=1, col=2)
            fig_elbow.update_layout(title="K Selection — Elbow & Silhouette", template="plotly_white", height=350)
            st.plotly_chart(fig_elbow, use_container_width=True)

            best_sil_k = list(k_range)[np.argmax(sil_scores)]
            st.success(f"Best K by Silhouette Score: **{best_sil_k}** (score = {max(sil_scores):.4f})")

            # ── K-Means with chosen K ──
            km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
            tmp_cl = tmp_cl.copy()
            tmp_cl["Cluster"] = km_final.fit_predict(X_cl).astype(str)

            col_x_cl, col_y_cl = st.columns(2)
            with col_x_cl:
                cx = st.selectbox("X axis", cl_feat, key="cx")
            with col_y_cl:
                cy = st.selectbox("Y axis", cl_feat, index=min(1, len(cl_feat)-1), key="cy")

            fig_km = px.scatter(tmp_cl, x=cx, y=cy, color="Cluster", template="plotly_white",
                                title=f"K-Means Clusters (K={chosen_k})", opacity=0.7)
            st.plotly_chart(fig_km, use_container_width=True)

            # ── Cluster profile table ──
            profile = tmp_cl.groupby("Cluster")[cl_feat].mean().round(2)
            st.subheader("📋 Cluster Profiles (Mean Values)")
            st.dataframe(profile.style.background_gradient(cmap="Blues"), use_container_width=True)

            # ── Hierarchical Clustering + Dendrogram ──
            st.subheader("🌳 Hierarchical Clustering — Dendrogram Validation")
            sample_n = min(200, len(X_cl))
            X_sample = X_cl[:sample_n]
            Z = linkage(X_sample, method=linkage_method)

            # Use scipy's dendrogram labels with Plotly figure factory
            try:
                labels_hier = [str(i) for i in range(sample_n)]
                fig_dend = ff.create_dendrogram(X_sample, linkagefun=lambda x: linkage(x, method=linkage_method),
                                                 color_threshold=0.7 * max(Z[:, 2]))
                fig_dend.update_layout(title=f"Dendrogram (linkage='{linkage_method}', n={sample_n})",
                                       template="plotly_white", height=500,
                                       xaxis=dict(showticklabels=False))
                st.plotly_chart(fig_dend, use_container_width=True)
            except Exception as e:
                st.warning(f"Dendrogram rendering note: {e}")

            # Agglomerative labels for heatmap
            hc = AgglomerativeClustering(n_clusters=chosen_k, linkage=linkage_method)
            hc_labels = hc.fit_predict(X_cl)
            tmp_cl["HC_Cluster"] = hc_labels.astype(str)

            fig_hc = px.scatter(tmp_cl, x=cx, y=cy, color="HC_Cluster", symbol="HC_Cluster",
                                template="plotly_white", title=f"Agglomerative Clustering (K={chosen_k})", opacity=0.7)
            st.plotly_chart(fig_hc, use_container_width=True)

            # ── KMeans vs HC comparison ──
            from sklearn.metrics import adjusted_rand_score
            ari = adjusted_rand_score(tmp_cl["Cluster"].astype(int), hc_labels)
            st.info(f"**Adjusted Rand Index (KMeans vs Hierarchical):** {ari:.4f}  — 1.0 = identical, 0 = random")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🛒 Association Rule Mining")

    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        cat_cols_ar = fdf.select_dtypes(include="object").columns.tolist()
        if len(cat_cols_ar) < 2:
            st.warning("Need at least 2 categorical columns for association rules.")
        else:
            ar1, ar2 = st.columns(2)
            with ar1:
                basket_cols = st.multiselect("Basket columns", cat_cols_ar, default=cat_cols_ar[:3], key="ar_basket")
            with ar2:
                min_support = st.slider("Min Support", 0.01, 0.5, 0.05, 0.01, key="ar_support")

            min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05, key="ar_conf")
            min_lift       = st.slider("Min Lift", 1.0, 10.0, 1.2, 0.1, key="ar_lift")

            if st.button("⛏️ Mine Rules", key="ar_run") and len(basket_cols) >= 2:
                with st.spinner("Mining associations…"):
                    tmp_ar = fdf[basket_cols].dropna().astype(str)
                    transactions = tmp_ar.apply(lambda r: [f"{c}={v}" for c, v in r.items()], axis=1).tolist()

                    te = TransactionEncoder()
                    te_arr = te.fit_transform(transactions)
                    basket_df = pd.DataFrame(te_arr, columns=te.columns_)

                    freq_items = apriori(basket_df, min_support=min_support, use_colnames=True)
                    if len(freq_items) == 0:
                        st.warning("No frequent itemsets found. Lower the support threshold.")
                    else:
                        rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
                        rules = rules[rules["confidence"] >= min_confidence].sort_values("lift", ascending=False)

                        st.success(f"Found **{len(rules)}** rules from **{len(freq_items)}** frequent itemsets.")

                        # ── Rules table ──
                        rules_disp = rules.copy()
                        rules_disp["antecedents"] = rules_disp["antecedents"].apply(lambda x: ", ".join(list(x)))
                        rules_disp["consequents"]  = rules_disp["consequents"].apply(lambda x: ", ".join(list(x)))
                        cols_show = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
                        cols_show = [c for c in cols_show if c in rules_disp.columns]
                        st.dataframe(rules_disp[cols_show].head(30).style.background_gradient(subset=["lift","confidence"], cmap="Greens"), use_container_width=True)

                        # ── Scatter plot: Support vs Confidence vs Lift ──
                        st.subheader("🔵 Scatter Plot: Support × Confidence × Lift")
                        sc1c, sc2c = st.columns(2)
                        with sc1c:
                            scatter_x = st.selectbox("X axis", ["support","confidence","lift","leverage"], index=0, key="ar_sx")
                        with sc2c:
                            scatter_y = st.selectbox("Y axis", ["support","confidence","lift","leverage"], index=1, key="ar_sy")

                        rules_disp["rule"] = rules_disp["antecedents"] + " → " + rules_disp["consequents"]
                        fig_ar = px.scatter(rules_disp.head(100), x=scatter_x, y=scatter_y,
                                            size="lift", color="lift",
                                            hover_data=["rule","support","confidence","lift"],
                                            color_continuous_scale="Viridis",
                                            title=f"Association Rules — {scatter_x} vs {scatter_y} (size=lift)",
                                            template="plotly_white", height=500)
                        st.plotly_chart(fig_ar, use_container_width=True)

                        # ── Network-style bubble chart ──
                        st.subheader("📊 Lift Distribution")
                        fig_lift = px.histogram(rules_disp, x="lift", nbins=30, color_discrete_sequence=["#2563eb"],
                                                template="plotly_white", title="Lift Distribution")
                        st.plotly_chart(fig_lift, use_container_width=True)

                        # ── Top rules bar ──
                        top_rules = rules_disp.head(15)
                        fig_top = px.bar(top_rules, x="lift", y="rule", orientation="h",
                                         color="confidence", color_continuous_scale="Blues",
                                         title="Top 15 Rules by Lift", template="plotly_white", height=500)
                        st.plotly_chart(fig_top, use_container_width=True)

    except ImportError:
        st.error("mlxtend is required. Add it to requirements.txt.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 6 – WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("🎛️ What-If Simulator with Predictive Outcomes")

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    st.markdown("""
    Use the controls below to simulate customer profiles and predict outcomes using a trained model.
    All dropdowns and sliders are dynamic — trained on your actual dataset.
    """)

    # ── Identify target ──
    num_cols_sim = fdf.select_dtypes(include=np.number).columns.tolist()
    cat_cols_sim = fdf.select_dtypes(include="object").columns.tolist()
    bin_cols_sim = [c for c in num_cols_sim if fdf[c].nunique() == 2]

    sim_targets = bin_cols_sim + cat_cols_sim
    sim_features_num = [c for c in num_cols_sim if c not in bin_cols_sim]
    sim_features_cat = cat_cols_sim

    with st.expander("⚙️ Simulator Settings", expanded=True):
        ws1, ws2 = st.columns(2)
        with ws1:
            sim_target = st.selectbox("Prediction target", sim_targets, key="sim_target")
        with ws2:
            num_feat_sel = st.multiselect("Numeric features", sim_features_num, default=sim_features_num[:5], key="sim_num_feat")

        cat_feat_sel = st.multiselect("Categorical features", sim_features_cat, default=sim_features_cat[:3], key="sim_cat_feat")

    # ── Build model once ──
    @st.cache_resource
    def train_sim_model(target, num_feats, cat_feats, data_hash):
        tmp = fdf[num_feats + cat_feats + [target]].dropna()
        X = tmp[num_feats + cat_feats].copy()
        y = tmp[target]
        le_map = {}
        for c in cat_feats:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            le_map[c] = le
        is_classif = y.dtype == object or y.nunique() <= 10
        if y.dtype == object:
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
        else:
            le_y = None
        if is_classif:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        return model, le_map, le_y, is_classif, tmp

    data_hash = str(len(fdf))
    try:
        model_sim, le_map_sim, le_y_sim, is_classif_sim, train_data = train_sim_model(
            sim_target, num_feat_sel, cat_feat_sel, data_hash
        )
        st.success("✅ Predictive model trained on your data.")
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("🎚️ Customer Profile Builder")

    input_vals = {}

    # Numeric sliders
    if num_feat_sel:
        num_cols_ui = st.columns(min(3, len(num_feat_sel)))
        for i, feat in enumerate(num_feat_sel):
            col = num_cols_ui[i % len(num_cols_ui)]
            mn = float(fdf[feat].min())
            mx = float(fdf[feat].max())
            med = float(fdf[feat].median())
            with col:
                input_vals[feat] = st.slider(feat.replace("_"," ").title(), mn, mx, med, key=f"sim_slider_{feat}")

    # Categorical dropdowns
    if cat_feat_sel:
        cat_cols_ui = st.columns(min(3, len(cat_feat_sel)))
        for i, feat in enumerate(cat_feat_sel):
            col = cat_cols_ui[i % len(cat_cols_ui)]
            options = sorted(fdf[feat].dropna().unique().tolist())
            with col:
                input_vals[feat] = st.selectbox(feat.replace("_"," ").title(), options, key=f"sim_select_{feat}")

    if st.button("🔮 Predict Outcome", key="sim_predict"):
        try:
            row = {}
            for feat in num_feat_sel:
                row[feat] = input_vals[feat]
            for feat in cat_feat_sel:
                le = le_map_sim[feat]
                val = input_vals[feat]
                if val in le.classes_:
                    row[feat] = le.transform([str(val)])[0]
                else:
                    row[feat] = 0

            X_input = pd.DataFrame([row])[num_feat_sel + cat_feat_sel]

            st.markdown("---")
            r1, r2, r3 = st.columns(3)

            if is_classif_sim:
                pred_class = model_sim.predict(X_input)[0]
                pred_proba = model_sim.predict_proba(X_input)[0]
                classes    = model_sim.classes_

                if le_y_sim:
                    label = le_y_sim.inverse_transform([pred_class])[0]
                else:
                    label = str(pred_class)

                with r1:
                    st.markdown(f"<div class='metric-card'><h3>Predicted Class</h3><h1 style='color:#2563eb'>{label}</h1></div>", unsafe_allow_html=True)
                with r2:
                    max_prob = pred_proba.max()
                    st.markdown(f"<div class='metric-card'><h3>Confidence</h3><h1 style='color:#10b981'>{max_prob*100:.1f}%</h1></div>", unsafe_allow_html=True)

                # Probability bar chart
                if le_y_sim:
                    cls_labels = [str(le_y_sim.inverse_transform([c])[0]) for c in classes]
                else:
                    cls_labels = [str(c) for c in classes]
                fig_prob = px.bar(x=cls_labels, y=pred_proba, color=cls_labels,
                                  title="Class Probability Distribution", template="plotly_white",
                                  labels={"x":"Class","y":"Probability"})
                st.plotly_chart(fig_prob, use_container_width=True)

            else:
                pred_val = model_sim.predict(X_input)[0]
                with r1:
                    st.markdown(f"<div class='metric-card'><h3>Predicted {sim_target}</h3><h1 style='color:#2563eb'>{pred_val:,.2f}</h1></div>", unsafe_allow_html=True)

                # Compare to distribution
                actual_vals = fdf[sim_target].dropna()
                percentile  = (actual_vals < pred_val).mean() * 100
                with r2:
                    st.markdown(f"<div class='metric-card'><h3>Percentile</h3><h1 style='color:#10b981'>{percentile:.0f}th</h1></div>", unsafe_allow_html=True)

                fig_dist = px.histogram(actual_vals, nbins=40, template="plotly_white",
                                        title=f"Where does this prediction fall in the distribution?")
                fig_dist.add_vline(x=pred_val, line_color="red", line_width=3,
                                   annotation_text=f"Your prediction: {pred_val:,.2f}", annotation_position="top right")
                st.plotly_chart(fig_dist, use_container_width=True)

            # ── Feature importance for this model ──
            st.subheader("📊 Feature Importance")
            fi_sim = pd.Series(model_sim.feature_importances_, index=num_feat_sel + cat_feat_sel).sort_values(ascending=True)
            fig_fi_sim = px.bar(fi_sim, orientation="h", template="plotly_white",
                                title="Feature Importance (Random Forest)", labels={"value":"Importance","index":"Feature"})
            st.plotly_chart(fig_fi_sim, use_container_width=True)

            # ── Sensitivity analysis ──
            if num_feat_sel:
                st.subheader("📐 Sensitivity Analysis")
                sens_feat = st.selectbox("Vary this feature", num_feat_sel, key="sens_feat")
                n_steps = 20
                sens_range = np.linspace(float(fdf[sens_feat].min()), float(fdf[sens_feat].max()), n_steps)
                sens_preds = []
                for v in sens_range:
                    r_sens = row.copy()
                    r_sens[sens_feat] = v
                    X_s = pd.DataFrame([r_sens])[num_feat_sel + cat_feat_sel]
                    if is_classif_sim:
                        pp = model_sim.predict_proba(X_s)[0].max()
                    else:
                        pp = model_sim.predict(X_s)[0]
                    sens_preds.append(pp)

                fig_sens = px.line(x=sens_range, y=sens_preds, template="plotly_white",
                                   title=f"Sensitivity: How {sens_feat} affects the prediction",
                                   labels={"x": sens_feat, "y": "Predicted Value / Confidence"})
                fig_sens.add_vline(x=input_vals[sens_feat], line_dash="dash", line_color="red",
                                   annotation_text="Current value")
                st.plotly_chart(fig_sens, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
