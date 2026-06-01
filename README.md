# UAE Customer Intelligence Dashboard

## What's Inside
- `app.py` — the full Streamlit app (all fixes + enhancements)
- `requirements.txt` — Python dependencies
- `uae_customers_2000_improved.csv` — sample dataset

## Changes Made

### 🐛 Bug Fix
- **AdaBoost error fixed**: Removed deprecated `algorithm="SAMME"` parameter
  (removed in scikit-learn ≥ 1.6). The app now uses the default which is
  equivalent and works on all recent scikit-learn versions.

### ✅ All Features Included

1. **Classification (Tab: 🎯)** — 8 algorithms compared:
   Random Forest, Gradient Boosting, AdaBoost, Logistic Regression,
   Decision Tree, KNN, Naive Bayes, SVM — with Accuracy/Precision/Recall/F1
   comparison table + grouped bar chart + combined ROC curves.

2. **ARIMA Forecasting (Tab: 📉)** — Configurable p/q orders, forecast horizon,
   95% confidence intervals, residual diagnostics.

3. **Drilled-Down EDA (Tab: 🔍)** — Cross-segment heatmap, Sunburst drill-down,
   cross-segment summary statistics, outlier detection.

4. **Hierarchical Clustering + Dendrogram (Tab: 👥)** — Interactive dendrogram
   with linkage method selector, Agglomerative vs K-Means silhouette comparison.

5. **Association Rules Scatter Plot (Tab: 🔗)** — Support vs Confidence bubble
   scatter (size = Lift) with hover details.

6. **What-If Simulator (Tab: 🎛)** — Dropdowns for categoricals, sliders for
   numerics, gauge chart, sensitivity analysis curve.

7. **Elbow Method + Silhouette Score (Tab: 👥)** — Side-by-side elbow (inertia)
   and silhouette score plots with best-k annotation.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud
1. Push to a GitHub repo
2. Connect at share.streamlit.io
3. Set main file to `app.py`
