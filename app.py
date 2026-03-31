# ============================================================
# 🥭 MLOrgano — Product Demand Forecast (Streamlit UI)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sqlite3, re, os, glob
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLOrgano — Demand Forecast",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a2e1a 0%, #14532d 100%);
    border-right: 1px solid #166534;
}
[data-testid="stSidebar"] * { color: #dcfce7 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label {
    color: #86efac !important;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 1px dashed #4ade80;
    border-radius: 10px;
}

/* Main background */
.main { background: #f7fef9; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #14532d 0%, #166534 60%, #15803d 100%);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: "🥭";
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.18;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 32px;
    font-weight: 800;
    color: white;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    color: #86efac;
    font-size: 14px;
    margin: 0;
    font-weight: 400;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}
.metric-card {
    flex: 1;
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #dcfce7;
    box-shadow: 0 2px 8px rgba(20,83,45,0.06);
}
.metric-card .m-label {
    font-size: 11px;
    font-weight: 700;
    color: #15803d;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}
.metric-card .m-value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #14532d;
}
.metric-card .m-sub {
    font-size: 12px;
    color: #6b7280;
    margin-top: 2px;
}

/* Result card — HIGH demand */
.result-high {
    background: linear-gradient(135deg, #14532d 0%, #166534 100%);
    border-radius: 18px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 8px 32px rgba(20,83,45,0.25);
}
/* Result card — LOW demand */
.result-low {
    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
    border-radius: 18px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 18px;
    box-shadow: 0 8px 32px rgba(127,29,29,0.25);
}
.demand-badge {
    font-family: 'Syne', sans-serif;
    font-size: 34px;
    font-weight: 800;
    margin: 8px 0 16px;
    letter-spacing: -0.5px;
}
.forecast-month-label {
    font-size: 11px;
    opacity: 0.7;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.conf-label { font-size: 13px; opacity: 0.85; margin-bottom: 8px; }
.conf-bar-wrap {
    background: rgba(255,255,255,0.2);
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-bottom: 20px;
}
.conf-bar {
    height: 8px;
    border-radius: 99px;
    background: #86efac;
}
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}
.stat-tile {
    background: rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 12px 16px;
}
.stat-tile .s-label { font-size: 11px; opacity: 0.7; margin-bottom: 4px; }
.stat-tile .s-value {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
}

/* Advice boxes */
.advice-box {
    background: #dcfce7;
    border: 1.5px solid #86efac;
    border-radius: 14px;
    padding: 18px 22px;
    color: #14532d;
    font-size: 14px;
    line-height: 1.8;
    margin-bottom: 16px;
}
.advice-box.warn {
    background: #fff7ed;
    border-color: #fdba74;
    color: #7c2d12;
}

/* Table tweaks */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #15803d, #16a34a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 32px !important;
    width: 100%;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 14px rgba(22,163,74,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(22,163,74,0.45) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #f0fdf4;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #15803d;
}
.stTabs [aria-selected="true"] {
    background: #15803d !important;
    color: white !important;
}

/* Section labels */
.sec-label {
    font-size: 11px;
    font-weight: 700;
    color: #15803d;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def mysql_to_sqlite(sql):
    sql = re.sub(r"ENGINE\s*=\s*\w+", "", sql)
    sql = re.sub(r"DEFAULT CHARSET\s*=\s*\w+", "", sql)
    sql = re.sub(r"COLLATE\s*=?\s*[\w_]+", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"`(\w+)`\s+int\s+NOT NULL\s+AUTO_INCREMENT",
                 r"\1 INTEGER PRIMARY KEY AUTOINCREMENT", sql, flags=re.IGNORECASE)
    sql = re.sub(r"AUTO_INCREMENT\s*=?\s*\d*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r",?\s*PRIMARY KEY\s*\(\s*`?\w+`?\s*\)", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r",?\s*UNIQUE KEY\s+`?\w+`?\s*\([^)]*\)", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r",?\s*KEY\s+`?\w+`?\s*\([^)]*\)", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r",?\s*INDEX\s+`?\w+`?\s*\([^)]*\)", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"`", "", sql)
    sql = re.sub(r"\bUNSIGNED\b", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"--.*?\n", "\n", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r",\s*\)", "\n)", sql)
    return sql


@st.cache_resource
def load_and_train(file_bytes_dict):
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = OFF;")

    for fname, raw_bytes in file_bytes_dict.items():
        raw = raw_bytes.decode("utf-8", errors="ignore")
        clean = mysql_to_sqlite(raw)
        clean = clean.replace(")\nCREATE", ");\nCREATE")
        for stmt in [s.strip() for s in clean.split(";") if s.strip()]:
            try:
                stmt = stmt.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS")
                cursor.execute(stmt)
            except:
                pass
    conn.commit()

    def load_table(name):
        try:
            return pd.read_sql(f"SELECT * FROM {name}", conn)
        except:
            return None

    products = load_table("products_details")
    orders   = load_table("confirm_order")
    reviews  = load_table("review")

    if products is None or orders is None or reviews is None:
        return None

    # ── Coerce numerics ──
    for col in ["product_price", "product_discount", "after_discount_price", "product_quantity"]:
        products[col] = pd.to_numeric(products[col], errors="coerce")
    for col in ["total_price", "quantity", "product_id"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")

    # ── Aggregations ──
    order_count  = orders.groupby("product_id")["order_id"].count().rename("order_count")
    total_qty    = orders.groupby("product_id")["quantity"].sum().rename("total_qty_sold")
    total_rev    = orders.groupby("product_id")["total_price"].sum().rename("total_revenue")
    avg_rating   = reviews.groupby("product_id")["rating"].mean().rename("avg_rating")
    review_count = reviews.groupby("product_id")["rating"].count().rename("review_count")

    df = products.copy()
    df = df.join(order_count,  on="product_id", how="left")
    df = df.join(total_qty,    on="product_id", how="left")
    df = df.join(total_rev,    on="product_id", how="left")
    df = df.join(avg_rating,   on="product_id", how="left")
    df = df.join(review_count, on="product_id", how="left")
    df[["order_count","total_qty_sold","total_revenue","avg_rating","review_count"]] = \
        df[["order_count","total_qty_sold","total_revenue","avg_rating","review_count"]].fillna(0)

    # ── Engineered features ──
    df["discount_ratio"]    = (df["product_discount"] / 100).clip(0, 1)
    df["savings_per_unit"]  = df["product_price"] - df["after_discount_price"]
    df["value_score"]       = (df["savings_per_unit"] / df["product_price"].replace(0, np.nan)).clip(0, 1)
    df["revenue_per_order"] = (df["total_revenue"] / df["order_count"].replace(0, np.nan)).fillna(0)
    df["avg_qty_per_order"] = (df["total_qty_sold"] / df["order_count"].replace(0, np.nan)).fillna(0)

    le_cat = LabelEncoder()
    df["category_enc"] = le_cat.fit_transform(df["product_category"].fillna("Unknown"))

    # ── Label ──
    nonzero   = df[df["total_qty_sold"] > 0]["total_qty_sold"]
    threshold = nonzero.median() if len(nonzero) > 0 else 1
    df["high_demand"] = (df["total_qty_sold"] >= threshold).astype(int)
    if df["high_demand"].nunique() < 2:
        threshold = df["total_qty_sold"].mean()
        df["high_demand"] = (df["total_qty_sold"] >= threshold).astype(int)

    FEATURES = [
        "product_price", "after_discount_price",
        "discount_ratio", "savings_per_unit", "value_score",
        "product_quantity", "avg_rating", "review_count",
        "revenue_per_order", "avg_qty_per_order", "category_enc"
    ]

    X = df[FEATURES].fillna(0)
    y = df["high_demand"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                target_names=["Low Demand","High Demand"], zero_division=0)

    product_stats = df.set_index("product_id")

    return {
        "products": products,
        "df": df,
        "model": model,
        "scaler": scaler,
        "le_cat": le_cat,
        "FEATURES": FEATURES,
        "product_stats": product_stats,
        "accuracy": acc,
        "report": report,
        "threshold": threshold,
        "orders": orders,
    }


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥭 MLOrgano")
    st.markdown("**Demand Forecast Dashboard**")
    st.markdown("---")

    st.markdown("### 📂 Upload SQL Files")
    st.markdown("Upload all 11 `.sql` files exported from your database.")

    uploaded_files = st.file_uploader(
        "Drop SQL files here",
        type=["sql"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("---")
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) loaded")
    else:
        st.info("Upload SQL files to begin")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#4ade80;opacity:0.7;'>MLOrgano Admin · v2.0</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <div class="hero-title">Demand Forecast Dashboard</div>
  <div class="hero-sub">Admin tool · Predict next-month demand before stocking up</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
    <div style="background:white;border:2px dashed #86efac;border-radius:18px;
                padding:60px 40px;text-align:center;">
        <div style="font-size:56px;margin-bottom:16px;">📂</div>
        <div style="font-family:'Syne',sans-serif;font-size:22px;font-weight:700;
                    color:#14532d;margin-bottom:8px;">Upload your SQL files to get started</div>
        <div style="color:#6b7280;font-size:14px;">
            Use the sidebar on the left to upload all 11 MLOrgano SQL files
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & train ──
file_bytes_dict = {f.name: f.read() for f in uploaded_files}

with st.spinner("🔄 Loading data & training model..."):
    result = load_and_train(file_bytes_dict)

if result is None:
    st.error("❌ Could not load required tables. Make sure all SQL files are uploaded.")
    st.stop()

products      = result["products"]
df            = result["df"]
model         = result["model"]
scaler        = result["scaler"]
le_cat        = result["le_cat"]
FEATURES      = result["FEATURES"]
product_stats = result["product_stats"]
orders        = result["orders"]

# ── Top metrics ──
total_products  = len(products)
total_orders    = len(orders)
high_dem_count  = int(df["high_demand"].sum())
avg_rating_all  = df["avg_rating"].mean()

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card">
    <div class="m-label">Total Products</div>
    <div class="m-value">{total_products}</div>
    <div class="m-sub">across all categories</div>
  </div>
  <div class="metric-card">
    <div class="m-label">Total Orders</div>
    <div class="m-value">{total_orders}</div>
    <div class="m-sub">historical records</div>
  </div>
  <div class="metric-card">
    <div class="m-label">High Demand Products</div>
    <div class="m-value">{high_dem_count}</div>
    <div class="m-sub">based on sales threshold</div>
  </div>
  <div class="metric-card">
    <div class="m-label">Model Accuracy</div>
    <div class="m-value">{result['accuracy']:.0%}</div>
    <div class="m-sub">XGBoost classifier</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(["🔍 Forecast", "📊 Products", "🤖 Model Info"])

# ═══════════════════════════════════════════════════
# TAB 1 — FORECAST
# ═══════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1.2], gap="large")

    with col_form:
        st.markdown("#### Configure Forecast")

        product_names = products["product_name"].tolist()
        selected_product = st.selectbox("Select Product", product_names)

        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        selected_month = st.selectbox("Forecast Month", month_names, index=4)

        new_discount = st.slider("Planned Discount (%)", 0, 50, 10, 1)

        planned_stock = st.slider("Planned Stock (units)", 50, 500, 200, 10)

        st.markdown("<br>", unsafe_allow_html=True)
        forecast_btn = st.button("🔍 Forecast Demand", use_container_width=True)

    with col_result:
        st.markdown("#### Forecast Result")

        if forecast_btn:
            try:
                prod_row = products[products["product_name"] == selected_product].iloc[0]

                new_after_disc     = float(prod_row["product_price"]) * (1 - new_discount / 100)
                new_savings        = float(prod_row["product_price"]) - new_after_disc
                new_value_score    = (new_savings / float(prod_row["product_price"])) if prod_row["product_price"] > 0 else 0
                new_discount_ratio = new_discount / 100

                pid  = prod_row["product_id"]
                hist = product_stats.loc[pid] if pid in product_stats.index else None

                avg_rat  = float(hist["avg_rating"])       if hist is not None else 0
                rev_cnt  = float(hist["review_count"])     if hist is not None else 0
                rev_po   = float(hist["revenue_per_order"]) if hist is not None else 0
                aqpo     = float(hist["avg_qty_per_order"]) if hist is not None else 0
                past_qty = float(hist["total_qty_sold"])   if hist is not None else 0
                past_ord = float(hist["order_count"])      if hist is not None else 0

                cat_enc = int(le_cat.transform(
                    [prod_row["product_category"]
                     if prod_row["product_category"] in le_cat.classes_
                     else le_cat.classes_[0]]
                )[0])

                row_df = pd.DataFrame([{
                    "product_price":        float(prod_row["product_price"]),
                    "after_discount_price": new_after_disc,
                    "discount_ratio":       new_discount_ratio,
                    "savings_per_unit":     new_savings,
                    "value_score":          new_value_score,
                    "product_quantity":     planned_stock,
                    "avg_rating":           avg_rat,
                    "review_count":         rev_cnt,
                    "revenue_per_order":    rev_po,
                    "avg_qty_per_order":    aqpo,
                    "category_enc":         cat_enc,
                }])[FEATURES]

                row_s = scaler.transform(row_df)
                pred  = int(model.predict(row_s)[0])
                proba = model.predict_proba(row_s)[0]
                conf  = float(proba[pred])
                conf_pct = round(conf * 100, 1)
                bar_w    = round(conf * 100)

                is_high  = pred == 1
                label    = "🔥 HIGH DEMAND" if is_high else "📉 LOW DEMAND"
                res_cls  = "result-high" if is_high else "result-low"

                st.markdown(f"""
                <div class="{res_cls}">
                  <div class="forecast-month-label">Demand Forecast — {selected_month}</div>
                  <div class="demand-badge">{label}</div>
                  <div class="conf-label">Model confidence: <b>{conf_pct}%</b></div>
                  <div class="conf-bar-wrap">
                    <div class="conf-bar" style="width:{bar_w}%"></div>
                  </div>
                  <div class="stat-grid">
                    <div class="stat-tile">
                      <div class="s-label">Past Orders</div>
                      <div class="s-value">{int(past_ord)}</div>
                    </div>
                    <div class="stat-tile">
                      <div class="s-label">Units Sold (hist.)</div>
                      <div class="s-value">{int(past_qty)}</div>
                    </div>
                    <div class="stat-tile">
                      <div class="s-label">Planned Discount</div>
                      <div class="s-value">{new_discount}%</div>
                    </div>
                    <div class="stat-tile">
                      <div class="s-label">Price After Disc.</div>
                      <div class="s-value">₹{new_after_disc:,.0f}</div>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if is_high:
                    est_units = max(int(aqpo * 1.2), planned_stock)
                    st.markdown(f"""
                    <div class="advice-box">
                      <b>📋 Admin Recommendation</b><br>
                      ✅ Strong sales predicted for <b>{selected_month}</b>.<br>
                      📦 Consider stocking at least <b>{est_units} units</b> (20% above avg order qty).<br>
                      💰 At ₹{new_after_disc:,.0f}/unit with {new_discount}% discount, estimated revenue:
                         <b>₹{new_after_disc * est_units:,.0f}</b>.<br>
                      ⭐ Avg rating: <b>{avg_rat:.1f}/5</b> from {int(rev_cnt)} reviews — promote confidently.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="advice-box warn">
                      <b>⚠️ Admin Recommendation</b><br>
                      📉 Low demand predicted for <b>{selected_month}</b>.<br>
                      🏷️ Try increasing the discount above <b>{new_discount + 5}%</b> to stimulate sales.<br>
                      📦 Avoid overstocking — keep stock below <b>{min(planned_stock, 150)} units</b>.<br>
                      📣 Consider bundling with a high-demand product to boost visibility.
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Forecast error: {e}")
        else:
            st.markdown("""
            <div style="background:#f0fdf4;border:1.5px dashed #86efac;border-radius:14px;
                        padding:40px 24px;text-align:center;color:#15803d;">
                <div style="font-size:36px;margin-bottom:12px;">🔍</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:16px;">
                    Configure and click Forecast
                </div>
                <div style="font-size:13px;margin-top:6px;opacity:0.7;">
                    Results will appear here
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# TAB 2 — PRODUCTS
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown("#### Product Inventory & Demand Overview")

    display_df = df[[
        "product_name", "product_category",
        "product_price", "product_discount", "after_discount_price",
        "total_qty_sold", "order_count", "avg_rating", "high_demand"
    ]].copy()

    display_df.columns = [
        "Product", "Category", "Price (₹)", "Discount (%)",
        "After Disc. (₹)", "Units Sold", "Orders", "Avg Rating", "High Demand"
    ]
    display_df["High Demand"] = display_df["High Demand"].map({1: "✅ Yes", 0: "❌ No"})

    # Category filter
    cats = ["All"] + sorted(df["product_category"].dropna().unique().tolist())
    sel_cat = st.selectbox("Filter by Category", cats)
    if sel_cat != "All":
        display_df = display_df[display_df["Category"] == sel_cat]

    st.dataframe(display_df, use_container_width=True, height=420)

    # Category breakdown
    st.markdown("#### Sales by Category")
    cat_sales = df.groupby("product_category")["total_qty_sold"].sum().sort_values(ascending=False)
    st.bar_chart(cat_sales)


# ═══════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ═══════════════════════════════════════════════════
with tab3:
    st.markdown("#### Model Performance")

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{result['accuracy']:.2%}")
    c2.metric("Demand Threshold", f"{result['threshold']:.0f} units")
    c3.metric("High Demand Products", f"{int(df['high_demand'].sum())} / {len(df)}")

    st.markdown("**Classification Report**")
    st.code(result["report"], language=None)

    st.markdown("**Feature Importance**")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))