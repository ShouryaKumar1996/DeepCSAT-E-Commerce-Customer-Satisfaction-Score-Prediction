"""
src/streamlit_app.py  —  DeepCSAT Dashboard
=============================================
Run from the DeepCSAT/ root folder:
    streamlit run src/streamlit_app.py

Expects this folder structure:
    DeepCSAT/
    ├── src/streamlit_app.py     <- this file
    ├── models/
    │   ├── best_deepcsat_model.keras
    │   ├── scaler.pkl
    │   └── label_encoders.pkl
    └── outputs/                 <- charts saved here by notebook
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# ── Resolve paths relative to this file ───────────────────────────────────────
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SRC_DIR)
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DeepCSAT", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.big-title{font-size:2.1rem;font-weight:700;color:#1F3864;}
.sec-hdr{font-size:1rem;font-weight:600;color:#2E75B6;
         border-bottom:2px solid #D6E4F0;padding-bottom:3px;margin-bottom:8px;}
</style>""", unsafe_allow_html=True)

CSAT_COLORS = {1:"#E74C3C",2:"#E67E22",3:"#F39C12",4:"#27AE60",5:"#2ECC71"}
CSAT_LABELS = {1:"😡 Very Dissatisfied",2:"😞 Dissatisfied",3:"😐 Neutral",
               4:"😊 Satisfied",5:"😄 Very Satisfied"}

CATEGORICAL_FEATURES = ["channel_name","category","Sub-category","Tenure Bucket","Agent Shift",
                         "price_bucket","resp_speed_cat","channel_cat"]
NUMERICAL_FEATURES   = ["response_time_min","survey_lag_days","issue_hour","issue_dow",
                         "issue_day","issue_week","is_weekend","is_night_issue",
                         "order_to_issue_days","sent_neg","sent_neu","sent_pos","sent_compound",
                         "remark_len","remark_word_cnt","has_remark","agent_csat_mean",
                         "agent_csat_std","agent_volume","manager_csat_mean","Item_price",
                         "subcat_count","has_order_datetime","has_city","has_product_cat",
                         "has_item_price"]

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf
    model  = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_deepcsat_model.keras"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    le     = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    return model, scaler, le

try:
    model, scaler, le_dict = load_model()
    READY = True
except Exception as e:
    READY = False; ERR = str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────
def derive(p):
    price = float(p.get("Item_price",999))
    resp  = float(p.get("response_time_min",10))
    chan  = str(p.get("channel_name","Inbound"))
    cat   = str(p.get("category","Order Related"))
    p["price_bucket"]   = ("very_low" if price<500 else "low" if price<1000
                           else "mid" if price<2000 else "high" if price<5000 else "very_high")
    p["resp_speed_cat"] = ("instant" if resp<5 else "fast" if resp<15
                           else "moderate" if resp<30 else "slow" if resp<60 else "very_slow")
    p["channel_cat"]    = f"{chan}_{cat}"
    return p

def do_predict(payload):
    payload = derive(payload)
    n_cat   = len(CATEGORICAL_FEATURES)
    cat_vals = []
    for col in CATEGORICAL_FEATURES:
        le  = le_dict[col]
        val = str(payload.get(col,"Unknown"))
        val = val if val in le.classes_ else le.classes_[0]
        cat_vals.append(int(le.transform([val])[0]))
    num_vals = [float(payload.get(col,0.0)) for col in NUMERICAL_FEATURES]
    vec      = np.array(cat_vals+num_vals, dtype=np.float32).reshape(1,-1)
    vec[0,n_cat:] = scaler.transform(vec[:,n_cat:])[0]
    return model.predict(vec, verbose=0)[0]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 DeepCSAT")
    st.caption("Customer Satisfaction Predictor")
    st.divider()
    page = st.radio("", ["🏠  Predict CSAT","📦  Batch Predict",
                         "📊  Model Performance","💡  About"],
                    label_visibility="collapsed")
    st.divider()
    if READY:
        st.success("✅ Model loaded")
        st.caption(f"Params: {model.count_params():,}")
    else:
        st.error("❌ Model not found")
        st.caption("Run the notebook first,\nthen restart this app.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Predict CSAT":
    st.markdown('<p class="big-title">🛒 Real-Time CSAT Prediction</p>', unsafe_allow_html=True)
    st.markdown("Fill in the interaction details and click **Predict**.")
    st.divider()

    if not READY:
        st.error(f"**Model not loaded.**\n\nError: `{ERR}`\n\n"
                 "Run all cells in `notebooks/DeepCSAT_Capstone_Notebook.ipynb` first, "
                 "then restart this app.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<p class="sec-hdr">Interaction</p>', unsafe_allow_html=True)
        channel  = st.selectbox("Contact Channel", ["Inbound","Outcall","Email"])
        category = st.selectbox("Issue Category", [
            "Returns","Order Related","Cancellation","Product Queries",
            "Payments related","Refund Related","Feedback",
            "Offers & Cashback","Shopzilla Related","App/website",
            "Onboarding related","Others"])
        subcat   = st.selectbox("Sub-Category", [
            "Delivery Related","Order Status Enquiry","Refund Enquiry",
            "Product Enquiry","Return Request","Cancellation Request",
            "Payment Issue","App Issue","Login Issue","Others"])
        price    = st.number_input("Item Price (₹)", 0, 200000, 999, 100)

    with c2:
        st.markdown('<p class="sec-hdr">Agent</p>', unsafe_allow_html=True)
        tenure  = st.selectbox("Tenure Bucket", ["On Job Training","0-30","31-60","61-90",">90"])
        shift   = st.selectbox("Agent Shift", ["Morning","Afternoon","Evening","Split","Night"])
        ag_mean = st.slider("Agent Historical CSAT Mean", 1.0, 5.0, 4.2, 0.1)
        ag_std  = st.slider("Agent CSAT Std Dev", 0.0, 2.0, 0.8, 0.1)
        ag_vol  = st.number_input("Agent Interactions So Far", 0, 10000, 150)
        mg_mean = st.slider("Manager Team CSAT Mean", 1.0, 5.0, 4.1, 0.1)

    with c3:
        st.markdown('<p class="sec-hdr">Timing & Sentiment</p>', unsafe_allow_html=True)
        resp_t  = st.number_input("Response Time (min)", 0, 300, 12)
        hour    = st.slider("Hour Issue Reported", 0, 23, 10)
        dow     = st.selectbox("Day of Week", [0,1,2,3,4,5,6],
                               format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        lag     = st.number_input("Survey Lag (days)", 0, 30, 0)
        sent    = st.slider("Remark Sentiment (−1 to +1)", -1.0, 1.0, 0.0, 0.05)
        has_rem = st.checkbox("Customer Left a Remark")

    st.divider()
    if st.button("🔮  Predict CSAT Score", use_container_width=True, type="primary"):
        payload = {
            "channel_name":channel, "category":category, "Sub-category":subcat,
            "Tenure Bucket":tenure, "Agent Shift":shift, "Item_price":price,
            "response_time_min":resp_t, "survey_lag_days":lag,
            "issue_hour":hour, "issue_dow":dow, "issue_day":15, "issue_week":30,
            "is_weekend":1 if dow>=5 else 0,
            "is_night_issue":1 if (hour>=22 or hour<=5) else 0,
            "order_to_issue_days":3,
            "sent_neg":round(max(0.0,-sent),4), "sent_neu":round(1-abs(sent),4),
            "sent_pos":round(max(0.0,sent),4),  "sent_compound":sent,
            "remark_len":60 if has_rem else 0,  "remark_word_cnt":10 if has_rem else 0,
            "has_remark":1 if has_rem else 0,
            "agent_csat_mean":ag_mean, "agent_csat_std":ag_std, "agent_volume":ag_vol,
            "manager_csat_mean":mg_mean, "subcat_count":8,
            "has_order_datetime":0, "has_city":0, "has_product_cat":0, "has_item_price":1,
        }
        with st.spinner("Running model..."):
            proba = do_predict(payload)
            pred  = int(proba.argmax()) + 1
            conf  = float(proba.max())

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
            <div style='background:{CSAT_COLORS[pred]};padding:28px;border-radius:12px;
                        text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.15);'>
                <div style='font-size:64px;font-weight:900;color:white;line-height:1;'>{pred}</div>
                <div style='font-size:15px;color:rgba(255,255,255,0.9);margin-top:6px;'>{CSAT_LABELS[pred]}</div>
                <div style='font-size:13px;color:rgba(255,255,255,0.72);margin-top:4px;'>
                    Confidence: {conf*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown("**Probability across all 5 classes**")
            fig, ax = plt.subplots(figsize=(6,3))
            ax.barh([f"CSAT-{i+1}" for i in range(5)], proba,
                    color=[CSAT_COLORS[i+1] for i in range(5)], edgecolor="white")
            for i, v in enumerate(proba):
                ax.text(v+0.008, i, f"{v*100:.1f}%", va="center", fontsize=10)
            ax.set_xlim(0,1.15); ax.set_xlabel("Probability", fontsize=10)
            ax.tick_params(labelsize=10); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.divider()
        if pred<=2:   st.error(f"🚨 HIGH RISK — CSAT {pred}. Escalate to supervisor immediately.")
        elif pred==3: st.warning(f"⚠️ MODERATE RISK — CSAT {pred}. Monitor this interaction.")
        else:         st.success(f"✅ LOW RISK — CSAT {pred}. Interaction is on track.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦  Batch Predict":
    st.markdown('<p class="big-title">📦 Batch CSAT Prediction</p>', unsafe_allow_html=True)
    st.markdown("Upload a CSV — every row is scored and returned as a downloadable file.")
    if not READY: st.error("Model not loaded. Run the notebook first."); st.stop()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_b = pd.read_csv(uploaded)
        st.info(f"Loaded **{len(df_b):,} rows** × {df_b.shape[1]} columns")
        st.dataframe(df_b.head(5), use_container_width=True)

        if st.button("▶  Score All Rows", type="primary"):
            preds, confs, risks = [], [], []
            bar = st.progress(0)
            for idx, row in df_b.iterrows():
                try:
                    proba = do_predict(row.to_dict())
                    p     = int(proba.argmax()) + 1
                    preds.append(p); confs.append(round(float(proba.max()),4))
                    risks.append("HIGH" if p<=2 else "MODERATE" if p==3 else "LOW")
                except Exception:
                    preds.append(None); confs.append(None); risks.append("ERROR")
                bar.progress((idx+1)/len(df_b))

            df_b["predicted_csat"]=preds; df_b["confidence"]=confs; df_b["risk_flag"]=risks
            st.success(f"Scored {len(preds):,} rows.")
            c1,c2,c3 = st.columns(3)
            c1.metric("High Risk (1-2)", risks.count("HIGH"), f"{risks.count('HIGH')/len(risks)*100:.1f}%")
            c2.metric("Moderate (3)", risks.count("MODERATE"))
            c3.metric("Low Risk (4-5)", risks.count("LOW"))

            fig, ax = plt.subplots(figsize=(6,3))
            s = pd.Series(preds).dropna().astype(int).value_counts().sort_index()
            ax.bar([f"CSAT-{i}" for i in s.index], s.values,
                   color=[CSAT_COLORS[i] for i in s.index], edgecolor="black")
            ax.set_title("Predicted CSAT Distribution"); ax.tick_params(labelsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.download_button("⬇️  Download Results", df_b.to_csv(index=False),
                               "deepcsat_predictions.csv","text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Model Performance":
    st.markdown('<p class="big-title">📊 Model Performance Dashboard</p>', unsafe_allow_html=True)
    st.divider()

    st.subheader("Model Comparison Table")
    df_res = pd.DataFrame([
        {"Model":"Dummy Classifier (always CSAT-5)","Train Acc":"—","Test Acc":"0.6940",
         "B-V Gap":"—","Weighted F1":"0.5686","MAE":"0.7579"},
        {"Model":"Logistic Regression","Train Acc":"~0.38","Test Acc":"~0.38",
         "B-V Gap":"~0.004","Weighted F1":"~0.45","MAE":"~1.21"},
        {"Model":"Random Forest","Train Acc":"~0.82","Test Acc":"~0.58",
         "B-V Gap":"~0.24","Weighted F1":"~0.59","MAE":"~0.86"},
        {"Model":"Best ANN — DeepCSAT ✅","Train Acc":"see notebook",
         "Test Acc":"see notebook","B-V Gap":"see notebook",
         "Weighted F1":"see notebook","MAE":"see notebook"},
    ])
    st.dataframe(df_res, use_container_width=True, hide_index=True)
    st.divider()

    # Show charts saved by the notebook
    st.subheader("Training & Evaluation Charts")
    tab1, tab2, tab3, tab4 = st.tabs(["Training Curves","Confusion Matrix",
                                       "Model Comparison","Bias-Variance Gap"])
    chart_map = {
        tab1: "training_curves.png",
        tab2: "confusion_matrix.png",
        tab3: "final_comparison.png",
        tab4: "bias_variance_gap.png",
    }
    for tab, fname in chart_map.items():
        with tab:
            path = os.path.join(OUTPUTS_DIR, fname)
            if os.path.exists(path):
                st.image(path, use_column_width=True)
            else:
                st.info(f"`outputs/{fname}` not found. Run the notebook first.")
    st.divider()

    st.subheader("Key Finding — Class Weight Calibration")
    st.info("""
`compute_class_weight('balanced')` creates a **46× weight ratio**:
CSAT-2 = **13.4×**, CSAT-5 = **0.29×**

This **destroys ANN gradient descent** — accuracy dropped to 41% (below dummy at 69.4%).
Random Forest handles it through bagging; gradient descent cannot.

**Fix:** mild weights `{CSAT-1: 2.0, CSAT-2: 3.0, CSAT-3: 2.5, CSAT-4: 2.0, CSAT-5: 0.5}` — max ratio **6×**
    """)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡  About":
    st.markdown('<p class="big-title">💡 About DeepCSAT</p>', unsafe_allow_html=True)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Project Summary")
        st.markdown("""
**DeepCSAT** predicts Customer Satisfaction (CSAT) scores for *Shopzilla*
from interaction metadata in real time.

| | |
|---|---|
| Dataset | ~86,000 interactions, 1 month |
| Target | CSAT Score 1–5 |
| Features | 34 engineered features |
| Best Model | Deep ANN — 4 residual blocks |

**EDA Hypotheses:**

| | Verdict |
|---|---|
| H1: Response Speed → CSAT | Partially Confirmed (r = −0.148) |
| H2: Agent Tenure → CSAT | ✅ Confirmed — 50% gap OJT vs senior |
| H3: Category → CSAT | ✅ Confirmed — 0.8+ point spread |
        """)
    with c2:
        st.subheader("Architecture")
        st.code("""
Input (34 features)
Dense(1024) + BatchNorm + Swish + Dropout(0.20)
Residual Block 1 — Dense(512) + Skip
Residual Block 2 — Dense(512) + Skip
Residual Block 3 — Dense(256) + Skip
Residual Block 4 — Dense(256) + Skip
Dense(128) + GELU + Dropout(0.10)
Dense(64)  + GELU
Dense(5)   + Softmax
        """, language="text")

        # Show EDA charts if available
        st.subheader("Sample EDA Charts")
        chart1 = os.path.join(OUTPUTS_DIR, "chart1_csat_distribution.png")
        chart2 = os.path.join(OUTPUTS_DIR, "chart2_category_csat.png")
        if os.path.exists(chart1):
            st.image(chart1, caption="CSAT Score Distribution", use_column_width=True)
        if os.path.exists(chart2):
            st.image(chart2, caption="Mean CSAT by Category", use_column_width=True)
