import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import hashlib
import joblib
from datetime import datetime

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeGuard – Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
VECT_FILE  = os.path.join(BASE_DIR, "vectorizer.pkl")
HIST_FILE  = os.path.join(BASE_DIR, "history.json")
FAKE_CSV   = os.path.join(BASE_DIR, "fake.csv")
TRUE_CSV   = os.path.join(BASE_DIR, "true.csv")

# ─── Helpers: persistence ──────────────────────────────────────────────────────
def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

# ─── User management ───────────────────────────────────────────────────────────
def load_users():
    return load_json(USERS_FILE, {})

def save_users(users):
    save_json(USERS_FILE, users)

def register_user(username, password, email):
    users = load_users()
    if username in users or username == "admin":
        return False, "Username already exists."
    users[username] = {"password": hash_pw(password), "email": email, "created": str(datetime.now())}
    save_users(users)
    return True, "Registered successfully!"

def verify_user(username, password):
    if username == "admin" and password == "admin123":
        return True, "admin"
    users = load_users()
    if username in users and users[username]["password"] == hash_pw(password):
        return True, "user"
    return False, None

# ─── History ───────────────────────────────────────────────────────────────────
def load_history():
    return load_json(HIST_FILE, [])

def save_prediction(username, title, text, result, confidence):
    history = load_history()
    history.append({
        "user": username,
        "title": title[:80] if title else "",
        "text_snippet": text[:120] if text else "",
        "result": result,
        "confidence": round(confidence * 100, 2),
        "timestamp": str(datetime.now())
    })
    save_json(HIST_FILE, history)

# ─── Model training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
        model = joblib.load(MODEL_FILE)
        vect  = joblib.load(VECT_FILE)
        return model, vect, "loaded"

    if not os.path.exists(FAKE_CSV) or not os.path.exists(TRUE_CSV):
        return None, None, "missing_data"

    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline

    fake = pd.read_csv(FAKE_CSV)
    true = pd.read_csv(TRUE_CSV)
    fake["label"] = 1   # 1 = Fake
    true["label"] = 0   # 0 = Real

    df = pd.concat([fake, true], ignore_index=True)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).str.strip()
    df = df[df["content"].str.len() > 10]

    X = df["content"]
    y = df["label"]

    vect  = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    X_vec = vect.fit_transform(X)

    model = LogisticRegression(max_iter=1000, C=5, solver="lbfgs")
    model.fit(X_vec, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vect,  VECT_FILE)
    return model, vect, "trained"

def predict(model, vect, title, text):
    content = (title + " " + text).strip()
    X = vect.transform([content])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return ("FAKE", prob[1]) if pred == 1 else ("REAL", prob[0])

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0f1117; color: #e2e8f0; }

/* Cards */
.card {
    background: #1e2130;
    border: 1px solid #2d3348;
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* Result badges */
.badge-fake {
    display: inline-block;
    background: linear-gradient(135deg,#ff4d6d,#c9184a);
    color: #fff;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 10px 32px;
    border-radius: 40px;
    letter-spacing: 2px;
    box-shadow: 0 4px 18px rgba(255,77,109,0.4);
}
.badge-real {
    display: inline-block;
    background: linear-gradient(135deg,#06d6a0,#118ab2);
    color: #fff;
    font-size: 1.6rem;
    font-weight: 700;
    padding: 10px 32px;
    border-radius: 40px;
    letter-spacing: 2px;
    box-shadow: 0 4px 18px rgba(6,214,160,0.4);
}

/* Confidence bar */
.conf-bar-wrap { background:#2d3348; border-radius:8px; height:14px; margin-top:8px; }
.conf-bar      { height:14px; border-radius:8px; transition: width .5s ease; }

/* Metric tiles */
.metric-tile {
    background: #252a3d;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    border: 1px solid #2d3348;
}
.metric-val  { font-size: 2rem; font-weight: 700; color: #7c83fd; }
.metric-label{ font-size: .8rem; color: #94a3b8; margin-top: 4px; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #161824 !important; }
section[data-testid="stSidebar"] .stButton button {
    width: 100%; border-radius: 8px; background: #252a3d;
    color: #e2e8f0; border: 1px solid #2d3348; margin-bottom: 4px;
}
section[data-testid="stSidebar"] .stButton button:hover { background: #7c83fd; color:#fff; }

/* Primary button */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#7c83fd,#5c63d8);
    color: #fff; border: none; border-radius: 10px;
    padding: 10px 28px; font-weight: 600; font-size: 1rem;
    box-shadow: 0 4px 14px rgba(124,131,253,0.4);
}
div.stButton > button[kind="primary"]:hover { opacity: .9; }

/* Inputs */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea {
    background: #252a3d !important;
    color: #e2e8f0 !important;
    border: 1px solid #3a4060 !important;
    border-radius: 8px !important;
}

/* Hide default menu */
#MainMenu, footer { visibility: hidden; }

h1,h2,h3 { color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ─────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in  = False
    st.session_state.username   = ""
    st.session_state.role       = ""
    st.session_state.page       = "login"
    st.session_state.result     = None
    st.session_state.model_ready= False

# ─── Load model once ───────────────────────────────────────────────────────────
model, vect, model_status = load_or_train_model()
if model is not None:
    st.session_state.model_ready = True

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGES
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    col1, col2, col3 = st.columns([1,1.4,1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 40px 0 10px'>
            <div style='font-size:3.5rem'>🛡️</div>
            <h1 style='font-size:2rem; margin:0'>FakeGuard</h1>
            <p style='color:#94a3b8; margin-top:4px'>AI-Powered Fake News Detector</p>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_reg = st.tabs(["🔑  Login", "📝  Register"])

        with tab_login:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            username = st.text_input("Username", key="li_user", placeholder="Enter username")
            password = st.text_input("Password", type="password", key="li_pass", placeholder="Enter password")
            if st.button("Login", type="primary", use_container_width=True):
                ok, role = verify_user(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username  = username
                    st.session_state.role      = role
                    st.session_state.page      = "admin" if role == "admin" else "detect"
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_reg:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            r_user  = st.text_input("Username",  key="r_user",  placeholder="Choose a username")
            r_email = st.text_input("Email",      key="r_email", placeholder="your@email.com")
            r_pass  = st.text_input("Password",   type="password", key="r_pass",  placeholder="Min 6 characters")
            r_pass2 = st.text_input("Confirm PW", type="password", key="r_pass2", placeholder="Repeat password")
            if st.button("Create Account", type="primary", use_container_width=True):
                if not r_user or not r_email or not r_pass:
                    st.error("All fields are required.")
                elif len(r_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                elif r_pass != r_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(r_user, r_pass, r_email)
                    if ok:
                        st.success(f"✅ {msg} You can now login.")
                    else:
                        st.error(f"❌ {msg}")
            st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR (authenticated)
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align:center; padding:20px 0 10px'>
            <div style='font-size:2.5rem'>🛡️</div>
            <div style='font-size:1.1rem; font-weight:700; color:#7c83fd'>FakeGuard</div>
            <div style='font-size:.8rem; color:#94a3b8; margin-top:4px'>
                👤 {st.session_state.username}
                {"&nbsp;&nbsp;<span style='background:#7c83fd;color:#fff;padding:2px 8px;border-radius:10px;font-size:.7rem'>ADMIN</span>" if st.session_state.role=="admin" else ""}
            </div>
        </div>
        <hr style='border-color:#2d3348; margin: 10px 0 16px'>
        """, unsafe_allow_html=True)

        if st.button("🔍  Detect News"):
            st.session_state.page = "detect"
            st.session_state.result = None
            st.rerun()
        if st.button("📜  My History"):
            st.session_state.page = "history"
            st.rerun()
        if st.session_state.role == "admin":
            if st.button("⚙️  Admin Panel"):
                st.session_state.page = "admin"
                st.rerun()

        st.markdown("<hr style='border-color:#2d3348; margin:16px 0'>", unsafe_allow_html=True)

        # Model status
        status_color = "#06d6a0" if st.session_state.model_ready else "#ff4d6d"
        status_text  = "Model Ready ✓" if st.session_state.model_ready else "Model Not Loaded"
        st.markdown(f"""
        <div style='font-size:.75rem; color:{status_color}; text-align:center; padding:6px;
             background:#1e2130; border-radius:8px; border:1px solid #2d3348'>
            {status_text}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪  Logout"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  DETECT PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_detect():
    st.markdown("## 🔍 Fake News Detector")
    st.markdown("<p style='color:#94a3b8'>Paste a news article below to analyse its authenticity.</p>", unsafe_allow_html=True)

    if not st.session_state.model_ready:
        st.warning("⚠️ Model not available. Please ensure **fake.csv** and **true.csv** are in the app directory, then restart.")
        if model_status == "missing_data":
            st.info(f"Expected files at:\n- `{FAKE_CSV}`\n- `{TRUE_CSV}`")
        return

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    title = st.text_input("📰 Article Title", placeholder="Enter article headline…")
    text  = st.text_area("📄 Article Body",  placeholder="Paste full article text here…", height=220)

    col_a, col_b = st.columns([1,5])
    with col_a:
        analyse = st.button("🔍 Analyse", type="primary")
    with col_b:
        if st.button("🔄 Clear"):
            st.session_state.result = None
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if analyse:
        if not title.strip() and not text.strip():
            st.error("Please enter a title or article text.")
        else:
            with st.spinner("Analysing article…"):
                label, conf = predict(model, vect, title, text)
            st.session_state.result = (label, conf, title, text)
            save_prediction(st.session_state.username, title, text, label, conf)

    if st.session_state.result:
        label, conf, _, _ = st.session_state.result
        is_fake = label == "FAKE"

        st.markdown("---")
        st.markdown("### 📊 Analysis Result")

        c1, c2 = st.columns([1, 1])
        with c1:
            badge = f"<span class='badge-fake'>🚨 FAKE NEWS</span>" if is_fake else f"<span class='badge-real'>✅ REAL NEWS</span>"
            st.markdown(f"<div style='margin-bottom:16px'>{badge}</div>", unsafe_allow_html=True)

            conf_pct = conf * 100
            bar_color = "#ff4d6d" if is_fake else "#06d6a0"
            st.markdown(f"""
            <p style='color:#94a3b8; margin-bottom:4px'>Confidence: <strong style='color:#e2e8f0'>{conf_pct:.1f}%</strong></p>
            <div class='conf-bar-wrap'>
                <div class='conf-bar' style='width:{conf_pct:.1f}%; background:{bar_color}'></div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            risk_level = "High Risk" if conf > 0.85 else ("Medium Risk" if conf > 0.60 else "Low Risk")
            risk_color = "#ff4d6d" if conf > 0.85 else ("#f9c74f" if conf > 0.60 else "#06d6a0")
            st.markdown(f"""
            <div class='card' style='padding:20px'>
                <div style='font-size:.85rem; color:#94a3b8'>Risk Level</div>
                <div style='font-size:1.4rem; font-weight:700; color:{risk_color}'>{risk_level}</div>
                <hr style='border-color:#2d3348; margin:12px 0'>
                <div style='font-size:.85rem; color:#94a3b8'>Verdict</div>
                <div style='font-size:.95rem; color:#e2e8f0'>
                    {"This article shows strong indicators of being fabricated or misleading." if is_fake
                     else "This article appears to be credible and factually presented."}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Tips
        if is_fake:
            st.markdown("""
            <div class='card' style='border-left:4px solid #ff4d6d'>
            <strong>⚠️ What to watch for:</strong><br>
            <span style='color:#94a3b8'>Sensational headlines · Emotional language · Missing sources · 
            Unverified quotes · Politically charged framing</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='card' style='border-left:4px solid #06d6a0'>
            <strong>✅ Good signs detected:</strong><br>
            <span style='color:#94a3b8'>Neutral tone · Factual language · Credible structure · 
            Attribution patterns consistent with real journalism</span>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HISTORY PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_history():
    st.markdown("## 📜 My Detection History")
    history = load_history()
    user_hist = [h for h in history if h["user"] == st.session_state.username]

    if not user_hist:
        st.info("No history yet. Start detecting articles!")
        return

    user_hist = list(reversed(user_hist))
    total = len(user_hist)
    fakes = sum(1 for h in user_hist if h["result"] == "FAKE")
    reals = total - fakes

    c1, c2, c3 = st.columns(3)
    for col, val, label in [(c1, total, "Total Checked"), (c2, fakes, "Fake Detected"), (c3, reals, "Real Detected")]:
        with col:
            color = "#7c83fd" if label == "Total Checked" else ("#ff4d6d" if label == "Fake Detected" else "#06d6a0")
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-val' style='color:{color}'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for h in user_hist:
        is_fake  = h["result"] == "FAKE"
        badge_bg = "#3d1a25" if is_fake else "#0d2d24"
        badge_cl = "#ff4d6d" if is_fake else "#06d6a0"
        badge_tx = "🚨 FAKE" if is_fake else "✅ REAL"
        st.markdown(f"""
        <div class='card' style='padding:16px 22px; margin-bottom:12px'>
            <div style='display:flex; justify-content:space-between; align-items:center'>
                <div style='font-weight:600; font-size:.95rem; max-width:70%'>{h['title'] or "—"}</div>
                <span style='background:{badge_bg}; color:{badge_cl}; padding:4px 14px;
                    border-radius:20px; font-size:.8rem; font-weight:700'>{badge_tx}</span>
            </div>
            <div style='color:#94a3b8; font-size:.8rem; margin-top:6px'>{h['text_snippet']}…</div>
            <div style='color:#4a5568; font-size:.75rem; margin-top:8px'>
                Confidence: {h['confidence']}% &nbsp;·&nbsp; {h['timestamp'][:19]}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
def page_admin():
    st.markdown("## ⚙️ Admin Panel")
    users   = load_users()
    history = load_history()

    # Summary metrics
    total_users  = len(users)
    total_checks = len(history)
    total_fake   = sum(1 for h in history if h["result"] == "FAKE")
    total_real   = total_checks - total_fake

    cols = st.columns(4)
    metrics = [
        ("👥 Users", total_users, "#7c83fd"),
        ("🔍 Total Checks", total_checks, "#5c63d8"),
        ("🚨 Fake Detected", total_fake, "#ff4d6d"),
        ("✅ Real Detected", total_real, "#06d6a0"),
    ]
    for col, (label, val, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-val' style='color:{color}'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["👥 Users", "📊 All Detections", "🗑️ Management"])

    with tab1:
        st.markdown("### Registered Users")
        if users:
            df_users = pd.DataFrame([
                {"Username": u, "Email": d.get("email","—"), "Registered": d.get("created","—")[:19]}
                for u, d in users.items()
            ])
            st.dataframe(df_users, use_container_width=True, hide_index=True)
        else:
            st.info("No registered users yet.")

    with tab2:
        st.markdown("### Detection Log")
        if history:
            df_hist = pd.DataFrame(history)
            df_hist = df_hist[["user","title","result","confidence","timestamp"]].copy()
            df_hist.columns = ["User","Title","Result","Confidence %","Timestamp"]
            df_hist["Timestamp"] = df_hist["Timestamp"].str[:19]
            df_hist = df_hist.iloc[::-1].reset_index(drop=True)

            # colour result column
            def color_result(val):
                return "color: #ff4d6d; font-weight:700" if val=="FAKE" else "color:#06d6a0; font-weight:700"

            st.dataframe(
                df_hist.style.applymap(color_result, subset=["Result"]),
                use_container_width=True, hide_index=True
            )

            # Per-user breakdown
            st.markdown("### Per-User Statistics")
            if history:
                user_stats = {}
                for h in history:
                    u = h["user"]
                    user_stats.setdefault(u, {"Total":0,"Fake":0,"Real":0})
                    user_stats[u]["Total"] += 1
                    if h["result"] == "FAKE":
                        user_stats[u]["Fake"] += 1
                    else:
                        user_stats[u]["Real"] += 1
                df_stats = pd.DataFrame([{"User":k,**v} for k,v in user_stats.items()])
                st.dataframe(df_stats, use_container_width=True, hide_index=True)
        else:
            st.info("No detections recorded yet.")

    with tab3:
        st.markdown("### Danger Zone")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("<div class='card' style='border-left:4px solid #ff4d6d'>", unsafe_allow_html=True)
            st.markdown("**🗑️ Clear Detection History**")
            st.markdown("<span style='color:#94a3b8;font-size:.85rem'>Permanently removes all detection records.</span>", unsafe_allow_html=True)
            if st.button("Clear All History", key="clr_hist"):
                save_json(HIST_FILE, [])
                st.success("Detection history cleared.")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='card' style='border-left:4px solid #ff4d6d'>", unsafe_allow_html=True)
            st.markdown("**👤 Delete a User**")
            user_list = list(users.keys())
            if user_list:
                del_user = st.selectbox("Select user", user_list, key="del_user")
                if st.button("Delete User", key="do_del"):
                    del users[del_user]
                    save_users(users)
                    st.success(f"User '{del_user}' deleted.")
                    st.rerun()
            else:
                st.info("No users to delete.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Retrain model
        st.markdown("<div class='card' style='border-left:4px solid #7c83fd; margin-top:16px'>", unsafe_allow_html=True)
        st.markdown("**🔁 Retrain Model**")
        st.markdown("<span style='color:#94a3b8;font-size:.85rem'>Deletes cached model and forces retraining from CSV files.</span>", unsafe_allow_html=True)
        if st.button("Retrain Model", key="retrain"):
            for f in [MODEL_FILE, VECT_FILE]:
                if os.path.exists(f):
                    os.remove(f)
            st.cache_resource.clear()
            st.success("Cache cleared. Restart the app to retrain.")
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    page_login()
else:
    render_sidebar()
    page = st.session_state.page
    if page == "detect":
        page_detect()
    elif page == "history":
        page_history()
    elif page == "admin":
        if st.session_state.role == "admin":
            page_admin()
        else:
            st.error("Access denied.")
            st.session_state.page = "detect"
            st.rerun()
    else:
        page_detect()
