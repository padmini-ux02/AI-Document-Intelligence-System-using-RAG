"""
Nexus AI – Document Intelligence System
Premium Streamlit UI with full RAG backend
"""
import os
import tempfile
import base64
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import RAGSystem

# Load environment variables
load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nexus AI | Document Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Image Loader Helpers ─────────────────────────────────────────────────────
def get_base64_image(image_path):
    if os.path.exists(image_path):
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode("utf-8")
        except Exception:
            pass
    return ""

app_bg_base64 = get_base64_image("app_background.png")
sidebar_bg_base64 = get_base64_image("sidebar_bg.png")
rag_icon_base64 = get_base64_image("rag_icon.png")

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

/* Main background image with fallback */
.stApp {{
    background: linear-gradient(135deg, #0a0a1a 0%, #0d0d2b 50%, #050518 100%) !important;
    background-image: url("data:image/png;base64,{app_bg_base64}") !important;
    background-repeat: no-repeat !important;
    background-position: center center !important;
    background-size: cover !important;
    background-attachment: fixed !important;
}}

/* Sidebar background image with fallback */
section[data-testid="stSidebar"] {{
    background: rgba(5,5,30,0.95) !important;
    background-image: url("data:image/png;base64,{sidebar_bg_base64}") !important;
    background-repeat: no-repeat !important;
    background-position: center center !important;
    background-size: cover !important;
    border-right: 1px solid rgba(0,242,255,0.2) !important;
}}

section[data-testid="stSidebar"] > div {{
    background: transparent !important;
}}

/* Hero */
.hero-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(2.2rem, 6vw, 3.8rem);
    font-weight: 700;
    text-align: center;
    background: linear-gradient(90deg, #00f2ff 0%, #7b2fff 50%, #ff2ff7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    padding: 25px 0 5px;
    letter-spacing: 3px;
    text-shadow: 0 0 30px rgba(0, 242, 255, 0.2);
}}
.hero-sub {{
    text-align: center;
    color: #a0aec0;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 35px;
    letter-spacing: 4px;
    text-transform: uppercase;
}}

/* GLASS cards */
.glass-card {{
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(0, 242, 255, 0.2);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
}}
.glass-card:hover {{
    border-color: rgba(0, 242, 255, 0.6);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0, 242, 255, 0.2);
}}
.card-title {{
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #00f2ff;
    margin-bottom: 10px;
    font-weight: 700;
}}
.card-body {{ color: #e2e8f0; font-size: 0.92rem; line-height: 1.7; }}

/* Metric badge */
.metric-badge {{
    display: inline-block;
    background: linear-gradient(90deg, rgba(0,242,255,0.15), rgba(123,47,255,0.15));
    border: 1px solid rgba(0,242,255,0.4);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: #00f2ff;
    margin: 4px;
    transition: all 0.3s ease;
}}
.metric-badge:hover {{
    border-color: #00f2ff;
    background: rgba(0,242,255,0.25);
}}

/* Chat bubbles */
.user-bubble {{
    background: linear-gradient(135deg, #7b2fff 0%, #4a00c8 100%);
    color: #fff;
    padding: 15px 20px;
    border-radius: 20px 20px 4px 20px;
    margin: 10px 0;
    box-shadow: 0 4px 20px rgba(123,47,255,0.35);
    font-size: 0.98rem;
}}
.bot-bubble {{
    background: rgba(0, 242, 255, 0.05);
    border: 1px solid rgba(0, 242, 255, 0.25);
    color: #e2f1f6;
    padding: 15px 20px;
    border-radius: 4px 20px 20px 20px;
    margin: 10px 0;
    font-size: 0.98rem;
    line-height: 1.7;
    backdrop-filter: blur(4px);
}}
.bot-bubble strong {{ color: #00f2ff; }}
.source-pill {{
    display: inline-block;
    background: rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(0, 242, 255, 0.3);
    border-left: 4px solid #00f2ff;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.85rem;
    color: #cbd5e1;
    margin: 6px 0;
    line-height: 1.6;
}}
.confidence-bar-wrap {{ 
    display: flex; align-items: center; gap: 12px; margin-top: 10px;
}}
.confidence-bar {{
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #00f2ff, #7b2fff);
}}

/* Welcome screen box styling */
.welcome-box {{
    background: rgba(10, 10, 30, 0.6) !important;
    border: 1px solid rgba(0, 242, 255, 0.3) !important;
    border-radius: 24px !important;
    padding: 45px 30px !important;
    text-align: center !important;
    margin-top: 25px !important;
    backdrop-filter: blur(12px) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5) !important;
}}
.welcome-avatar-container {{
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 25px;
}}
.welcome-avatar {{
    width: 130px;
    height: 130px;
    border-radius: 50%;
    object-fit: cover;
    border: 3px solid #00f2ff;
    box-shadow: 0 0 25px rgba(0, 242, 255, 0.5);
    animation: float 4s ease-in-out infinite;
}}
@keyframes float {{
    0% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-10px); }}
    100% {{ transform: translateY(0px); }}
}}
.welcome-title {{
    font-family: 'Orbitron', sans-serif;
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    text-shadow: 0 0 15px rgba(0, 242, 255, 0.6);
    margin-bottom: 12px;
}}
.welcome-desc {{
    font-size: 1.15rem;
    color: #cbd5e1;
    line-height: 1.7;
    margin-bottom: 30px;
    font-weight: 300;
    max-width: 700px;
    margin-left: auto;
    margin-right: auto;
}}

/* Features strip & chips */
.feat-strip {{
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 35px;
}}
.feat-chip {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(0, 242, 255, 0.25);
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 0.9rem;
    color: #e2e8f0;
    display: flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}}
.feat-chip:hover {{
    background: rgba(0, 242, 255, 0.12);
    border-color: #00f2ff;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 242, 255, 0.3);
}}
.feat-chip-icon {{
    font-size: 1.1rem;
}}

/* Step items */
.steps-container {{
    max-width: 600px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
}}
.step-item {{
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(0, 242, 255, 0.15) !important;
    border-left: 4px solid #00f2ff !important;
    border-radius: 10px !important;
    padding: 15px 20px !important;
    text-align: left !important;
    color: #cbd5e1 !important;
    font-size: 0.95rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    transition: background 0.3s ease !important;
}}
.step-item:hover {{
    background: rgba(0, 242, 255, 0.04) !important;
}}
.step-num {{
    font-size: 1.25rem;
    font-weight: 700;
    color: #00f2ff;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(90deg, #00b4d8, #7b2fff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 1px !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 180, 216, 0.25) !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 0 25px rgba(0, 180, 216, 0.5) !important;
}}

/* INPUT FIELD STYLING: Force White background and Black text when typing */
.stTextInput input, .stTextArea textarea, div[data-testid="stChatInput"] textarea {{
    background-color: #ffffff !important;
    color: #000000 !important;
    caret-color: #000000 !important;
    font-weight: 500 !important;
}}

.stTextInput input::placeholder, .stTextArea textarea::placeholder, div[data-testid="stChatInput"] textarea::placeholder {{
    color: #64748b !important;
    opacity: 0.9 !important;
}}

/* Containers of the text inputs */
.stTextInput > div > div, .stTextArea > div > div {{
    background-color: #ffffff !important;
    border: 2px solid rgba(0, 242, 255, 0.5) !important;
    border-radius: 10px !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1) !important;
}}

.stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {{
    border-color: #7b2fff !important;
    box-shadow: 0 0 12px rgba(123, 47, 255, 0.4) !important;
}}

/* Streamlit chat input outer container styling */
div[data-testid="stChatInput"] {{
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}}

div[data-testid="stChatInput"] > div {{
    background-color: #ffffff !important;
    border: 2px solid #00f2ff !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
    padding: 4px 8px !important;
}}

div[data-testid="stChatInput"] > div:focus-within {{
    border-color: #7b2fff !important;
    box-shadow: 0 0 15px rgba(123, 47, 255, 0.5) !important;
}}

.stFileUploader {{ 
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px dashed rgba(0, 242, 255, 0.3) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {{ 
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    gap: 8px !important;
    border-bottom: none !important;
}}

.stTabs [data-baseweb="tab"] {{
    border-radius: 8px !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.98rem !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}}

.stTabs [data-baseweb="tab"]:hover {{
    color: #00f2ff !important;
    background: rgba(0, 242, 255, 0.05) !important;
}}

.stTabs [aria-selected="true"] {{
    background: rgba(0, 242, 255, 0.15) !important;
    color: #00f2ff !important;
    border-bottom: 2px solid #00f2ff !important;
}}

/* Divider */
.custom-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,242,255,0.3), transparent);
    margin: 20px 0;
}}
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "history" not in st.session_state:
    st.session_state.history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "summary" not in st.session_state:
    st.session_state.summary = None
if "keywords" not in st.session_state:
    st.session_state.keywords = []
if "topics" not in st.session_state:
    st.session_state.topics = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-family:'Orbitron',sans-serif; color:#00f2ff; font-size:1.1rem; letter-spacing:2px">NEXUS AI</div>
        <div style="color:#4a6070; font-size:0.7rem; letter-spacing:1px; margin-top:3px">DOCUMENT INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card-title">⚙️ Engine Mode</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-... (optional, for best results)",
        help="Without a key, the system uses a local TF-IDF engine. With a key, it uses GPT for answers."
    )

    if api_key:
        st.success("✅ OpenAI mode active")
    else:
        st.info("🔵 Local mode — no key required")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drag & Drop files here",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.caption(f"📁 {len(uploaded_files)} file(s) selected")

    btn_activate = st.button("⚡ BUILD KNOWLEDGE BASE", use_container_width=True, disabled=not bool(uploaded_files))

    if btn_activate and uploaded_files:
        with st.status("🔄 Building intelligence...", expanded=True) as status:
            # Save to temp
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            for f in uploaded_files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                file_paths.append(path)
                st.write(f"📄 Loaded: **{f.name}**")

            # Init RAG
            st.write("🧠 Initializing engine...")
            st.session_state.rag = RAGSystem(api_key=api_key if api_key else None)

            st.write("⚙️ Chunking & Indexing...")
            try:
                count = st.session_state.rag.process_and_index(file_paths)
                st.session_state.chunk_count = count

                st.write("📊 Extracting insights...")
                st.session_state.summary = st.session_state.rag.generate_summary()
                st.session_state.keywords = st.session_state.rag.extract_keywords()
                st.session_state.topics = st.session_state.rag.extract_topics()
                st.session_state.indexed = True
                st.session_state.history = []
                status.update(label=f"✅ {count} chunks indexed!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Build Failed", state="error", expanded=True)
                st.error(str(e))
                st.session_state.indexed = False
                st.stop()

    if st.session_state.indexed:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="padding:12px">
            <div class="card-title">📊 Knowledge Base</div>
            <div class="card-body">
              <b style="color:#00f2ff">{st.session_state.chunk_count}</b> chunks indexed<br>
              <b style="color:#7b2fff">{len(st.session_state.history) // 2}</b> Q&As this session
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear & Reset", use_container_width=True):
            for key in ["rag", "history", "indexed", "summary", "keywords", "topics", "chunk_count"]:
                if key in ["rag", "summary"]:
                    st.session_state[key] = None
                elif key in ["history", "keywords", "topics"]:
                    st.session_state[key] = []
                elif key == "indexed":
                    st.session_state[key] = False
                elif key == "chunk_count":
                    st.session_state[key] = 0
            st.rerun()

# ─── Main UI ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">NEXUS AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Retrieval Augmented Generation · Document Intelligence System</p>', unsafe_allow_html=True)

# ─── WELCOME SCREEN ───────────────────────────────────────────────────────────
if not st.session_state.indexed:
    st.markdown(f"""
    <div class="welcome-box">
        <div class="welcome-avatar-container">
            <img src="data:image/png;base64,{rag_icon_base64}" class="welcome-avatar" alt="RAG Icon">
        </div>
        <div class="welcome-title">Intelligence Awaits</div>
        <p class="welcome-desc">
            Transform your documents into an interactive knowledge base<br>
            powered by next-generation Retrieval Augmented Generation.
        </p>
        
        <div class="feat-strip">
            <div class="feat-chip"><span class="feat-chip-icon">🧠</span> Hybrid BM25 + TF-IDF</div>
            <div class="feat-chip"><span class="feat-chip-icon">🔍</span> Smart Re-ranking</div>
            <div class="feat-chip"><span class="feat-chip-icon">💬</span> Conversation Memory</div>
            <div class="feat-chip"><span class="feat-chip-icon">📊</span> Live Analytics</div>
        </div>
        
        <div class="steps-container">
            <div class="step-item"><span class="step-num">①</span> Upload PDF, DOCX, or TXT files from the sidebar</div>
            <div class="step-item"><span class="step-num">②</span> Click <b style="color:#00ffff;">⚡ BUILD KNOWLEDGE BASE</b> to index</div>
            <div class="step-item"><span class="step-num">③</span> Ask anything about your documents...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── DASHBOARD ───────────────────────────────────────────────────────────────
else:
    # --- TABS ---
    tab_chat, tab_insights = st.tabs(["💬 Neural Chat", "📊 Document Insights"])

    # ══ TAB 1: CHAT ══════════════════════════════════════════════════════════
    with tab_chat:
        # Render history
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-bubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                ans = msg["content"].replace("\n", "<br>")
                conf = msg.get("confidence", 0)
                pct = int(conf * 100)
                color = "#00f2ff" if pct >= 70 else "#ffb300" if pct >= 40 else "#ff4444"
                st.markdown(f"""
                <div class="bot-bubble">
                    🧠 {ans}
                    <div class="confidence-bar-wrap" style="margin-top:10px">
                        <span style="font-size:0.7rem; color:#4a7080; min-width:80px">Confidence</span>
                        <div class="confidence-bar" style="width:{pct}%; background:linear-gradient(90deg,{color},#7b2fff)"></div>
                        <span style="font-size:0.7rem; color:{color}; min-width:35px">{pct}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if msg.get("sources"):
                    with st.expander("📍 Source Passages"):
                        for s in msg["sources"]:
                            score_pct = int(s['score'] * 100)
                            st.markdown(f"""
                            <div class="source-pill">
                                <span style="color:#00f2ff; font-size:0.7rem">📄 {s['source']} &nbsp;|&nbsp; Relevance: {score_pct}%</span><br>
                                {s['content'][:300]}{'...' if len(s['content']) > 300 else ''}
                            </div>
                            """, unsafe_allow_html=True)

    # ══ TAB 2: INSIGHTS ══════════════════════════════════════════════════════
    with tab_insights:
        c1, c2 = st.columns([2, 1])

        with c1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">📋 Document Summary</div>
                <div class="card-body">{st.session_state.summary or "No summary available."}</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            kw_html = "".join(f'<span class="metric-badge">{k}</span>' for k in st.session_state.keywords)
            tp_html = "".join(f'<span class="metric-badge" style="border-color:rgba(123,47,255,0.4); color:#a060ff">{t}</span>' for t in st.session_state.topics)

            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">🏷️ Key Signals</div>
                <div style="line-height:2;">{kw_html or "<span style='color:#4a6070'>None detected</span>"}</div>
            </div>
            <div class="glass-card">
                <div class="card-title">🌐 Named Entities / Topics</div>
                <div style="line-height:2;">{tp_html or "<span style='color:#4a6070'>None detected</span>"}</div>
            </div>
            """, unsafe_allow_html=True)

    # ─── GLOBAL CHAT INPUT ───────────────────────────────────────────────────
    # Note: st.chat_input MUST be at the top level, outside of tabs/columns.
    query = st.chat_input("Ask anything about your documents...")
    if query:
        st.session_state.history.append({"role": "user", "content": query})
        with st.spinner("🧠 Processing..."):
            result = st.session_state.rag.get_response(query)
        st.session_state.history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"]
        })
        st.rerun()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 40px 0 10px; color:#2a3a4a; font-size:0.7rem; letter-spacing:2px">
    NEXUS AI v2.0 &nbsp;·&nbsp; RAG PIPELINE &nbsp;·&nbsp; ZERO TORCH DEPENDENCY
</div>
""", unsafe_allow_html=True)
