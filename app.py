"""
app.py
──────
YouTube RAG Chatbot — Streamlit Application
Run with:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from rag_engine import YouTubeRAG, extract_video_id, fetch_transcript
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YT Chat — YouTube AI Chatbot",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

/* ── CSS Variables ── */
:root {
    --bg:          #F7F5F2;
    --bg-card:     #FFFFFF;
    --bg-sidebar:  #FEFEFE;
    --accent:      #E05252;
    --accent-soft: #FFF0F0;
    --accent-2:    #F4A261;
    --text-1:      #1A1A2E;
    --text-2:      #5C5C7A;
    --text-3:      #A0A0B8;
    --border:      rgba(26,26,46,0.08);
    --border-md:   rgba(26,26,46,0.13);
    --green:       #2DB87B;
    --green-soft:  #EDFAF4;
    --shadow-sm:   0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:   0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04);
    --shadow-lg:   0 12px 40px rgba(0,0,0,0.10);
    --radius-sm:   8px;
    --radius-md:   14px;
    --radius-lg:   20px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text-1) !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── App Header ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.logo-pill {
    background: linear-gradient(135deg, #E05252, #F4A261);
    border-radius: var(--radius-sm);
    width: 42px; height: 42px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(224,82,82,0.30);
    color: white;
}
.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.3px;
    color: var(--text-1);
    margin: 0; line-height: 1.2;
}
.app-sub {
    font-size: 11px;
    color: var(--text-3);
    font-family: 'DM Mono', monospace;
    margin: 0;
    margin-top: 2px;
}

/* ── Sidebar section labels ── */
.section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: var(--text-3);
    margin: 1.2rem 0 0.6rem;
}

/* ── Video chips ── */
.video-chip {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: box-shadow 0.18s;
}
.video-chip:hover { box-shadow: var(--shadow-sm); }
.video-chip-id {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--accent);
    font-weight: 500;
}
.video-chip-status { font-size: 11px; font-weight: 500; }
.status-ready   { color: var(--green); }
.status-pending { color: var(--text-3); }
.status-error   { color: var(--accent); }

/* ── Text input ── */
.stTextInput > div > div > input {
    background: var(--bg) !important;
    border: 1.5px solid var(--border-md) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 14px !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
.stTextInput > div > div > input::placeholder {
    color: var(--text-3) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(224,82,82,0.10) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #E05252, #E8694A) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100% !important;
    padding: 0.6rem 1.1rem !important;
    box-shadow: 0 3px 10px rgba(224,82,82,0.25) !important;
    transition: all 0.2s !important;
    letter-spacing: 0.1px !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 18px rgba(224,82,82,0.30) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }
.stButton > button:disabled {
    background: var(--border) !important;
    color: var(--text-3) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Radio (language toggle) ── */
.stRadio > div {
    flex-direction: row !important;
    gap: 1rem;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 8px 12px;
}
.stRadio label { font-size: 13px !important; color: var(--text-2) !important; font-weight: 500 !important; }
.stRadio [data-checked="true"] label { color: var(--text-1) !important; font-weight: 600 !important; }

/* ── Chat container ── */
.chat-container {
    max-width: 740px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 18px;
    padding-bottom: 2.5rem;
}

/* ── Message rows ── */
.msg-row-user {
    display: flex;
    flex-direction: row-reverse;
    gap: 10px;
    align-items: flex-start;
    animation: fadeUp 0.25s ease both;
}
.msg-row-bot {
    display: flex;
    flex-direction: row;
    gap: 10px;
    align-items: flex-start;
    animation: fadeUp 0.25s ease both;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Avatars ── */
.avatar {
    width: 36px; height: 36px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
    flex-shrink: 0;
    letter-spacing: -0.5px;
}
.avatar-user {
    background: var(--bg);
    border: 1.5px solid var(--border-md);
    color: var(--text-2);
}
.avatar-bot {
    background: linear-gradient(135deg, #E05252, #F4A261);
    color: white;
    box-shadow: 0 3px 8px rgba(224,82,82,0.25);
}

/* ── Bubbles ── */
.bubble-user {
    background: linear-gradient(135deg, #E05252, #E8694A);
    border-radius: 16px 4px 16px 16px;
    padding: 13px 18px;
    font-size: 15px;
    line-height: 1.65;
    color: white;
    max-width: 520px;
    box-shadow: 0 4px 12px rgba(224,82,82,0.20);
    font-weight: 400;
}
.bubble-bot {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px 16px 16px 16px;
    padding: 13px 18px;
    font-size: 15px;
    line-height: 1.75;
    color: var(--text-1);
    max-width: 560px;
    box-shadow: var(--shadow-sm);
}
.bubble-bot.urdu {
    font-family: 'Noto Nastaliq Urdu', serif !important;
    direction: rtl;
    text-align: right;
    line-height: 2.5;
    font-size: 16px;
}
.lang-tag {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--text-3);
    margin-top: 5px;
    padding-left: 2px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem 3rem;
    color: var(--text-3);
}
.empty-icon {
    font-size: 52px;
    margin-bottom: 20px;
    filter: drop-shadow(0 4px 12px rgba(224,82,82,0.2));
}
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text-1);
    margin-bottom: 10px;
}
.empty-sub {
    font-size: 15px;
    line-height: 1.8;
    color: var(--text-2);
}

/* ── Quick chips (empty state) ── */
.chips-row {
    display: flex; flex-wrap: wrap; gap: 9px;
    justify-content: center;
    margin-top: 1.4rem;
}
.chip {
    background: var(--bg-card);
    border: 1.5px solid var(--border-md);
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 13px;
    color: var(--text-2);
    cursor: pointer;
    font-weight: 500;
    transition: all 0.18s;
    box-shadow: var(--shadow-sm);
}
.chip:hover {
    border-color: var(--accent);
    color: var(--accent);
    box-shadow: 0 3px 10px rgba(224,82,82,0.12);
}

/* ── Status bar ── */
.status-bar {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 11px 18px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: var(--text-2);
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: var(--shadow-sm);
}

/* ── Progress / spinner ── */
.stProgress > div > div { background-color: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Alerts ── */
.stAlert {
    border-radius: var(--radius-sm) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-md); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-3); }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1rem 0 !important;
}

/* ── Footer info ── */
.footer-info {
    font-size: 11px;
    color: var(--text-3);
    font-family: 'DM Mono', monospace;
    line-height: 2;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    margin-top: 0.5rem;
}
.footer-info span {
    color: var(--accent);
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ─────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "rag": YouTubeRAG(),
        "videos": {},          # id → {"status": ..., "error": ...}
        "messages": [],        # [{"role": "user"|"bot", "content": str}]
        "is_built": False,
        "language": "English",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="app-header">
        <div class="logo-pill">▶</div>
        <div>
            <p class="app-title">YT Chat</p>
            <p class="app-sub">YouTube AI Chatbot</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Language ──
    st.markdown('<div class="section-label">🌐 Response Language</div>', unsafe_allow_html=True)
    lang = st.radio(
        "language",
        options=["English", "Urdu — اردو"],
        index=0 if st.session_state.language == "English" else 1,
        label_visibility="collapsed",
        horizontal=True,
    )
    st.session_state.language = "Urdu" if "Urdu" in lang else "English"

    st.markdown("---")

    # ── Add video ──
    st.markdown('<div class="section-label">📹 Add YouTube Videos</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "url_input",
        placeholder="youtube.com/watch?v=... or Video ID",
        label_visibility="collapsed",
    )

    col_add, col_clear = st.columns([3, 1])
    with col_add:
        if st.button("＋ Add Video", use_container_width=True):
            vid_id = extract_video_id(url_input) if url_input.strip() else None
            if not vid_id:
                st.error("Invalid URL or ID. Try again.")
            elif vid_id in st.session_state.videos:
                st.warning("Already added!")
            else:
                st.session_state.videos[vid_id] = {"status": "pending", "error": ""}
                st.rerun()

    with col_clear:
        if st.button("🗑", use_container_width=True, help="Clear all videos"):
            st.session_state.videos.clear()
            st.session_state.rag.clear()
            st.session_state.is_built = False
            st.session_state.messages.clear()
            st.rerun()

    # ── Video list ──
    if st.session_state.videos:
        st.markdown('<div class="section-label">📋 Loaded Videos</div>', unsafe_allow_html=True)
        to_remove = []
        for vid_id, info in st.session_state.videos.items():
            status = info["status"]
            icon = "✓" if status == "ready" else ("✗" if status == "error" else "○")
            css_class = f"status-{'ready' if status=='ready' else 'error' if status=='error' else 'pending'}"
            col_v, col_x = st.columns([5, 1])
            with col_v:
                st.markdown(f"""
                <div class="video-chip">
                    <span class="video-chip-id">{vid_id}</span>
                    <span class="video-chip-status {css_class}">{icon} {status}</span>
                </div>
                """, unsafe_allow_html=True)
            with col_x:
                if st.button("×", key=f"rm_{vid_id}", help=f"Remove {vid_id}"):
                    to_remove.append(vid_id)
        for vid_id in to_remove:
            del st.session_state.videos[vid_id]
            st.session_state.rag.transcripts.pop(vid_id, None)
            st.session_state.is_built = False
            st.rerun()

    st.markdown("---")

    # ── Load button ──
    can_load = len(st.session_state.videos) > 0
    if st.button(
        "⚡ Load & Analyze Videos",
        disabled=not can_load,
        use_container_width=True,
    ):
        st.session_state.rag.clear()
        st.session_state.is_built = False
        st.session_state.messages.clear()

        progress = st.progress(0, text="Starting...")
        status_text = st.empty()
        step = 0
        total_videos = len(st.session_state.videos)
        fetch_steps = total_videos
        total_steps = fetch_steps + 4  # +4 for build steps

        # Fetch transcripts
        for vid_id in list(st.session_state.videos.keys()):
            status_text.markdown(f"⏳ Fetching transcript for `{vid_id}`...")
            st.session_state.videos[vid_id]["status"] = "loading"
            ok, result = fetch_transcript(vid_id)
            if ok:
                st.session_state.rag.add_transcript(vid_id, result)
                st.session_state.videos[vid_id]["status"] = "ready"
                st.session_state.videos[vid_id]["error"] = ""
            else:
                st.session_state.videos[vid_id]["status"] = "error"
                st.session_state.videos[vid_id]["error"] = result
            step += 1
            progress.progress(int(step / total_steps * 100), text=f"Fetched {step}/{fetch_steps} videos")

        ready_videos = [v for v, i in st.session_state.videos.items() if i["status"] == "ready"]

        if not ready_videos:
            st.error("No transcripts could be fetched. Check the video IDs.")
            progress.empty()
            status_text.empty()
        else:
            # Build RAG pipeline
            def on_progress(msg):
                global step
                step += 1
                progress.progress(
                    min(int(step / total_steps * 100), 99),
                    text=msg
                )
                status_text.markdown(f"⚙️ {msg}")

            try:
                st.session_state.rag.build(progress_callback=on_progress)
                st.session_state.is_built = True
                progress.progress(100, text="Done!")
                status_text.empty()
                st.success(f"✅ {len(ready_videos)} video(s) ready! Ask anything →")
            except Exception as e:
                st.error(f"Build failed: {e}")
                progress.empty()
                status_text.empty()

    # ── Footer info ──
    st.markdown("---")
    st.markdown("""
    <div class="footer-info">
        Model: <span>llama-3.3-70b</span><br>
        Embeddings: <span>all-MiniLM-L6-v2</span><br>
        Vector DB: <span>FAISS</span><br>
        Framework: <span>LangChain</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

# ── Status bar ──
ready_count = sum(1 for v in st.session_state.videos.values() if v["status"] == "ready")
lang_display = "اردو" if st.session_state.language == "Urdu" else "English"
dot = "🟢" if st.session_state.is_built else "⚪"

st.markdown(f"""
<div class="status-bar">
    {dot} &nbsp;
    <strong style="color:var(--text-1)">{ready_count}</strong>&nbsp;video(s) loaded &nbsp;·&nbsp;
    Answering in&nbsp;<strong style="color:var(--accent)">{lang_display}</strong>
    {"&nbsp;·&nbsp;<span style='color:var(--green);font-weight:600'>Ready to chat</span>" if st.session_state.is_built else ""}
</div>
""", unsafe_allow_html=True)

# ── Empty state ──
if not st.session_state.is_built and not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">▶️</div>
        <div class="empty-title">No videos loaded yet</div>
        <div class="empty-sub">
            Add YouTube video links in the sidebar,<br>
            click <strong>Load &amp; Analyze Videos</strong>,<br>
            then ask anything about them!
        </div>
        <div class="chips-row">
            <span class="chip">Summarize the video</span>
            <span class="chip">What are the key topics?</span>
            <span class="chip">Who is mentioned?</span>
            <span class="chip">Main takeaways?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Chat messages ──
if st.session_state.messages:
    is_urdu = st.session_state.language == "Urdu"
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.messages:
        content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        if msg["role"] == "user":
            chat_html += f"""
            <div class="msg-row-user">
                <div class="avatar avatar-user">U</div>
                <div class="bubble-user">{content}</div>
            </div>"""
        else:
            urdu_class = "urdu" if is_urdu else ""
            lang_tag = "اردو" if is_urdu else "English"
            chat_html += f"""
            <div class="msg-row-bot">
                <div class="avatar avatar-bot">▶</div>
                <div>
                    <div class="bubble-bot {urdu_class}">{content}</div>
                    <div class="lang-tag">{lang_tag}</div>
                </div>
            </div>"""
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)

# ── Chat input ──
if st.session_state.is_built:
    # Quick suggestion chips (only if chat is empty)
    if not st.session_state.messages:
        suggestions = [
            "Can you summarize this video?",
            "What are the main topics discussed?",
            "Who are the key people mentioned?",
            "What are the top 5 takeaways?",
        ]
        cols = st.columns(len(suggestions))
        for i, (col, sug) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(sug, key=f"chip_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": sug})
                    with st.spinner("Thinking..."):
                        answer = st.session_state.rag.ask(sug, st.session_state.language)
                    st.session_state.messages.append({"role": "bot", "content": answer})
                    st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Main input
    with st.form("chat_form", clear_on_submit=True):
        placeholder_text = (
            "ویڈیو کے بارے میں کچھ بھی پوچھیں..."
            if st.session_state.language == "Urdu"
            else "Ask anything about the videos..."
        )
        user_q = st.text_input(
            "question",
            placeholder=placeholder_text,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send →", use_container_width=False)

    if submitted and user_q.strip():
        st.session_state.messages.append({"role": "user", "content": user_q.strip()})
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag.ask(
                    user_q.strip(), st.session_state.language
                )
            except Exception as e:
                answer = f"Error: {e}"
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.rerun()

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑 Clear Chat", use_container_width=False):
            st.session_state.messages.clear()
            st.rerun()



