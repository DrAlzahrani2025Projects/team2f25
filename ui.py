# ui.py — Streamlit UI helpers (no business logic)
from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# --- avatar/logo image paths (change if you store elsewhere) ---
WOLF_LOGO = "assets/wolf.png"
STUDENT_AVATAR = "assets/student.png"
TITLE = "assets/csusb_bg.png"

# =====================================================================
# Theming & CSS
# =====================================================================

def _read_b64(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return base64.b64encode(p.read_bytes()).decode()
    except Exception:
        return None


def inject_css(path: str = "styles.css"):
    """Inject external styles if available, then our base theme for polish."""
    # External stylesheet first (project-specific overrides)
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

    # Base theme (modern cards, chips, chat bubbles, subtle animations)
    st.markdown(
        r"""
        <style>
          :root {
            --ui-bg: #0b1220;           /* app bg (seen if no bg image) */
            --ui-card: rgba(255,255,255,0.66);
            --ui-card-dark: rgba(16,24,39,0.72);
            --ui-muted: #94a3b8;        /* slate-400 */
            --ui-fg: #0f172a;           /* slate-900 */
            --ui-fg-invert: #f8fafc;    /* near-white */
            --ui-primary: #2563eb;      /* indigo-600 */
            --ui-primary-2: #1e40af;    /* indigo-800 */
            --ui-accent: #06b6d4;       /* cyan-500 */
            --ui-ring: rgba(37,99,235,0.35);
            --ui-shadow: 0 10px 32px rgba(2,6,23,.18);
            --ui-radius: 16px;
          }

          /* Make core containers translucent */
          .block-container { padding-top: 1.2rem; }
          [data-testid="stSidebar"] { background: linear-gradient(180deg, rgba(255,255,255,.75), rgba(255,255,255,.55)); backdrop-filter: blur(8px); }
          [data-testid="stHeader"] { background: transparent; }

          /* Cards */
          .ui-card { background: var(--ui-card); border-radius: var(--ui-radius); box-shadow: var(--ui-shadow); padding: 1rem 1.1rem; }
          .ui-card.dark { background: var(--ui-card-dark); color: var(--ui-fg-invert); }
          .ui-subtle { color: var(--ui-muted); font-size: .92rem; }

          /* Badges */
          .badge { display:inline-flex;align-items:center;gap:.5rem;background:var(--ui-primary);color:white;padding:.35rem .65rem;border-radius:.6rem;font-weight:700;letter-spacing:.2px }
          .subbadge { display:inline-block;background:#e5edff;color:#1e40af;padding:.18rem .45rem;border-radius:.35rem;font-weight:700;font-size:.8rem;margin-bottom:.35rem }

          /* Chips */
          .chip { display:inline-flex;align-items:center;gap:.4rem;border:1px solid rgba(15,23,42,.08);border-radius:999px;padding:.28rem .6rem; backdrop-filter: blur(2px); }

          /* Chat bubbles */
          [data-testid="stChatMessage"] { padding-top: .25rem; }
          [data-testid="stChatMessage"] > div { border-radius: var(--ui-radius); box-shadow: var(--ui-shadow); }
          [data-testid="stChatMessage"] .stMarkdown { font-size: 1rem; }
          [data-testid="stChatMessage"]:has(.st-emotion-cache-15k0t6l) { /* assistant bubble */ }
          
          /* Tighten DF header */
          .stDataFrame { border-radius: var(--ui-radius); box-shadow: var(--ui-shadow); }

          /* Link list in chat */
          .ui-links ul { list-style: none; padding-left: 0; }
          .ui-links li { margin: .35rem 0; }
          .ui-links a { text-decoration: none; border-bottom: 1px dashed rgba(37,99,235,.35); }
          .ui-links a:hover { border-bottom-style: solid; }

          /* Buttons */
          .stButton > button { border-radius: 12px; box-shadow: var(--ui-shadow); border: 1px solid rgba(15,23,42,.08); }
          .stButton > button:focus { outline: 2px solid var(--ui-ring); }

          /* Metric tiles */
          .metric { display:flex; flex-direction:column; gap:.1rem; padding:.7rem 1rem; border-radius: 14px; background:linear-gradient(180deg, rgba(255,255,255,.9), rgba(255,255,255,.6)); box-shadow: var(--ui-shadow); }
          .metric .k { font-size: 1.8rem; font-weight: 800; letter-spacing: .2px; }
          .metric .l { color: var(--ui-muted); font-size: .85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_badge_css():
    # kept for backward compatibility (moved into base theme above)
    st.markdown("""
    <style>
    .badge{display:inline-flex;align-items:center;gap:.5rem;background:#2563eb;color:white;padding:.35rem .65rem;border-radius:.6rem;font-weight:700;letter-spacing:.2px}
    .subbadge{display:inline-block;background:#e5edff;color:#1e40af;padding:.18rem .45rem;border-radius:.35rem;font-weight:700;font-size:.8rem;margin-bottom:.35rem}
    </style>
    """, unsafe_allow_html=True)


def set_app_background(image_path: str, darken: float = 0.45):
    """Full-page background with an adjustable dark overlay."""
    b64 = _read_b64(image_path)
    if not b64:
        # graceful fallback: gradient
        st.markdown(
    f"""
    <style>
      html, body, [data-testid="stAppViewContainer"], .stApp {{
        height: 100%;
        background:
          linear-gradient(rgba(0,0,0,{darken}), rgba(0,0,0,{darken})),
          url("data:image/png;base64,{b64}") center / cover no-repeat !important; /* removed 'fixed' */
        isolation: isolate; /* new stacking context helps WebKit compositing */
      }}
      .block-container, .main, [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: transparent !important;
      }}
      [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.70) !important;
        -webkit-backdrop-filter: blur(8px);
        backdrop-filter: blur(8px);
      }}

      /* Safari: force no blur & ensure solid background to avoid black boxes */
      @supports (-webkit-touch-callout: none){{
        [data-testid="stSidebar"]{{
          -webkit-backdrop-filter:none; backdrop-filter:none;
          background: rgba(255,255,255,.88) !important;
          @supports not (-webkit-touch-callout: none){{
        html, body, [data-testid="stAppViewContainer"], .stApp{{
          background-attachment: fixed;  /* keep parallax on non-Safari */
        }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

        return

    st.markdown(
        f"""
        <style>
          html, body, [data-testid="stAppViewContainer"], .stApp {{
            height: 100%;
            background: linear-gradient(rgba(0,0,0,{darken}), rgba(0,0,0,{darken})),
                        url("data:image/png;base64,{b64}") center / cover fixed no-repeat !important;
          }}
          .block-container, .main, [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
          [data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.70) !important; -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =====================================================================
# Small helpers
# =====================================================================

def _img_tag(path: str, max_height: int = 260, radius: int = 16, shadow: bool = False) -> str:
    b64 = _read_b64(path)
    if not b64:
        return ""
    shadow_css = "box-shadow: var(--ui-shadow);" if shadow else ""
    return (
        f"<img src='data:image/png;base64,{b64}' style='width:100%;max-height:{max_height}px;object-fit:contain;"
        f"border-radius:{radius}px;{shadow_css}padding-bottom:2px;' />"
    )


def _avatar_for(role: str):
    """Student avatar for user; wolf for assistant; fallback to emoji."""
    if role == "user":
        return str(Path(STUDENT_AVATAR)) if Path(STUDENT_AVATAR).exists() else "🧑"
    return str(Path(WOLF_LOGO)) if Path(WOLF_LOGO).exists() else "🐺"

# =====================================================================
# Header / Chat
# =====================================================================

def header(
    app_title: str,
    source_url: str,
    image_path: str | None = None,
    show_text: bool = True,
    show_caption: bool = True,
    max_height: int = 260,
):
    """Hero header with optional image & caption."""
    if image_path:
        p = Path(image_path)
        if p.exists():
            b64 = _read_b64(image_path) or ""
            st.markdown(
                f"""
                <style>
                  .title-logo, .title-logo img {{ box-shadow: none !important; border-radius: 0 !important; background: transparent !important; filter: none !important; }}
                </style>
                <div class="title-logo" style="margin: 8px 0 8px; display:flex; justify-content:center; align-items:center; line-height:0; overflow:visible;">
                  <img src="data:image/png;base64,{b64}" alt="App header image" style="max-height:{max_height}px; height:auto; max-width:95vw; object-fit:contain; display:block; padding-bottom:2px;" />
                </div>
                """,
                unsafe_allow_html=True,
            )
            if show_text:
                st.title(app_title)
            if show_caption:
                st.caption(f"🎯 CSUSB CSE internship links • Source: {source_url}")
            return

    # Fallback banner or plain title
    TITLE_tag = _img_tag(TITLE, max_height=180, radius=0, shadow=False)
    if TITLE_tag:
        st.markdown(f"<div style='margin:0 0 .8rem 0;max-width:990px'>{TITLE_tag}</div>", unsafe_allow_html=True)
        if show_caption:
            st.caption(f"🎯 CSUSB CSE internship links • Source: {source_url}")
        return

    if show_text:
        st.title(app_title)
    if show_caption:
        st.caption(f"🎯 CSUSB CSE internship links • Source: {source_url}")


def render_msg(role: str, content: str):
    with st.chat_message(role, avatar=_avatar_for(role)):
        st.markdown(content)


def render_history(messages):
    for m in messages:
        render_msg(m["role"], m["content"])

# =====================================================================
# Résumé Sidebar (optional helper; not called by core flow)
# =====================================================================

def show_resume_sidebar(on_extract, on_llm_extract, on_save):
    """
    Renders a sidebar uploader and persists parsed résumé to session_state.
    Callbacks: on_extract(file)->text, on_llm_extract(text)->data, on_save(data,text)->None
    """
    if "resume_uploader_key" not in st.session_state:
        st.session_state["resume_uploader_key"] = "resume_uploader_0"

    up = st.sidebar.file_uploader(
        "Upload your résumé (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        key=st.session_state["resume_uploader_key"],
        help="Drag & drop a file here",
    )

    if up is not None:
        with st.spinner("Extracting résumé…"):
            text = on_extract(up)
            data = on_llm_extract(text)
            on_save(data, text)
            st.session_state["resume_text"] = text
            st.session_state["resume_data"] = data
        st.sidebar.success("✅ Résumé saved!")
        st.session_state["resume_uploader_key"] = f"resume_uploader_{int(time.time()*1000)}"
        st.rerun()

    if st.session_state.get("resume_data"):
        with st.sidebar.expander("📄 Résumé Info"):
            data = st.session_state["resume_data"]
            cols = st.columns(2)
            with cols[0]:
                st.markdown("<div class='subbadge'>Name</div>", unsafe_allow_html=True)
                st.write(data.get("name") or "—")
            with cols[1]:
                st.markdown("<div class='subbadge'>Email</div>", unsafe_allow_html=True)
                st.write(data.get("email") or "—")
            st.markdown("<div class='subbadge'>Top Skills</div>", unsafe_allow_html=True)
            if data.get("skills"):
                chips = " ".join([f"<span class='chip'>{s}</span>" for s in map(str, data["skills"][:8])])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.write("—")

# =====================================================================
# Results table (Found Links)
# =====================================================================

def render_found_links_table(results_df: pd.DataFrame):
    if results_df is None or results_df.empty:
        st.info("No links found on the CSUSB CSE page right now.")
        return

    df = results_df.copy()
    for col in ["title", "company", "link"]:
        if col not in df.columns:
            df[col] = ""

    # Header badge
    st.markdown('<div class="badge">🧭 Found Links</div>', unsafe_allow_html=True)
    st.write("")

    # Quick filters (client-side; UI-only)
    with st.expander("🔎 Refine (client-side)", expanded=False):
        q = st.text_input("Search title/company")
        host_choices: List[str] = sorted(list({str(x) for x in df.get("host", pd.Series(dtype=str)).astype(str) if x}))
        host_sel = st.multiselect("Filter by host", host_choices, max_selections=6)
        if q:
            ql = q.lower().strip()
            mask = (
                df.get("title").astype(str).str.lower().str.contains(ql, na=False) |
                df.get("company").astype(str).str.lower().str.contains(ql, na=False)
            )
            df = df[mask]
        if host_sel:
            df = df[df.get("host").astype(str).isin(set(host_sel))]

    # Metrics
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown("<div class='metric'><div class='l'>Total Links</div><div class='k'>"+str(len(df))+"</div></div>", unsafe_allow_html=True)
    with c2:
        uniq_comp = df["company"].replace("", pd.NA).dropna().nunique()
        st.markdown("<div class='metric'><div class='l'>Companies</div><div class='k'>"+str(uniq_comp)+"</div></div>", unsafe_allow_html=True)
    with c3:
        uniq_hosts = df.get("host", pd.Series(dtype=str)).replace("", pd.NA).dropna().nunique()
        st.markdown("<div class='metric'><div class='l'>Hosts</div><div class='k'>"+str(uniq_hosts)+"</div></div>", unsafe_allow_html=True)

    display_df = df.rename(columns={"title": "Link Text", "company": "Company", "link": "Visit"})[
        ["Link Text", "Company", "Visit"]
    ].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Visit": st.column_config.LinkColumn("Visit", help="Open link in a new tab"),
            "Link Text": st.column_config.TextColumn("Link Text", width="large"),
            "Company": st.column_config.TextColumn("Company", width="medium"),
        },
    )

    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv,
        file_name="csusb_links.csv",
        mime="text/csv",
    )

# =====================================================================
# Chat link list
# =====================================================================

def render_links_in_chat(results_df: pd.DataFrame, limit: int = 50):
    """Renders a clickable bullet list of links *inside the chat*."""
    if results_df is None or results_df.empty:
        render_msg("assistant", "No links found on the CSUSB CSE page right now.")
        return

    df = results_df.copy()
    for col in ["title", "company", "link"]:
        if col not in df.columns:
            df[col] = ""

    lines: List[str] = []
    for _, row in df.head(limit).iterrows():
        title = str(row.get("title") or "").strip() or "Career Page"
        company = str(row.get("company") or "").strip()
        url = str(row.get("link") or "").strip()
        if not url:
            continue
        if company:
            lines.append(f"- [{title}]({url}) — **{company}**")
        else:
            lines.append(f"- [{title}]({url})")

    md = "\n".join(lines) if lines else "_No links to display._"
    with st.chat_message("assistant", avatar=_avatar_for("assistant")):
        st.markdown(f"<div class='ui-links'>{md}</div>", unsafe_allow_html=True)
