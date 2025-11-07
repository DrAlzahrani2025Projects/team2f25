# ui.py
# Streamlit UI helpers only (no business logic)
from __future__ import annotations

import time
import base64
from pathlib import Path
import pandas as pd
import streamlit as st

# --- avatar/logo image paths (change if you store elsewhere) ---
WOLF_LOGO = "assets/wolf.png"
STUDENT_AVATAR = "assets/student.png"
TITLE = "assets/csusb_bg.png"


# ------------------------------------------------------------
# Background / CSS
# ------------------------------------------------------------


def set_app_background(image_path: str, darken: float = 0.45):
    """
    Full-page background with an adjustable dark overlay.
    darken: 0.0 (none) … 0.7 (very dark)
    """
    p = Path(image_path)
    if not p.exists():
        st.warning(f"Background not found: {p.resolve()}")
        return
    b64 = base64.b64encode(p.read_bytes()).decode()

    st.markdown(f"""
    <style>
      html, body, [data-testid="stAppViewContainer"], .stApp {{
        height: 100%;
        background: linear-gradient(rgba(0,0,0,{darken}), rgba(0,0,0,{darken})),
                    url("data:image/png;base64,{b64}") center / cover fixed no-repeat !important;
      }}

      /* Let the background show through */
      .block-container, .main, [data-testid="stHeader"], [data-testid="stToolbar"] {{
        background: transparent !important;
      }}

      /* Optional: make the sidebar readable */
      [data-testid="stSidebar"] {{
        background: rgba(255,255,255,0.65) !important;
        -webkit-backdrop-filter: blur(6px);
        backdrop-filter: blur(6px);
      }}
    </style>
    """, unsafe_allow_html=True)


def inject_css(path: str = "styles.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass


def inject_badge_css():
    st.markdown("""
    <style>
    .badge {
      display:inline-flex;align-items:center;gap:.5rem;
      background:#2563eb;color:white;padding:.4rem .7rem;border-radius:.6rem;
      font-weight:700;letter-spacing:.2px
    }
    .subbadge {
      display:inline-block;background:#e5edff;color:#1e40af;
      padding:.15rem .4rem;border-radius:.35rem;font-weight:700;font-size:.8rem;margin-bottom:.35rem
    }
    </style>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# Small helpers (NEW)
# ------------------------------------------------------------
def _img_tag(path: str, max_height: int = 260, radius: int = 16, shadow=True) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    shadow_css = "box-shadow: 0 10px 32px rgba(0,0,0,.18);" if shadow else ""
    return (
        f"<img src='data:image/png;base64,{b64}' "
        f"style='width:100%;max-height:{max_height}px;object-fit:cover;"
        f"border-radius:{radius}px;{shadow_css}' />"
    )

def _avatar_for(role: str):
    """Student avatar for user; wolf for assistant; fallback to emoji."""
    if role == "user":
        return str(Path(STUDENT_AVATAR)) if Path(STUDENT_AVATAR).exists() else "🧑"
    return str(Path(WOLF_LOGO)) if Path(WOLF_LOGO).exists() else "🐺"


# ------------------------------------------------------------
# Header / Chat
# ------------------------------------------------------------
def header(app_title: str, source_url: str, image_path: str | None = None,
           show_text: bool = True, show_caption: bool = True, max_height: int = 260):
    """
    Renders the app header. If image_path is provided and exists,
    show a centered title image with no shadow. Otherwise try the legacy TITLE banner.
    """

    # Preferred hero image
    if image_path:
        p = Path(image_path)
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            st.markdown(
                f"""
                <style>
                  .title-logo, .title-logo img {{
                    box-shadow: none !important;
                    border-radius: 0 !important;
                    background: transparent !important;
                    filter: none !important;
                  }}
                </style>
                <div class="title-logo"
                     style="margin: 8px 0 6px; display:flex; justify-content:center; align-items:center;">
                  <img src="data:image/png;base64,{b64}"
                       alt="App header image"
                       style="height:{max_height}px; max-width:95vw; object-fit:contain; display:block;" />
                </div>
                """,
                unsafe_allow_html=True
            )
            if show_text:
                st.title(app_title)
            if show_caption:
                st.caption(f"🎯 CSUSB CSE internship links • Source: {source_url}")
            return  # done

    # Fallback: original banner (if available), else plain title
    TITLE_tag = None
    try:
        # radius=0 to avoid rounded corners/shadow in legacy banner too
        TITLE_tag = _img_tag(TITLE, max_height=180, radius=0)  # may raise if TITLE/_img_tag undefined
    except Exception:
        TITLE_tag = None

    if TITLE_tag:
        st.markdown(
            f"<div style='margin:0 0 .8rem 0;max-width:990px'>{TITLE_tag}</div>",
            unsafe_allow_html=True
        )
        if show_caption:
            st.caption(f"🎯 CSUSB CSE internship links • Source: {source_url}")
        return

    # Final fallback: plain text title + caption
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


# ------------------------------------------------------------
# Résumé Upload (sidebar)
# ------------------------------------------------------------
def show_resume_sidebar(on_extract, on_llm_extract, on_save):
    """
    Renders a sidebar uploader and persists parsed résumé to session_state.
    Callbacks:
      - on_extract(file) -> text
      - on_llm_extract(text) -> data dict
      - on_save(data, text) -> None
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📎 Resume Upload")

    if "resume_uploader_key" not in st.session_state:
        st.session_state["resume_uploader_key"] = "resume_uploader_0"

    up = st.sidebar.file_uploader(
        "Upload your résumé",
        type=["pdf", "docx", "txt"],
        key=st.session_state["resume_uploader_key"],
        help="Upload PDF, DOCX, or TXT file",
    )

    if up is not None:
        with st.spinner("Extracting résumé..."):
            text = on_extract(up)
            data = on_llm_extract(text)
            on_save(data, text)
            st.session_state["resume_text"] = text
            st.session_state["resume_data"] = data
        st.sidebar.success("✅ Résumé saved!")
        st.session_state["resume_uploader_key"] = f"resume_uploader_{int(time.time()*1000)}"
        st.experimental_rerun()

    if st.session_state.get("resume_data"):
        with st.sidebar.expander("📄 Résumé Info"):
            data = st.session_state["resume_data"]
            if data.get("name"):
                st.write(f"**Name:** {data['name']}")
            if data.get("email"):
                st.write(f"**Email:** {data['email']}")
            if data.get("skills"):
                st.write(f"**Skills:** {', '.join(list(map(str, data['skills']))[:5])}")


# ------------------------------------------------------------
# Results table (Found Links)
# ------------------------------------------------------------
def render_found_links_table(results_df: pd.DataFrame):
    if results_df is None or results_df.empty:
        st.info("No links found on the CSUSB CSE page right now.")
        return

    # Normalize
    df = results_df.copy()
    for col in ["title", "company", "link"]:
        if col not in df.columns:
            df[col] = ""

    # Header badge
    st.markdown('<div class="badge">🧭 Found Links</div>', unsafe_allow_html=True)
    st.write("")

    # Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="subbadge">Total Links</div>', unsafe_allow_html=True)
        st.markdown(f"<h2>{len(df)}</h2>", unsafe_allow_html=True)
    with col2:
        unique_companies = df["company"].replace("", pd.NA).dropna().nunique()
        st.markdown('<div class="subbadge">Companies</div>', unsafe_allow_html=True)
        st.markdown(f"<h2>{unique_companies}</h2>", unsafe_allow_html=True)

    display_df = df.rename(
        columns={"title": "Link Text", "company": "Company", "link": "Visit"}
    )[["Link Text", "Company", "Visit"]].copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Visit": st.column_config.LinkColumn("Visit", help="Click to open link"),
            "Link Text": st.column_config.TextColumn("Link Text", width="medium"),
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


# ------------------------------------------------------------
# Chat list of links
# ------------------------------------------------------------
def render_links_in_chat(results_df: pd.DataFrame, limit: int = 50):
    """
    Renders a clickable bullet list of links *inside the chat*.
    Columns expected: title, company, link (gracefully handled if missing).
    """
    if results_df is None or results_df.empty:
        render_msg("assistant", "No links found on the CSUSB CSE page right now.")
        return

    df = results_df.copy()
    for col in ["title", "company", "link"]:
        if col not in df.columns:
            df[col] = ""

    lines = []
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
        st.markdown(md)
