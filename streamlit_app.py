"""
Streamlit app for Titanic Chat Agent — standalone version for Streamlit Cloud.
Calls the LangChain agent directly (no separate FastAPI backend needed).
"""

from __future__ import annotations

import asyncio
import base64
import os
import uuid

import nest_asyncio
import streamlit as st

nest_asyncio.apply()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Titanic Chat Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Groq key from Streamlit secrets into env before importing agent ────

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# ── Lazy agent import (after env is set) ─────────────────────────────────────

from backend.agent import ask_agent  # noqa: E402

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stCaption {
        color: #a0a0a0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #333355 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #2e2e4e;
        color: #ffffff !important;
        border: 1px solid #444466;
        border-radius: 6px;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #3e3e6e;
        border-color: #6666aa;
    }
    .stChatMessage { border-radius: 10px; }
    img { border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.15); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg",
        use_container_width=True,
    )
    st.title("Titanic Chat Agent")
    st.markdown(
        """
        Ask me anything about the **Titanic passenger dataset**.

        **Example questions:**
        - What percentage of passengers were male?
        - Show me a histogram of passenger ages
        - What was the average ticket fare?
        - How many passengers embarked from each port?
        - What was the survival rate by class?
        - Show a bar chart of survival by gender
        """
    )
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    st.divider()
    st.caption("Built with FastAPI · LangChain · Streamlit")

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Titanic Dataset Chat Agent")
st.caption("Ask questions in plain English and get text answers with visualisations")

# ── Render chat history ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        for chart_b64 in msg.get("charts", []):
            st.image(base64.b64decode(chart_b64), use_container_width=True)

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about the Titanic dataset..."):
    st.session_state.messages.append({"role": "user", "text": prompt, "charts": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analysing the dataset..."):
            try:
                result = asyncio.run(
                    ask_agent(prompt, session_id=st.session_state.session_id)
                )
                answer_text = result.get("text", "Sorry, I could not process that.")
                charts = result.get("charts", [])

                st.markdown(answer_text)
                for chart_b64 in charts:
                    st.image(base64.b64decode(chart_b64), use_container_width=True)

                st.session_state.messages.append(
                    {"role": "assistant", "text": answer_text, "charts": charts}
                )
            except Exception as exc:
                err = f"Error: {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "text": err, "charts": []}
                )
