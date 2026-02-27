"""
Streamlit frontend for the Titanic Chat Agent.

Renders a clean chat interface that sends questions to the FastAPI backend
and displays text answers + inline chart images.
"""

from __future__ import annotations

import base64
import os
import uuid

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🚢 Titanic Chat Agent",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Main container */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Chart images */
    .chart-container img {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
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
    st.title("🚢 Titanic Chat Agent")
    st.markdown(
        """
        Ask me anything about the **Titanic passenger dataset**!

        **Example questions:**
        - *What percentage of passengers were male?*
        - *Show me a histogram of passenger ages*
        - *What was the average ticket fare?*
        - *How many passengers embarked from each port?*
        - *What was the survival rate by class?*
        - *Show a bar chart of survival by gender*
        """
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
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

st.title("🚢 Titanic Dataset Chat Agent")
st.caption("Ask questions in plain English · Get text answers & visualisations")

# ── Render chat history ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])
        for chart_b64 in msg.get("charts", []):
            st.image(base64.b64decode(chart_b64), use_container_width=True)

# ── Chat input ────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about the Titanic dataset…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "text": prompt, "charts": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    with st.chat_message("assistant"):
        with st.spinner("Analysing the dataset…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                answer_text = data.get("text", "Sorry, I couldn't process that.")
                charts = data.get("charts", [])

                st.markdown(answer_text)
                for chart_b64 in charts:
                    st.image(base64.b64decode(chart_b64), use_container_width=True)

                st.session_state.messages.append(
                    {"role": "assistant", "text": answer_text, "charts": charts}
                )

            except requests.exceptions.ConnectionError:
                err = "⚠️ Cannot reach the backend. Make sure the FastAPI server is running on `{}`".format(
                    BACKEND_URL
                )
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "text": err, "charts": []}
                )
            except Exception as exc:
                err = f"⚠️ Error: {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "text": err, "charts": []}
                )
