"""
LangChain agent that analyses the Titanic dataset.

Exposes two tools to the ReAct agent:
  1. query_dataframe  – run a Pandas expression and return text results
  2. create_chart     – generate a matplotlib/seaborn visualisation, return the
                        image as a base-64 PNG string
"""

from __future__ import annotations

import base64
import io
import os
import re
import textwrap
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from backend.config import get_settings

# ── Load dataset once ─────────────────────────────────────────────────────────
settings = get_settings()

_CSV_PATH = Path(__file__).resolve().parent.parent / settings.TITANIC_CSV_PATH
_df: pd.DataFrame | None = None


def _get_df() -> pd.DataFrame:
    """Lazily load (and cache) the Titanic CSV."""
    global _df
    if _df is None:
        _df = pd.read_csv(_CSV_PATH)
    return _df


def _df_summary() -> str:
    """Return a compact description of the dataframe for LLM context."""
    df = _get_df()
    buf = io.StringIO()
    df.info(buf=buf)
    info_str = buf.getvalue()
    desc = df.describe(include="all").to_string()
    columns = ", ".join(df.columns.tolist())
    sample = df.head(3).to_string()
    return textwrap.dedent(f"""\
    === Titanic DataFrame Info ===
    Columns: {columns}
    Shape: {df.shape[0]} rows × {df.shape[1]} columns

    {info_str}

    === Statistical Summary ===
    {desc}

    === Sample Rows ===
    {sample}
    """)


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def query_dataframe(pandas_expression: str) -> str:
    """Execute a Pandas expression on the Titanic DataFrame and return the text result.

    The DataFrame is available as `df`. Some examples:
      - df['Sex'].value_counts(normalize=True) * 100
      - df['Age'].mean()
      - df.groupby('Embarked').size()
      - df[df['Survived'] == 1].shape[0]
      - len(df[df['Sex'] == 'male']) / len(df) * 100

    Always use `df` to reference the Titanic DataFrame.
    Return the result as a string.
    """
    df = _get_df()  # noqa: F841 – used inside eval
    try:
        result = eval(pandas_expression)  # noqa: S307
        if isinstance(result, pd.DataFrame):
            return result.to_string()
        if isinstance(result, pd.Series):
            return result.to_string()
        return str(result)
    except Exception as exc:
        return f"Error executing expression: {exc}\n{traceback.format_exc()}"


@tool
def create_chart(python_code: str) -> str:
    """Generate a matplotlib/seaborn chart from the Titanic DataFrame and return the
    image encoded as a base64 PNG string.

    Available variables:
      - df   : the Titanic pandas DataFrame
      - plt  : matplotlib.pyplot
      - sns  : seaborn
      - pd   : pandas

    Guidelines:
      - Always create a new figure: plt.figure(figsize=(10, 6))
      - Add a descriptive title with plt.title(...)
      - Add axis labels with plt.xlabel(...) / plt.ylabel(...)
      - Use plt.tight_layout() before the end
      - Do NOT call plt.show() or plt.savefig() – the system handles that.

    Example code:
      plt.figure(figsize=(10, 6))
      sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='steelblue')
      plt.title('Distribution of Passenger Ages')
      plt.xlabel('Age')
      plt.ylabel('Count')
      plt.tight_layout()
    """
    df = _get_df()
    try:
        plt.close("all")
        exec_globals = {"df": df, "plt": plt, "sns": sns, "pd": pd}
        exec(python_code, exec_globals)  # noqa: S102

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close("all")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return f"CHART_BASE64:{b64}"
    except Exception as exc:
        plt.close("all")
        return f"Error creating chart: {exc}\n{traceback.format_exc()}"


# ── Agent construction ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
You are **TitanicBot**, a friendly and knowledgeable data-analysis assistant.
You have access to the famous Titanic passenger dataset loaded as a Pandas DataFrame.

{df_summary}

RULES:
1. When the user asks a factual/numeric question, use the `query_dataframe` tool with
   a valid Pandas expression to compute the answer. Then present the result in clear,
   conversational English.
2. When the user asks for a visualisation (chart, plot, histogram, bar chart, pie chart,
   etc.), use the `create_chart` tool. Write clean matplotlib/seaborn code.
   If the question also requires a textual answer, call `query_dataframe` FIRST and
   then `create_chart`.
3. Always be accurate. Never guess numbers – always query the DataFrame.
4. If a query fails, inspect the error and try a corrected expression.
5. Format numbers nicely (e.g. 2 decimal places for percentages).
6. If the user's question is ambiguous, make a reasonable assumption and state it.
7. Be concise but friendly.
""")


def build_agent():
    """Create and return a LangGraph ReAct agent with Titanic-analysis tools."""
    settings = get_settings()

    llm = ChatGroq(
        model=settings.GROQ_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )

    tools = [query_dataframe, create_chart]
    checkpointer = MemorySaver()

    system_message = _SYSTEM_PROMPT.format(df_summary=_df_summary())

    agent = create_react_agent(
        llm,
        tools,
        checkpointer=checkpointer,
        state_modifier=system_message,
    )
    return agent


# Module-level singleton
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


async def ask_agent(question: str, session_id: str = "default") -> dict:
    """Send a question to the agent and return structured output.

    Returns
    -------
    dict  with keys:
        text   : str            – the conversational answer
        charts : list[str]      – list of base64-encoded PNG images (may be empty)
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": session_id}}

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    # Extract the final AI message
    messages = result["messages"]
    ai_text_parts: list[str] = []
    charts: list[str] = []

    for msg in messages:
        # Tool messages may contain chart base64
        content = msg.content if hasattr(msg, "content") else ""
        if isinstance(content, str) and "CHART_BASE64:" in content:
            b64 = content.split("CHART_BASE64:", 1)[1].strip()
            charts.append(b64)

    # The last AI message is the final answer
    final_msg = messages[-1]
    text = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    return {"text": text, "charts": charts}
