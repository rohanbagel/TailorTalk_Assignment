# 🚢 Titanic Dataset Chat Agent

A friendly chatbot that analyses the Titanic passenger dataset. Ask questions in
plain English and get **text answers** plus **visual charts**.

## Tech Stack

| Layer     | Technology            |
|-----------|-----------------------|
| Backend   | Python · FastAPI      |
| Agent     | LangChain · LangGraph |
| Frontend  | Streamlit             |
| Data      | Pandas · Seaborn      |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Start the backend (FastAPI)

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Start the frontend (Streamlit) — in a separate terminal

```bash
streamlit run frontend/app.py --server.port 8501
```

### 5. Open the app

Navigate to **http://localhost:8501** in your browser.

## Example Questions

- "What percentage of passengers were male on the Titanic?"
- "Show me a histogram of passenger ages"
- "What was the average ticket fare?"
- "How many passengers embarked from each port?"
- "What was the survival rate by passenger class?"
- "Show a bar chart comparing survival by gender"

## Project Structure

```
├── backend/
│   ├── __init__.py
│   ├── config.py          # Pydantic settings
│   ├── agent.py           # LangChain agent + tools
│   └── main.py            # FastAPI application
├── frontend/
│   └── app.py             # Streamlit chat interface
├── data/
│   └── titanic.csv        # Titanic dataset
├── requirements.txt
├── .env.example
└── README.md
```

## Architecture

```
┌──────────────┐       POST /chat        ┌──────────────────┐
│   Streamlit  │ ──────────────────────▶  │   FastAPI         │
│   Frontend   │ ◀──────────────────────  │   Backend         │
└──────────────┘    {text, charts[]}      │                   │
                                          │  ┌──────────────┐ │
                                          │  │  LangChain   │ │
                                          │  │  ReAct Agent  │ │
                                          │  │              │ │
                                          │  │  Tools:      │ │
                                          │  │  • query_df  │ │
                                          │  │  • create_   │ │
                                          │  │    chart     │ │
                                          │  └──────────────┘ │
                                          └──────────────────┘
```

## How It Works

1. User types a question in the Streamlit chat interface
2. Streamlit sends a POST request to the FastAPI `/chat` endpoint
3. FastAPI invokes the LangChain ReAct agent with the question
4. The agent decides which tool(s) to call:
   - **query_dataframe**: executes a Pandas expression on the Titanic DataFrame
   - **create_chart**: generates a matplotlib/seaborn visualisation (returned as base64 PNG)
5. The agent composes a friendly text answer from tool results
6. FastAPI returns `{text, charts[]}` to Streamlit
7. Streamlit renders the text and any chart images inline
