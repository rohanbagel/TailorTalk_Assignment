Titanic Dataset Chat Agent

Chatbot for exploring the Titanic passenger dataset. Ask questions and get text answers or charts.

## Setup
1. Install dependencies:
   pip install -r requirements.txt
2. Copy .env.example to .env and add your API key.
3. Start backend:
   uvicorn backend.main:app --reload --port 8000
4. Start frontend:
   streamlit run frontend/app.py --server.port 8501


Open locally: http://localhost:8501
Hosted app: https://tailortalkassignmentinternshala.streamlit.app/

## Example Questions
- What percentage of passengers were male?
- Show a histogram of ages
- What was the average ticket fare?

## Files
- backend/: FastAPI + agent
- frontend/: Streamlit UI
- data/titanic.csv: Dataset
- requirements.txt
