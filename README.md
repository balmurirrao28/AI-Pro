# ScheduleAI — Agentic RAG Schedule Assistant

A Render-ready Agentic RAG schedule assistant for the next 30 days.

## What it does

- Stores schedule events in **ChromaDB**.
- Uses a lightweight deterministic local embedding function, so it does **not** need an API key or external embedding model.
- Uses RAG/vector retrieval for open-ended schedule questions.
- Has exactly two schedule tools:
  - `get_schedule` — retrieve schedule by date, time, type, or natural-language query.
  - `update_schedule` — add, update/move, or remove events.
- Calculates free windows such as Friday afternoon.
- Supports natural dates such as today, tomorrow, weekdays, and `August 15`.
- Keeps the schedule inside a rolling 30-day window.
- Includes sample meetings, workshops, tasks, appointments, and college entries.
- Includes a responsive web interface and `/health` endpoint.

## Example queries

- What do I have scheduled tomorrow?
- Am I free Friday afternoon?
- Add a meeting on August 15 at 3 PM.
- Move my meeting from 2 PM to 4 PM.
- Remove my workshop.
- Show my meetings.
- When does college end today?

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

Health check: `http://localhost:8000/health`

## Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

No `.env` file is required. Optional environment variables:

- `TIMEZONE` (default: `Asia/Kolkata`)
- `CHROMA_PATH` (default: `./chroma_db`)

## Project files

- `app.py` — complete FastAPI application, UI, RAG pipeline, and agent tools.
- `requirements.txt` — Python dependencies.
- `final_deployed_url.txt` — final Render URL placeholder to be replaced after deployment.
