# College Schedule AI

An Agentic RAG-based Schedule Assistant that manages a student's
30-day college schedule using FastAPI and ChromaDB.

## Features

- Agentic schedule assistant
- ChromaDB vector database
- RAG-based schedule retrieval
- `get_schedule` tool
- `update_schedule` tool
- Add schedule entries
- Update/move schedule entries
- Remove schedule entries
- College ending-time questions
- 30-day schedule management
- Responsive web interface
- Google Login interface
- Render deployment ready

## College Schedule

The assistant understands the following college timings:

| Day | College Ends |
|---|---|
| Monday | 4:00 PM |
| Tuesday | 4:00 PM |
| Wednesday | 4:00 PM |
| Thursday | 3:00 PM |
| Friday | 3:00 PM |
| Saturday | No College |
| Sunday | No College |

Example questions:

- When is my college ending today?
- When does college end tomorrow?
- When does college end on Monday?
- When does college end on Friday?
- What is my college schedule this week?
- Am I free Friday afternoon?
- Add a meeting on August 15 at 3 PM.
- Move my meeting from 2 PM to 4 PM.
- Remove my meeting.

## Agent Tools

### 1. get_schedule

Retrieves relevant schedule information using ChromaDB
semantic search based on:

- User query
- Date
- Time
- Morning
- Afternoon

### 2. update_schedule

Manages schedule entries:

- Add
- Update
- Move
- Remove
- Delete

## Technology

- Python
- FastAPI
- ChromaDB
- HTML
- CSS
- JavaScript
- Uvicorn

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
