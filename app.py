import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any

import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Agentic RAG Schedule Assistant", version="1.0.0")

# -------------------------------------------------------------------
# Sample schedule: 30 days from 2026-08-12 through 2026-09-10.
# Chroma stores the searchable schedule documents.
# -------------------------------------------------------------------
START = datetime(2026, 8, 12)
END = START + timedelta(days=29)

SAMPLE_EVENTS = [
    {"title": "Team Stand-up", "date": "2026-08-12", "time": "10:00", "type": "meeting", "duration": 30},
    {"title": "RAG Architecture Workshop", "date": "2026-08-13", "time": "14:00", "type": "workshop", "duration": 120},
    {"title": "Doctor Appointment", "date": "2026-08-14", "time": "11:30", "type": "appointment", "duration": 60},
    {"title": "Project Planning Meeting", "date": "2026-08-15", "time": "15:00", "type": "meeting", "duration": 60},
    {"title": "Submit Project Report", "date": "2026-08-17", "time": "17:00", "type": "task", "duration": 30},
    {"title": "Client Review", "date": "2026-08-18", "time": "16:00", "type": "meeting", "duration": 60},
    {"title": "Python Workshop", "date": "2026-08-20", "time": "10:00", "type": "workshop", "duration": 90},
    {"title": "Dentist Appointment", "date": "2026-08-22", "time": "12:00", "type": "appointment", "duration": 60},
    {"title": "Sprint Retrospective", "date": "2026-08-25", "time": "15:00", "type": "meeting", "duration": 60},
    {"title": "Prepare Presentation", "date": "2026-08-27", "time": "18:00", "type": "task", "duration": 60},
    {"title": "Product Demo", "date": "2026-09-01", "time": "11:00", "type": "meeting", "duration": 60},
    {"title": "AI Research Workshop", "date": "2026-09-03", "time": "14:30", "type": "workshop", "duration": 120},
    {"title": "Monthly Planning", "date": "2026-09-05", "time": "10:00", "type": "meeting", "duration": 60},
]

chroma = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db"))
collection = chroma.get_or_create_collection(name="schedule")


def valid_date(value: str) -> bool:
    try:
        d = datetime.strptime(value, "%Y-%m-%d")
        return START.date() <= d.date() <= END.date()
    except ValueError:
        return False


def seed_database():
    if collection.count() > 0:
        return
    ids, docs, metas = [], [], []
    for e in SAMPLE_EVENTS:
        eid = str(uuid.uuid4())
        ids.append(eid)
        docs.append(
            f"{e['title']} on {e['date']} at {e['time']}. "
            f"Type: {e['type']}. Duration: {e['duration']} minutes."
        )
        metas.append({**e, "id": eid})
    collection.add(ids=ids, documents=docs, metadatas=metas)


seed_database()


def all_events() -> list[dict[str, Any]]:
    data = collection.get(include=["metadatas"])
    return [dict(x) for x in data["metadatas"]]


def event_text(e: dict[str, Any]) -> str:
    return (
        f"{e['title']} on {e['date']} at {e['time']} "
        f"({e['type']}, {e.get('duration', 60)} minutes)"
    )


def add_to_index(e: dict[str, Any]):
    eid = e.get("id") or str(uuid.uuid4())
    e["id"] = eid
    collection.add(
        ids=[eid],
        documents=[event_text(e)],
        metadatas=[e],
    )


def replace_in_index(e: dict[str, Any]):
    collection.delete(ids=[e["id"]])
    add_to_index(e)


def parse_date(text: str) -> str | None:
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"\b(august|september)\s+(\d{1,2})\b", text, re.I)
    if m:
        month = 8 if m.group(1).lower() == "august" else 9
        return f"2026-{month:02d}-{int(m.group(2)):02d}"
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    lower = text.lower()
    base = START
    if "tomorrow" in lower:
        return (base + timedelta(days=1)).strftime("%Y-%m-%d")
    if "today" in lower:
        return base.strftime("%Y-%m-%d")
    for day, idx in weekdays.items():
        if day in lower:
            delta = (idx - base.weekday()) % 7
            if "next " + day in lower and delta == 0:
                delta = 7
            return (base + timedelta(days=delta)).strftime("%Y-%m-%d")
    return None


def parse_time(text: str) -> str | None:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text, re.I)
    if not m:
        return None
    h, minute, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    if ap == "pm" and h != 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return f"{h:02d}:{minute:02d}"


def get_schedule(query: str = "", date: str | None = None,
                 start_time: str | None = None, end_time: str | None = None):
    """Tool 1: retrieve schedule information with Chroma RAG."""
    q = query or date or "schedule"
    result = collection.query(query_texts=[q], n_results=min(8, max(1, collection.count())))
    hits = result.get("metadatas", [[]])[0]
    if date:
        hits = [e for e in hits if e.get("date") == date]
    if start_time:
        hits = [e for e in hits if e.get("time", "00:00") >= start_time]
    if end_time:
        hits = [e for e in hits if e.get("time", "23:59") <= end_time]
    return sorted(hits, key=lambda x: (x.get("date", ""), x.get("time", "")))


def update_schedule(action: str, event: dict[str, Any] | None = None,
                    event_id: str | None = None, changes: dict[str, Any] | None = None):
    """Tool 2: add, update, or remove schedule entries."""
    if action == "add":
        e = dict(event or {})
        if not e.get("title") or not e.get("date") or not e.get("time"):
            raise ValueError("title, date and time are required")
        if not valid_date(e["date"]):
            raise ValueError("Date must be within the 30-day schedule window.")
        e.setdefault("type", "meeting")
        e.setdefault("duration", 60)
        add_to_index(e)
        return e

    events = all_events()
    target = next((x for x in events if x.get("id") == event_id), None)
    if not target:
        raise ValueError("Schedule entry not found.")

    if action == "remove":
        collection.delete(ids=[target["id"]])
        return target

    if action == "update":
        target.update(changes or {})
        if not valid_date(target["date"]):
            raise ValueError("Updated date must be within the 30-day schedule window.")
        replace_in_index(target)
        return target

    raise ValueError("action must be add, update, or remove")


def find_matching_event(text: str):
    events = all_events()
    lower = text.lower()
    candidates = [e for e in events if e["title"].lower() in lower]
    if candidates:
        return candidates[0]
    d = parse_date(text)
    t = parse_time(text)
    if d:
        same_day = [e for e in events if e["date"] == d]
        if t:
            same_day.sort(key=lambda e: abs(
                int(e["time"][:2]) * 60 + int(e["time"][3:]) -
                (int(t[:2]) * 60 + int(t[3:]))
            ))
        return same_day[0] if same_day else None
    return None


def agent(message: str):
    """Small agentic router: decides whether to retrieve or mutate."""
    lower = message.lower()

    mutation_words = ["add ", "create ", "schedule ", "book ", "move ",
                      "reschedule ", "update ", "change ", "remove ",
                      "delete ", "cancel "]
    is_mutation = any(w in lower for w in mutation_words)

    # UPDATE: move/reschedule/change an existing event.
    if is_mutation and any(w in lower for w in ["move ", "reschedule ", "change "]):
        old = find_matching_event(message)
        new_time = parse_time(message)
        new_date = parse_date(message)
        if old and (new_time or new_date):
            changes = {}
            if new_time:
                changes["time"] = new_time
            if new_date:
                changes["date"] = new_date
            updated = update_schedule("update", event_id=old["id"], changes=changes)
            return {"tool": "update_schedule", "action": "update", "result": updated}

    # REMOVE
    if is_mutation and any(w in lower for w in ["remove ", "delete ", "cancel "]):
        old = find_matching_event(message)
        if old:
            removed = update_schedule("remove", event_id=old["id"])
            return {"tool": "update_schedule", "action": "remove", "result": removed}

    # ADD
    if is_mutation and any(w in lower for w in ["add ", "create ", "schedule ", "book "]):
        d = parse_date(message)
        t = parse_time(message)
        if d and t:
            title = "New Meeting"
            m = re.search(r"(?:add|create|schedule|book)\s+(?:a\s+)?(.+?)\s+on\s+"
                          r"(?:august|september|\d{4}[-/])", message, re.I)
            if m:
                title = m.group(1).strip()
            event = {"title": title, "date": d, "time": t, "type": "meeting", "duration": 60}
            return {"tool": "update_schedule", "action": "add",
                    "result": update_schedule("add", event=event)}

    # GET: date, time, free/busy, or semantic query.
    d = parse_date(message)
    if "afternoon" in lower:
        return {"tool": "get_schedule", "result": get_schedule(message, d, "12:00", "17:00")}
    if "morning" in lower:
        return {"tool": "get_schedule", "result": get_schedule(message, d, "08:00", "12:00")}
    return {"tool": "get_schedule", "result": get_schedule(message, d)}


class ChatRequest(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html>
<html><head><title>Schedule Assistant</title>
<style>
body{font-family:Arial;max-width:850px;margin:40px auto;padding:0 18px;background:#f6f7fb}
.card{background:white;padding:28px;border-radius:16px;box-shadow:0 5px 25px #0001}
input{width:78%;padding:14px;border:1px solid #ddd;border-radius:9px}
button{padding:14px 18px;border:0;border-radius:9px;cursor:pointer}
#out{white-space:pre-wrap;margin-top:20px;line-height:1.5}
</style></head>
<body><div class="card">
<h1>Agentic RAG Schedule Assistant</h1>
<p>Ask about your schedule or add, move, and remove events.</p>
<input id="q" placeholder="What do I have scheduled tomorrow?">
<button onclick="ask()">Ask</button><div id="out"></div>
</div><script>
async function ask(){
 const q=document.getElementById('q').value;
 const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:q})});
 const d=await r.json(); document.getElementById('out').textContent=d.answer;
}
</script></body></html>"""


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        result = agent(req.message)
        items = result["result"]
        if isinstance(items, dict):
            answer = f"{items['title']} — {items['date']} at {items['time']}."
        elif not items:
            answer = "No matching schedule entries found."
        else:
            lines = [event_text(x) for x in items]
            answer = "\n".join(lines)
        return {"answer": answer, "tool": result["tool"], "data": items}
    except Exception as exc:
        return {"answer": f"Could not complete the request: {exc}", "error": str(exc)}


@app.get("/health")
def health():
    return {"status": "ok", "events": collection.count(), "window": {
        "start": START.strftime("%Y-%m-%d"), "end": END.strftime("%Y-%m-%d")
    }}
