import os
import re
import uuid
import hashlib
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="ScheduleAI", version="2.0")
TZ = ZoneInfo(os.getenv("TIMEZONE", "Asia/Kolkata"))
DB_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "schedule_assistant_v1"
VECTOR_SIZE = 96

db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_or_create_collection(COLLECTION_NAME)


def now():
    return datetime.now(TZ)


def day_start():
    return now().replace(hour=0, minute=0, second=0, microsecond=0)


def date_str(value):
    return value.strftime("%Y-%m-%d")


def display_time(value):
    h, m = map(int, value.split(":"))
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def event_text(e):
    return f"{e['title']} {e['type']} on {e['date']} at {e['time']} for {e['duration']} minutes"


def embed(text):
    # Deterministic local vectors: no API key and no model download.
    v = [0.0] * VECTOR_SIZE
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return v
    for token in tokens:
        digest = hashlib.sha256(token.encode()).digest()
        for i in range(4):
            idx = int.from_bytes(digest[i * 2:i * 2 + 2], "big") % VECTOR_SIZE
            v[idx] += 1.0 if digest[i + 8] % 2 else -1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def all_events():
    return collection.get(include=["metadatas"]).get("metadatas", [])


def store_event(event):
    event = dict(event)
    event.setdefault("id", str(uuid.uuid4()))
    event.setdefault("type", "meeting")
    event.setdefault("duration", 60)
    event.setdefault("source", "user")
    collection.upsert(ids=[event["id"]], documents=[event_text(event)], embeddings=[embed(event_text(event))], metadatas=[event])
    return event


def delete_event(event_id):
    collection.delete(ids=[event_id])


def college_hours(d):
    # Mon-Wed: 9:20 AM-4 PM; Thu-Sat: 9:20 AM-3 PM; Sun: no college.
    if d.weekday() <= 2:
        return "09:20", "16:00"
    if d.weekday() <= 5:
        return "09:20", "15:00"
    return None


def seed_schedule():
    if collection.count() > 0:
        return
    start = day_start()
    for offset in range(30):
        d = start + timedelta(days=offset)
        hours = college_hours(d)
        if hours:
            begin, finish = hours
            h1, m1 = map(int, begin.split(":"))
            h2, m2 = map(int, finish.split(":"))
            duration = (h2 * 60 + m2) - (h1 * 60 + m1)
            store_event({"title": "College", "date": date_str(d), "time": begin, "type": "college", "duration": duration, "source": "system"})
    examples = [
        ("Team Meeting", 1, "14:00", "meeting", 60),
        ("AI Workshop", 3, "11:00", "workshop", 90),
        ("Doctor Appointment", 5, "10:30", "appointment", 45),
        ("Project Task", 2, "17:30", "task", 60),
    ]
    for title, offset, when, kind, duration in examples:
        d = start + timedelta(days=offset)
        store_event({"title": title, "date": date_str(d), "time": when, "type": kind, "duration": duration, "source": "sample"})


seed_schedule()

MONTHS = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6,"july":7,"august":8,"september":9,"october":10,"november":11,"december":12}
WEEKDAYS = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}


def resolve_date(text):
    text = text.lower()
    base = day_start()
    if "today" in text: return date_str(base)
    if "tomorrow" in text: return date_str(base + timedelta(days=1))
    m = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if m:
        try: return date_str(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=TZ))
        except ValueError: return None
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b", text)
    if m:
        month, day = MONTHS[m.group(1)], int(m.group(2)); year = base.year
        try: d = datetime(year, month, day, tzinfo=TZ)
        except ValueError: return None
        if d.date() < base.date(): d = d.replace(year=year + 1)
        return date_str(d)
    for name, weekday in WEEKDAYS.items():
        if re.search(r"\bnext\s+" + name + r"\b", text):
            delta = (weekday - base.weekday()) % 7 or 7
            return date_str(base + timedelta(days=delta))
        if re.search(r"\b" + name + r"\b", text):
            delta = (weekday - base.weekday()) % 7
            return date_str(base + timedelta(days=delta))
    return None


def resolve_times(text):
    result = []
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text.lower()):
        hour, minute = int(m.group(1)), int(m.group(2) or 0)
        if hour > 12 or minute > 59: continue
        if m.group(3) == "pm" and hour != 12: hour += 12
        if m.group(3) == "am" and hour == 12: hour = 0
        result.append(f"{hour:02d}:{minute:02d}")
    return result


def in_window(d):
    start = day_start().date(); target = datetime.strptime(d, "%Y-%m-%d").date()
    return start <= target <= start + timedelta(days=29)


def get_schedule(query="", date=None, start=None, end=None, event_type=None):
    """Tool 1: exact schedule filters first, then Chroma vector retrieval."""
    events = all_events()
    if date: events = [e for e in events if e.get("date") == date]
    if event_type: events = [e for e in events if e.get("type") == event_type]
    if start:
        s = int(start[:2]) * 60 + int(start[3:])
        events = [e for e in events if int(e["time"][:2]) * 60 + int(e["time"][3:]) >= s]
    if end:
        e_min = int(end[:2]) * 60 + int(end[3:])
        events = [e for e in events if int(e["time"][:2]) * 60 + int(e["time"][3:]) <= e_min]
    if date or event_type or start or end:
        return sorted(events, key=lambda e: (e["date"], e["time"]))
    if not events: return []
    result = collection.query(query_embeddings=[embed(query or "schedule")], n_results=min(8, len(events)), include=["metadatas", "distances"])
    return sorted(result.get("metadatas", [[]])[0], key=lambda e: (e["date"], e["time"]))


def update_schedule(action, event=None, event_id=None, changes=None):
    """Tool 2: add, update/move, or remove schedule entries."""
    if action == "add": return store_event(event or {})
    if not event_id: raise ValueError("Event id is required.")
    stored = collection.get(ids=[event_id], include=["metadatas"])
    metas = stored.get("metadatas", [])
    if not metas: raise ValueError("Schedule entry not found.")
    old = dict(metas[0])
    if action == "remove": delete_event(event_id); return old
    if action == "update": old.update(changes or {}); return store_event(old)
    raise ValueError("Unknown update action.")


def find_event(text, old_time=None, kind=None):
    events = all_events(); date = resolve_date(text); lower = text.lower()
    if date: events = [e for e in events if e["date"] == date]
    if old_time: events = [e for e in events if e["time"] == old_time]
    if kind: events = [e for e in events if e["type"] == kind]
    scored = []
    for e in events:
        score = 0; title = e["title"].lower()
        if title in lower: score += 10
        if e["type"] in lower: score += 5
        if e["time"] in lower: score += 3
        if e["source"] != "system": score += 1
        scored.append((score, e))
    scored.sort(key=lambda x: (-x[0], x[1]["date"], x[1]["time"]))
    return scored[0][1] if scored and scored[0][0] > 0 else (events[0] if len(events) == 1 else None)


def event_type(text):
    for word, kind in [("appointment","appointment"),("workshop","workshop"),("task","task"),("meeting","meeting")]:
        if word in text.lower(): return kind
    return "meeting"


def free_slots(date, start="12:00", end="17:00"):
    events = get_schedule(date=date); start_m = int(start[:2])*60 + int(start[3:]); end_m = int(end[:2])*60 + int(end[3:]); busy=[]
    for e in events:
        a=int(e["time"][:2])*60+int(e["time"][3:]); b=a+int(e.get("duration",60))
        if b > start_m and a < end_m: busy.append((max(start_m,a), min(end_m,b), e))
    busy.sort(); free=[]; cursor=start_m
    for a,b,_ in busy:
        if a>cursor: free.append((cursor,a))
        cursor=max(cursor,b)
    if cursor<end_m: free.append((cursor,end_m))
    return busy, free


def hm(minutes): return display_time(f"{minutes//60:02d}:{minutes%60:02d}")


def agent(message):
    """Agent/router: selects a retrieval answer or one of the two schedule tools."""
    text = message.strip(); low = text.lower(); date = resolve_date(text); times = resolve_times(text)
    if low in {"hi","hello","hey"}: return "chat", "Hi! Ask me about your next 30 days, free time, or add/move/remove an event."
    if "college" in low and any(x in low for x in ["end","ending","ends","finish","time"]):
        d=date or date_str(day_start()); dt=datetime.strptime(d,"%Y-%m-%d"); hours=college_hours(dt)
        return "answer", (f"College ends at {display_time(hours[1])} on {dt.strftime('%A')} ({d})." if hours else f"No college is scheduled on {dt.strftime('%A')} ({d}).")
    if "free" in low:
        d=date or date_str(day_start()); start,end=("12:00","17:00") if "afternoon" in low else ("09:00","21:00"); busy,free=free_slots(d,start,end)
        if not busy: return "free_time", f"You're free from {display_time(start)} to {display_time(end)} on {d}."
        lines=[f"On {datetime.strptime(d,'%Y-%m-%d').strftime('%A')}, you're busy:"]
        lines += [f"• {e['title']} ({display_time(e['time'])}–{hm(b)})" for a,b,e in busy]
        lines.append("Free windows: " + (", ".join(f"{hm(a)}–{hm(b)}" for a,b in free) if free else "none"))
        return "free_time", "\n".join(lines)
    if any(x in low for x in ["move ","reschedule ","change "]):
        if len(times)<2: return "clarify", "Tell me the event and both times, for example: “Move my meeting from 2 PM to 4 PM.”"
        old_time,new_time=times[0],times[-1]; kind=event_type(low) if any(k in low for k in ["meeting","workshop","appointment","task"]) else None; target=find_event(low,old_time=old_time,kind=kind)
        if not target: return "clarify", f"I couldn't find an event at {display_time(old_time)}. Try including its name or date."
        changes={"time":new_time};
        if date: changes["date"]=date
        updated=update_schedule("update",event_id=target["id"],changes=changes)
        return "updated", f"Moved {updated['title']} to {display_time(updated['time'])} on {updated['date']}."
    if any(x in low for x in ["remove ","delete ","cancel "]):
        target=find_event(low,old_time=times[0] if times else None)
        if not target: return "clarify", "Tell me which event to remove, for example: “Remove my doctor appointment tomorrow.”"
        removed=update_schedule("remove",event_id=target["id"]); return "removed", f"Removed {removed['title']} from {removed['date']} at {display_time(removed['time'])}."
    if re.match(r"^(add|create|schedule|book)\b",low):
        if not date or not times: return "clarify", "Please include the date and time, for example: “Add a meeting tomorrow at 3 PM.”"
        if not in_window(date): return "clarify", f"{date} is outside the next-30-days schedule window."
        title=re.sub(r"^(add|create|schedule|book)\s+","",text,flags=re.I)
        title=re.sub(r"\b(?:on|for)\s+(?:today|tomorrow|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?)\b","",title,flags=re.I)
        title=re.sub(r"\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)\b","",title,flags=re.I); title=re.sub(r"^(a|an|the)\s+","",title,flags=re.I).strip(" ,.-") or "New Event"
        e=update_schedule("add",{"title":title,"date":date,"time":times[0],"type":event_type(low),"duration":60,"source":"user"})
        return "added", f"Added {e['title']} for {date} at {display_time(e['time'])}."
    kind=None
    for word in ["meeting","workshop","appointment","task"]:
        if re.search(r"\b"+word+r"s?\b",low): kind=word; break
    if date: return "get_schedule", get_schedule(message,date=date)
    if kind: return "get_schedule", get_schedule(message,event_type=kind)
    return "get_schedule", get_schedule(message)


def answer(message, tool, result):
    if tool in {"chat","clarify","answer","free_time","added","updated","removed"}: return result
    if not result:
        low=message.lower()
        for word in ["meeting","workshop","appointment","task"]:
            if word in low: return f"You don't have any {word}s scheduled in the requested period."
        return "Nothing is scheduled in the requested period."
    if len(result)==1:
        e=result[0]; return f"{e['title']} — {e['date']} at {display_time(e['time'])} ({e['duration']} min)."
    return "\n".join(f"• {e['title']} — {e['date']} at {display_time(e['time'])} ({e['duration']} min)" for e in result)


class ChatRequest(BaseModel):
    message: str


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScheduleAI — Agentic RAG</title><style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#182033;--muted:#6d7890;--line:#e4e7ef;--accent:#6658e8;--dark:#11162a}*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink)}.app{display:flex;min-height:100vh}.side{width:255px;background:var(--dark);color:#fff;padding:28px 18px}.brand{font-size:25px;font-weight:800;margin-bottom:34px}.brand i{color:#8f84ff;font-style:normal}.nav{padding:12px 14px;border-radius:12px;color:#aeb7cf;margin:6px 0}.nav.on{background:#28234b;color:#fff}.note{margin-top:28px;padding:16px;border:1px solid #303750;border-radius:15px;color:#b9c1d4;font-size:12px;line-height:1.7}.main{width:min(1050px,100%);margin:auto;padding:34px}.top{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}h1{font-size:38px;line-height:1.1;margin:0 0 8px}.sub{color:var(--muted)}.pill{background:#e7f8ef;color:#247d49;border-radius:30px;padding:9px 13px;font-size:12px;font-weight:800;white-space:nowrap}.cards{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}.quick{border:1px solid var(--line);background:#fff;border-radius:12px;padding:11px 14px;cursor:pointer}.chat{background:#fff;border:1px solid var(--line);border-radius:22px;min-height:470px;padding:22px;box-shadow:0 16px 40px #1b23400b}.msg{max-width:80%;padding:14px 17px;border-radius:16px;margin:10px 0;white-space:pre-wrap;line-height:1.55}.bot{background:#f0f2f7}.user{margin-left:auto;background:var(--accent);color:#fff}.composer{display:flex;gap:10px;margin-top:14px}.composer input{flex:1;border:1px solid var(--line);border-radius:14px;padding:16px;font-size:15px;outline:none}.send{border:0;background:var(--accent);color:#fff;border-radius:14px;padding:0 25px;font-weight:800;cursor:pointer}.meta{text-align:center;color:#8b94a8;font-size:12px;margin-top:12px}@media(max-width:720px){.side{display:none}.main{padding:22px 14px}h1{font-size:30px}.pill{display:none}.msg{max-width:94%}.send{padding:0 18px}}
</style></head><body><div class="app"><aside class="side"><div class="brand">Schedule<i>AI</i></div><div class="nav on">✦ Assistant</div><div class="nav">▣ 30-Day Schedule</div><div class="nav">◷ Agent Tools</div><div class="note"><b>Try:</b><br>What do I have tomorrow?<br>Am I free Friday afternoon?<br>Add a meeting tomorrow at 3 PM.<br>Move my meeting from 2 PM to 4 PM.<br>Remove my workshop.</div></aside><main class="main"><div class="top"><div><h1>Your schedule, understood.</h1><div class="sub">Agentic routing + ChromaDB RAG for your next 30 days.</div></div><div class="pill">● SYSTEM ONLINE</div></div><div class="cards"><button class="quick" onclick="ask('What do I have today?')">📅 Today</button><button class="quick" onclick="ask('What do I have tomorrow?')">Tomorrow</button><button class="quick" onclick="ask('Am I free Friday afternoon?')">Free Friday</button><button class="quick" onclick="ask('When does college end today?')">College</button><button class="quick" onclick="ask('Show my meetings')">Meetings</button></div><div id="chat" class="chat"><div class="msg bot">Hi! I’m ScheduleAI.\n\nI can retrieve your schedule with RAG or update it when you ask. Try one of the examples above.</div></div><div class="composer"><input id="q" placeholder="Ask about your schedule..." onkeydown="if(event.key==='Enter')send()"><button class="send" onclick="send()">Send</button></div><div class="meta">FastAPI • ChromaDB • RAG • get_schedule • update_schedule • Asia/Kolkata</div></main></div><script>function add(text,cls){const x=document.createElement('div');x.className='msg '+cls;x.textContent=text;document.getElementById('chat').appendChild(x);x.scrollIntoView({behavior:'smooth'});return x}function ask(t){document.getElementById('q').value=t;send()}async function send(){const i=document.getElementById('q'),v=i.value.trim();if(!v)return;i.value='';add(v,'user');const b=add('Thinking…','bot');try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})});const d=await r.json();b.textContent=d.answer||'No answer returned.'}catch(e){b.textContent='The assistant could not be reached. Please try again.'}}</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        tool, result = agent(request.message)
        return {"answer": answer(request.message, tool, result), "tool": tool, "data": result}
    except Exception as exc:
        return {"answer": "I couldn't complete that request. Please try again.", "error": str(exc)}


@app.get("/health")
def health():
    return {"status":"ok","today":date_str(day_start()),"timezone":str(TZ),"events":collection.count(),"tools":["get_schedule","update_schedule"],"rag":"ChromaDB vector retrieval with deterministic local embeddings"}
