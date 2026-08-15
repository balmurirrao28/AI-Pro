import os
import re
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="ScheduleAI")

TZ = ZoneInfo(os.getenv("TIMEZONE", "Asia/Kolkata"))
DB_PATH = os.getenv("CHROMA_PATH", "./chroma_db")

db = chromadb.PersistentClient(path=DB_PATH)
collection = db.get_or_create_collection("schedule_dynamic_v7")


def now():
    return datetime.now(TZ)


def today():
    return now().replace(
        hour=0, minute=0, second=0, microsecond=0
    )


def fmt_date(d):
    return d.strftime("%Y-%m-%d")


def fmt_time(value):
    hour, minute = map(int, value.split(":"))
    suffix = "AM" if hour < 12 else "PM"
    display_hour = hour % 12 or 12
    return f"{display_hour}:{minute:02d} {suffix}"


def event_text(event):
    return (
        f"{event['title']} on {event['date']} at {event['time']}. "
        f"{event['type']} for {event['duration']} minutes."
    )


def add_event(
    title,
    date,
    time,
    event_type,
    duration=60,
    source="user",
):
    event = {
        "id": str(uuid.uuid4()),
        "title": title,
        "date": date,
        "time": time,
        "type": event_type,
        "duration": int(duration),
        "source": source,
    }

    collection.add(
        ids=[event["id"]],
        documents=[event_text(event)],
        metadatas=[event],
    )
    return event


def all_events():
    return collection.get(include=["metadatas"]).get("metadatas", [])


def refresh_college_schedule():
    """
    Rebuild only system college/movie entries from today forward.
    User-created events are never removed.
    This makes the 30-day window roll automatically every day.
    """
    existing = all_events()
    old_system = [
        e["id"] for e in existing
        if e.get("source") == "system"
    ]

    if old_system:
        collection.delete(ids=old_system)

    start = today()

    for offset in range(30):
        d = start + timedelta(days=offset)
        weekday = d.weekday()

        if weekday <= 2:
            duration = 400
        elif weekday <= 5:
            duration = 340
        else:
            continue

        add_event(
            "College",
            fmt_date(d),
            "09:20",
            "college",
            duration,
            "system",
        )

    # Movie is always on the next/current Friday at 7:20 PM.
    days_to_friday = (4 - start.weekday()) % 7
    movie = start + timedelta(days=days_to_friday)

    if movie <= start + timedelta(days=29):
        add_event(
            "Movie",
            fmt_date(movie),
            "19:20",
            "movie",
            150,
            "system",
        )


refresh_college_schedule()


# ============================================================
# TOOL 1: get_schedule
# ============================================================

def get_schedule(
    query="",
    date=None,
    start=None,
    end=None,
    event_type=None,
):
    """
    Retrieves schedule information.
    Exact metadata filtering is used for dates/types/times.
    Chroma semantic retrieval is used only for open-ended questions.
    """
    events = all_events()
    text = (query or "").lower()

    if event_type:
        events = [
            e for e in events
            if e.get("type") == event_type
        ]

    if date:
        events = [
            e for e in events
            if e.get("date") == date
        ]

    if start:
        events = [
            e for e in events
            if e.get("time", "") >= start
        ]

    if end:
        events = [
            e for e in events
            if e.get("time", "") <= end
        ]

    if event_type or date or start or end:
        return sorted(
            events,
            key=lambda e: (e["date"], e["time"])
        )

    if not events:
        return []

    result = collection.query(
        query_texts=[query or "schedule"],
        n_results=min(5, len(events)),
    )

    return sorted(
        result.get("metadatas", [[]])[0],
        key=lambda e: (e["date"], e["time"])
    )


# ============================================================
# TOOL 2: update_schedule
# ============================================================

def update_schedule(
    action,
    event=None,
    event_id=None,
    changes=None,
):
    if action == "add":
        data = dict(event or {})
        data.setdefault("id", str(uuid.uuid4()))
        data.setdefault("type", "meeting")
        data.setdefault("duration", 60)
        data["source"] = "user"

        collection.add(
            ids=[data["id"]],
            documents=[event_text(data)],
            metadatas=[data],
        )
        return data

    stored = collection.get(
        ids=[event_id],
        include=["metadatas"],
    )

    if not stored.get("metadatas"):
        raise ValueError("Schedule entry not found.")

    old = dict(stored["metadatas"][0])

    if action == "remove":
        collection.delete(ids=[event_id])
        return old

    if action == "update":
        old.update(changes or {})
        old["source"] = "user"

        collection.delete(ids=[event_id])
        collection.add(
            ids=[event_id],
            documents=[event_text(old)],
            metadatas=[old],
        )
        return old

    raise ValueError("Unknown schedule action.")


# ============================================================
# DATE / TIME PARSING
# ============================================================

MONTHS = {
    "january": 1, "february": 2, "march": 3,
    "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def resolve_date(text):
    text = text.lower()
    base = today()

    if "today" in text:
        return fmt_date(base)

    if "tomorrow" in text:
        return fmt_date(base + timedelta(days=1))

    numeric = re.search(
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        text,
    )
    if numeric:
        return (
            f"{int(numeric.group(1)):04d}-"
            f"{int(numeric.group(2)):02d}-"
            f"{int(numeric.group(3)):02d}"
        )

    month_match = re.search(
        r"\b("
        + "|".join(MONTHS)
        + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        text,
    )

    if month_match:
        month = MONTHS[month_match.group(1)]
        day = int(month_match.group(2))
        year = base.year

        candidate = datetime(
            year, month, day, tzinfo=TZ
        )

        if candidate.date() < base.date():
            year += 1

        return f"{year:04d}-{month:02d}-{day:02d}"

    for name, weekday in WEEKDAYS.items():
        if re.search(r"\b" + name + r"\b", text):
            delta = (weekday - base.weekday()) % 7

            if re.search(
                r"\bnext\s+" + name + r"\b",
                text,
            ):
                delta = delta or 7

            return fmt_date(
                base + timedelta(days=delta)
            )

    return None


def resolve_times(text):
    found = []

    pattern = (
        r"\b(\d{1,2})"
        r"(?::(\d{2}))?"
        r"\s*(am|pm)\b"
    )

    for match in re.finditer(
        pattern,
        text.lower(),
    ):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)

        if match.group(3) == "pm" and hour != 12:
            hour += 12

        if match.group(3) == "am" and hour == 12:
            hour = 0

        found.append(f"{hour:02d}:{minute:02d}")

    return found


# ============================================================
# EVENT MATCHING
# ============================================================

def find_event(text, old_time=None, event_type=None):
    events = all_events()
    date = resolve_date(text)
    lower = text.lower()

    if event_type:
        events = [
            e for e in events
            if e.get("type") == event_type
        ]

    if date:
        events = [
            e for e in events
            if e.get("date") == date
        ]

    if old_time:
        events = [
            e for e in events
            if e.get("time") == old_time
        ]

    # Prefer title words that are actually present in the question.
    meaningful = []
    for event in events:
        title = event["title"].lower()
        if title in lower:
            meaningful.append(event)

    if len(meaningful) == 1:
        return meaningful[0]

    if len(events) == 1:
        return events[0]

    return None


def event_type_from_text(text):
    if "movie" in text or "film" in text:
        return "movie"
    if "appointment" in text:
        return "appointment"
    if "workshop" in text:
        return "workshop"
    if "task" in text:
        return "task"
    if "college" in text:
        return "college"
    return "meeting"


# ============================================================
# AGENT
# ============================================================

def agent(message):
    # Refresh system entries on every request so the 30-day window
    # never becomes stale.
    refresh_college_schedule()

    text = message.lower().strip()
    date = resolve_date(text)
    times = resolve_times(text)

    # College questions are handled directly, not by semantic search.
    if (
        "college" in text
        and any(
            word in text
            for word in [
                "end", "ending", "ends",
                "finish", "finishing",
                "time", "timing",
            ]
        )
    ):
        requested = date or fmt_date(today())
        d = datetime.strptime(requested, "%Y-%m-%d")

        if d.weekday() <= 2:
            ending = "4:00 PM"
        elif d.weekday() <= 5:
            ending = "3:00 PM"
        else:
            ending = "No college scheduled"

        return (
            "college",
            {
                "date": requested,
                "day": d.strftime("%A"),
                "ending": ending,
            },
        )

    # Week questions.
    if (
        "this week" in text
        or "weekly schedule" in text
        or "college schedule this week" in text
    ):
        base = today()
        monday = base - timedelta(days=base.weekday())

        result = []

        for i in range(6):
            d = monday + timedelta(days=i)

            if d.weekday() <= 2:
                ending = "4:00 PM"
            else:
                ending = "3:00 PM"

            result.append(
                {
                    "day": d.strftime("%A"),
                    "date": fmt_date(d),
                    "start": "9:20 AM",
                    "end": ending,
                }
            )

        return "week", result

    # Incomplete commands must NEVER fall through to RAG.
    if text in {
        "schedule",
        "schedule it",
        "add",
        "add it",
        "book",
        "book it",
        "create",
        "create it",
    }:
        return (
            "clarification",
            "Sure! What would you like to schedule? "
            "Please provide the event and date/time.",
        )

    # Move / reschedule / change.
    if any(
        word in text
        for word in [
            "move ", "reschedule ", "change ",
        ]
    ):
        old_time = times[0] if len(times) > 1 else None
        new_time = times[-1] if times else None

        target = find_event(
            text,
            old_time=old_time,
        )

        if target and new_time:
            changes = {"time": new_time}

            if date:
                changes["date"] = date

            return (
                "update_schedule",
                update_schedule(
                    "update",
                    event_id=target["id"],
                    changes=changes,
                ),
            )

        return (
            "clarification",
            "I can move it. Please tell me the event "
            "and the new date/time.",
        )

    # Remove / delete / cancel.
    if any(
        word in text
        for word in [
            "remove ", "delete ", "cancel ",
        ]
    ):
        target = find_event(
            text,
            old_time=times[0] if times else None,
        )

        if target:
            return (
                "update_schedule",
                update_schedule(
                    "remove",
                    event_id=target["id"],
                ),
            )

        return (
            "clarification",
            "Which event would you like me to remove?",
        )

    # Add / create / schedule / book.
    add_command = any(
        text.startswith(word)
        for word in [
            "add ",
            "create ",
            "schedule ",
            "book ",
        ]
    )

    if add_command:
        if not date or not times:
            return (
                "clarification",
                "What would you like to schedule? "
                "Please include the date and time.",
            )

        title = re.sub(
            r"^\s*(add|create|schedule|book)\s+",
            "",
            message,
            flags=re.I,
        ).strip()

        title = re.sub(
            r"^(a|an|the)\s+",
            "",
            title,
            flags=re.I,
        ).strip()

        # Remove common date/time wording from title.
        title = re.sub(
            r"\s+(?:on|for)\s+"
            r"(?:today|tomorrow|"
            r"(?:next\s+)?(?:monday|tuesday|wednesday|"
            r"thursday|friday|saturday|sunday))",
            "",
            title,
            flags=re.I,
        )

        title = re.sub(
            r"\s+(?:today|tomorrow|"
            r"(?:next\s+)?(?:monday|tuesday|wednesday|"
            r"thursday|friday|saturday|sunday))"
            r"\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)",
            "",
            title,
            flags=re.I,
        )

        title = re.sub(
            r"\s+(?:at|on)\s+\d{1,2}"
            r"(?::\d{2})?\s*(?:am|pm)",
            "",
            title,
            flags=re.I,
        )

        title = title.strip(" ,.-") or "New Event"

        return (
            "update_schedule",
            update_schedule(
                "add",
                {
                    "title": title,
                    "date": date,
                    "time": times[0],
                    "type": event_type_from_text(text),
                    "duration": 60,
                },
            ),
        )

    # Explicit event-type retrieval.
    type_words = {
        "workshop": "workshop",
        "workshops": "workshop",
        "meeting": "meeting",
        "meetings": "meeting",
        "appointment": "appointment",
        "appointments": "appointment",
        "task": "task",
        "tasks": "task",
        "movie": "movie",
        "movies": "movie",
        "film": "movie",
    }

    for word, kind in type_words.items():
        if re.search(r"\b" + word + r"\b", text):
            return (
                "get_schedule",
                get_schedule(
                    message,
                    date=date,
                    event_type=kind,
                ),
            )

    # Free-time questions.
    if "free" in text:
        start = "12:00" if "afternoon" in text else None
        end = "17:00" if "afternoon" in text else None

        return (
            "free_time",
            get_schedule(
                message,
                date=date,
                start=start,
                end=end,
            ),
        )

    # Today/tomorrow/general date queries.
    if date:
        return (
            "get_schedule",
            get_schedule(
                message,
                date=date,
            ),
        )

    # Final RAG retrieval.
    return (
        "get_schedule",
        get_schedule(message),
    )


# ============================================================
# ANSWERS
# ============================================================

def make_answer(question, tool, result):
    text = question.lower()

    if tool == "clarification":
        return result

    if tool == "college":
        if result["ending"] == "No college scheduled":
            return (
                f"{result['day']} ({result['date']}): "
                "No college is scheduled."
            )

        return (
            f"Your college ends at {result['ending']} "
            f"on {result['day']} ({result['date']})."
        )

    if tool == "week":
        return "\n".join(
            f"• {item['day']} ({item['date']}): "
            f"9:20 AM – {item['end']}"
            for item in result
        )

    if tool == "free_time":
        if not result:
            return "You are free during that requested period."

        return "\n".join(
            f"• {e['title']} — {e['date']} at {fmt_time(e['time'])}"
            for e in result
        )

    if isinstance(result, dict):
        return (
            f"Done — {result['title']} is scheduled for "
            f"{result['date']} at {fmt_time(result['time'])}."
        )

    if not result:
        if "workshop" in text:
            return "You don't have any workshops scheduled."
        if "meeting" in text:
            return "You don't have any meetings scheduled."
        if "appointment" in text:
            return "You don't have any appointments scheduled."
        if "task" in text:
            return "You don't have any tasks scheduled."
        if "movie" in text or "film" in text:
            return "You don't have any movies scheduled."
        if "today" in text:
            return "You have nothing scheduled today."
        return "No matching schedule entries found."

    if len(result) == 1:
        e = result[0]
        return (
            f"{e['title']} is scheduled for "
            f"{e['date']} at {fmt_time(e['time'])}."
        )

    return "\n".join(
        f"• {e['title']} — {e['date']} at {fmt_time(e['time'])}"
        for e in result
    )


# ============================================================
# API
# ============================================================

class ChatRequest(BaseModel):
    message: str


PAGE = r"""
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ScheduleAI</title>
<style>
*{box-sizing:border-box}
body{
margin:0;font-family:Inter,Arial,sans-serif;
background:linear-gradient(135deg,#f7f8ff,#eef2ff);
color:#172033
}
.layout{display:flex;min-height:100vh}
.sidebar{
width:250px;background:#11162a;color:#fff;
padding:28px 18px;box-shadow:10px 0 35px #1d24420f
}
.logo{font-size:25px;font-weight:800;margin-bottom:40px}
.logo span{color:#8b7cff}
.nav{padding:13px 15px;margin:7px 0;border-radius:12px;color:#aeb7cf}
.nav.active{background:#292451;color:#fff}
.tip{
margin-top:35px;padding:18px;border:1px solid #303750;
border-radius:16px;background:#171d34;color:#aeb7cf;
font-size:12px;line-height:1.8
}
.main{flex:1;max-width:1100px;margin:auto;padding:38px}
.login{
background:white;border:1px solid #e1e5ef;border-radius:18px;
padding:14px 18px;display:flex;align-items:center;
justify-content:space-between;gap:20px;margin-bottom:25px;
box-shadow:0 10px 30px #252b5109
}
.login small{display:block;color:#8791a5;margin-top:4px}
.google{
background:white;border:1px solid #d9deea;border-radius:10px;
padding:10px 16px;font-weight:700;cursor:pointer
}
.header{display:flex;justify-content:space-between;align-items:flex-start}
h1{font-size:38px;margin:0 0 8px}
.subtitle{color:#77829b}
.status{
background:#e6f8ee;color:#25824b;padding:9px 13px;
border-radius:30px;font-size:12px;font-weight:700
}
.quick{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}
.quick button{
background:white;border:1px solid #dfe4ee;border-radius:12px;
padding:11px 15px;cursor:pointer;transition:.2s
}
.quick button:hover{transform:translateY(-2px);box-shadow:0 8px 20px #26345c12}
.chat{
background:#fff;border:1px solid #e1e6ef;border-radius:22px;
min-height:430px;padding:24px;box-shadow:0 15px 45px #252b5112
}
.message{
max-width:78%;padding:15px 18px;border-radius:16px;
margin:12px 0;white-space:pre-wrap;line-height:1.6
}
.bot{background:#f1f3f8}
.user{margin-left:auto;background:#6757e8;color:#fff}
.composer{display:flex;gap:10px;margin-top:15px}
.composer input{
flex:1;padding:17px;border:1px solid #dce2ec;border-radius:14px;
font-size:15px;outline:none
}
.composer input:focus{border-color:#8b7cff;box-shadow:0 0 0 4px #8b7cff14}
.send{
border:0;border-radius:14px;background:#6757e8;color:#fff;
padding:0 25px;font-weight:800;cursor:pointer
}
.footer{text-align:center;color:#8a94a8;font-size:12px;margin-top:12px}
@media(max-width:720px){
.sidebar{display:none}.main{padding:22px 14px}h1{font-size:29px}
.status{display:none}.message{max-width:95%}
.login{align-items:flex-start;flex-direction:column}
.google{width:100%}.send{padding:0 18px}
}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
<div class="logo">Schedule<span>AI</span></div>
<div class="nav active">✦ Assistant</div>
<div class="nav">▣ My Schedule</div>
<div class="nav">◷ 30-Day Planner</div>
<div class="tip">
<b>Try asking</b><br><br>
When does college end today?<br>
When does college end Friday?<br>
What is my college schedule this week?<br>
When is my movie scheduled?<br>
Schedule a doctor appointment Friday at 11 AM.<br>
Move my movie to 8 PM.
</div>
</aside>

<main class="main">
<div class="login">
<div>
<strong>Welcome to ScheduleAI</strong>
<small>Your personal 30-day schedule assistant</small>
</div>
<button class="google" onclick="googleLogin()">G&nbsp; Sign in with Google</button>
</div>

<div class="header">
<div>
<h1>Your schedule, understood.</h1>
<div class="subtitle">
Ask naturally. The agent retrieves or updates the right schedule entry.
</div>
</div>
<div class="status">● RAG ONLINE</div>
</div>

<div class="quick">
<button onclick="ask('What do I have today?')">📅 Today</button>
<button onclick="ask('What do I have tomorrow?')">Tomorrow</button>
<button onclick="ask('When does college end today?')">🎓 College</button>
<button onclick="ask('When is my movie scheduled?')">🎬 Movie</button>
<button onclick="ask('What is my college schedule this week?')">This Week</button>
</div>

<div id="chat" class="chat">
<div class="message bot">
Hi! I'm your ScheduleAI assistant.

Ask me about college, movies, meetings, appointments, tasks, free time, or events you want to add, move, or remove.
</div>
</div>

<div class="composer">
<input id="question" placeholder="Ask: When is my movie scheduled?"
onkeydown="if(event.key==='Enter')send()">
<button class="send" onclick="send()">Send</button>
</div>

<div class="footer">
Dynamic dates • Rolling 30 days • ChromaDB RAG • get_schedule • update_schedule
</div>
</main>
</div>

<script>
function addMessage(text,type){
const box=document.createElement("div");
box.className="message "+type;
box.textContent=text;
document.getElementById("chat").appendChild(box);
box.scrollIntoView({behavior:"smooth"});
}

function ask(text){
document.getElementById("question").value=text;
send();
}

function googleLogin(){
alert("Google sign-in requires your Firebase/Google OAuth configuration. The interface is ready.");
}

async function send(){
const input=document.getElementById("question");
const value=input.value.trim();
if(!value)return;

input.value="";
addMessage(value,"user");
addMessage("Thinking...","bot");

const messages=document.querySelectorAll(".message");
const bot=messages[messages.length-1];

try{
const response=await fetch("/chat",{
method:"POST",
headers:{"Content-Type":"application/json"},
body:JSON.stringify({message:value})
});
const data=await response.json();
bot.textContent=data.answer;
}catch(error){
bot.textContent="Unable to reach the assistant.";
}
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        tool, result = agent(request.message)
        return {
            "answer": make_answer(
                request.message,
                tool,
                result,
            ),
            "tool": tool,
            "data": result,
        }
    except Exception as error:
        return {
            "answer": f"Request failed: {error}",
            "error": str(error),
        }


@app.get("/health")
def health():
    refresh_college_schedule()
    return {
        "status": "ok",
        "today": fmt_date(today()),
        "timezone": str(TZ),
        "events": collection.count(),
        "tools": ["get_schedule", "update_schedule"],
    }
