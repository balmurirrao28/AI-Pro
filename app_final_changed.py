import os
import re
import uuid
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Schedule AI")

# Dynamic schedule window
TZ = ZoneInfo(os.getenv("TIMEZONE", "Asia/Kolkata"))


def today():
    return datetime.now(TZ).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )


def date_string(d):
    return d.strftime("%Y-%m-%d")


db = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH", "./chroma_db")
)

# Fresh collection prevents old fixed-date records from leaking in.
collection = db.get_or_create_collection(
    "schedule_dynamic_v6"
)


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
    source="user"
):
    event = {
        "id": str(uuid.uuid4()),
        "title": title,
        "date": date,
        "time": time,
        "type": event_type,
        "duration": duration,
        "source": source
    }

    collection.add(
        ids=[event["id"]],
        documents=[event_text(event)],
        metadatas=[event]
    )

    return event


def refresh_system_schedule():
    data = collection.get(include=["metadatas"])

    old_ids = [
        event["id"]
        for event in data.get("metadatas", [])
        if event.get("source") == "system"
    ]

    if old_ids:
        collection.delete(ids=old_ids)

    start = today()

    # Monday-Wednesday: 9:20 AM - 4:00 PM
    # Thursday-Saturday: 9:20 AM - 3:00 PM
    # Sunday: No college
    for i in range(30):
        d = start + timedelta(days=i)

        if d.weekday() <= 2:
            duration = 400
        elif d.weekday() <= 5:
            duration = 340
        else:
            continue

        add_event(
            "College",
            date_string(d),
            "09:20",
            "college",
            duration,
            "system"
        )

    # Dynamic movie on the upcoming/current Friday at 7:20 PM.
    days_to_friday = (4 - start.weekday()) % 7
    movie_date = start + timedelta(days=days_to_friday)

    add_event(
        "Movie",
        date_string(movie_date),
        "19:20",
        "movie",
        150,
        "system"
    )


refresh_system_schedule()



# ============================================================
# TOOL 1: GET SCHEDULE
# ============================================================

def get_schedule(
    query="",
    date=None,
    start=None,
    end=None
):
    """
    Retrieves relevant schedule information using ChromaDB semantic search.
    Date/time filters are applied after retrieval so unrelated entries
    are never returned for a specific date or time window.
    """

    count = collection.count()

    if count == 0:
        return []

    results_count = min(max(count, 1), 5)

    results = collection.query(
        query_texts=[query or "college schedule"],
        n_results=results_count
    )

    items = results.get("metadatas", [[]])[0]

    if date:
        items = [
            event for event in items
            if event["date"] == date
        ]

    if start:
        items = [
            event for event in items
            if event["time"] >= start
        ]

    if end:
        items = [
            event for event in items
            if event["time"] <= end
        ]

    return sorted(
        items,
        key=lambda x: (x["date"], x["time"])
    )



# ============================================================
# TOOL 2: UPDATE SCHEDULE
# ============================================================

def update_schedule(
    action,
    event=None,
    event_id=None,
    changes=None
):
    """
    Adds, updates, or removes schedule entries.
    """

    if action == "add":

        new_event = dict(event)

        new_event.setdefault(
            "id",
            str(uuid.uuid4())
        )

        new_event.setdefault(
            "type",
            "meeting"
        )

        new_event.setdefault(
            "duration",
            60
        )

        collection.add(
            ids=[new_event["id"]],
            documents=[event_text(new_event)],
            metadatas=[new_event],
        )

        return new_event

    stored = collection.get(
        ids=[event_id],
        include=["metadatas"]
    )

    if not stored["metadatas"]:
        raise ValueError(
            "Schedule entry not found."
        )

    old_event = dict(
        stored["metadatas"][0]
    )

    if action == "remove":

        collection.delete(
            ids=[event_id]
        )

        return old_event

    if action == "update":

        old_event.update(
            changes or {}
        )

        collection.delete(
            ids=[event_id]
        )

        collection.add(
            ids=[event_id],
            documents=[event_text(old_event)],
            metadatas=[old_event],
        )

        return old_event

    raise ValueError(
        "Unknown schedule action."
    )


# ============================================================
# DATE PARSER
# ============================================================

def get_date(text):
    text = text.lower()
    current = today()

    if "today" in text:
        return date_string(current)

    if "tomorrow" in text:
        return date_string(current + timedelta(days=1))

    match = re.search(
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        text
    )

    if match:
        return (
            f"{int(match.group(1)):04d}-"
            f"{int(match.group(2)):02d}-"
            f"{int(match.group(3)):02d}"
        )

    months = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12
    }

    match = re.search(
        r"\b(" + "|".join(months.keys()) +
        r")\s+(\d{1,2})\b",
        text
    )

    if match:
        month = months[match.group(1)]
        year = current.year

        try:
            candidate = datetime(
                year,
                month,
                int(match.group(2)),
                tzinfo=TZ
            )
            if candidate.date() < current.date():
                year += 1
        except ValueError:
            return None

        return (
            f"{year:04d}-{month:02d}-"
            f"{int(match.group(2)):02d}"
        )

    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5,
        "sunday": 6
    }

    for name, number in weekdays.items():
        if name in text:
            delta = (number - current.weekday()) % 7
            if "next " + name in text:
                delta = delta or 7

            return date_string(
                current + timedelta(days=delta)
            )

    return None



# ============================================================
# TIME PARSER
# ============================================================

def get_times(text):

    times = []

    pattern = (
        r"\b(\d{1,2})"
        r"(?::(\d{2}))?"
        r"\s*(am|pm)\b"
    )

    for match in re.finditer(
        pattern,
        text.lower()
    ):

        hour = int(match[1])
        minute = int(match[2] or 0)
        period = match[3]

        if period == "pm" and hour != 12:
            hour += 12

        if period == "am" and hour == 12:
            hour = 0

        times.append(
            f"{hour:02d}:{minute:02d}"
        )

    return times


# ============================================================
# EVENT SEARCH
# ============================================================

def all_events():

    return collection.get(
        include=["metadatas"]
    )["metadatas"]


def find_event(
    text,
    old_time=None,
    event_type=None
):
    lower = text.lower()

    items = collection.get(
        include=["metadatas"]
    ).get("metadatas", [])

    date = get_date(text)

    if event_type:
        items = [
            e for e in items
            if e.get("type") == event_type
        ]

    if date:
        items = [
            e for e in items
            if e.get("date") == date
        ]

    if old_time:
        items = [
            e for e in items
            if e.get("time") == old_time
        ]

    for event in items:
        if event["title"].lower() in lower:
            return event

    if len(items) == 1:
        return items[0]

    return None



# ============================================================
# AGENT
# ============================================================

def agent(message):
    text = message.lower()
    date = get_date(text)
    times = get_times(text)

    # College ending questions
    if (
        "college" in text
        and any(
            word in text
            for word in [
                "end", "ending", "ends",
                "finish", "finishing",
                "time", "timing"
            ]
        )
    ):
        requested = date or date_string(today())
        d = datetime.strptime(requested, "%Y-%m-%d")

        if d.weekday() <= 2:
            result = {
                "date": requested,
                "day": d.strftime("%A"),
                "start": "9:20 AM",
                "end": "4:00 PM"
            }
        elif d.weekday() <= 5:
            result = {
                "date": requested,
                "day": d.strftime("%A"),
                "start": "9:20 AM",
                "end": "3:00 PM"
            }
        else:
            result = {
                "date": requested,
                "day": d.strftime("%A"),
                "start": None,
                "end": "No college"
            }

        return "college", result

    # This week
    if "this week" in text or "weekly schedule" in text:
        current = today()
        monday = current - timedelta(days=current.weekday())
        result = []

        for i in range(6):
            d = monday + timedelta(days=i)

            result.append({
                "day": d.strftime("%A"),
                "date": date_string(d),
                "start": "9:20 AM",
                "end": (
                    "4:00 PM"
                    if d.weekday() <= 2
                    else "3:00 PM"
                )
            })

        return "week", result

    # Movie queries
    if "movie" in text or "film" in text:
        return (
            "get_schedule",
            get_schedule(
                message,
                date=date,
                event_type="movie"
            )
        )

    # Event type queries
    type_map = {
        "workshop": "workshop",
        "workshops": "workshop",
        "meeting": "meeting",
        "meetings": "meeting",
        "appointment": "appointment",
        "appointments": "appointment",
        "task": "task",
        "tasks": "task"
    }

    for word, kind in type_map.items():
        if word in text:
            return (
                "get_schedule",
                get_schedule(
                    message,
                    date=date,
                    event_type=kind
                )
            )

    # Move / reschedule
    if any(
        word in text
        for word in [
            "move ",
            "reschedule ",
            "change "
        ]
    ):
        old_time = times[0] if len(times) > 1 else None
        new_time = times[-1] if times else None

        target = find_event(
            text,
            old_time=old_time
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
                    changes=changes
                )
            )

        return (
            "clarification",
            "Which event would you like to move, "
            "and what should its new time be?"
        )

    # Remove / delete / cancel
    if any(
        word in text
        for word in [
            "remove ",
            "delete ",
            "cancel "
        ]
    ):
        target = find_event(
            text,
            old_time=times[0] if times else None
        )

        if target:
            return (
                "update_schedule",
                update_schedule(
                    "remove",
                    event_id=target["id"]
                )
            )

        return (
            "clarification",
            "Which event would you like to remove?"
        )

    # Add / create / schedule / book
    if any(
        word in text
        for word in [
            "add ",
            "create ",
            "schedule ",
            "book "
        ]
    ):
        if date and times:
            title = re.sub(
                r"^\s*(add|create|schedule|book)\s+",
                "",
                message,
                flags=re.I
            ).strip()

            title = re.sub(
                r"^(a|an|the)\s+",
                "",
                title,
                flags=re.I
            ).strip()

            # Remove common date/time tails from title.
            title = re.sub(
                r"\s+(?:today|tomorrow|"
                r"(?:next\s+)?(?:monday|tuesday|wednesday|"
                r"thursday|friday|saturday|sunday))"
                r"(?:\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm))?",
                "",
                title,
                flags=re.I
            )

            title = re.sub(
                r"\s+(?:on|for)\s+.*$",
                "",
                title,
                flags=re.I
            ).strip(" ,.-")

            title = title or "New Event"

            if "movie" in text or "film" in text:
                event_type = "movie"
            elif "appointment" in text:
                event_type = "appointment"
            elif "workshop" in text:
                event_type = "workshop"
            elif "task" in text:
                event_type = "task"
            else:
                event_type = "meeting"

            return (
                "update_schedule",
                update_schedule(
                    "add",
                    {
                        "title": title,
                        "date": date,
                        "time": times[0],
                        "type": event_type,
                        "duration": 60
                    }
                )
            )

        return (
            "clarification",
            "Sure! What would you like to schedule? "
            "Please provide the event and date/time."
        )

    # Explicitly stop vague commands from reaching RAG.
    if text.strip() in {
        "schedule",
        "schedule it",
        "add",
        "add it",
        "book",
        "book it",
        "create",
        "create it"
    }:
        return (
            "clarification",
            "Sure! What would you like to schedule? "
            "Please provide the event and date/time."
        )

    # Free time
    if "free" in text:
        start = None
        end = None

        if "afternoon" in text:
            start = "12:00"
            end = "17:00"

        return (
            "get_schedule",
            get_schedule(
                message,
                date=date,
                start=start,
                end=end
            )
        )

    # General RAG retrieval
    return (
        "get_schedule",
        get_schedule(
            message,
            date=date
        )
    )



def make_answer(question, tool, result):
    text = question.lower()

    if tool == "clarification":
        return result

    if tool == "college":
        if result["end"] == "No college":
            return (
                f"{result['day']} ({result['date']}): "
                "No college is scheduled."
            )

        return (
            f"Your college is from {result['start']} "
            f"until {result['end']} on "
            f"{result['day']} ({result['date']})."
        )

    if tool == "week":
        return "\n".join(
            f"• {item['day']} — "
            f"{item['date']}: "
            f"{item['start']} – {item['end']}"
            for item in result
        )

    if isinstance(result, dict):
        return (
            f"Done — {result['title']} "
            f"is scheduled for {result['date']} at "
            f"{result['time']}."
        )

    if not result:
        if "workshop" in text:
            return "You don't have any workshops scheduled."
        if "meeting" in text:
            return "You don't have any meetings scheduled."
        if "appointment" in text:
            return (
                "You don't have any appointments scheduled. "
                "You can say: Schedule a doctor appointment "
                "Friday at 11 AM."
            )
        if "task" in text:
            return "You don't have any tasks scheduled."
        if "movie" in text:
            return "You don't have any movies scheduled."
        if "free" in text:
            return "You are free during that requested period."
        if "today" in text:
            return "You have nothing scheduled today."

        return "No matching schedule entries found."

    if len(result) == 1:
        event = result[0]
        return (
            f"{event['title']} is scheduled for "
            f"{event['date']} at {event['time']}."
        )

    return "\n".join(
        f"• {event['title']} — "
        f"{event['date']} at {event['time']}"
        for event in result
    )



# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str


# ============================================================
# NEW UI
# ============================================================

PAGE = r"""
<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>Schedule AI</title>

<style>

*{
box-sizing:border-box
}

body{
margin:0;
background:#f5f7fb;
color:#172033;
font-family:Arial,sans-serif
}

.layout{
display:flex;
min-height:100vh
}

.sidebar{
width:240px;
background:white;
border-right:1px solid #e5e9f2;
padding:28px 18px
}

.logo{
font-size:23px;
font-weight:800;
margin-bottom:35px
}

.logo span{
color:#6757e8
}

.nav{
padding:12px 14px;
margin:6px 0;
border-radius:10px;
color:#65708a
}

.nav.active{
background:#eeeafd;
color:#4f40c7;
font-weight:700
}

.tip{
margin-top:35px;
background:#f4f2ff;
border-radius:14px;
padding:15px;
color:#646b80;
font-size:12px;
line-height:1.7
}

.main{
max-width:1000px;
width:100%;
margin:auto;
padding:42px
}

.header{
display:flex;
justify-content:space-between;
align-items:center
}

h1{
margin:0;
font-size:34px
}

.subtitle{
color:#7a8498;
margin-top:8px
}

.status{
background:#e9f8ef;
color:#23844a;
padding:8px 12px;
border-radius:20px;
font-size:12px
}

.quick{
display:flex;
gap:9px;
flex-wrap:wrap;
margin:28px 0
}

.quick button{
border:1px solid #dfe4ee;
background:white;
border-radius:9px;
padding:10px 14px;
cursor:pointer
}

.chat{
background:white;
border:1px solid #e1e6ef;
border-radius:18px;
padding:22px;
min-height:420px;
box-shadow:0 8px 30px #26345c0b
}

.message{
max-width:78%;
padding:14px 17px;
border-radius:14px;
margin:12px 0;
white-space:pre-wrap;
line-height:1.55
}

.bot{
background:#f1f3f8
}

.user{
margin-left:auto;
background:#6757e8;
color:white
}

.composer{
display:flex;
gap:10px;
margin-top:15px
}

.composer input{
flex:1;
border:1px solid #dce2ec;
border-radius:12px;
padding:16px;
font-size:15px
}

.composer button{
border:0;
background:#6757e8;
color:white;
border-radius:12px;
padding:0 24px;
font-weight:700;
cursor:pointer
}

.footer{
font-size:12px;
color:#8b94a8;
margin-top:12px
}

.login-card{
background:white;border:1px solid #e1e6ef;border-radius:16px;
padding:16px 20px;margin-bottom:25px;display:flex;
justify-content:space-between;align-items:center;
box-shadow:0 5px 20px #26345c0b
}
.login-card strong{display:block;font-size:15px;margin-bottom:5px}
.login-card small{color:#8992a5}
.google-login{background:white;border:1px solid #d9deea;border-radius:10px;
padding:11px 16px;cursor:pointer;font-weight:600;display:flex;
align-items:center;gap:9px}
.google-icon{font-weight:800;font-size:18px}
.google-login:hover{background:#f7f8fc}

@media(max-width:700px){

.sidebar{
display:none
}

.main{
padding:22px 15px
}

h1{
font-size:27px
}

.message{
max-width:95%
}

}

</style>

</head>

<body>

<div class="layout">

<aside class="sidebar">

<div class="logo">
Schedule<span>AI</span>
</div>

<div class="nav active">
✦ Assistant
</div>

<div class="nav">
▣ My Schedule
</div>

<div class="nav">
◷ 30-Day Planner
</div>

<div class="tip">

<b>Try asking</b>

<br><br>

When does college end today?

<br>

When does college end Friday?

<br>

When does college end tomorrow?

<br>

What is my schedule this week?

</div>

</aside>

<main class="main">

<div class="login-card">
<div>
<strong>Welcome to ScheduleAI</strong>
<small>Sign in to manage your personal schedule</small>
</div>
<button class="google-login" onclick="googleLogin()">
<span class="google-icon">G</span> Sign in with Google
</button>
</div>

<div class="header">

<div>

<h1>
Your college schedule, understood.
</h1>

<div class="subtitle">
Ask about college timings, free time, or your schedule.
</div>

</div>

<div class="status">
● RAG ONLINE
</div>

</div>

<div class="quick">

<button onclick="ask('When does college end today?')">
🎓 College Today
</button>

<button onclick="ask('When does college end tomorrow?')">
📅 Tomorrow
</button>

<button onclick="ask('When does college end Friday?')">
🕒 Friday
</button>

<button onclick="ask('What is my schedule this week?')">
📚 This Week
</button>

</div>

<div id="chat" class="chat">

<div class="message bot">

Hi! I’m your ScheduleAI assistant.\n\nI can tell you when college ends, check your free time, and add, move, or remove events.

</div>

</div>

<div class="composer">

<input
id="question"
placeholder="Ask about your schedule..."
onkeydown="if(event.key==='Enter') send()"
>

<button onclick="send()">
Send
</button>

</div>

<div class="footer">

FastAPI + ChromaDB • College timetable • Agent tools: get_schedule, update_schedule

</div>

</main>

</div>

<script>

function addMessage(text,type){

const box=document.createElement("div");

box.className="message "+type;

box.textContent=text;

document.getElementById("chat").appendChild(box);

box.scrollIntoView({
behavior:"smooth"
});

}

function ask(text){

document.getElementById("question").value=text;

send();

}

function googleLogin(){
alert("Google sign-in requires your Firebase/Google OAuth configuration. The login interface is ready.");
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

const response=await fetch(
"/chat",
{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify({
message:value
})
}
);

const data=await response.json();

bot.textContent=data.answer;

}catch(error){

bot.textContent=
"Unable to reach the assistant.";

}

}

</script>

</body>

</html>
"""


@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return PAGE


@app.post("/chat")
def chat(request: ChatRequest):

    try:

        tool,result=agent(
            request.message
        )

        return {
            "answer":make_answer(
                request.message,
                tool,
                result
            ),
            "tool":tool,
            "data":result
        }

    except Exception as error:

        return {
            "answer":
            f"Request failed: {error}",
            "error":str(error)
        }


@app.get("/health")
def health():

    return {
        "status":"ok",
        "tools":[
            "get_schedule",
            "update_schedule"
        ],
        "events":collection.count()
    }