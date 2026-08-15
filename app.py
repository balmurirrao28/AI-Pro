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

# ============================================================
# DYNAMIC DATE
# ============================================================

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


# ============================================================
# CHROMADB
# ============================================================

db = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH", "./chroma_db")
)

collection = db.get_or_create_collection(
    "schedule_dynamic_v5"
)


def event_text(e):
    return (
        f"{e['title']} on {e['date']} at {e['time']}. "
        f"{e['type']} for {e['duration']} minutes."
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


# ============================================================
# DYNAMIC SYSTEM SCHEDULE
# ============================================================

def refresh_system_schedule():
    data = collection.get(
        include=["metadatas"]
    )

    old_ids = []

    for event in data.get("metadatas", []):
        if event.get("source") == "system":
            old_ids.append(event["id"])

    if old_ids:
        collection.delete(ids=old_ids)

    start = today()

    # Monday-Wednesday: 9:20 AM - 4:00 PM
    # Thursday-Saturday: 9:20 AM - 3:00 PM
    # Sunday: no college

    for i in range(30):
        d = start + timedelta(days=i)
        weekday = d.weekday()

        if weekday <= 2:
            duration = 400
        elif weekday <= 5:
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

    # Dynamic movie every upcoming/current Friday
    days_to_friday = (4 - start.weekday()) % 7

    movie_date = start + timedelta(
        days=days_to_friday
    )

    if movie_date <= start + timedelta(days=29):
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
# TOOL 1 - GET SCHEDULE
# ============================================================

def get_schedule(
    query="",
    date=None,
    start=None,
    end=None,
    event_type=None
):
    """
    Retrieves relevant schedule information.
    Explicit date/type filters are applied before returning results.
    ChromaDB is used for open-ended RAG retrieval.
    """

    text = (query or "").lower()

    requested_type = event_type

    if not requested_type:
        if "workshop" in text:
            requested_type = "workshop"
        elif "meeting" in text:
            requested_type = "meeting"
        elif "appointment" in text:
            requested_type = "appointment"
        elif "task" in text:
            requested_type = "task"
        elif "movie" in text or "film" in text:
            requested_type = "movie"
        elif "college" in text:
            requested_type = "college"

    all_data = collection.get(
        include=["metadatas"]
    )

    items = all_data.get(
        "metadatas",
        []
    )

    # Explicit filters
    if requested_type:
        items = [
            e for e in items
            if e.get("type") == requested_type
        ]

    if date:
        items = [
            e for e in items
            if e.get("date") == date
        ]

    if start:
        items = [
            e for e in items
            if e.get("time", "") >= start
        ]

    if end:
        items = [
            e for e in items
            if e.get("time", "") <= end
        ]

    # If explicit filters were used, return exact matches.
    if requested_type or date or start or end:
        return sorted(
            items,
            key=lambda x: (
                x["date"],
                x["time"]
            )
        )

    # Otherwise use ChromaDB semantic retrieval.
    if not items:
        return []

    results = collection.query(
        query_texts=[
            query or "schedule"
        ],
        n_results=min(5, len(items))
    )

    results = results.get(
        "metadatas",
        [[]]
    )[0]

    return sorted(
        results,
        key=lambda x: (
            x["date"],
            x["time"]
        )
    )


# ============================================================
# TOOL 2 - UPDATE SCHEDULE
# ============================================================

def update_schedule(
    action,
    event=None,
    event_id=None,
    changes=None
):
    """
    Adds, updates, moves or removes schedule entries.
    """

    if action == "add":

        new_event = dict(event or {})

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

        new_event["source"] = "user"

        collection.add(
            ids=[new_event["id"]],
            documents=[
                event_text(new_event)
            ],
            metadatas=[new_event]
        )

        return new_event

    stored = collection.get(
        ids=[event_id],
        include=["metadatas"]
    )

    if not stored.get("metadatas"):
        raise ValueError(
            "Schedule entry not found."
        )

    old = dict(
        stored["metadatas"][0]
    )

    if action == "remove":

        collection.delete(
            ids=[event_id]
        )

        return old

    if action == "update":

        old.update(
            changes or {}
        )

        collection.delete(
            ids=[event_id]
        )

        collection.add(
            ids=[event_id],
            documents=[
                event_text(old)
            ],
            metadatas=[old]
        )

        return old

    raise ValueError(
        "Invalid schedule action."
    )


# ============================================================
# DATE PARSING
# ============================================================

def resolve_date(text):
    text = text.lower()
    current = today()

    if "today" in text:
        return date_string(current)

    if "tomorrow" in text:
        return date_string(
            current + timedelta(days=1)
        )

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
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12
    }

    match = re.search(
        r"\b("
        + "|".join(months.keys())
        + r")\s+(\d{1,2})\b",
        text
    )

    if match:

        month = months[
            match.group(1)
        ]

        year = current.year

        candidate = datetime(
            year,
            month,
            int(match.group(2)),
            tzinfo=TZ
        )

        if candidate.date() < current.date():
            year += 1

        return (
            f"{year:04d}-"
            f"{month:02d}-"
            f"{int(match.group(2)):02d}"
        )

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6
    }

    for name, number in weekdays.items():

        if name not in text:
            continue

        delta = (
            number - current.weekday()
        ) % 7

        if "next " + name in text:
            delta = delta or 7

        return date_string(
            current + timedelta(days=delta)
        )

    return None


# ============================================================
# TIME PARSING
# ============================================================

def resolve_times(text):

    results = []

    pattern = (
        r"\b(\d{1,2})"
        r"(?::(\d{2}))?"
        r"\s*(am|pm)\b"
    )

    for match in re.finditer(
        pattern,
        text.lower()
    ):

        hour = int(match.group(1))
        minute = int(
            match.group(2) or 0
        )

        period = match.group(3)

        if period == "pm" and hour != 12:
            hour += 12

        if period == "am" and hour == 12:
            hour = 0

        results.append(
            f"{hour:02d}:{minute:02d}"
        )

    return results


# ============================================================
# FIND EVENT
# ============================================================

def find_event(
    text,
    old_time=None,
    event_type=None
):
    items = collection.get(
        include=["metadatas"]
    ).get("metadatas", [])

    lower = text.lower()
    date = resolve_date(text)

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

    # Exact title match
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

    date = resolve_date(text)
    times = resolve_times(text)

    # --------------------------------------------------------
    # COLLEGE ENDING
    # --------------------------------------------------------

    if (
        "college" in text
        and any(
            word in text
            for word in [
                "end",
                "ending",
                "ends",
                "finish",
                "finishing",
                "time",
                "timing"
            ]
        )
    ):

        requested = (
            date
            or date_string(today())
        )

        d = datetime.strptime(
            requested,
            "%Y-%m-%d"
        )

        if d.weekday() <= 2:

            return (
                "college",
                {
                    "date": requested,
                    "day": d.strftime("%A"),
                    "start": "9:20 AM",
                    "end": "4:00 PM"
                }
            )

        if d.weekday() <= 5:

            return (
                "college",
                {
                    "date": requested,
                    "day": d.strftime("%A"),
                    "start": "9:20 AM",
                    "end": "3:00 PM"
                }
            )

        return (
            "college",
            {
                "date": requested,
                "day": d.strftime("%A"),
                "start": None,
                "end": "No college"
            }
        )

    # --------------------------------------------------------
    # THIS WEEK
    # --------------------------------------------------------

    if (
        "this week" in text
        or "weekly schedule" in text
    ):

        current = today()

        monday = (
            current
            - timedelta(
                days=current.weekday()
            )
        )

        result = []

        for i in range(6):

            d = monday + timedelta(days=i)

            if d.weekday() <= 2:
                ending = "4:00 PM"
            else:
                ending = "3:00 PM"

            result.append({
                "day": d.strftime("%A"),
                "date": date_string(d),
                "start": "9:20 AM",
                "end": ending
            })

        return (
            "week",
            result
        )

    # --------------------------------------------------------
    # MOVIE
    # --------------------------------------------------------

    if (
        "movie" in text
        or "film" in text
    ):

        return (
            "get_schedule",
            get_schedule(
                message,
                date=date,
                event_type="movie"
            )
        )

    # --------------------------------------------------------
    # OTHER TYPES
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MOVE / RESCHEDULE
    # --------------------------------------------------------

    if any(
        word in text
        for word in [
            "move ",
            "reschedule ",
            "change "
        ]
    ):

        old_time = (
            times[0]
            if len(times) > 1
            else None
        )

        new_time = (
            times[-1]
            if times
            else None
        )

        target = find_event(
            text,
            old_time
        )

        if target and new_time:

            changes = {
                "time": new_time
            }

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

    # --------------------------------------------------------
    # DELETE
    # --------------------------------------------------------

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
            times[0] if times else None
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
            "update_schedule",
            None
        )

    # --------------------------------------------------------
    # ADD EVENT
    # --------------------------------------------------------

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

            match = re.search(
                r"(?:add|create|schedule|book)"
                r"\s+(?:a\s+)?(.+?)"
                r"\s+(?:on|for)\s+",
                message,
                re.I
            )

            title = (
                match.group(1).strip()
                if match
                else "New Event"
            )

            if "movie" in text:
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

    # --------------------------------------------------------
    # FREE TIME
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GENERAL RETRIEVAL
    # --------------------------------------------------------

    return (
        "get_schedule",
        get_schedule(
            message,
            date=date
        )
    )


# ============================================================
# NATURAL ANSWERS
# ============================================================

def make_answer(
    question,
    tool,
    result
):

    text = question.lower()

    if tool == "college":

        if result["end"] == "No college":

            return (
                f"{result['day']} "
                f"({result['date']}): "
                "No college is scheduled."
            )

        return (
            f"Your college is from "
            f"{result['start']} until "
            f"{result['end']} on "
            f"{result['day']} "
            f"({result['date']})."
        )

    if tool == "week":

        return "\n".join(
            f"• {item['day']} — "
            f"{item['date']}: "
            f"{item['start']} – "
            f"{item['end']}"
            for item in result
        )

    if isinstance(result, dict):

        return (
            f"Done — {result['title']} "
            f"is scheduled for "
            f"{result['date']} at "
            f"{result['time']}."
        )

    if not result:

        if "workshop" in text:
            return (
                "You don't have any "
                "workshops scheduled."
            )

        if "meeting" in text:
            return (
                "You don't have any "
                "meetings scheduled."
            )

        if "appointment" in text:
            return (
                "You don't have any "
                "appointments scheduled."
            )

        if "task" in text:
            return (
                "You don't have any "
                "tasks scheduled."
            )

        if "movie" in text:
            return (
                "You don't have any "
                "movies scheduled."
            )

        if "free" in text:
            return (
                "You are free during "
                "that requested period."
            )

        if "today" in text:
            return (
                "You have nothing "
                "scheduled today."
            )

        return (
            "No matching schedule "
            "entries found."
        )

    if len(result) == 1:

        event = result[0]

        return (
            f"{event['title']} is scheduled "
            f"for {event['date']} at "
            f"{event['time']}."
        )

    return "\n".join(
        f"• {event['title']} — "
        f"{event['date']} at "
        f"{event['time']}"
        for event in result
    )


# ============================================================
# API MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str


# ============================================================
# UI
# ============================================================

PAGE = r"""
<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>ScheduleAI</title>

<style>

*{
box-sizing:border-box
}

body{
margin:0;
font-family:Inter,Arial,sans-serif;
background:
linear-gradient(135deg,#f7f8ff,#eef2ff);
color:#172033
}

.layout{
display:flex;
min-height:100vh
}

.sidebar{
width:250px;
background:#11162a;
color:white;
padding:28px 18px
}

.logo{
font-size:25px;
font-weight:800;
margin-bottom:40px
}

.logo span{
color:#8b7cff
}

.nav{
padding:13px 15px;
margin:7px 0;
border-radius:12px;
color:#aeb7cf
}

.nav.active{
background:#292451;
color:white
}

.side-card{
margin-top:35px;
padding:18px;
border:1px solid #303750;
border-radius:16px;
background:#171d34;
color:#aeb7cf;
font-size:12px;
line-height:1.8
}

.main{
flex:1;
max-width:1100px;
margin:auto;
padding:38px
}

.login{
background:white;
border:1px solid #e1e5ef;
border-radius:18px;
padding:14px 18px;
display:flex;
align-items:center;
justify-content:space-between;
gap:20px;
margin-bottom:25px;
box-shadow:0 10px 30px #252b5109
}

.login small{
display:block;
color:#8791a5;
margin-top:4px
}

.google{
background:white;
border:1px solid #d9deea;
border-radius:10px;
padding:10px 16px;
font-weight:700;
cursor:pointer
}

.header{
display:flex;
justify-content:space-between;
align-items:flex-start
}

h1{
font-size:38px;
margin:0 0 8px
}

.subtitle{
color:#77829b
}

.status{
background:#e6f8ee;
color:#25824b;
padding:9px 13px;
border-radius:30px;
font-size:12px;
font-weight:700
}

.quick{
display:flex;
gap:10px;
flex-wrap:wrap;
margin:22px 0
}

.quick button{
background:white;
border:1px solid #dfe4ee;
border-radius:12px;
padding:11px 15px;
cursor:pointer;
transition:.2s
}

.quick button:hover{
transform:translateY(-2px);
box-shadow:0 8px 20px #26345c12
}

.chat{
background:white;
border:1px solid #e1e6ef;
border-radius:22px;
min-height:430px;
padding:24px;
box-shadow:0 15px 45px #252b5112
}

.message{
max-width:78%;
padding:15px 18px;
border-radius:16px;
margin:12px 0;
white-space:pre-wrap;
line-height:1.6
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
padding:17px;
border:1px solid #dce2ec;
border-radius:14px;
font-size:15px;
outline:none
}

.composer input:focus{
border-color:#8b7cff;
box-shadow:0 0 0 4px #8b7cff14
}

.send{
border:0;
border-radius:14px;
background:#6757e8;
color:white;
padding:0 25px;
font-weight:800;
cursor:pointer
}

.footer{
text-align:center;
color:#8a94a8;
font-size:12px;
margin-top:12px
}

@media(max-width:720px){

.sidebar{
display:none
}

.main{
padding:22px 14px
}

h1{
font-size:29px
}

.status{
display:none
}

.message{
max-width:95%
}

.login{
flex-direction:column;
align-items:flex-start
}

.google{
width:100%
}

.send{
padding:0 18px
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

<div class="side-card">

<b>Ask naturally</b>

<br><br>

When is my movie scheduled?

<br>

What do I have today?

<br>

Am I free Friday afternoon?

<br>

When does college end tomorrow?

<br>

Add a movie Friday at 7:20 PM.

<br>

Move my movie to 8 PM.

</div>

</aside>

<main class="main">

<div class="login">

<div>

<strong>
Welcome to ScheduleAI
</strong>

<small>
Your personal 30-day schedule assistant
</small>

</div>

<button
class="google"
onclick="googleLogin()"
>
G&nbsp; Sign in with Google
</button>

</div>

<div class="header">

<div>

<h1>
Your schedule, understood.
</h1>

<div class="subtitle">

Ask a question and let the agent find
or update the right event.

</div>

</div>

<div class="status">
● RAG ONLINE
</div>

</div>

<div class="quick">

<button
onclick="ask('What do I have today?')"
>
📅 Today
</button>

<button
onclick="ask('What do I have tomorrow?')"
>
Tomorrow
</button>

<button
onclick="ask('When does college end today?')"
>
🎓 College
</button>

<button
onclick="ask('When is my movie scheduled?')"
>
🎬 Movie
</button>

<button
onclick="ask('What is my college schedule this week?')"
>
This Week
</button>

</div>

<div
id="chat"
class="chat"
>

<div class="message bot">

Hi! I'm your ScheduleAI assistant.

Ask me about your college, movie,
free time, or any event you want to
add, move, or remove.

</div>

</div>

<div class="composer">

<input
id="question"
placeholder="Ask: When is my movie scheduled?"
onkeydown="if(event.key==='Enter')send()"
>

<button
class="send"
onclick="send()"
>
Send
</button>

</div>

<div class="footer">

Dynamic dates • ChromaDB RAG •
get_schedule • update_schedule

</div>

</main>

</div>

<script>

function addMessage(text,type){

const box=document.createElement("div");

box.className="message "+type;

box.textContent=text;

document
.getElementById("chat")
.appendChild(box);

box.scrollIntoView({
behavior:"smooth"
});

}

function ask(text){

document
.getElementById("question")
.value=text;

send();

}

function googleLogin(){

alert(
"Google sign-in needs your Firebase/Google OAuth configuration."
);

}

async function send(){

const input=
document.getElementById("question");

const value=input.value.trim();

if(!value)return;

input.value="";

addMessage(
value,
"user"
);

addMessage(
"Thinking...",
"bot"
);

const messages=
document.querySelectorAll(".message");

const bot=
messages[messages.length-1];

try{

const response=
await fetch(
"/chat",
{
method:"POST",
headers:{
"Content-Type":
"application/json"
},
body:JSON.stringify({
message:value
})
}
);

const data=
await response.json();

bot.textContent=
data.answer;

}

catch(error){

bot.textContent=
"Unable to reach the assistant.";

}

}

</script>

</body>

</html>
"""


# ============================================================
# ROUTES
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():
    return PAGE


@app.post("/chat")
def chat(request: ChatRequest):

    try:

        tool, result = agent(
            request.message
        )

        return {
            "answer": make_answer(
                request.message,
                tool,
                result
            ),
            "tool": tool,
            "data": result
        }

    except Exception as error:

        return {
            "answer":
            f"Request failed: {error}",
            "error": str(error)
        }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "tools": [
            "get_schedule",
            "update_schedule"
        ],
        "events": collection.count(),
        "today": date_string(today()),
        "timezone": str(TZ)
    }
