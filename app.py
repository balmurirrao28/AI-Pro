import os
import re
import uuid
from datetime import datetime, timedelta

import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Schedule AI")

# 30-day schedule
START = datetime(2026, 8, 12)
END = START + timedelta(days=29)

SEED = []
current = START
while current <= END:
    if current.weekday() <= 2:
        end_time = "16:00"
        duration = 420
    elif current.weekday() <= 4:
        end_time = "15:00"
        duration = 360
    else:
        current += timedelta(days=1)
        continue
    SEED.append((
        "College",
        current.strftime("%Y-%m-%d"),
        "09:00",
        "college",
        duration
    ))
    current += timedelta(days=1)

db = chromadb.PersistentClient(
    path=os.getenv("CHROMA_PATH", "./chroma_db")
)

collection = db.get_or_create_collection("college_schedule")


def event_text(event):
    return (
        f"{event['title']} on {event['date']} at {event['time']}. "
        f"{event['type']} for {event['duration']} minutes."
    )


def seed_database():
    if collection.count() > 0:
        return

    for title, date, time, kind, duration in SEED:
        event = {
            "id": str(uuid.uuid4()),
            "title": title,
            "date": date,
            "time": time,
            "type": kind,
            "duration": duration,
        }

        collection.add(
            ids=[event["id"]],
            documents=[event_text(event)],
            metadatas=[event],
        )


seed_database()


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
    Retrieves relevant schedule information
    using ChromaDB semantic search.
    """

    count = collection.count()
    results_count = min(max(count, 1), 20)

    results = collection.query(
        query_texts=[query or date or "schedule"],
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

    if "today" in text:
        return START.strftime("%Y-%m-%d")

    if "tomorrow" in text:
        return (
            START + timedelta(days=1)
        ).strftime("%Y-%m-%d")

    match = re.search(
        r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b",
        text
    )

    if match:
        return (
            f"{int(match[1]):04d}-"
            f"{int(match[2]):02d}-"
            f"{int(match[3]):02d}"
        )

    match = re.search(
        r"\b(august|september)\s+(\d{1,2})\b",
        text
    )

    if match:

        month = (
            8
            if match[1].lower() == "august"
            else 9
        )

        return (
            f"2026-{month:02d}-"
            f"{int(match[2]):02d}"
        )

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    for day, number in weekdays.items():

        if day in text:

            difference = (
                number - START.weekday()
            ) % 7

            if (
                "next " + day in text
                and difference == 0
            ):
                difference = 7

            return (
                START + timedelta(days=difference)
            ).strftime("%Y-%m-%d")

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


def find_event(text, old_time=None):

    lower = text.lower()

    for event in all_events():

        if event["title"].lower() in lower:
            return event

    date = get_date(text)

    candidates = [
        event
        for event in all_events()
        if not date
        or event["date"] == date
    ]

    if old_time:

        candidates = [
            event
            for event in candidates
            if event["time"] == old_time
        ]

    if len(candidates) == 1:
        return candidates[0]

    return None


# ============================================================
# AGENT
# ============================================================

def agent(message):

    text = message.lower()

    date = get_date(text)
    times = get_times(text)

    mutation_words = [
        "add ",
        "create ",
        "schedule ",
        "book ",
        "move ",
        "reschedule ",
        "change ",
        "update ",
        "remove ",
        "delete ",
        "cancel ",
    ]

    is_mutation = any(
        word in text
        for word in mutation_words
    )

    # MOVE / RESCHEDULE
    if (
        is_mutation
        and any(
            word in text
            for word in [
                "move ",
                "reschedule ",
                "change "
            ]
        )
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

            result = update_schedule(
                "update",
                event_id=target["id"],
                changes=changes
            )

            return (
                "update_schedule",
                result
            )

    # REMOVE
    if (
        is_mutation
        and any(
            word in text
            for word in [
                "remove ",
                "delete ",
                "cancel "
            ]
        )
    ):

        target = find_event(
            text,
            times[0] if times else None
        )

        if target:

            result = update_schedule(
                "remove",
                event_id=target["id"]
            )

            return (
                "update_schedule",
                result
            )

    # ADD
    if (
        is_mutation
        and any(
            word in text
            for word in [
                "add ",
                "create ",
                "schedule ",
                "book "
            ]
        )
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
                else "New Meeting"
            )

            if title.lower() == "a meeting":
                title = "New Meeting"

            event = {
                "title": title,
                "date": date,
                "time": times[0],
                "type": "meeting",
                "duration": 60,
            }

            result = update_schedule(
                "add",
                event=event
            )

            return (
                "update_schedule",
                result
            )

    # COLLEGE ENDING / COLLEGE SCHEDULE
    if any(word in text for word in [
        "college", "college ending", "college ends",
        "college ending time", "when does college end",
        "when is my college ending"
    ]):
        requested_date = date or START.strftime("%Y-%m-%d")
        day = datetime.strptime(requested_date, "%Y-%m-%d")
        if day.weekday() <= 2:
            ending = "4:00 PM"
        elif day.weekday() <= 4:
            ending = "3:00 PM"
        else:
            ending = "No college scheduled"
        return (
            "college_schedule",
            {
                "date": requested_date,
                "day": day.strftime("%A"),
                "ending": ending
            }
        )

    # RETRIEVAL
    if "afternoon" in text:

        return (
            "get_schedule",
            get_schedule(
                message,
                date,
                "12:00",
                "17:00"
            )
        )

    if "morning" in text:

        return (
            "get_schedule",
            get_schedule(
                message,
                date,
                "08:00",
                "12:00"
            )
        )

    return (
        "get_schedule",
        get_schedule(
            message,
            date
        )
    )


def make_answer(
    question,
    tool,
    result
):

    if tool == "college_schedule":
        if result["ending"] == "No college scheduled":
            return (
                f"{result['day']} ({result['date']}): "
                "No college is scheduled."
            )
        return (
            f"Your college ends at {result['ending']} "
            f"on {result['day']} ({result['date']})."
        )

    if isinstance(result, dict):

        return (
            f"Done — {result['title']} "
            f"is scheduled for "
            f"{result['date']} at "
            f"{result['time']}."
        )

    if not result:

        if "free" in question.lower():

            return (
                "You are free during "
                "that requested period."
            )

        return (
            "No matching schedule "
            "entries found."
        )

    return "\n".join(
        "• " + event_text(event)
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

When is my college ending today?

<br>

When does college end on Friday?

<br>

When does college end tomorrow?

<br>

What is my college schedule this week?

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

<button onclick="ask('What do I have scheduled tomorrow?')">
College Today
</button>

<button onclick="ask('When does college end on Friday?')">
Friday afternoon
</button>

<button onclick="ask('Show my workshops')">
Workshops
</button>

<button onclick="ask('Show my meetings')">
Meetings
</button>

</div>

<div id="chat" class="chat">

<div class="message bot">

Hi! I’m your AI College Schedule Assistant.\n\nI can tell you when college ends, check your schedule, and manage events.

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
