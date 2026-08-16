import os,re,uuid,hashlib,math
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app=FastAPI(title="ScheduleAI",version="3.1")
TZ=ZoneInfo(os.getenv("TIMEZONE","Asia/Kolkata")); db=chromadb.PersistentClient(path="./chroma_db")
col=db.get_or_create_collection("schedule_assistant")
MONTHS={m.lower():i for i,m in enumerate("January February March April May June July August September October November December".split(),1)}
DAYS={d.lower():i for i,d in enumerate("Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split())}

def today(): return datetime.now(TZ).replace(hour=0,minute=0,second=0,microsecond=0)
def ds(d): return d.strftime("%Y-%m-%d")
def pretty(t):
 h,m=map(int,t.split(":"));return f"{h%12 or 12}:{m:02d} {'AM' if h<12 else 'PM'}"
def mins(t): h,m=map(int,t.split(":"));return h*60+m
def vec(s):
 v=[0.0]*96
 for w in re.findall(r"[a-z0-9]+",s.lower()):
  b=hashlib.sha256(w.encode()).digest()
  for i in range(4):v[int.from_bytes(b[i*2:i*2+2],"big")%96]+=1 if b[i+8]&1 else -1
 n=math.sqrt(sum(x*x for x in v)) or 1;return [x/n for x in v]
def doc(e):return f"{e['title']} {e['type']} on {e['date']} at {e['time']} for {e['duration']} minutes"
def all_events():return col.get(include=["metadatas"]).get("metadatas",[])
def save(e):
 e=dict(e);e.setdefault("id",str(uuid.uuid4()));e.setdefault("type","meeting");e.setdefault("duration",60);e.setdefault("source","user")
 col.upsert(ids=[e["id"]],documents=[doc(e)],embeddings=[vec(doc(e))],metadatas=[e]);return e

def parse_date(q):
 q=q.lower();b=today()
 if "today" in q:return ds(b)
 if "tomorrow" in q:return ds(b+timedelta(days=1))
 m=re.search(r"\b(20\d\d)[-/](\d{1,2})[-/](\d{1,2})\b",q)
 if m:
  try:return ds(datetime(int(m[1]),int(m[2]),int(m[3]),tzinfo=TZ))
  except:return None
 m=re.search(r"\b("+"|".join(MONTHS)+r")\s+(\d{1,2})(?:st|nd|rd|th)?\b",q)
 if m:
  try:
   d=datetime(b.year,MONTHS[m[1]],int(m[2]),tzinfo=TZ)
   if d.date()<b.date():d=d.replace(year=d.year+1)
   return ds(d)
  except:return None
 m=re.search(r"\bon\s+(\d{1,2})(?:st|nd|rd|th)?\b",q)
 if m:
  for i in range(30):
   d=b+timedelta(days=i)
   if d.day==int(m[1]):return ds(d)
 for n,w in DAYS.items():
  if re.search(r"\bnext\s+"+n+r"\b",q):return ds(b+timedelta(days=(w-b.weekday())%7 or 7))
  if re.search(r"\b"+n+r"\b",q):return ds(b+timedelta(days=(w-b.weekday())%7))
 return None

def times(q):
 out=[]
 for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b",q.lower()):
  h,mi=int(m[1]),int(m[2] or 0)
  if h>12 or mi>59:continue
  if m[3]=="pm" and h!=12:h+=12
  if m[3]=="am" and h==12:h=0
  out.append(f"{h:02d}:{mi:02d}")
 return out

def get_schedule(query="",date=None,kind=None):
 data=all_events()
 if date:data=[e for e in data if e["date"]==date]
 if kind:data=[e for e in data if e["type"]==kind]
 if date or kind:return sorted(data,key=lambda e:(e["date"],e["time"]))
 if not data:return []
 r=col.query(query_embeddings=[vec(query or "schedule")],n_results=min(8,len(data)),include=["metadatas"])
 return sorted(r.get("metadatas",[[]])[0],key=lambda e:(e["date"],e["time"]))

def update_schedule(action,event_id=None,event=None,changes=None):
 if action=="add":return save(event or {})
 found=col.get(ids=[event_id],include=["metadatas"]).get("metadatas",[])
 if not found:raise ValueError("Schedule entry not found")
 old=dict(found[0])
 if action=="remove":col.delete(ids=[event_id]);return old
 old.update(changes or {});return save(old)

def seed():
 if col.count():return
 b=today();samples=[("Team Meeting",1,"14:00","meeting",60),("Project Task",2,"17:30","task",60),("AI Workshop",3,"11:00","workshop",90),("Doctor Appointment",5,"10:30","appointment",45)]
 for title,off,t,k,d in samples:save({"title":title,"date":ds(b+timedelta(days=off)),"time":t,"type":k,"duration":d,"source":"sample"})
 for i in range(30):
  d=b+timedelta(days=i)
  if d.weekday()<6:save({"title":"College","date":ds(d),"time":"09:20","type":"college","duration":400,"source":"system"})
seed()

def find_event(q,old=None):
 data=all_events();date=parse_date(q);low=q.lower()
 if date:data=[e for e in data if e["date"]==date]
 if old:data=[e for e in data if e["time"]==old]
 scored=[]
 for e in data:
  s=(10 if e["title"].lower() in low else 0)+(5 if e["type"] in low else 0)+(3 if e["time"] in low else 0);scored.append((s,e))
 scored.sort(key=lambda x:(-x[0],x[1]["date"],x[1]["time"]))
 return scored[0][1] if scored and scored[0][0] else (data[0] if len(data)==1 else None)

def free_windows(date,start="12:00",end="17:00"):
 busy=[]
 for e in get_schedule(date=date):
  a=mins(e["time"]);z=a+int(e["duration"])
  if z>mins(start) and a<mins(end):busy.append((max(a,mins(start)),min(z,mins(end)),e))
 busy.sort();open_= [];cur=mins(start)
 for a,z,e in busy:
  if a>cur:open_.append((cur,a))
  cur=max(cur,z)
 if cur<mins(end):open_.append((cur,mins(end)))
 return busy,open_
def clock(x):return pretty(f"{x//60:02d}:{x%60:02d}")

def agent(q):
 low=q.lower().strip();date=parse_date(low);ts=times(low)
 # Natural-language date request must never fall through to vector search.
 if re.search(r"\b(today'?s date|date today|what date is it|what is the date today)\b",low):
  return "direct",today().strftime("Today is %A, %B %-d, %Y.")
 if low in {"hi","hello","hey"}:return "direct","Hi! I’m ScheduleAI. I can find, add, move, or remove events for your next 30 days."
 if "free" in low:
  d=date or ds(today());start,end=("12:00","17:00") if "afternoon" in low else ("09:00","21:00");busy,open_=free_windows(d,start,end)
  if not busy:return "direct",f"You’re free from {pretty(start)} to {pretty(end)} on {d}."
  b="\n".join(f"• {e['title']} — {pretty(e['time'])}–{clock(z)}" for a,z,e in busy);f=", ".join(f"{clock(a)}–{clock(z)}" for a,z in open_) or "none"
  return "direct",f"Busy on {d}:\n{b}\n\nFree windows: {f}"
 if any(x in low for x in ("move ","reschedule ","change ")):
  if len(ts)<2:return "direct","Please give me the old and new time, e.g. “Move my meeting from 2 PM to 4 PM.”"
  e=find_event(low,ts[0])
  if not e:return "direct","I couldn’t find that event. Include its name or date and try again."
  changes={"time":ts[-1]};
  if date:changes["date"]=date
  e=update_schedule("update",e["id"],changes=changes);return "direct",f"Done. {e['title']} is now at {pretty(e['time'])} on {e['date']}."
 if any(x in low for x in ("remove ","delete ","cancel ")):
  e=find_event(low,ts[0] if ts else None)
  if not e:return "direct","Tell me which event to remove, including its name or date."
  e=update_schedule("remove",e["id"]);return "direct",f"Removed {e['title']} from {e['date']} at {pretty(e['time'])}."
 if re.match(r"^(add|create|schedule|book)\b",low):
  if not date or not ts:return "direct","Please include both the date and time, e.g. “Add a meeting tomorrow at 3 PM.”"
  if not today().date()<=datetime.strptime(date,"%Y-%m-%d").date()<=today().date()+timedelta(days=29):return "direct","Please choose a date within the next 30 days."
  title=re.sub(r"^(add|create|schedule|book)\s+","",q,flags=re.I);title=re.sub(r"\s+(on|for)\s+[^@]+(?=\s+at\s+\d)","",title,flags=re.I);title=re.sub(r"\s+at\s+\d{1,2}(?::\d{2})?\s*(am|pm)","",title,flags=re.I).strip(" ,.-") or "New Event"
  e=update_schedule("add",event={"title":title,"date":date,"time":ts[0],"type":"meeting","duration":60,"source":"user"});return "direct",f"Added {e['title']} for {e['date']} at {pretty(e['time'])}."
 kind=next((k for k in ("meeting","workshop","appointment","task") if re.search(r"\b"+k+r"s?\b",low)),None)
 data=get_schedule(q,date=date,kind=kind)
 if ts:data=[e for e in data if mins(e["time"])==mins(ts[0])]
 return "get_schedule",data

def answer(tool,result):
 if tool=="direct":return result
 if not result:return "Nothing matching that request is scheduled."
 return "\n".join(f"• {e['title']} — {e['date']} at {pretty(e['time'])} ({e['duration']} min)" for e in result)

class Chat(BaseModel):message:str
PAGE=r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScheduleAI</title><style>
:root{--bg:#f5f7fc;--ink:#172033;--muted:#778199;--line:#e4e8f0;--a:#635bdf;--dark:#11172a;--soft:#f0f1f8}*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Arial;background:var(--bg);color:var(--ink)}.app{display:flex;min-height:100vh}.side{width:252px;background:linear-gradient(180deg,#11172a,#171d35);color:#fff;padding:28px 18px}.brand{font-size:25px;font-weight:850;margin:2px 8px 34px}.brand span{color:#9a91ff}.nav{padding:12px 14px;color:#aeb7cf;border-radius:12px;margin:6px 0;font-size:14px}.active{background:#2b264d;color:#fff}.tip{margin-top:28px;padding:16px;border:1px solid #303750;border-radius:16px;color:#c4cada;font-size:12px;line-height:1.8}.main{width:min(1120px,100%);margin:auto;padding:34px}.head{display:flex;justify-content:space-between;gap:18px}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.12em;color:var(--a);text-transform:uppercase;margin-bottom:9px}.head h1{font-size:40px;margin:0 0 9px;letter-spacing:-1px}.sub{color:var(--muted)}.status{background:#e9f8f0;color:#1f8b55;padding:9px 13px;border-radius:30px;font-size:11px;font-weight:850}.quick{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin:25px 0 16px}.quick button{border:1px solid var(--line);background:#fff;border-radius:15px;padding:14px;text-align:left;cursor:pointer;box-shadow:0 5px 20px #17203a08}.qtitle{font-weight:800;font-size:13px}.qdesc{display:block;color:var(--muted);font-size:11px;margin-top:4px}.chat{background:#fff;border:1px solid var(--line);border-radius:24px;min-height:450px;padding:22px;box-shadow:0 18px 50px #17203a0d}.welcome{display:flex;gap:13px;background:linear-gradient(135deg,#f3f2ff,#f7f8fc);border:1px solid #e6e4fb;padding:18px;border-radius:18px}.avatar{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--a),#8a7ff0);color:#fff;display:grid;place-items:center;font-weight:900}.welcome strong{display:block;margin-bottom:4px}.welcome span{color:var(--muted);font-size:13px;line-height:1.5}.msg{max-width:84%;padding:14px 17px;border-radius:17px;margin:10px 0;line-height:1.55;white-space:pre-wrap;font-size:14px}.bot{background:var(--soft);border-bottom-left-radius:6px}.user{margin-left:auto;background:linear-gradient(135deg,var(--a),#766de7);color:#fff;border-bottom-right-radius:6px}.events{display:grid;gap:9px}.event{display:flex;align-items:center;gap:12px;padding:12px;background:#fff;border:1px solid #e6e9f0;border-radius:14px}.datebox{width:54px;text-align:center;background:#f1f0ff;border-radius:10px;padding:7px 4px;color:var(--a);font-weight:850;font-size:11px}.eventmain{flex:1}.eventtitle{font-weight:800;font-size:13px}.meta{color:var(--muted);font-size:11px;margin-top:3px}.tag{font-size:10px;padding:5px 8px;border-radius:20px;background:#f0f1f5;color:#626d83;text-transform:capitalize}.composer{display:flex;gap:10px;margin-top:14px;background:#fff;border:1px solid var(--line);padding:7px;border-radius:17px}.composer input{flex:1;border:0;background:transparent;padding:13px;font-size:14px;outline:none}.send{border:0;background:var(--a);color:#fff;border-radius:12px;padding:0 22px;font-weight:850;cursor:pointer}.foot{text-align:center;color:#929bae;font-size:11px;margin-top:11px}@media(max-width:820px){.side{display:none}.main{padding:24px 14px}.head h1{font-size:31px}.quick{grid-template-columns:repeat(2,1fr)}.status{display:none}.msg{max-width:94%}}@media(max-width:460px){.quick{grid-template-columns:1fr}}
</style></head><body><div class="app"><aside class="side"><div class="brand">Schedule<span>AI</span></div><div class="nav active">✦ &nbsp;Assistant</div><div class="nav">▣ &nbsp;30-Day Schedule</div><div class="nav">◷ &nbsp;Agent Tools</div><div class="tip"><b>Try asking</b><br>What do I have tomorrow?<br>Am I free Friday afternoon?<br>Add a meeting tomorrow at 3 PM.<br>Move my meeting from 2 PM to 4 PM.<br>Remove my workshop.</div></aside><main class="main"><div class="head"><div><div class="eyebrow">Agentic Schedule Assistant</div><h1>Your schedule, understood.</h1><div class="sub">Smart retrieval and schedule updates powered by ChromaDB RAG.</div></div><div class="status">● SYSTEM ONLINE</div></div><div class="quick"><button onclick="ask('What do I have today?')"><span class="qtitle">📅 Today</span><span class="qdesc">See today’s events</span></button><button onclick="ask('What do I have tomorrow?')"><span class="qtitle">Tomorrow</span><span class="qdesc">Plan the next day</span></button><button onclick="ask('Am I free Friday afternoon?')"><span class="qtitle">Free time</span><span class="qdesc">Check Friday afternoon</span></button><button onclick="ask('Show my meetings')"><span class="qtitle">Meetings</span><span class="qdesc">Find scheduled meetings</span></button></div><div id="chat" class="chat"><div class="welcome"><div class="avatar">AI</div><div><strong>Hi! I’m ScheduleAI.</strong><span>I can retrieve your schedule with RAG or update it when you ask. Choose a quick action or type a request below.</span></div></div></div><div class="composer"><input id="q" placeholder="Ask: What do I have tomorrow?" autocomplete="off"><button class="send" onclick="send()">Send</button></div><div class="foot">FastAPI • ChromaDB • Agentic RAG • get_schedule • update_schedule</div></main></div><script>
function ask(t){document.getElementById('q').value=t;send()}function add(t,c){let x=document.createElement('div');x.className='msg '+c;x.textContent=t;document.getElementById('chat').appendChild(x);x.scrollIntoView({behavior:'smooth'});return x}function esc(s){return String(s).replace(/[&<>\"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]))}function t12(t){let p=t.split(':'),h=+p[0];return (h%12||12)+':'+p[1]+' '+(h<12?'AM':'PM')}function addEvents(a){let w=document.createElement('div');w.className='msg bot';let g=document.createElement('div');g.className='events';a.forEach(e=>{let r=document.createElement('div');r.className='event';let d=new Date(e.date+'T00:00:00');r.innerHTML='<div class="datebox">'+d.toLocaleDateString('en-US',{month:'short'})+'<br>'+d.getDate()+'</div><div class="eventmain"><div class="eventtitle">'+esc(e.title)+'</div><div class="meta">'+esc(e.date)+' • '+t12(e.time)+' • '+e.duration+' min</div></div><div class="tag">'+esc(e.type)+'</div>';g.appendChild(r)});w.appendChild(g);document.getElementById('chat').appendChild(w);w.scrollIntoView({behavior:'smooth'})}async function send(){let i=document.getElementById('q'),v=i.value.trim();if(!v)return;i.value='';add(v,'user');let b=add('Thinking…','bot');try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})}),d=await r.json();b.remove();Array.isArray(d.data)&&d.data.length?addEvents(d.data):add(d.answer||'No answer returned.','bot')}catch(e){b.remove();add('Unable to reach the assistant. Please try again.','bot')}}document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')send()});
</script></body></html>'''

@app.get("/",response_class=HTMLResponse)
def home():return PAGE
@app.post("/chat")
def chat(c:Chat):
 try:
  tool,result=agent(c.message);return {"answer":answer(tool,result),"tool":tool,"data":result}
 except Exception as e:return {"answer":"I couldn’t complete that request. Please try again.","error":str(e)}
@app.get("/health")
def health():return {"status":"ok","events":col.count(),"tools":["get_schedule","update_schedule"],"rag":"ChromaDB"}
