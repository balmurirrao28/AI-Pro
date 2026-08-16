import os,re,uuid,hashlib,math,secrets
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import chromadb,httpx
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from authlib.integrations.starlette_client import OAuth

app=FastAPI(title="ScheduleAI",version="5.0")
app.add_middleware(SessionMiddleware,secret_key=os.getenv("SESSION_SECRET",secrets.token_hex(32)),https_only=False,same_site="lax")
TZ=ZoneInfo(os.getenv("TIMEZONE","Asia/Kolkata"));db=chromadb.PersistentClient(path=os.getenv("CHROMA_PATH","./chroma_db"));col=db.get_or_create_collection("schedule_assistant")
GOOGLE_ID=os.getenv("GOOGLE_CLIENT_ID","");GOOGLE_SECRET=os.getenv("GOOGLE_CLIENT_SECRET","");BASE=os.getenv("APP_URL","").rstrip("/");oauth=OAuth()
if GOOGLE_ID and GOOGLE_SECRET: oauth.register(name="google",client_id=GOOGLE_ID,client_secret=GOOGLE_SECRET,server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",client_kwargs={"scope":"openid email profile https://www.googleapis.com/auth/calendar"})
MONTHS={m.lower():i for i,m in enumerate("January February March April May June July August September October November December".split(),1)}
DAYS={d.lower():i for i,d in enumerate("Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split())}
KINDS=("meeting","workshop","task","appointment")
def today():return datetime.now(TZ).replace(hour=0,minute=0,second=0,microsecond=0)
def ds(d):return d.strftime("%Y-%m-%d")
def pretty(t):
 h,m=map(int,t.split(":"));return f"{h%12 or 12}:{m:02d} {'AM' if h<12 else 'PM'}"
def mins(t):h,m=map(int,t.split(":"));return h*60+m
def now_text():return datetime.now(TZ).strftime("It is %I:%M %p on %A, %B %d, %Y (%Z).")
def vec(s):
 v=[0.0]*96
 for w in re.findall(r"[a-z0-9]+",s.lower()):
  b=hashlib.sha256(w.encode()).digest()
  for i in range(4):v[int.from_bytes(b[i*2:i*2+2],"big")%96]+=1 if b[i+8]&1 else -1
 n=math.sqrt(sum(x*x for x in v)) or 1;return [x/n for x in v]
def doc(e):return f"{e['title']} {e['type']} on {e['date']} at {e['time']} for {e['duration']} minutes"
def all_events():return col.get(include=["metadatas"]).get("metadatas",[])
def save(e):
 e=dict(e);e.setdefault("id",str(uuid.uuid4()));e.setdefault("duration",60);e.setdefault("source","user")
 col.upsert(ids=[e["id"]],documents=[doc(e)],embeddings=[vec(doc(e))],metadatas=[e]);return e
def cleanup_system():
 r=col.get(include=["metadatas"]);ids=[i for i,m in zip(r.get("ids",[]),r.get("metadatas",[])) if m.get("source")=="system"]
 if ids:col.delete(ids=ids)
def seed():
 if col.count():return
 b=today()
 for title,off,t,k,d in [("Team Meeting",1,"14:00","meeting",60),("Project Task",2,"17:30","task",60),("AI Workshop",3,"11:00","workshop",90),("Doctor Appointment",5,"10:30","appointment",45)]:save({"title":title,"date":ds(b+timedelta(days=off)),"time":t,"type":k,"duration":d,"source":"sample"})
cleanup_system();seed()
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
   d=datetime(b.year,MONTHS[m[1]],int(m[2]),tzinfo=TZ);return ds(d.replace(year=d.year+1) if d.date()<b.date() else d)
  except:return None
 for n,w in DAYS.items():
  if re.search(r"\bnext\s+"+n+r"\b",q):return ds(b+timedelta(days=(w-b.weekday())%7 or 7))
  if re.search(r"\b"+n+r"\b",q):return ds(b+timedelta(days=(w-b.weekday())%7))
 return None
def times(q):
 out=[]
 for m in re.finditer(r"\b(\d{1,2})(?:[:\s](\d{2}))?\s*(am|pm)\b",q.lower()):
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
def find_event(q,old=None):
 data=all_events();date=parse_date(q);low=q.lower()
 if date:data=[e for e in data if e["date"]==date]
 if old:data=[e for e in data if e["time"]==old]
 scored=[]
 for e in data:
  score=(5 if e["type"] in low else 0)+sum(2 for w in re.findall(r"[a-z0-9]+",e["title"].lower()) if w in low)
  scored.append((score,e))
 scored.sort(key=lambda x:(-x[0],x[1]["date"],x[1]["time"]))
 return scored[0][1] if scored and scored[0][0]>0 else (data[0] if len(data)==1 else None)
def free_windows(date,start="09:00",end="21:00"):
 busy=[]
 for e in get_schedule(date=date):
  a=mins(e["time"]);z=a+int(e["duration"])
  if z>mins(start) and a<mins(end):busy.append((max(a,mins(start)),min(z,mins(end)),e))
 busy.sort();free=[];cur=mins(start)
 for a,z,_ in busy:
  if a>cur:free.append((cur,a))
  cur=max(cur,z)
 if cur<mins(end):free.append((cur,mins(end)))
 return busy,free
def clock(x):return pretty(f"{x//60:02d}:{x%60:02d}")

def pending(request,q):
 p=request.session.get("pending_add")
 if not p:return None
 low=q.lower().strip()
 if p["step"]=="type":
  k=next((x for x in KINDS if re.search(r"\b"+x+r"s?\b",low)),None)
  if not k:return "Please choose a type: meeting, workshop, task, or appointment."
  p.update(type=k,step="title");request.session["pending_add"]=p;return f"Got it — a {k}. What should I call it?"
 if p["step"]=="title":
  p.update(title=q.strip(),step="date");request.session["pending_add"]=p;return "What date should I schedule it?"
 if p["step"]=="date":
  d=parse_date(low)
  if not d:return "I couldn't understand that date. Try tomorrow, Friday, or August 20."
  p.update(date=d,step="time");request.session["pending_add"]=p;return "What time should it start? For example: 3 PM or 7:20 PM."
 if p["step"]=="time":
  ts=times(low)
  if not ts:return "Please give a time such as 3 PM or 7:20 PM."
  p.update(time=ts[0],step="duration");request.session["pending_add"]=p;return "How long should it last? For example: 30 minutes or 1 hour."
 if p["step"]=="duration":
  m=re.search(r"(\d+)\s*(?:min|mins|minutes)",low);h=re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)",low)
  dur=int(m[1]) if m else int(float(h[1])*60) if h else None
  if not dur:return "Please give a duration such as 30 minutes or 1 hour."
  if not today().date()<=datetime.strptime(p["date"],"%Y-%m-%d").date()<=today().date()+timedelta(days=29):request.session.pop("pending_add",None);return "Please choose a date within the next 30 days."
  e=save({**p,"duration":dur,"source":"user"});request.session.pop("pending_add",None)
  return f"Added {e['title']} ({e['type']}) for {e['date']} at {pretty(e['time'])}, {e['duration']} minutes."
 request.session.pop("pending_add",None);return None

def agent(request,q):
 low=q.lower().strip();date=parse_date(low);ts=times(low)
 p=pending(request,q)
 if p is not None:return "direct",p
 if re.search(r"\b(time|current time|what time|what's the time|what is the time)\b",low):return "direct",now_text()
 if re.search(r"\b(today'?s date|date today|what date is it|what is the date today)\b",low):return "direct",today().strftime("Today is %A, %B %d, %Y.")
 if low in {"hi","hello","hey"}:return "direct","Hi! I’m ScheduleAI. I can find, add, move, or remove events."
 if re.match(r"^(add|create|schedule|book)\b",low):
  kind=next((k for k in KINDS if re.search(r"\b"+k+r"s?\b",low)),None)
  if not kind:request.session["pending_add"]={"step":"type"};return "direct","Sure. What type of schedule is it: meeting, workshop, task, or appointment?"
  title=re.sub(r"^(add|create|schedule|book)\s+","",q,flags=re.I);title=re.sub(r"\b(meeting|workshop|task|appointment)\b","",title,flags=re.I).strip(" ,.-") or kind.title()
  if not date:request.session["pending_add"]={"step":"date","type":kind,"title":title};return "direct","What date should I schedule it?"
  if not ts:request.session["pending_add"]={"step":"time","type":kind,"title":title,"date":date};return "direct","What time should it start?"
  e=save({"title":title,"type":kind,"date":date,"time":ts[0],"duration":60,"source":"user"});return "direct",f"Added {e['title']} ({e['type']}) for {e['date']} at {pretty(e['time'])}."
 if re.search(r"\b(move|movie|reschedule|change)\b",low):
  if len(ts)<2:return "direct","Tell me the event, current time, and new time. Example: Move my meeting from 2 PM to 4 PM."
  e=find_event(low,ts[0])
  if not e:return "direct","I couldn't find that event. Include its name or date."
  ch={"time":ts[-1]};
  if date:ch["date"]=date
  e=update_schedule("update",e["id"],changes=ch);return "direct",f"Done. {e['title']} is now on {e['date']} at {pretty(e['time'])}."
 if re.search(r"\b(remove|delete|cancel)\b",low):
  e=find_event(low,ts[0] if ts else None)
  if not e:return "direct","Tell me which event to remove, including its name or date."
  e=update_schedule("remove",e["id"]);return "direct",f"Removed {e['title']} from {e['date']} at {pretty(e['time'])}."
 if "free" in low:
  d=date or ds(today());start,end=("12:00","17:00") if "afternoon" in low else ("09:00","21:00");busy,free=free_windows(d,start,end)
  if not busy:return "direct",f"You’re free from {pretty(start)} to {pretty(end)} on {d}."
  b="\n".join(f"• {e['title']} — {pretty(e['time'])}–{clock(z)}" for a,z,e in busy);f=", ".join(f"{clock(a)}–{clock(z)}" for a,z in free) or "none";return "direct",f"Busy on {d}:\n{b}\n\nFree windows: {f}"
 kind=next((k for k in KINDS if re.search(r"\b"+k+r"s?\b",low)),None);data=get_schedule(q,date=date,kind=kind)
 if ts:data=[e for e in data if mins(e["time"])==mins(ts[0])]
 return "get_schedule",data
def answer(tool,result):
 if tool=="direct":return result
 if not result:return "Nothing matching that request is scheduled."
 return "\n".join(f"• {e['title']} — {e['date']} at {pretty(e['time'])} ({e['duration']} min)" for e in result)
async def calendar_events(request):
 token=request.session.get("google_token")
 if not token:return None
 try:
  start=datetime.now(TZ).isoformat();end=(datetime.now(TZ)+timedelta(days=30)).isoformat()
  async with httpx.AsyncClient(timeout=15) as c:r=await c.get("https://www.googleapis.com/calendar/v3/calendars/primary/events",headers={"Authorization":"Bearer "+token},params={"timeMin":start,"timeMax":end,"singleEvents":"true","orderBy":"startTime","maxResults":100})
  return r.json().get("items",[]) if r.status_code==200 else None
 except Exception:return None
@app.get("/auth/google")
async def google_login(request:Request):
 if not GOOGLE_ID or not GOOGLE_SECRET:return HTMLResponse("<h3>Google Calendar is not configured.</h3>",503)
 redirect=BASE+"/auth/google/callback" if BASE else str(request.url_for("google_callback"));return await oauth.google.authorize_redirect(request,redirect)
@app.get("/auth/google/callback",name="google_callback")
async def google_callback(request:Request):
 token=await oauth.google.authorize_access_token(request);request.session["google_token"]=token["access_token"];return RedirectResponse("/")
@app.get("/auth/logout")
async def logout(request:Request):request.session.clear();return RedirectResponse("/")
@app.get("/api/google-calendar")
async def google_calendar(request:Request):
 data=await calendar_events(request);return {"connected":data is not None,"events":data or []}
class Chat(BaseModel):message:str
PAGE='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScheduleAI</title><style>body{margin:0;font-family:Arial,sans-serif;background:#f5f7fc;color:#172033}.app{display:flex;min-height:100vh}.side{width:240px;background:#12182b;color:white;padding:28px 18px}.brand{font-size:25px;font-weight:800;margin-bottom:30px}.brand span{color:#9a91ff}.nav{padding:12px;margin:6px 0;color:#b9c0d3}.active{background:#2b264d;border-radius:12px;color:white}.tip{margin-top:25px;border:1px solid #303750;border-radius:14px;padding:14px;color:#c4cada;font-size:12px;line-height:1.8}.main{width:min(1100px,100%);margin:auto;padding:30px}.top{display:flex;justify-content:space-between;gap:15px}.eyebrow{font-size:12px;color:#635bdf;font-weight:800}.head{font-size:38px;margin:8px 0}.sub{color:#778199}.btn{border:1px solid #e4e8f0;background:white;border-radius:12px;padding:11px 14px;text-decoration:none;color:#172033;font-weight:700;cursor:pointer}.quick{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:22px 0}.quick button{border:1px solid #e4e8f0;background:white;border-radius:14px;padding:14px;text-align:left}.qdesc{display:block;color:#778199;font-size:11px;margin-top:4px}.chat{background:white;border:1px solid #e4e8f0;border-radius:22px;min-height:430px;padding:20px}.msg{max-width:84%;padding:14px 17px;border-radius:17px;margin:10px 0;white-space:pre-wrap;line-height:1.55;font-size:14px}.bot{background:#f0f1f8}.user{margin-left:auto;background:#635bdf;color:white}.composer{display:flex;gap:8px;background:white;border:1px solid #e4e8f0;padding:7px;border-radius:16px;margin-top:12px}.composer input{flex:1;border:0;outline:0;padding:13px;font-size:14px}.send{border:0;background:#635bdf;color:white;border-radius:11px;padding:0 20px;font-weight:800}.events{display:grid;gap:8px}.event{display:flex;gap:12px;align-items:center;border:1px solid #e6e9f0;border-radius:13px;padding:11px}.datebox{width:52px;text-align:center;background:#f1f0ff;border-radius:9px;padding:7px;color:#635bdf;font-weight:800;font-size:11px}.eventmain{flex:1}.eventtitle{font-weight:800;font-size:13px}.meta{color:#778199;font-size:11px;margin-top:3px}.tag{font-size:10px;background:#f0f1f5;border-radius:20px;padding:5px 8px}@media(max-width:800px){.side{display:none}.main{padding:20px 12px}.quick{grid-template-columns:repeat(2,1fr)}.top{flex-direction:column}.head{font-size:30px}}@media(max-width:480px){.quick{grid-template-columns:1fr}}</style></head><body><div class="app"><aside class="side"><div class="brand">Schedule<span>AI</span></div><div class="nav active">✦ &nbsp;Assistant</div><div class="nav">▣ &nbsp;30-Day Schedule</div><div class="nav">◷ &nbsp;Agent Tools</div><div class="tip"><b>Try asking</b><br>Add schedule<br>Add a meeting tomorrow at 3 PM<br>Move my meeting from 2 PM to 4 PM<br>What do I have tomorrow?<br>What is the time?</div></aside><main class="main"><div class="top"><div><div class="eyebrow">AGENTIC SCHEDULE ASSISTANT</div><div class="head">Your schedule, understood.</div><div class="sub">RAG + Google Calendar + voice commands.</div></div><div><button class="btn" onclick="voice()">🎙 Voice</button> <a class="btn" href="/auth/google">Connect Google Calendar</a></div></div><div class="quick"><button onclick="ask('What do I have today?')"><b>📅 Today</b><span class="qdesc">See today’s events</span></button><button onclick="ask('What do I have tomorrow?')"><b>Tomorrow</b><span class="qdesc">Plan the next day</span></button><button onclick="ask('Am I free Friday afternoon?')"><b>Free time</b><span class="qdesc">Check Friday afternoon</span></button><button onclick="ask('Show my meetings')"><b>Meetings</b><span class="qdesc">Find scheduled meetings</span></button></div><div id="chat" class="chat"><div class="msg bot"><b>Hi! I’m ScheduleAI.</b><br>I will not create an event until the required details are collected. Say <b>“add schedule”</b> and I will ask for type, title, date, time, and duration.</div></div><div class="composer"><input id="q" placeholder="Ask: What do I have tomorrow?"><button class="btn" onclick="voice()">🎙</button><button class="send" onclick="send()">Send</button></div></main></div><script>function ask(t){q.value=t;send()}function add(t,c){let x=document.createElement('div');x.className='msg '+c;x.textContent=t;chat.appendChild(x);x.scrollIntoView({behavior:'smooth'});return x}function esc(s){return String(s).replace(/[&<>\"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]))}function t12(t){let[h,m]=t.split(':');h=+h;return(h%12||12)+':'+m+' '+(h<12?'AM':'PM')}function card(a){let x=document.createElement('div');x.className='msg bot';let w=document.createElement('div');w.className='events';a.forEach(e=>{let r=document.createElement('div');r.className='event';let d=new Date(e.date+'T00:00:00');r.innerHTML='<div class="datebox">'+d.toLocaleDateString('en-US',{month:'short'})+'<br>'+d.getDate()+'</div><div class="eventmain"><div class="eventtitle">'+esc(e.title)+'</div><div class="meta">'+e.date+' • '+t12(e.time)+' • '+e.duration+' min</div></div><div class="tag">'+esc(e.type)+'</div>';w.appendChild(r)});x.appendChild(w);chat.appendChild(x);x.scrollIntoView({behavior:'smooth'})}async function send(){let v=q.value.trim();if(!v)return;q.value='';add(v,'user');let b=add('Thinking…','bot');try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})});let d=await r.json();b.remove();if(Array.isArray(d.data)&&d.data.length)card(d.data);else add(d.answer||'No answer returned.','bot')}catch(e){b.remove();add('Unable to reach the assistant.','bot')}}q.addEventListener('keydown',e=>{if(e.key==='Enter')send()});function voice(){const S=window.SpeechRecognition||window.webkitSpeechRecognition;if(!S){add('Voice input is not supported. Try Chrome or Edge.','bot');return}const r=new S();r.lang='en-IN';r.onresult=e=>{q.value=e.results[0][0].transcript;send()};r.start()}</script></body></html>'''
@app.get("/",response_class=HTMLResponse)
def home():return PAGE
@app.post("/chat")
def chat(c:Chat,request:Request):
 try:
  tool,result=agent(request,c.message);return {"answer":answer(tool,result),"tool":tool,"data":result}
 except Exception as e:return {"answer":"I couldn’t complete that request. Please try again.","error":str(e)}
@app.get("/health")
def health():return {"status":"ok","events":col.count(),"tools":["get_schedule","update_schedule"],"google_calendar":bool(GOOGLE_ID and GOOGLE_SECRET)}
