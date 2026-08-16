import os,re,uuid,hashlib,math,secrets
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import chromadb,httpx
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from authlib.integrations.starlette_client import OAuth

app=FastAPI(title="ScheduleAI",version="4.1")
app.add_middleware(SessionMiddleware,secret_key=os.getenv("SESSION_SECRET",secrets.token_hex(32)),https_only=False,same_site="lax")
TZ=ZoneInfo(os.getenv("TIMEZONE","Asia/Kolkata"));db=chromadb.PersistentClient(path=os.getenv("CHROMA_PATH","./chroma_db"));col=db.get_or_create_collection("schedule_assistant")
GOOGLE_ID=os.getenv("GOOGLE_CLIENT_ID","");GOOGLE_SECRET=os.getenv("GOOGLE_CLIENT_SECRET","");BASE=os.getenv("APP_URL","").rstrip("/");oauth=OAuth()
if GOOGLE_ID and GOOGLE_SECRET: oauth.register(name="google",client_id=GOOGLE_ID,client_secret=GOOGLE_SECRET,server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",client_kwargs={"scope":"openid email profile https://www.googleapis.com/auth/calendar"})
MONTHS={m.lower():i for i,m in enumerate("January February March April May June July August September October November December".split(),1)}
DAYS={d.lower():i for i,d in enumerate("Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split())}
def today():return datetime.now(TZ).replace(hour=0,minute=0,second=0,microsecond=0)
def now_text():return datetime.now(TZ).strftime("It is %I:%M %p on %A, %B %d, %Y (%Z).")
def ds(d):return d.strftime("%Y-%m-%d")
def pretty(t):
 h,m=map(int,t.split(":"));return f"{h%12 or 12}:{m:02d} {'AM' if h<12 else 'PM'}"
def mins(t):h,m=map(int,t.split(":"));return h*60+m
def vec(s):
 v=[0.0]*96
 for w in re.findall(r"[a-z0-9]+",s.lower()):
  b=hashlib.sha256(w.encode()).digest()
  for i in range(4):v[int.from_bytes(b[i*2:i*2+2],"big")%96]+=1 if b[i+8]&1 else -1
 n=math.sqrt(sum(x*x for x in v)) or 1;return [x/n for x in v]
def doc(e):return f"{e['title']} {e['type']} on {e['date']} at {e['time']} for {e['duration']} minutes"
def all_events():return col.get(include=["metadatas"]).get("metadatas",[])
def save(e):
 e=dict(e);e.setdefault("id",str(uuid.uuid4()));e.setdefault("type","meeting");e.setdefault("duration",60);e.setdefault("source","user");col.upsert(ids=[e["id"]],documents=[doc(e)],embeddings=[vec(doc(e))],metadatas=[e]);return e
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
 m=re.search(r"\bon\s+(\d{1,2})(?:st|nd|rd|th)?\b",q)
 if m:
  for i in range(30):
   d=b+timedelta(days=i)
   if d.day==int(m[1]):return ds(d)
 for n,w in DAYS.items():
  if re.search(r"\bnext\s+"+n+r"\b",q):return ds(b+timedelta(days=(w-b.weekday())%7 or 7))
  if re.search(r"\b"+n+r"\b",q):return ds(b+timedelta(days=(w-b.weekday())%7))
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
 r=col.query(query_embeddings=[vec(query or "schedule")],n_results=min(8,len(data)),include=["metadatas"]);return sorted(r.get("metadatas",[[]])[0],key=lambda e:(e["date"],e["time"]))
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
 scored=[((10 if e["title"].lower() in low else 0)+(5 if e["type"] in low else 0)+(3 if e["time"] in low else 0),e) for e in data];scored.sort(key=lambda x:(-x[0],x[1]["date"],x[1]["time"]));return scored[0][1] if scored and scored[0][0] else (data[0] if len(data)==1 else None)
def free_windows(date,start="12:00",end="17:00"):
 busy=[]
 for e in get_schedule(date=date):
  a=mins(e["time"]);z=a+int(e["duration"])
  if z>mins(start) and a<mins(end):busy.append((max(a,mins(start)),min(z,mins(end)),e))
 busy.sort();open_=[];cur=mins(start)
 for a,z,_ in busy:
  if a>cur:open_.append((cur,a))
  cur=max(cur,z)
 if cur<mins(end):open_.append((cur,mins(end)))
 return busy,open_
def clock(x):return pretty(f"{x//60:02d}:{x%60:02d}")
def agent(q):
 low=q.lower().strip();date=parse_date(low);ts=times(low)
 # Handle time BEFORE RAG. This prevents queries like "time?" from returning random events.
 if re.fullmatch(r"(?:time|time\?|current\s+time|what\s+time|what\s+time\s+is\s+it|what(?:'s|\s+is)\s+the\s+time|tell\s+me\s+the\s+time)[?.!\s]*",low):return "direct",now_text()
 if re.search(r"\b(today'?s date|date today|what date is it|what is the date today)\b",low):return "direct",today().strftime("Today is %A, %B %d, %Y.")
 if low in {"hi","hello","hey"}:return "direct","Hi! I’m ScheduleAI. I can find, add, move, or remove events for your next 30 days."
 if "free" in low:
  d=date or ds(today());a,z=("12:00","17:00") if "afternoon" in low else ("09:00","21:00");busy,open_=free_windows(d,a,z)
  if not busy:return "direct",f"You’re free from {pretty(a)} to {pretty(z)} on {d}."
  b="\n".join(f"• {e['title']} — {pretty(e['time'])}–{clock(z)}" for a,z,e in busy);f=", ".join(f"{clock(a)}–{clock(z)}" for a,z in open_) or "none";return "direct",f"Busy on {d}:\n{b}\n\nFree windows: {f}"
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
  title=re.sub(r"^(add|create|schedule|book)\s+","",q,flags=re.I);title=re.sub(r"\s+(on|for)\s+[^@]+(?=\s+at\s+\d)","",title,flags=re.I);title=re.sub(r"\s+at\s+\d{1,2}(?::\d{2})?\s*(am|pm)","",title,flags=re.I).strip(" ,.-") or "New Event";e=update_schedule("add",event={"title":title,"date":date,"time":ts[0],"type":"meeting","duration":60,"source":"user"});return "direct",f"Added {e['title']} for {e['date']} at {pretty(e['time'])}."
 kind=next((k for k in ("meeting","workshop","appointment","task") if re.search(r"\b"+k+r"s?\b",low)),None);data=get_schedule(q,date=date,kind=kind)
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
  async with httpx.AsyncClient(timeout=15) as c:
   r=await c.get("https://www.googleapis.com/calendar/v3/calendars/primary/events",headers={"Authorization":"Bearer "+token},params={"timeMin":start,"timeMax":end,"singleEvents":"true","orderBy":"startTime","maxResults":100})
  return r.json().get("items",[]) if r.status_code==200 else None
 except Exception:return None
@app.get("/auth/google")
async def google_login(request:Request):
 if not GOOGLE_ID or not GOOGLE_SECRET:return HTMLResponse("<h3>Google Calendar is not configured.</h3><p>Add GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in Render.</p>",503)
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
PAGE='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScheduleAI</title><style>:root{--a:#635bdf;--dark:#11172a;--bg:#f5f7fc;--ink:#172033;--muted:#778199;--line:#e4e8f0}*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Arial;background:var(--bg);color:var(--ink)}.app{display:flex;min-height:100vh}.side{width:250px;background:linear-gradient(180deg,#11172a,#191f38);color:#fff;padding:28px 18px}.brand{font-size:25px;font-weight:850;margin:2px 8px 32px}.brand span{color:#9b92ff}.nav{padding:12px 14px;color:#b3bdd2;border-radius:12px;margin:6px 0}.active{background:#2b264d;color:#fff}.tip{margin-top:26px;padding:15px;border:1px solid #343b55;border-radius:15px;color:#c5cde0;font-size:12px;line-height:1.8}.main{width:min(1120px,100%);margin:auto;padding:30px}.top{display:flex;justify-content:space-between;gap:18px;align-items:flex-start}.eyebrow{font-size:11px;font-weight:850;letter-spacing:.13em;color:var(--a);text-transform:uppercase}.head{font-size:38px;font-weight:850;margin:7px 0}.sub{color:var(--muted)}.actions{display:flex;gap:8px}.btn{border:1px solid var(--line);background:#fff;border-radius:12px;padding:11px 14px;text-decoration:none;color:var(--ink);font-weight:750;cursor:pointer}.primary{background:var(--a);color:#fff;border-color:var(--a)}.quick{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:23px 0 15px}.quick button{border:1px solid var(--line);background:#fff;border-radius:15px;padding:14px;text-align:left;cursor:pointer}.quick b{display:block}.quick span{display:block;color:var(--muted);font-size:11px;margin-top:4px}.status{color:#18834e;font-size:12px;font-weight:800;margin:10px 2px}.chat{background:#fff;border:1px solid var(--line);border-radius:24px;min-height:450px;padding:22px;box-shadow:0 18px 50px #17203a0d}.welcome{padding:18px;border-radius:18px;background:linear-gradient(135deg,#f2f1ff,#f8f9fc);border:1px solid #e5e3fa}.msg{max-width:86%;margin:10px 0;padding:14px 17px;border-radius:17px;white-space:pre-wrap;line-height:1.55;font-size:14px}.user{margin-left:auto;background:linear-gradient(135deg,var(--a),#786fea);color:#fff}.bot{background:#f0f1f7}.events{display:grid;gap:9px}.event{display:flex;align-items:center;gap:12px;padding:12px;border:1px solid #e6e9f0;border-radius:14px;background:#fff}.date{width:55px;text-align:center;background:#f0efff;color:var(--a);border-radius:10px;padding:7px;font-weight:850;font-size:11px}.eventmain{flex:1}.eventmain b{font-size:13px}.meta{font-size:11px;color:var(--muted);margin-top:3px}.tag{font-size:10px;background:#f0f1f5;padding:5px 8px;border-radius:20px}.composer{display:flex;gap:8px;margin-top:14px;background:#fff;border:1px solid var(--line);border-radius:17px;padding:7px}.composer input{flex:1;border:0;outline:0;padding:13px;font-size:14px}.send{border:0;background:var(--a);color:#fff;border-radius:12px;padding:0 20px;font-weight:850}.foot{text-align:center;color:#929bae;font-size:11px;margin-top:10px}@media(max-width:820px){.side{display:none}.main{padding:22px 14px}.head{font-size:30px}.quick{grid-template-columns:repeat(2,1fr)}.top{flex-direction:column}.actions{width:100%}}@media(max-width:480px){.quick{grid-template-columns:1fr}.actions{flex-wrap:wrap}} </style></head><body><div class="app"><aside class="side"><div class="brand">Schedule<span>AI</span></div><div class="nav active">✦ &nbsp;Assistant</div><div class="nav">▣ &nbsp;30-Day Schedule</div><div class="nav">◷ &nbsp;Agent Tools</div><div class="tip"><b>Try asking</b><br>What do I have tomorrow?<br>Am I free Friday afternoon?<br>Add a meeting tomorrow at 3 PM.<br>Move my meeting from 2 PM to 4 PM.<br>What time is it?</div></aside><main class="main"><div class="top"><div><div class="eyebrow">Agentic Schedule Assistant</div><div class="head">Your schedule, understood.</div><div class="sub">ChromaDB RAG + Google Calendar + voice commands.</div></div><div class="actions"><button class="btn" onclick="voice()">🎙 Voice</button><a class="btn" href="/auth/google">G&nbsp; Connect Google</a></div></div><div id="status"></div><div class="quick"><button onclick="ask('What do I have today?')"><b>📅 Today</b><span>See today’s events</span></button><button onclick="ask('What do I have tomorrow?')"><b>Tomorrow</b><span>Plan the next day</span></button><button onclick="ask('Am I free Friday afternoon?')"><b>Free time</b><span>Check availability</span></button><button onclick="ask('Show my meetings')"><b>Meetings</b><span>Find meetings</span></button></div><div id="chat" class="chat"><div class="welcome"><b>Hi! I’m ScheduleAI.</b><br><span>Ask about your schedule, add or move an event, connect Google Calendar, or use voice.</span></div></div><div class="composer"><input id="q" placeholder="Ask about your schedule..." autocomplete="off"><button class="btn" onclick="voice()">🎙</button><button class="send" onclick="send()">Send</button></div><div class="foot">FastAPI • ChromaDB • Agentic RAG • Google Calendar • Voice</div></main></div><script>const q=document.getElementById('q'),chat=document.getElementById('chat'),status=document.getElementById('status');function ask(t){q.value=t;send()}function msg(t,c){let x=document.createElement('div');x.className='msg '+c;x.textContent=t;chat.appendChild(x);x.scrollIntoView({behavior:'smooth'});return x}function esc(s){return String(s).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}function t12(t){let [h,m]=t.split(':');h=+h;return(h%12||12)+':'+m+' '+(h<12?'AM':'PM')}function cards(items){let x=document.createElement('div');x.className='msg bot';let w=document.createElement('div');w.className='events';items.forEach(e=>{let r=document.createElement('div');r.className='event';let d=new Date(e.date+'T00:00:00');r.innerHTML='<div class="date">'+d.toLocaleDateString('en-US',{month:'short'})+'<br>'+d.getDate()+'</div><div class="eventmain"><b>'+esc(e.title)+'</b><div class="meta">'+esc(e.date)+' • '+t12(e.time)+' • '+e.duration+' min</div></div><div class="tag">'+esc(e.type)+'</div>';w.appendChild(r)});x.appendChild(w);chat.appendChild(x);x.scrollIntoView({behavior:'smooth'})}async function send(){let v=q.value.trim();if(!v)return;q.value='';msg(v,'user');let b=msg('Thinking…','bot');try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})});let d=await r.json();b.remove();if(Array.isArray(d.data)&&d.data.length)cards(d.data);else msg(d.answer||'No answer returned.','bot')}catch(e){b.remove();msg('Unable to reach ScheduleAI. Please try again.','bot')}}q.addEventListener('keydown',e=>{if(e.key==='Enter')send()});function voice(){let S=window.SpeechRecognition||window.webkitSpeechRecognition;if(!S){msg('Voice input is not supported here. Use Chrome or Edge.','bot');return}let r=new S();r.lang='en-IN';r.interimResults=false;r.onstart=()=>msg('🎙 Listening…','bot');r.onresult=e=>{q.value=e.results[0][0].transcript;send()};r.onerror=()=>msg('I could not hear that. Please try again.','bot');r.start()}async function google(){try{let r=await fetch('/api/google-calendar'),d=await r.json();if(d.connected){status.innerHTML='<div class="status">✓ Google Calendar connected • '+d.events.length+' upcoming events</div>';if(d.events.length)cards(d.events.map(e=>{let v=(e.start||{}).dateTime||e.start?.date||'';let p=v.replace('Z','').split('T');return{title:e.summary||'Google Calendar event',date:p[0],time:(p[1]||'00:00').slice(0,5),duration:60,type:'Google Calendar'}}).slice(0,12))}}catch(e){}}google();</script></body></html>'''
@app.get("/",response_class=HTMLResponse)
def home():return PAGE
@app.post("/chat")
def chat(c:Chat):
 try:tool,result=agent(c.message);return {"answer":answer(tool,result),"tool":tool,"data":result}
 except Exception as e:return {"answer":"I couldn’t complete that request. Please try again.","error":str(e)}
@app.get("/health")
def health():return {"status":"ok","events":col.count(),"tools":["get_schedule","update_schedule"],"google_calendar":bool(GOOGLE_ID and GOOGLE_SECRET),"voice":"browser SpeechRecognition","timezone":str(TZ)}
