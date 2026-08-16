import os,re,uuid,hashlib,math
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import chromadb
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app=FastAPI(title="ScheduleAI",version="2.2")
TZ=ZoneInfo(os.getenv("TIMEZONE","Asia/Kolkata"))
db=chromadb.PersistentClient(path=os.getenv("CHROMA_PATH","./chroma_db"))
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
  for i in range(4): v[int.from_bytes(b[i*2:i*2+2],'big')%96]+=1 if b[i+8]&1 else -1
 n=math.sqrt(sum(x*x for x in v)) or 1;return [x/n for x in v]
def et(e): return f"{e['title']} {e['type']} on {e['date']} at {e['time']} for {e['duration']} minutes"
def all_events(): return col.get(include=['metadatas']).get('metadatas',[])
def save(e):
 e=dict(e);e.setdefault('id',str(uuid.uuid4()));e.setdefault('type','meeting');e.setdefault('duration',60);e.setdefault('source','user')
 col.upsert(ids=[e['id']],documents=[et(e)],embeddings=[vec(et(e))],metadatas=[e]);return e

def parse_date(q):
 q=q.lower();b=today()
 if 'today' in q:return ds(b)
 if 'tomorrow' in q:return ds(b+timedelta(days=1))
 m=re.search(r'\b(20\d\d)[-/](\d{1,2})[-/](\d{1,2})\b',q)
 if m:
  try:return ds(datetime(int(m[1]),int(m[2]),int(m[3]),tzinfo=TZ))
  except ValueError:return None
 m=re.search(r'\b('+ '|'.join(MONTHS)+r')\s+(\d{1,2})(?:st|nd|rd|th)?\b',q)
 if m:
  try:
   d=datetime(b.year,MONTHS[m[1]],int(m[2]),tzinfo=TZ)
   if d.date()<b.date():d=d.replace(year=d.year+1)
   return ds(d)
  except ValueError:return None
 for n,w in DAYS.items():
  if re.search(r'\bnext\s+'+n+r'\b',q):return ds(b+timedelta(days=(w-b.weekday())%7 or 7))
  if re.search(r'\b'+n+r'\b',q):return ds(b+timedelta(days=(w-b.weekday())%7))
 return None

def times(q):
 out=[]
 for m in re.finditer(r'\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b',q.lower()):
  h,mi=int(m[1]),int(m[2] or 0)
  if h>12 or mi>59:continue
  if m[3]=='pm' and h!=12:h+=12
  if m[3]=='am' and h==12:h=0
  out.append(f'{h:02d}:{mi:02d}')
 return out

def get_schedule(query='',date=None,kind=None,start=None,end=None):
 data=all_events()
 if date:data=[e for e in data if e['date']==date]
 if kind:data=[e for e in data if e['type']==kind]
 if start:data=[e for e in data if mins(e['time'])>=mins(start)]
 if end:data=[e for e in data if mins(e['time'])<=mins(end)]
 if date or kind or start or end:return sorted(data,key=lambda e:(e['date'],e['time']))
 if not data:return []
 r=col.query(query_embeddings=[vec(query or 'schedule')],n_results=min(8,len(data)),include=['metadatas'])
 return sorted(r.get('metadatas',[[]])[0],key=lambda e:(e['date'],e['time']))

def update_schedule(action,event_id=None,event=None,changes=None):
 if action=='add':return save(event or {})
 found=col.get(ids=[event_id],include=['metadatas']).get('metadatas',[])
 if not found:raise ValueError('Schedule entry not found')
 old=dict(found[0])
 if action=='remove':col.delete(ids=[event_id]);return old
 old.update(changes or {});return save(old)

def seed():
 if col.count():return
 b=today();samples=[('Team Meeting',1,'14:00','meeting',60),('AI Workshop',3,'11:00','workshop',90),('Doctor Appointment',5,'10:30','appointment',45),('Project Task',2,'17:30','task',60)]
 for title,off,t,k,d in samples:save({'title':title,'date':ds(b+timedelta(days=off)),'time':t,'type':k,'duration':d,'source':'sample'})
 for i in range(30):
  d=b+timedelta(days=i)
  if d.weekday()<6:
   end=16 if d.weekday()<3 else 15;save({'title':'College','date':ds(d),'time':'09:20','type':'college','duration':end*60-560,'source':'system'})
seed()

def find(q,old=None):
 data=all_events();date=parse_date(q);low=q.lower()
 if date:data=[e for e in data if e['date']==date]
 if old:data=[e for e in data if e['time']==old]
 scored=[]
 for e in data:
  s=(10 if e['title'].lower() in low else 0)+(5 if e['type'] in low else 0)+(3 if e['time'] in low else 0)
  scored.append((s,e))
 scored.sort(key=lambda x:(-x[0],x[1]['date'],x[1]['time']))
 return scored[0][1] if scored and scored[0][0] else (data[0] if len(data)==1 else None)
def kind(q):
 for k in ('appointment','workshop','task','meeting'):
  if k in q.lower():return k
 return 'meeting'
def free(date,start,end):
 busy=[]
 for e in get_schedule(date=date):
  a=mins(e['time']);z=a+int(e['duration'])
  if z>mins(start) and a<mins(end):busy.append((max(a,mins(start)),min(z,mins(end)),e))
 busy.sort();free=[];cur=mins(start)
 for a,z,_ in busy:
  if a>cur:free.append((cur,a))
  cur=max(cur,z)
 if cur<mins(end):free.append((cur,mins(end)))
 return busy,free
def clock(x):return pretty(f'{x//60:02d}:{x%60:02d}')

def agent(q):
 low=q.lower().strip();date=parse_date(low);ts=times(low)
 if low in {'hi','hello','hey'}:return 'direct','Hi! I’m ScheduleAI. Ask about your schedule or tell me what to add, move, or remove.'
 if 'free' in low:
  d=date or ds(today());a,z=('12:00','17:00') if 'afternoon' in low else ('09:00','21:00');busy,open_=free(d,a,z)
  if not busy:return 'direct',f"You’re free from {pretty(a)} to {pretty(z)} on {d}."
  b='\n'.join(f"• {e['title']} — {pretty(e['time'])}–{clock(z2)}" for a2,z2,e in busy);f=', '.join(f'{clock(a2)}–{clock(z2)}' for a2,z2 in open_) or 'none'
  return 'direct',f'Busy on {d}:\n{b}\n\nFree windows: {f}'
 if any(x in low for x in ('move ','reschedule ','change ')):
  if len(ts)<2:return 'direct','Please give me the old and new time, e.g. “Move my meeting from 2 PM to 4 PM.”'
  e=find(low,ts[0])
  if not e:return 'direct','I couldn’t find that event. Include its name or date and try again.'
  changes={'time':ts[-1]};
  if date:changes['date']=date
  e=update_schedule('update',e['id'],changes=changes);return 'direct',f"Done. {e['title']} is now at {pretty(e['time'])} on {e['date']}."
 if any(x in low for x in ('remove ','delete ','cancel ')):
  e=find(low,ts[0] if ts else None)
  if not e:return 'direct','Tell me which event to remove, including its name or date.'
  e=update_schedule('remove',e['id']);return 'direct',f"Removed {e['title']} from {e['date']} at {pretty(e['time'])}."
 if re.match(r'^(add|create|schedule|book)\b',low):
  if not date or not ts:return 'direct','Please include both the date and time, e.g. “Add a meeting tomorrow at 3 PM.”'
  if not today().date()<=datetime.strptime(date,'%Y-%m-%d').date()<= (today()+timedelta(days=29)).date():return 'direct','Please choose a date within the next 30 days.'
  title=re.sub(r'^(add|create|schedule|book)\s+','',q,flags=re.I);title=re.sub(r'\s+(?:on|for)\s+(?:today|tomorrow|(?:next\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)','',title,flags=re.I);title=re.sub(r'\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)','',title,flags=re.I);title=re.sub(r'^(a|an|the)\s+','',title,flags=re.I).strip(' ,.-') or 'New Event'
  e=update_schedule('add',event={'title':title,'date':date,'time':ts[0],'type':kind(low),'duration':60,'source':'user'});return 'direct',f"Added {e['title']} for {e['date']} at {pretty(e['time'])}."
 k=next((x for x in ('meeting','workshop','appointment','task') if re.search(r'\b'+x+r's?\b',low)),None)
 return 'get_schedule',get_schedule(q,date=date,kind=k)

def answer(q,tool,result):
 if tool=='direct':return result
 if not result:return 'Nothing matching that request is scheduled.'
 return '\n'.join(f"• {e['title']} — {e['date']} at {pretty(e['time'])} ({e['duration']} min)" for e in result)
class Chat(BaseModel):message:str

PAGE='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScheduleAI</title><style>:root{--a:#6658e8;--dark:#11162a;--bg:#f6f7fb}*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Arial;background:var(--bg);color:#182033}.app{display:flex;min-height:100vh}.side{width:250px;background:var(--dark);color:white;padding:28px 18px}.brand{font-size:25px;font-weight:800;margin-bottom:35px}.brand span{color:#9187ff}.nav{padding:12px 14px;color:#aeb7cf;border-radius:12px;margin:6px 0}.active{background:#29234c;color:white}.tip{margin-top:30px;padding:16px;border:1px solid #303750;border-radius:15px;color:#bdc4d5;font-size:12px;line-height:1.8}.main{max-width:1050px;width:100%;margin:auto;padding:35px}.head{display:flex;justify-content:space-between;gap:15px}.head h1{font-size:38px;margin:0 0 8px}.sub{color:#707b91}.status{background:#e6f8ee;color:#247b48;padding:9px 13px;border-radius:30px;font-size:12px;font-weight:800}.quick{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0}.quick button{border:1px solid #e0e4ed;background:white;border-radius:12px;padding:11px 14px;cursor:pointer}.chat{background:white;border:1px solid #e1e5ee;border-radius:22px;min-height:460px;padding:22px;box-shadow:0 15px 40px #17203a0d}.msg{max-width:80%;padding:14px 17px;border-radius:16px;margin:10px 0;line-height:1.55;white-space:pre-wrap}.bot{background:#f0f2f7}.user{margin-left:auto;background:var(--a);color:white}.composer{display:flex;gap:10px;margin-top:14px}.composer input{flex:1;padding:16px;border:1px solid #dce1ea;border-radius:14px;font-size:15px}.send{border:0;background:var(--a);color:white;border-radius:14px;padding:0 25px;font-weight:800;cursor:pointer}.foot{text-align:center;color:#8a93a5;font-size:12px;margin-top:12px}@media(max-width:720px){.side{display:none}.main{padding:22px 14px}.head h1{font-size:30px}.status{display:none}.msg{max-width:94%}.send{padding:0 18px}}</style></head><body><div class="app"><aside class="side"><div class="brand">Schedule<span>AI</span></div><div class="nav active">✦ Assistant</div><div class="nav">▣ 30-Day Schedule</div><div class="nav">◷ Agent Tools</div><div class="tip"><b>Try asking</b><br>What do I have tomorrow?<br>Am I free Friday afternoon?<br>Add a meeting tomorrow at 3 PM.<br>Move my meeting from 2 PM to 4 PM.<br>Remove my workshop.</div></aside><main class="main"><div class="head"><div><h1>Your schedule, understood.</h1><div class="sub">Agentic routing + ChromaDB RAG for your next 30 days.</div></div><div class="status">● SYSTEM ONLINE</div></div><div class="quick"><button onclick="ask('What do I have today?')">📅 Today</button><button onclick="ask('What do I have tomorrow?')">Tomorrow</button><button onclick="ask('Am I free Friday afternoon?')">Free Friday</button><button onclick="ask('Show my meetings')">Meetings</button></div><div id="chat" class="chat"><div class="msg bot"><b>Hi! I’m ScheduleAI.</b><br><br>I can retrieve your schedule with RAG or update it when you ask.<br><br>Try one of the examples above.</div></div><div class="composer"><input id="q" placeholder="Ask about your schedule..." onkeydown="if(event.key==='Enter')send()"><button class="send" onclick="send()">Send</button></div><div class="foot">FastAPI • ChromaDB • RAG • get_schedule • update_schedule</div></main></div><script>function add(t,c){const x=document.createElement('div');x.className='msg '+c;x.textContent=t;document.getElementById('chat').appendChild(x);x.scrollIntoView({behavior:'smooth'});return x}function ask(t){document.getElementById('q').value=t;send()}async function send(){const i=document.getElementById('q'),v=i.value.trim();if(!v)return;i.value='';add(v,'user');const b=add('Thinking…','bot');try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})});const d=await r.json();b.textContent=d.answer||'No answer returned.'}catch(e){b.textContent='Unable to reach the assistant. Please try again.'}}</script></body></html>'''

@app.get('/',response_class=HTMLResponse)
def home():return PAGE
@app.post('/chat')
def chat(c:Chat):
 try:
  tool,result=agent(c.message);return {'answer':answer(c.message,tool,result),'tool':tool,'data':result}
 except Exception as e:return {'answer':'I couldn’t complete that request. Please try again.','error':str(e)}
@app.get('/health')
def health():return {'status':'ok','events':col.count(),'tools':['get_schedule','update_schedule'],'rag':'ChromaDB'}
