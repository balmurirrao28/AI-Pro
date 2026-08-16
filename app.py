import os,re,uuid,hashlib,math,secrets
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import chromadb,httpx
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse,RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from authlib.integrations.starlette_client import OAuth
app=FastAPI(title='ScheduleAI',version='7.0')
app.add_middleware(SessionMiddleware,secret_key=os.getenv('SESSION_SECRET',secrets.token_hex(32)),same_site='lax')
TZ=ZoneInfo(os.getenv('TIMEZONE','Asia/Kolkata'));db=chromadb.PersistentClient(path=os.getenv('CHROMA_PATH','./chroma_db'));col=db.get_or_create_collection('schedule_assistant')
KINDS=('meeting','workshop','task','appointment');MONTHS={m.lower():i for i,m in enumerate('January February March April May June July August September October November December'.split(),1)};DAYS={d.lower():i for i,d in enumerate('Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split())}
GID=os.getenv('GOOGLE_CLIENT_ID','');GSECRET=os.getenv('GOOGLE_CLIENT_SECRET','');BASE=os.getenv('APP_URL','').rstrip('/');oauth=OAuth()
if GID and GSECRET:oauth.register(name='google',client_id=GID,client_secret=GSECRET,server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',client_kwargs={'scope':'openid email profile https://www.googleapis.com/auth/calendar'})
def today():return datetime.now(TZ).replace(hour=0,minute=0,second=0,microsecond=0)
def ds(d):return d.strftime('%Y-%m-%d')
def pretty(t):h,m=map(int,t.split(':'));return f'{h%12 or 12}:{m:02d} {"AM" if h<12 else "PM"}'
def mins(t):h,m=map(int,t.split(':'));return h*60+m
def now_text():return datetime.now(TZ).strftime('It is %I:%M %p on %A, %B %d, %Y (%Z).')
def vec(s):
 v=[0.0]*96
 for w in re.findall(r'[a-z0-9]+',s.lower()):
  b=hashlib.sha256(w.encode()).digest()
  for i in range(4):v[int.from_bytes(b[i*2:i*2+2],'big')%96]+=1 if b[i+8]&1 else -1
 n=math.sqrt(sum(x*x for x in v)) or 1;return [x/n for x in v]
def doc(e):return f"{e['title']} {e['type']} on {e['date']} at {e['time']} for {e['duration']} minutes"
def events():return col.get(include=['metadatas']).get('metadatas',[])
def save(e):
 e=dict(e);e.setdefault('id',str(uuid.uuid4()));e.setdefault('duration',60);e.setdefault('source','user');col.upsert(ids=[e['id']],documents=[doc(e)],embeddings=[vec(doc(e))],metadatas=[e]);return e
def seed():
 if col.count():return
 b=today()
 for title,off,t,k,d in [('Team Meeting',1,'14:00','meeting',60),('Project Task',2,'17:30','task',60),('AI Workshop',3,'11:00','workshop',90),('Doctor Appointment',5,'10:30','appointment',45)]:save({'title':title,'date':ds(b+timedelta(days=off)),'time':t,'type':k,'duration':d,'source':'sample'})
seed()
def parse_date(q):
 q=q.lower();b=today()
 if 'today' in q:return ds(b)
 if 'tomorrow' in q:return ds(b+timedelta(days=1))
 m=re.search(r'\b(20\d\d)[-/](\d{1,2})[-/](\d{1,2})\b',q)
 if m:
  try:return ds(datetime(int(m[1]),int(m[2]),int(m[3]),tzinfo=TZ))
  except:return None
 m=re.search(r'\b('+'|'.join(MONTHS)+r')\s+(\d{1,2})(?:st|nd|rd|th)?\b',q)
 if m:
  try:
   d=datetime(b.year,MONTHS[m[1]],int(m[2]),tzinfo=TZ);return ds(d if d.date()>=b.date() else d.replace(year=d.year+1))
  except:return None
 for n,w in DAYS.items():
  if re.search(r'\bnext\s+'+n+r'\b',q):return ds(b+timedelta(days=(w-b.weekday())%7 or 7))
  if re.search(r'\b'+n+r'\b',q):return ds(b+timedelta(days=(w-b.weekday())%7))
 return None
def times(q):
 out=[]
 for m in re.finditer(r'\b(\d{1,2})(?:[:\s](\d{2}))?\s*(am|pm)\b',q.lower()):
  h,mi=int(m[1]),int(m[2] or 0)
  if h>12 or mi>59:continue
  if m[3]=='pm' and h!=12:h+=12
  if m[3]=='am' and h==12:h=0
  out.append(f'{h:02d}:{mi:02d}')
 return out
def get_schedule(q='',date=None,kind=None):
 d=events()
 if date:d=[e for e in d if e['date']==date]
 if kind:d=[e for e in d if e['type']==kind]
 if date or kind:return sorted(d,key=lambda e:(e['date'],e['time']))
 if not d:return []
 r=col.query(query_embeddings=[vec(q or 'schedule')],n_results=min(8,len(d)),include=['metadatas']);return sorted(r.get('metadatas',[[]])[0],key=lambda e:(e['date'],e['time']))
def update_schedule(action,event_id,changes=None):
 found=col.get(ids=[event_id],include=['metadatas']).get('metadatas',[])
 if not found:raise ValueError('Schedule entry not found')
 old=dict(found[0])
 if action=='remove':col.delete(ids=[event_id]);return old
 old.update(changes or {});return save(old)
def find_event(q,old=None):
 d=events();date=parse_date(q);low=q.lower()
 if date:d=[e for e in d if e['date']==date]
 if old:d=[e for e in d if e['time']==old]
 scored=[]
 for e in d:
  s=(5 if e['type'] in low else 0)+sum(2 for w in re.findall(r'[a-z0-9]+',e['title'].lower()) if w in low);scored.append((s,e))
 scored.sort(key=lambda x:(-x[0],x[1]['date'],x[1]['time']));return scored[0][1] if scored and scored[0][0]>0 else (d[0] if len(d)==1 else None)
def free_windows(date,start='09:00',end='21:00'):
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
def pending(req,q):
 p=req.session.get('pending_add')
 if not p:return None
 low=q.lower().strip();step=p['step']
 if step=='type':
  k=next((x for x in KINDS if re.search(r'\b'+x+r's?\b',low)),None)
  if not k:return 'Choose: meeting, workshop, task, or appointment.'
  p.update(type=k,step='title');req.session['pending_add']=p;return f'Great — {k}. What should I call it?'
 if step=='title':p.update(title=q.strip(),step='date');req.session['pending_add']=p;return 'What date should I schedule it?'
 if step=='date':
  d=parse_date(low)
  if not d:return 'I could not understand that date. Try tomorrow, Friday, or August 20.'
  p.update(date=d,step='time');req.session['pending_add']=p;return 'What time should it start?'
 if step=='time':
  ts=times(low)
  if not ts:return 'Please give a time such as 3 PM or 7:20 PM.'
  p.update(time=ts[0],step='duration');req.session['pending_add']=p;return 'How long should it last? For example, 30 minutes or 1 hour.'
 if step=='duration':
  m=re.search(r'(\d+)\s*(?:min|mins|minutes)',low);h=re.search(r'(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)',low);dur=int(m[1]) if m else int(float(h[1])*60) if h else None
  if not dur:return 'Please give a duration such as 30 minutes or 1 hour.'
  d=datetime.strptime(p['date'],'%Y-%m-%d').date()
  if not today().date()<=d<=today().date()+timedelta(days=29):req.session.pop('pending_add',None);return 'Please choose a date within the next 30 days.'
  e=save({**p,'duration':dur,'source':'user'});req.session.pop('pending_add',None);return f"Added {e['title']} ({e['type']}) for {e['date']} at {pretty(e['time'])}, {dur} minutes."
def agent(req,q):
 low=q.lower().strip();date=parse_date(low);ts=times(low);p=pending(req,q)
 if p is not None:return 'direct',p
 if re.fullmatch(r"(?:time|time\?|current\s+time|what\s+time|what\s+time\s+is\s+it|what(?:'s|\s+is)\s+the\s+time|tell\s+me\s+the\s+time)[?.!\s]*",low):return 'direct',now_text()
 if re.search(r"\b(today'?s date|date today|what date is it|what is the date today)\b",low):return 'direct',today().strftime('Today is %A, %B %d, %Y.')
 if low in {'hi','hello','hey'}:return 'direct','Hi! I’m ScheduleAI. I can find, add, move, or remove events.'
 if re.match(r'^(add|create|schedule|book)\b',low):
  k=next((x for x in KINDS if re.search(r'\b'+x+r's?\b',low)),None)
  if not k:req.session['pending_add']={'step':'type'};return 'direct','Sure. What type is it: meeting, workshop, task, or appointment?'
  title=re.sub(r'^(add|create|schedule|book)\s+','',q,flags=re.I);title=re.sub(r'\b(meeting|workshop|task|appointment)\b','',title,flags=re.I).strip(' ,.-') or k.title()
  if not date:req.session['pending_add']={'step':'date','type':k,'title':title};return 'direct','What date should I schedule it?'
  if not ts:req.session['pending_add']={'step':'time','type':k,'title':title,'date':date};return 'direct','What time should it start?'
  req.session['pending_add']={'step':'duration','type':k,'title':title,'date':date,'time':ts[0]};return 'direct','How long should it last?'
 if re.search(r'\b(move|reschedule|change)\b',low):
  if len(ts)<2:return 'direct','Tell me the event, current time, and new time.'
  e=find_event(low,ts[0])
  if not e:return 'direct','I could not find that event. Include its name or date.'
  ch={'time':ts[-1]}
  if date:ch['date']=date
  e=update_schedule('update',e['id'],ch);return 'direct',f"Done. {e['title']} is now on {e['date']} at {pretty(e['time'])}."
 if re.search(r'\b(remove|delete|cancel)\b',low):
  e=find_event(low,ts[0] if ts else None)
  if not e:return 'direct','Tell me which event to remove, including its name or date.'
  e=update_schedule('remove',e['id']);return 'direct',f"Removed {e['title']} from {e['date']} at {pretty(e['time'])}."
 if 'free' in low:
  d=date or ds(today());a,z=('12:00','17:00') if 'afternoon' in low else ('09:00','21:00');busy,free=free_windows(d,a,z)
  if not busy:return 'direct',f'You’re free from {pretty(a)} to {pretty(z)} on {d}.'
  b='\n'.join(f"• {e['title']} — {pretty(e['time'])}–{clock(z2)}" for a2,z2,e in busy);f=', '.join(f'{clock(a2)}–{clock(z2)}' for a2,z2 in free) or 'none';return 'direct',f'Busy on {d}:\n{b}\n\nFree windows: {f}'
 k=next((x for x in KINDS if re.search(r'\b'+x+r's?\b',low)),None);data=get_schedule(q,date=date,kind=k)
 if ts:data=[e for e in data if mins(e['time'])==mins(ts[0])]
 return 'get_schedule',data
def answer(tool,result):
 if tool=='direct':return result
 if not result:return 'Nothing matching that request is scheduled.'
 return '\n'.join(f"• {e['title']} — {e['date']} at {pretty(e['time'])} ({e['duration']} min)" for e in result)
async def cal(req):
 token=req.session.get('google_token')
 if not token:return None
 try:
  async with httpx.AsyncClient(timeout=12) as c:r=await c.get('https://www.googleapis.com/calendar/v3/calendars/primary/events',headers={'Authorization':'Bearer '+token},params={'timeMin':datetime.now(TZ).isoformat(),'timeMax':(datetime.now(TZ)+timedelta(days=30)).isoformat(),'singleEvents':'true','orderBy':'startTime','maxResults':100})
  return r.json().get('items',[]) if r.status_code==200 else None
 except:return None
@app.get('/auth/google')
async def google(req:Request):
 if not GID or not GSECRET:return HTMLResponse('<h3>Google Calendar is not configured.</h3>',503)
 return await oauth.google.authorize_redirect(req,BASE+'/auth/google/callback')
@app.get('/auth/google/callback')
async def callback(req:Request):
 t=await oauth.google.authorize_access_token(req);req.session['google_token']=t['access_token'];return RedirectResponse('/')
@app.get('/auth/logout')
async def logout(req:Request):req.session.clear();return RedirectResponse('/')
@app.get('/api/google-calendar')
async def calendar(req:Request):
 d=await cal(req);return {'connected':d is not None,'events':d or []}
class Chat(BaseModel):message:str
PAGE='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ScheduleAI</title><style>:root{--bg:#f5f7fc;--ink:#172033;--muted:#778199;--line:#e4e8f0;--a:#635bdf;--soft:#f0f1f8}*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,Arial;background:var(--bg);color:var(--ink)}.app{display:flex;min-height:100vh}.side{width:252px;background:#11172a;color:#fff;padding:28px 18px}.brand{font-size:25px;font-weight:850;margin:2px 8px 34px}.brand span{color:#9a91ff}.nav{padding:12px 14px;color:#aeb7cf;border-radius:12px;margin:6px 0;font-size:14px}.active{background:#2b264d;color:#fff}.tip{margin-top:28px;padding:16px;border:1px solid #303750;border-radius:16px;color:#c4cada;font-size:12px;line-height:1.8}.main{width:min(1120px,100%);margin:auto;padding:26px 30px}.top{display:flex;justify-content:space-between;align-items:center;gap:12px}.eyebrow{font-size:12px;font-weight:800;letter-spacing:.12em;color:var(--a);text-transform:uppercase}.head{font-size:38px;margin:7px 0}.sub{color:var(--muted)}.auth{display:flex;gap:8px}.google,.voice{border:1px solid var(--line);background:#fff;border-radius:12px;padding:11px 14px;text-decoration:none;color:var(--ink);font-weight:750;cursor:pointer}.quick{display:grid;grid-template-columns:repeat(4,1fr);gap:11px;margin:22px 0 14px}.quick button{border:1px solid var(--line);background:#fff;border-radius:15px;padding:14px;text-align:left;cursor:pointer}.qtitle{font-weight:800}.qdesc{display:block;color:var(--muted);font-size:11px;margin-top:4px}.chat{background:#fff;border:1px solid var(--line);border-radius:24px;padding:20px;min-height:130px;max-height:52vh;overflow-y:auto;box-shadow:0 12px 35px #17203a0d}.welcome{display:flex;gap:13px;background:#f5f4ff;border:1px solid #e3e0ff;padding:16px;border-radius:18px}.avatar{width:38px;height:38px;border-radius:12px;background:var(--a);color:#fff;display:grid;place-items:center;font-weight:900;flex:none}.welcome strong{display:block;margin-bottom:4px}.welcome span{color:var(--muted);font-size:13px;line-height:1.5}.msg{max-width:84%;padding:13px 16px;border-radius:17px;margin:9px 0;line-height:1.55;white-space:pre-wrap;font-size:14px}.bot{background:var(--soft)}.user{margin-left:auto;background:var(--a);color:#fff}.events{display:grid;gap:9px}.event{display:flex;align-items:center;gap:12px;padding:11px;background:#fff;border:1px solid var(--line);border-radius:14px}.datebox{width:54px;text-align:center;background:#f1f0ff;border-radius:10px;padding:7px 4px;color:var(--a);font-weight:850;font-size:11px}.eventmain{flex:1}.eventtitle{font-weight:800;font-size:13px}.meta{color:var(--muted);font-size:11px;margin-top:3px}.tag{font-size:10px;padding:5px 8px;border-radius:20px;background:#f0f1f5;color:#626d83}.composer{display:flex;gap:8px;margin-top:12px;background:#fff;border:1px solid var(--line);padding:7px;border-radius:17px}.composer input{flex:1;border:0;background:transparent;padding:13px;font-size:14px;outline:none}.send{border:0;background:var(--a);color:#fff;border-radius:12px;padding:0 20px;font-weight:850}.foot{text-align:center;color:#929bae;font-size:11px;margin-top:8px}@media(max-width:820px){.side{display:none}.main{padding:20px 14px}.head{font-size:30px}.quick{grid-template-columns:repeat(2,1fr)}.top{align-items:flex-start}.auth{flex-direction:column}}@media(max-width:480px){.quick{grid-template-columns:1fr}}</style></head><body><div class="app"><aside class="side"><div class="brand">Schedule<span>AI</span></div><div class="nav active">✦ &nbsp;Assistant</div><div class="nav">▣ &nbsp;30-Day Schedule</div><div class="nav">◷ &nbsp;Agent Tools</div><div class="tip"><b>Try asking</b><br>What do I have tomorrow?<br>Am I free Friday afternoon?<br>Add a meeting tomorrow at 3 PM.<br>Move my meeting from 2 PM to 4 PM.<br>What is today's date?</div></aside><main class="main"><div class="top"><div><div class="eyebrow">Agentic Schedule Assistant</div><div class="head">Your schedule, understood.</div><div class="sub">RAG + Google Calendar + voice commands.</div></div><div class="auth"><button class="voice" onclick="voice()">🎙 Voice</button><a class="google" href="/auth/google">G&nbsp; Connect Google Calendar</a></div></div><div class="quick"><button onclick="ask('What do I have today?')"><span class="qtitle">📅 Today</span><span class="qdesc">See today’s events</span></button><button onclick="ask('What do I have tomorrow?')"><span class="qtitle">Tomorrow</span><span class="qdesc">Plan the next day</span></button><button onclick="ask('Am I free Friday afternoon?')"><span class="qtitle">Free time</span><span class="qdesc">Check availability</span></button><button onclick="ask('Show my meetings')"><span class="qtitle">Meetings</span><span class="qdesc">Find scheduled meetings</span></button></div><div id="chat" class="chat"><div class="welcome"><div class="avatar">AI</div><div><strong>Hi! I’m ScheduleAI.</strong><span>I will not create an event until the required details are collected. Say <b>“add schedule”</b> to start the guided flow.</span></div></div></div><div class="composer"><input id="q" placeholder="Ask: What do I have tomorrow?"><button class="voice" onclick="voice()">🎙</button><button class="send" onclick="send()">Send</button></div><div class="foot">RAG schedule retrieval • Guided event updates • 30-day planning</div></main></div><script>function ask(t){q.value=t;send()}function add(t,c){let x=document.createElement('div');x.className='msg '+c;x.textContent=t;chat.appendChild(x);chat.scrollTop=chat.scrollHeight;return x}function esc(s){return String(s).replace(/[&<>\"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}[m]))}function t12(t){let[h,m]=t.split(':');h=+h;return(h%12||12)+':'+m+' '+(h<12?'AM':'PM')}function card(a){let x=document.createElement('div');x.className='msg bot';let w=document.createElement('div');w.className='events';a.forEach(e=>{let r=document.createElement('div');r.className='event';let d=new Date(e.date+'T00:00:00');r.innerHTML='<div class="datebox">'+d.toLocaleDateString('en-US',{month:'short'})+'<br>'+d.getDate()+'</div><div class="eventmain"><div class="eventtitle">'+esc(e.title)+'</div><div class="meta">'+e.date+' • '+t12(e.time)+' • '+e.duration+' min</div></div><div class="tag">'+esc(e.type)+'</div>';w.appendChild(r)});x.appendChild(w);chat.appendChild(x);chat.scrollTop=chat.scrollHeight}async function send(){let v=q.value.trim();if(!v)return;q.value='';add(v,'user');let b=add('Thinking…','bot');try{let r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:v})});let d=await r.json();b.remove();if(Array.isArray(d.data)&&d.data.length)card(d.data);else add(d.answer||'No answer returned.','bot')}catch(e){b.remove();add('Unable to reach the assistant.','bot')}}q.addEventListener('keydown',e=>{if(e.key==='Enter')send()});function voice(){const S=window.SpeechRecognition||window.webkitSpeechRecognition;if(!S){add('Voice input is not supported. Try Chrome or Edge.','bot');return}const r=new S();r.lang='en-IN';r.onresult=e=>{q.value=e.results[0][0].transcript;send()};r.start()}</script></body></html>'''
@app.get('/',response_class=HTMLResponse)
def home():return PAGE
@app.post('/chat')
def chat(c:Chat,request:Request):
 try:
  tool,result=agent(request,c.message);return {'answer':answer(tool,result),'tool':tool,'data':result}
 except Exception as e:return {'answer':'I could not complete that request. Please try again.','error':str(e)}
@app.get('/health')
def health():return {'status':'ok','events':col.count(),'tools':['get_schedule','update_schedule'],'google_calendar':bool(GID and GSECRET)}
