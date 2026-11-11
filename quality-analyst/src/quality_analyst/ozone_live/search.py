"""One user-triggered public search, derived from the observed current form.

No scheduled/bulk acquisition, booking, general access permission claim, JS
execution or retry. Historical V policy/client are left unchanged.
"""
from datetime import date,datetime,timezone
from pathlib import Path
from uuid import uuid4
import json,re,threading,time
from urllib.parse import urljoin,urlsplit
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from .access import AccessPolicy,AccessError,safe_url,require_public_dns
from .client import Response
from .contracts import SearchSpec
from .real_contracts import SearchResult
from .real_parse import parse_results
from .observed_wire import controls,ajax_parameters,occupation_echo
from .normalize import normalize_name
from .adapter import MemoryCache
from quality_analyst.release_config import settings
from quality_analyst.ollama import compact,sha256
from quality_analyst.contracts import strict_json_loads

ROOT=Path(__file__).resolve().parents[3]
SESSIONS=settings().data_dir/'ozone_search_sessions'
BASE=settings().ozone_base_url+'/'
ROUTES=(('GET','/'),('GET','/recherchehotel/autocomplete'),('POST','/recherchehotel/searchRoom'),
        ('POST','/recherchehotel/SearchResult'),('POST','/recherchehotel/addRoomHome'),('POST','/recherchehotel/getAgeEnfantPanel'))
_LOAD=threading.Lock();_CACHE=MemoryCache();_RATE=threading.Lock();_LAST=[0.0]
SECRET=re.compile(r'csrf|token|sessionkey|cookie|password|authorization|api.key|nonce',re.I)

def redact(data):
    if isinstance(data,dict):return {k:'[not logged]' if SECRET.search(str(k)) else redact(v) for k,v in data.items()}
    if isinstance(data,list):return [redact(v) for v in data]
    return data

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def make_spec(destination,check_in,check_out,rooms):
    values={'destination':destination,'check_in':check_in.isoformat() if isinstance(check_in,date) else check_in,
        'check_out':check_out.isoformat() if isinstance(check_out,date) else check_out,
        'rooms':[r.model_dump() if hasattr(r,'model_dump') else r for r in rooms],
        'destination_id':None,'hotel_id':None,'hotel_name':None,'room_type':None,'board':None}
    values['travellers']=sum(r['adults']+len(r['child_ages']) for r in values['rooms'])
    spec=SearchSpec.model_validate_json(compact(values))
    if spec.check_in<=date.today():raise ValueError('FUTURE_CHECK_IN_REQUIRED')
    return spec


class SessionClient:
    def __init__(self,folder,*,cancel=None,session=None,resolver=require_public_dns):
        self.folder=folder;self.cancel=cancel or threading.Event();self.resolver=resolver;self.trace=[]
        self.session=session or requests.Session();self.session.trust_env=False
        self.session.headers.update({'User-Agent':'QualityAnalyst/0.2 (user-initiated public hotel search)',
            'Accept':'text/html,application/json;q=0.9','Accept-Encoding':'identity','Accept-Language':'fr,en;q=0.8'})
        self.session.mount('https://',HTTPAdapter(max_retries=Retry(total=0,redirect=0)))
        self.deadline=time.monotonic()+120;self.stopped=False;self.posted=set()
    def check(self):
        if self.cancel.is_set():raise AccessError('SEARCH_CANCELLED')
        if self.stopped:raise AccessError('SEARCH_STOPPED_AFTER_ACCESS_OR_HTTP_FAILURE')
        if len(self.trace)>=10 or time.monotonic()>self.deadline:raise AccessError('HTTP_SEARCH_BUDGET_EXHAUSTED')
    def request(self,method,path,*,data=None,multipart=False,ajax=False):
        self.check();target=safe_url(path,AccessPolicy(approved_routes=ROUTES),method=method)
        key=(method,urlsplit(target).path,compact(redact(data)))
        if method=='POST' and key in self.posted:raise AccessError('DUPLICATE_POST_REFUSED')
        if method=='POST':self.posted.add(key)
        with _RATE:
            delay=max(0,1-(time.monotonic()-_LAST[0]))
            if self.cancel.wait(delay):raise AccessError('SEARCH_CANCELLED')
            _LAST[0]=time.monotonic()
        self.check();self.resolver(urlsplit(target).hostname);self.check()
        row={'attempt':len(self.trace)+1,'method':method,'url':target,'data':redact(data),
             'started_at':datetime.now(timezone.utc).isoformat(),'encoding':'multipart/form-data' if multipart else 'form-urlencoded' if data else None}
        self.trace.append(row);save(self.folder/f"{row['attempt']:02d}_request.json",row)
        response=None
        try:
            remaining=self.deadline-time.monotonic()
            headers={'X-Requested-With':'XMLHttpRequest','Referer':BASE+'recherchehotel/searchRoom'} if ajax else {}
            args={'files':{k:(None,str(v)) for k,v in data.items()}} if multipart else {'data':data}
            response=self.session.request(method,target,timeout=(min(5,remaining),min(settings().ozone_request_timeout,remaining)),
                allow_redirects=False,stream=True,headers=headers,**args)
            row.update(http_status=response.status_code,content_type=response.headers.get('Content-Type','').split(';')[0],
                       session_cookie_count=len(self.session.cookies))
            if response.status_code in {401,403,407,429}:self.stopped=True;raise AccessError('ACCESS_REFUSED_NO_RETRY')
            # The observed path returned HTTP 200 at every stage. A changed redirect
            # contract is not followed toward login, a new host or another method.
            if response.status_code!=200:self.stopped=True;raise AccessError('HTTP_STATUS_'+str(response.status_code))
            if int(response.headers.get('Content-Length','0') or 0)>2*1024**2:raise AccessError('HTTP_BODY_LIMIT')
            body=bytearray()
            for chunk in response.iter_content(8192):
                if self.cancel.is_set():raise AccessError('SEARCH_CANCELLED')
                if time.monotonic()>self.deadline:raise AccessError('HTTP_SEARCH_DEADLINE')
                body.extend(chunk)
                if len(body)>2*1024**2:raise AccessError('HTTP_BODY_LIMIT')
            raw=bytes(body)
            if re.search(rb'cf-chl-|just a moment|verify you are human|captcha challenge',raw,re.I):
                self.stopped=True;raise AccessError('ACCESS_CHALLENGE_STOP')
            result=Response(target,raw,row['content_type'],datetime.now(timezone.utc))
            row.update(bytes_read=len(raw),response_sha256=result.sha256);return result
        except requests.RequestException as exc:
            row['error']=type(exc).__name__;self.stopped=True;raise AccessError('HTTP_TRANSPORT_FAILURE_NO_RETRY') from None
        except AccessError as exc:row['error']=str(exc);raise
        finally:
            if response is not None:response.close()
            row['ended_at']=datetime.now(timezone.utc).isoformat();save(self.folder/f"{row['attempt']:02d}_response.json",row)
    def close(self):self.session.close()


def resolve_destination(rows,destination):
    if not isinstance(rows,list):raise AccessError('AUTOCOMPLETE_ARRAY_REQUIRED')
    target=normalize_name(destination)
    candidates=[r for r in rows if isinstance(r,dict) and all(k in r for k in ['label','category','id'])
        and r['category']=='VILLE' and (normalize_name(r['label'])==target or normalize_name(r['label'].split(',')[0])==target)]
    if len(candidates)!=1:raise AccessError('DESTINATION_NOT_FOUND_OR_AMBIGUOUS')
    return candidates[0]


def bind_form(home,client,spec,selection):
    soup=BeautifulSoup(home.body,'lxml')
    matches=[f for f in soup.find_all('form') if urlsplit(urljoin(home.url,f.get('action',''))).path=='/recherchehotel/searchRoom']
    if len(matches)!=1 or matches[0].get('method','GET').upper()!='POST':raise AccessError('OBSERVED_POST_FORM_REQUIRED')
    form=matches[0];values=controls(form);text=home.body.decode('utf-8','replace')
    required={'hotel','du','au','date','adulte1','enfants1','nbChambre','termsearch','typesearch','idsearch'}
    if not required<=values.keys():raise AccessError('FORM_FIELDS_CHANGED')
    resolved=selection['id']
    if resolved in {'',None}:
        if not re.search(r'\$\([\'"]#idsearch[\'"]\)\.val\(ui\.item\.id\s*\|\|\s*0\)',text):
            raise AccessError('EMPTY_ID_WITHOUT_OBSERVED_FRONTEND_FALLBACK')
        resolved=0
    for index,room in enumerate(spec.rooms,1):
        if index>1:
            if '/recherchehotel/addRoomHome' not in text or not re.search(r"data:\s*'index='\s*\+\s*nb_chambre",text):
                raise AccessError('EXTRA_ROOM_WIRE_NOT_OBSERVED')
            part=client.request('POST','/recherchehotel/addRoomHome',data={'index':index-1},ajax=True)
            fragment=strict_json_loads(part.body)
            values.update(controls(BeautifulSoup(fragment['html'],'lxml')))
        for name,value in [(f'adulte{index}',room.adults),(f'enfants{index}',len(room.child_ages))]:
            if name not in values:raise AccessError('OCCUPANCY_CONTROL_NOT_RETURNED')
            values[name]=str(value)
        if room.child_ages:
            if '/recherchehotel/getAgeEnfantPanel' not in text or 'nb_enfant=' not in text:raise AccessError('CHILD_AGE_PANEL_CONTRACT_NOT_OBSERVED')
            panel=client.request('POST','/recherchehotel/getAgeEnfantPanel',data={'nb_enfant':len(room.child_ages),'index':index},ajax=True)
            fragment=strict_json_loads(panel.body);age_form=BeautifulSoup(fragment['html'],'lxml')
            nodes=age_form.select('select[name]')
            if len(nodes)!=len(room.child_ages):raise AccessError('AGE_CONTROL_COUNT_MISMATCH')
            for node,age in zip(nodes,room.child_ages):
                options={o.get('value') for o in node.find_all('option')}
                if str(age) not in options:raise AccessError('AGE_NOT_ALLOWED_BY_RETURNED_CONTROL')
                if node['name'] in values:raise AccessError('DUPLICATE_AGE_CONTROL')
                values[node['name']]=str(age)
    values.update(hotel=selection['label'],termsearch=selection['label'],typesearch=selection['category'],idsearch=str(resolved),
        du=spec.check_in.strftime('%d/%m/%Y'),au=spec.check_out.strftime('%d/%m/%Y'),
        date=spec.check_in.strftime('%d/%m/%Y')+' - '+spec.check_out.strftime('%d/%m/%Y'),nbChambre=str(len(spec.rooms)))
    return values,form.get('enctype')=='multipart/form-data'


def empty(spec,code,*,trace=(),session_id=None,seconds=0):
    return SearchResult(schema_version='qa.ozone-retrieval/0.1.0',data_status='unverifiable',origin='none',fetched_at=None,
        source_url=None,request_parameters=spec,cache_hit=False,cache_age=None,offers=[],evidence=[],destination_candidates=[],
        errors=[code],authorization_status='public_user_initiated_scope_only',trace=list(trace),reported_results=None,
        excluded_products=[],duration_seconds=float(seconds),contract_id='ozone.public-form.observed/0.2.0',search_session_id=session_id,
        nights=(spec.check_out-spec.check_in).days if spec else None)


def search_hotels(destination,check_in,check_out,rooms,*,force_live=False,cancel=None,session_root=SESSIONS,client_factory=SessionClient):
    """Explicit adult/child ages in every room; no destination/date/occupancy defaults."""
    started=time.monotonic();spec=None
    try:spec=make_spec(destination,check_in,check_out,rooms)
    except (ValueError,TypeError,KeyError):return empty(None,'SEARCH_PARAMETERS_MISSING_OR_INVALID')
    if not force_live:
        cached=_CACHE.get(spec.fingerprint)
        if cached and cached.data_status=='cached':return cached
    if not _LOAD.acquire(blocking=False):return empty(spec,'HTTP_SEARCH_ALREADY_RUNNING')
    folder=Path(session_root)/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ_')+uuid4().hex[:8])
    folder.mkdir(parents=True,exist_ok=False);client=None;result=None
    try:
        save(folder/'reservation.json',{'request_parameters':spec.model_dump(mode='json'),'maximum_http_attempts':10,
            'maximum_search_submissions':1,'automatic_retries':0,'scope':'public_user_triggered_search_only','cookies_persisted':False})
        client=client_factory(folder,cancel=cancel)
        home=client.request('GET',BASE)
        if '/recherchehotel/autocomplete' not in home.body.decode('utf-8','replace'):raise AccessError('AUTOCOMPLETE_ROUTE_NOT_OBSERVED')
        auto=client.request('GET','/recherchehotel/autocomplete');rows=strict_json_loads(auto.body)
        selection=resolve_destination(rows,spec.destination);save(folder/'resolved_destination.json',redact(selection))
        form,multipart=bind_form(home,client,spec,selection)
        page=client.request('POST','/recherchehotel/searchRoom',data=form,multipart=multipart)
        parameters=ajax_parameters(page.body.decode('utf-8','replace'));occupation_echo(parameters,spec)
        if parameters.get('termsearch')!=selection['label'] or parameters.get('typesearch')!=selection['category']:
            raise AccessError('SERVER_DESTINATION_SCOPE_MISMATCH')
        save(folder/'continuation_binding.json',{'source_sha256':page.sha256,'source_url':page.url,
            'parameters':redact(parameters),'occupancy_echo_verified':True,'same_session':True})
        response=client.request('POST','/recherchehotel/SearchResult',data=parameters,ajax=True)
        if 'json' not in response.content_type:raise AccessError('RESULT_CONTENT_TYPE_NOT_JSON')
        offers,evidence,excluded,count=parse_results(response,spec,context_verified=True)
        result=SearchResult(schema_version='qa.ozone-retrieval/0.1.0',data_status='fresh_live',origin='live',
            fetched_at=response.fetched_at,source_url=response.url,request_parameters=spec,cache_hit=False,cache_age=0.0,
            offers=offers,evidence=evidence,destination_candidates=[redact(selection)],errors=[] if offers else ['NO_VERIFIED_HOTEL_SUMMARY_NOT_UNAVAILABILITY'],
            authorization_status='public_user_initiated_scope_only',trace=client.trace,reported_results=count,excluded_products=excluded,
            duration_seconds=float(time.monotonic()-started),contract_id='ozone.public-form.observed/0.2.0',search_session_id=folder.name,
            nights=(spec.check_out-spec.check_in).days)
        if offers:_CACHE.put(spec.fingerprint,result)
    except AccessError as exc:result=empty(spec,str(exc),trace=client.trace if client else [],session_id=folder.name,seconds=time.monotonic()-started)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as exc:
        result=empty(spec,'CONTRACT_OR_PARSE_ERROR_'+type(exc).__name__,trace=client.trace if client else [],session_id=folder.name,seconds=time.monotonic()-started)
    finally:
        if client:client.close()
        _LOAD.release()
        if result:save(folder/'result.json',result.model_dump(mode='json'))
    return result
