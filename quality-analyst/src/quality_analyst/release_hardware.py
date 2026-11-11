"""Portable preflight, cgroup-aware RAM watchdog and sampled NVIDIA telemetry."""
from datetime import datetime,timezone
from pathlib import Path
import csv, io, shutil, subprocess, threading, time
import psutil
from quality_analyst.contracts import strict_json_loads
from quality_analyst.release_config import settings,ollama_connection

def probe_ram():
    try:
        vm=psutil.virtual_memory();remaining=[vm.available];limits=[]
        for limit_path,current_path in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
            p=Path(limit_path)
            if not p.exists(): continue
            value=p.read_text().strip()
            if value=='max': continue
            cap=int(value);used=int(Path(current_path).read_text())
            if 0<cap<2**60: remaining.append(max(0,cap-used));limits.append(cap)
        return {'status':'ok','total_bytes':min([vm.total]+limits),'available_bytes':min(remaining),
                'scope':'host_and_visible_cgroup_minimum_not_remote_server_private_memory'}
    except (OSError,ValueError): return {'status':'unavailable','reason':'MEMORY_QUERY_FAILED'}

def probe_gpu():
    executable=shutil.which('nvidia-smi')
    if not executable: return {'status':'unavailable','reason':'NVIDIA_SMI_NOT_FOUND','devices':[]}
    try:
        p=subprocess.run([executable,'--query-gpu=name,memory.total,memory.used,memory.free,driver_version',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        if p.returncode: raise ValueError()
        devices=[]
        for row in csv.reader(io.StringIO(p.stdout),skipinitialspace=True):
            if len(row)!=5: raise ValueError()
            total,used,free=(int(v) for v in row[1:4])
            if min(total,used,free)<0: raise ValueError()
            devices.append({'name':row[0],'total_mib':total,'used_mib':used,'free_mib':free,'driver_version':row[4].strip()})
        if not devices: raise ValueError()
        return {'status':'ok','devices':devices}
    except (OSError,ValueError,subprocess.TimeoutExpired): return {'status':'unavailable','reason':'NVIDIA_QUERY_FAILED','devices':[]}

def probe_api(endpoint):
    if endpoint not in {'version','tags','ps'}: raise ValueError('METADATA_ENDPOINT_REFUSED')
    c=ollama_connection(5)
    try:
        c.request('GET','/api/'+endpoint,headers={'Accept':'application/json'})
        r=c.getresponse();data=r.read(131073)
        if r.status!=200 or len(data)>131072: raise ValueError()
        data=strict_json_loads(data)
        if not isinstance(data,dict): raise ValueError()
        # Metadata only. No headers, paths, model template or arbitrary data logged.
        if endpoint in {'tags','ps'}:
            data={'models':[{k:v for k,v in m.items() if k in {'name','digest','size','size_vram','expires_at'}} for m in data.get('models',[])]}
        else:data={'version':data.get('version')}
        return {'status':'ok','data':data}
    except (OSError,ValueError):return {'status':'unavailable','reason':'OLLAMA_METADATA_UNAVAILABLE'}
    finally:c.close()

def collect_snapshot():
    return {'system':{'ram':probe_ram(),'gpu':probe_gpu() if settings().hardware_mode=='gpu' else {'status':'not_requested','devices':[]}},
            'ollama':{'api':{key:probe_api(key) for key in ('version','tags','ps')}},
            'resource_policy':settings().hardware_mode}

def preflight(key='qwen'):
    from quality_analyst.ollama import INSTRUCT_DIGEST
    if key!='qwen':raise ValueError('ONLY_PINNED_QWEN_SUPPORTED')
    snap=collect_snapshot();errors=[];api=snap['ollama']['api'];s=settings()
    if not all(v['status']=='ok' for v in api.values()):errors.append('OLLAMA_API_UNAVAILABLE')
    models=[m for m in api['tags'].get('data',{}).get('models',[]) if m.get('name')==s.ollama_model]
    model=models[0] if len(models)==1 else None
    if model is None:errors.append('MODEL_UNAVAILABLE_INSTALL_MANUALLY')
    elif model.get('digest')!=INSTRUCT_DIGEST:errors.append('MODEL_DIGEST_MISMATCH')
    # The pinned model may already be loaded by the Ozone advisor (Amira uses
    # the same local qwen): sharing those weights costs no extra memory and
    # Ollama queues the requests. Only a different loaded model blocks the run.
    loaded=api['ps'].get('data',{}).get('models')
    if loaded is None or any(m.get('name')!=s.ollama_model for m in loaded):errors.append('MODEL_NOT_CONFIRMED_IDLE')
    ram=snap['system']['ram'];minimum=3.5
    if s.hardware_mode=='cpu':
        # Advisor Qwen 3.5 is locally pinned by tag digest. Its exact KV layout
        # is not exposed by Ollama 0.33.3, so this is a conservative estimate.
        size=(model or {}).get('size')
        if type(size) is not int or size<=0:
            errors.append('MODEL_MEMORY_SIZE_UNAVAILABLE');size=2497293803
        kv_bytes=36*8*128*2*2*s.ollama_num_ctx
        planned=size+kv_bytes+1024**3
        minimum=max(s.cpu_ram_preflight_gib,planned/1024**3+s.cpu_ram_reserve_gib)
        snap['cpu_memory_plan']={'weights_bytes':size,'kv_estimate_bytes':kv_bytes,
            'workspace_allowance_bytes':1024**3,'reserve_gib':s.cpu_ram_reserve_gib,
            'effective_preflight_gib':minimum,'estimate_is_hard_limit':False}
        if api['version'].get('data',{}).get('version')!='0.33.3':
            errors.append('CPU_OLLAMA_VERSION_NOT_VALIDATED_FOR_NO_TRUNCATION')
    if ram.get('status')!='ok' or ram.get('available_bytes',0)<minimum*1024**3:
        errors.append(f'RAM_HEADROOM_BELOW_{minimum}_GIB_{s.hardware_mode.upper()}')
    if s.hardware_mode=='gpu' and not any(d['free_mib']>=6144 for d in snap['system']['gpu'].get('devices',[])):
        errors.append('VRAM_HEADROOM_BELOW_6144_MIB_OR_NOT_OBSERVABLE')
    return snap,errors,model

# Compatibility object: callers retain hardware.preflight.probe_* API.
class Probes: pass
probes=Probes()
for name in ('probe_ram','probe_gpu','probe_api','collect_snapshot'):setattr(probes,name,globals()[name])

class Sampler:
    def __init__(self):self.samples=[];self.stop=threading.Event();self.thread=threading.Thread(target=self.collect,daemon=True)
    def collect(self):
        while not self.stop.is_set() :
            self.samples.append({'observed_at':datetime.now(timezone.utc).isoformat(),'ram':probe_ram(),
                                 'gpu':probe_gpu() if settings().hardware_mode=='gpu' else {'status':'not_requested','devices':[]},'loaded_models':probe_api('ps')})
            self.stop.wait(1)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=12)


def confirm_idle(seconds=10):
    """Observe unloading after either success or cancellation; never generate."""
    deadline=time.monotonic()+seconds
    while True:
        state=probe_api('ps')
        if state.get('status')=='ok' and state.get('data',{}).get('models')==[]:
            return state,True
        if time.monotonic()>=deadline:return state,False
        time.sleep(.2)
