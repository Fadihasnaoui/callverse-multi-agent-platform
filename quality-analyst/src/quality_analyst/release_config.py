"""Validated release infrastructure settings. No network or writes on import."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
import http.client, json, logging, os, re

ROOT = Path(__file__).resolve().parents[2]

class ConfigurationError(ValueError): pass

def dotenv(path):
    """Small literal dotenv subset; no shell expansion or executable expressions."""
    values={}
    if not path.exists(): return values
    for index,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        line=line.strip()
        if not line or line.startswith('#'): continue
        if '=' not in line: raise ConfigurationError(f'.env ligne {index} : NAME=value requis')
        key,value=line.split('=',1);key=key.strip();value=value.strip()
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*',key) or key in values:
            raise ConfigurationError(f'.env ligne {index} : nom invalide ou doublon')
        if value.startswith('"'):
            try: value=json.loads(value)
            except ValueError: raise ConfigurationError(f'.env ligne {index} : guillemets invalides') from None
        elif value.startswith("'") and value.endswith("'"): value=value[1:-1]
        if not isinstance(value,str) or '\x00' in value or '\n' in value:
            raise ConfigurationError(f'.env ligne {index} : valeur littérale invalide')
        values[key]=value
    return values

@dataclass(frozen=True)
class Settings:
    ollama_model:str
    ollama_base_url:str
    ollama_mode:str
    hardware_mode:str
    streamlit_port:int
    ozone_base_url:str
    ozone_request_timeout:int
    ozone_cache_ttl:int
    app_env:str
    log_level:str
    data_dir:Path
    cpu_ram_preflight_gib:float
    cpu_ram_reserve_gib:float
    ollama_num_ctx:int
    ollama_num_predict:int
    ollama_generation_timeout:int

    ai_provider:str
    gemini_model:str
    gemini_request_timeout:int
    gemini_max_calls_per_day:int
    gemini_key_file:Path

    @property
    def generation_timeout(self):
        return self.ollama_generation_timeout if self.hardware_mode=='cpu' else 180

def load_settings(environ=None,env_file=None):
    values=dotenv(Path(env_file) if env_file else ROOT/'.env')
    if values.get('GEMINI_API_KEY') or values.get('GOOGLE_API_KEY'):
        raise ConfigurationError('Clé API interdite dans .env ; utiliser un fichier secret extérieur à la distribution')
    values.update(dict(os.environ if environ is None else environ))
    def get(key,default): return values.get(key,default)
    def integer(key,default,low,high):
        value=get(key,str(default))
        if not re.fullmatch(r'\d+',value) or not low<=int(value)<=high:
            raise ConfigurationError(f'{key} : entier entre {low} et {high} requis')
        return int(value)
    def gib(key,default,low,high):
        value=get(key,str(default))
        if not re.fullmatch(r'\d+(?:\.\d{1,3})?',value) or not low<=float(value)<=high:
            raise ConfigurationError(f'{key} : nombre entre {low} et {high} Gio requis')
        return float(value)
    def choice(key,default,choices):
        value=get(key,default)
        if value not in choices: raise ConfigurationError(f'{key} : choisir parmi '+', '.join(choices))
        return value
    mode=choice('OLLAMA_MODE','compose',('compose','host','local'))
    url=get('OLLAMA_BASE_URL','http://ollama:11434')
    try:
        u=urlsplit(url);port=u.port
        hosts={'compose':{'ollama'},'host':{'host.docker.internal'},'local':{'127.0.0.1','localhost','::1'}}[mode]
        if u.scheme!='http' or u.hostname not in hosts or u.username or u.password or u.path not in ('','/') or u.query or u.fragment or not port or not 1<=port<=65535:
            raise ValueError()
    except ValueError: raise ConfigurationError('OLLAMA_BASE_URL : URL HTTP sans identifiants, hôte autorisé et port explicite requis pour OLLAMA_MODE') from None
    # The integrated service uses the same local model as the Customer Advisor.
    model=get('OLLAMA_MODEL','qwen3.5:4b')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,149}',model):
        raise ConfigurationError('OLLAMA_MODEL : nom de modèle invalide ; identité Qwen épinglée toujours exigée')
    ozone=get('OZONE_BASE_URL','https://ozonevoyages.tn').rstrip('/')
    if ozone!='https://ozonevoyages.tn':
        raise ConfigurationError('OZONE_BASE_URL : seul https://ozonevoyages.tn est autorisé par le contrat observé')
    data=get('APP_DATA_DIR','runtime_data')
    if not data.strip() or '\x00' in data: raise ConfigurationError('APP_DATA_DIR : chemin requis')
    path=Path(data).expanduser()
    if not path.is_absolute(): path=ROOT/path
    path=path.resolve()
    if path==ROOT or any(path.is_relative_to(ROOT/p) for p in ('src','scripts','prompts')):
        raise ConfigurationError('APP_DATA_DIR : choisir un répertoire de données distinct du code')
    preflight=gib('CPU_RAM_PREFLIGHT_GIB',10,6,64)
    reserve=gib('CPU_RAM_RESERVE_GIB',4,2,16)
    ctx=integer('OLLAMA_NUM_CTX',8192,2048,8192)
    predict=integer('OLLAMA_NUM_PREDICT',2048,128,2048)
    if preflight<reserve+3.5:
        raise ConfigurationError('CPU_RAM_PREFLIGHT_GIB : réserve + au moins 3.5 Gio requis')
    if ctx-predict-256<1024:
        raise ConfigurationError('OLLAMA_NUM_CTX : prévoir num_predict + 256 + au moins 1024 tokens d’entrée')
    provider=choice('AI_PROVIDER','gemini',('gemini','ollama'))
    gemini_model=get('GEMINI_MODEL','gemini-3.5-flash-lite')
    if not re.fullmatch(r'gemini-[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}',gemini_model):
        raise ConfigurationError('GEMINI_MODEL : identifiant Gemini simple requis, sans URL')
    key_file=Path(get('GEMINI_API_KEY_FILE','/run/secrets/gemini_api_key'))
    return Settings(model,url.rstrip('/'),mode,choice('OLLAMA_HARDWARE_MODE','cpu',('cpu','gpu')),
        integer('STREAMLIT_PORT',8501,1024,65535),ozone,
        integer('OZONE_REQUEST_TIMEOUT',30,1,30),integer('OZONE_CACHE_TTL',60,1,60),
        choice('APP_ENV','production',('production','development')),
        choice('APP_LOG_LEVEL','INFO',('DEBUG','INFO','WARNING','ERROR')),path,preflight,reserve,ctx,predict,
        integer('OLLAMA_GENERATION_TIMEOUT',180,60,900),provider,gemini_model,
        integer('GEMINI_REQUEST_TIMEOUT',90,10,180),integer('GEMINI_MAX_CALLS_PER_DAY',20,1,100),key_file)

@lru_cache(maxsize=1)
def settings(): return load_settings()

def ollama_connection(timeout):
    u=urlsplit(settings().ollama_base_url)
    return http.client.HTTPConnection(u.hostname,u.port,timeout=timeout)

def configure_logging():
    # Do not log input, headers, cookies, model responses or config values.
    logging.basicConfig(level=settings().log_level,format='%(levelname)s %(name)s %(message)s')
    for name in ('urllib3','requests'): logging.getLogger(name).setLevel(logging.WARNING)
    for name in ('google','httpx','httpcore'): logging.getLogger(name).setLevel(logging.CRITICAL)
