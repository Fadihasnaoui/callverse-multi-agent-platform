import threading,time
from quality_analyst.release_config import settings
class MemoryCache:
    def __init__(self,clock=time.monotonic):self.clock=clock;self.lock=threading.Lock();self.items={}
    def get(self,key):
        with self.lock:
            value=self.items.get(key)
            if not value:return None
            at,result=value;age=self.clock()-at
            if age>settings().ozone_cache_ttl:return None
            return result.model_copy(deep=True,update={'data_status':'cached' if age<=settings().ozone_cache_ttl else 'stale','cache_hit':True,'cache_age':float(age)})
    def put(self,key,value):
        with self.lock:
            if len(self.items)>=8:self.items.pop(next(iter(self.items)))
            self.items[key]=(self.clock(),value.model_copy(deep=True))
