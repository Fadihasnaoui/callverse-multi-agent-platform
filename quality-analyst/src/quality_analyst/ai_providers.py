"""Small provider boundary; business prompts and verification stay shared."""
from typing import Protocol
from quality_analyst.ollama import AdapterError

class AIProvider(Protocol):
    name:str
    def execute(self,c,directory,*,stage='context',context=None,retrieval=None,
                cloud_consent=False,free_tier_confirmed=False): ...

class OllamaProvider:
    name='ollama'
    def execute(self,c,directory,*,stage='context',context=None,retrieval=None,**kwargs):
        from quality_analyst.ozone_runtime_v02 import execute
        return execute(c,directory,stage=stage,context=context,retrieval=retrieval)

class GeminiProvider:
    name='gemini'
    def execute(self,c,directory,*,stage='context',context=None,retrieval=None,
                cloud_consent=False,free_tier_confirmed=False):
        from quality_analyst.gemini_runtime import execute
        return execute(c,directory,stage=stage,context=context,retrieval=retrieval,
            cloud_consent=cloud_consent,free_tier_confirmed=free_tier_confirmed)

def get_provider(name)->AIProvider:
    if name=='ollama':return OllamaProvider()
    if name=='gemini':return GeminiProvider()
    raise AdapterError('AI_PROVIDER_REFUSED')
