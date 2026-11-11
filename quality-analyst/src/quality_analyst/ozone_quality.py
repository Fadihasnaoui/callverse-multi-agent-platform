from quality_analyst.ollama import compact,AdapterError
from quality_analyst.quote_resolver import QuoteProposal,QuoteResolver
from quality_analyst.evidence import EvidenceVerifier
from quality_analyst.input import MessageRegistry
def admission(payload,config):
    pb=len(payload['prompt'].encode());sb=len(compact(payload['format']).encode());estimate=(pb+sb+2)//3+256
    if pb>20000 or sb>13000 or estimate>config.input_token_budget:raise AdapterError('CONTEXT_LIMIT_NO_TRUNCATION')
    return {'estimated_tokens':estimate,'input_budget':config.input_token_budget,'proven_upper_bound':False,'prompt_bytes':pb,'schema_bytes':sb}

def quote_checker(c):
    resolver=QuoteResolver(EvidenceVerifier(MessageRegistry(c)));checks=[]
    def check(q,owner):
        result=resolver.resolve(QuoteProposal.model_validate_json(compact(q.model_dump(mode='json')))).to_dict();result['owner']=owner;checks.append(result)
        return result['status']=='accepted'
    return check,checks
