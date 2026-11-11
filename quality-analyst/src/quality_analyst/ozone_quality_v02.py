"""Free qualitative hypotheses with exact quotes and dated source comparisons.

No QA taxonomy, training, semantic oracle or promotion of a technically valid
citation into a confirmed advisor error. V/0.1 remains unchanged.
"""
from pathlib import Path
from datetime import datetime,timezone
from typing import Literal,Annotated
from urllib.parse import urlsplit
import re
from quality_analyst.release_config import settings
from pydantic import Field
from quality_analyst import ozone_quality as old,ozone_quality_contracts as base
from quality_analyst.contracts import strict_json_loads,Text
from quality_analyst.ollama import GenerationConfig,INSTRUCT_MODEL,compact,sha256,AdapterError
from quality_analyst.ozone_live.parse import money
from quality_analyst.ozone_live.normalize import normalize_name
from quality_analyst.occupancy import compare_occupancy

VERSION='qa.ozone_live_quality/0.2.0'
IMPLEMENTATION_VERSION=VERSION
ROOT=Path(__file__).resolve().parents[2]
CONTEXT=GenerationConfig(num_ctx=settings().ollama_num_ctx if settings().hardware_mode=='cpu' else 8192,num_predict=settings().ollama_num_predict if settings().hardware_mode=='cpu' else 2048,think=False,model=INSTRUCT_MODEL)
ANALYSIS=GenerationConfig(num_ctx=settings().ollama_num_ctx if settings().hardware_mode=='cpu' else 8192,num_predict=settings().ollama_num_predict if settings().hardware_mode=='cpu' else 2048,think=False,model=INSTRUCT_MODEL)
CONTEXT_PROMPT=ROOT/'prompts/ozone_context_v0.2.txt'
ANALYSIS_PROMPT=ROOT/'prompts/ozone_quality_v0.2.txt'


def models(c,comparison_ids=()):
    make,limit,literals=base.make,base.bounded,base.literals
    short=Annotated[Text,Field(max_length=140)];sentence=Annotated[Text,Field(max_length=300)]
    quote=make('SourceQuote',message_id=literals([m.message_id for m in c.messages]),
        role=Literal[tuple(sorted({m.role.value for m in c.messages}))],quote=Annotated[Text,Field(max_length=400)])
    field=make('TravelCriterion',field=Literal[base.FIELDS],value=short,owner=Literal['customer_request','advisor_claim'],citation=quote)
    claim=make('QuotedClaim',property=Literal['price_total','price_from','availability','booking','board','service','other'],
        value=short,hotel_name=short|None,currency=short|None,citation=quote)
    context=make('TravelContextV02',fields=limit(field,14),claims=limit(claim,4),missing_information=limit(short,5))
    finding=make('QualitativeHypothesis',title=short,description=sentence,consequence=short,
        basis=Literal['conversation','internal_inconsistency','ozone_difference'],citations=limit(quote,4,1),
        comparison_ids=limit(literals(comparison_ids),2 if comparison_ids else 0),recommendation=sentence)
    analysis=make('FreeReportV02',needs=sentence,handling=sentence,weaknesses=limit(finding,4),unverifiable=limit(short,5))
    return context,analysis


def check_context(parsed,c):
    check,checks=old.quote_checker(c);fields=[];claims=[]
    for index,field in enumerate(parsed.fields):
        exact=check(field.citation,'field/'+str(index));issues=[]
        if field.value not in field.citation.quote:issues.append('VALUE_NOT_EXACTLY_SOURCED')
        if field.citation.role!=('customer' if field.owner=='customer_request' else 'advisor'):issues.append('CRITERION_ROLE_MISMATCH')
        fields.append({'field':field.model_dump(mode='json'),'accepted':exact and not issues,'issues':issues})
    for index,claim in enumerate(parsed.claims):
        exact=check(claim.citation,'claim/'+str(index));issues=[]
        if claim.citation.role!='advisor':issues.append('ADVISOR_ROLE_REQUIRED')
        for key in ['value','hotel_name','currency']:
            value=getattr(claim,key)
            if value and value not in claim.citation.quote:issues.append(key.upper()+'_NOT_EXACTLY_SOURCED')
        if exact and not issues:claims.append(claim.model_dump(mode='json'))
    return {'fields':fields,'claims':claims,'checks':checks,'missing_information':parsed.missing_information,'semantic_verified':False}


def source_issues(retrieval):
    if retrieval is None or retrieval.origin!='live':return ['NO_LIVE_SOURCE']
    spec=retrieval.request_parameters;issues=[]
    if spec is None or retrieval.fetched_at is None:return ['INCOMPLETE_SOURCE_PROVENANCE']
    if urlsplit(retrieval.source_url or '').hostname!='ozonevoyages.tn':issues.append('UNAPPROVED_SOURCE_HOST')
    if (datetime.now(timezone.utc)-retrieval.fetched_at).total_seconds()<0:issues.append('FUTURE_SOURCE_TIMESTAMP')
    by_id={e.evidence_id:e for e in retrieval.evidence}
    for offer in retrieval.offers:
        if offer.hotel_id!='path:'+urlsplit(offer.detail_url or '').path:issues.append('HOTEL_PATH_IDENTITY_MISMATCH')
        if offer.search_fingerprint!=spec.fingerprint:issues.append('SEARCH_FINGERPRINT_MISMATCH')
        if offer.currency not in {None,'TND','EUR','USD'}:issues.append('UNRECOGNIZED_CURRENCY')
        for key in offer.evidence_ids:
            e=by_id.get(key)
            if e is None or e.source_url!=retrieval.source_url or e.fetched_at!=retrieval.fetched_at or e.request_fingerprint!=spec.fingerprint:
                issues.append('FIELD_PROVENANCE_MISMATCH')
        contexts=[by_id[k] for k in offer.evidence_ids if k in by_id and by_id[k].field=='request_context']
        expected={'check_in':spec.check_in.isoformat(),'check_out':spec.check_out.isoformat(),'rooms':[r.model_dump() for r in spec.rooms]}
        if not contexts or any(strict_json_loads(e.observed)!=expected for e in contexts):issues.append('OCCUPANCY_OR_DATES_PROOF_MISMATCH')
    return sorted(set(issues))


def comparisons(c,context,retrieval):
    rows=[];issues=source_issues(retrieval)
    spec=retrieval.request_parameters if retrieval else None
    age=(datetime.now(timezone.utc)-retrieval.fetched_at).total_seconds() if retrieval and retrieval.fetched_at else None
    transcript_dates=set(re.findall(r'\b20\d\d-\d\d-\d\d\b',' '.join(m.text for m in c.messages)))
    mismatched=bool(spec and transcript_dates-{spec.check_in.isoformat(),spec.check_out.isoformat()})
    for claim in (context or {}).get('claims',[]):
        row={'comparison_id':'C-'+sha256(compact(claim).encode())[:12],'claim':claim,'status':'unverifiable','reason':'NO_COMPARABLE_SOURCE',
            'evidence_ids':[],'historical_error_proven':False,'parameters_comparable':False,'descriptive_only':True,
            'fetched_at':retrieval.fetched_at.isoformat() if retrieval and retrieval.fetched_at else None,
            'source_url':retrieval.source_url if retrieval else None,'source_age_seconds':age,
            'temporal_scope':'snapshot_at_retrieval_not_historical_advisor_truth','identity_status':None}
        rows.append(row)
        if issues:row['reason']=';'.join(issues);continue
        if age is None or age>settings().ozone_cache_ttl or retrieval.data_status=='stale':row['reason']='SOURCE_SNAPSHOT_STALE';continue
        if mismatched:row['reason']='TRANSCRIPT_AND_SEARCH_DATES_DIFFER_OR_AMBIGUOUS';continue
        if not {spec.check_in.isoformat(),spec.check_out.isoformat()}<=transcript_dates:
            row['reason']='TRANSCRIPT_AND_SEARCH_DATES_NOT_ESTABLISHED';continue
        name=normalize_name(claim.get('hotel_name') or '')
        matched=[o for o in retrieval.offers if name and normalize_name(o.hotel_name)==name and o.context_verified]
        if len(matched)!=1:row['reason']='HOTEL_NAME_NOT_UNIQUE_EXACT_MATCH';continue
        offer=matched[0];row.update(offer_id=offer.offer_id,identity_status='unique_exact_quoted_name_and_returned_path_not_global_identity')
        by_id={e.evidence_id:e for e in retrieval.evidence}
        row['occupancy_check']=compare_occupancy(c,spec.rooms,claim)
        row['occupancy_check']['source_evidence_ids']=[k for k in offer.evidence_ids if by_id[k].field=='request_context']
        if row['occupancy_check']['status']!='compatible':
            row['reason']=row['occupancy_check']['reason'];continue
        if claim['property']=='availability':
            value=normalize_name(claim['value'])
            if re.search(r'\b(pas|not|cannot|non)\b',normalize_name(claim['citation']['quote'])):
                row['reason']='NEGATED_OR_QUALIFIED_CLAIM_NOT_AUTOMATICALLY_COMPARABLE';continue
            affirmed='available' if value in {'disponible','available','confirmee','confirme','confirmed','disponibilite confirmee'} else 'on_request' if value in {'sur demande','on request'} else None
            if affirmed and offer.availability!='unknown':
                row.update(status='source_differs' if affirmed!=offer.availability else 'consistent_with_source',
                    reason='DISPLAYED_HOTEL_STATUS_COMPARISON_ONLY_NOT_ROOM_BOOKING_CONFIRMATION',
                    evidence_ids=[k for k in offer.evidence_ids if by_id[k].field in {'hotel_name','hotel_id','availability','request_context'}],
                    parameters_comparable=True,displayed_source_state=offer.availability)
            else:row['reason']='CLAIM_OR_SOURCE_STATUS_NOT_EXPLICIT'
        elif claim['property'] in {'price_total','price_from'}:
            row['reason']='HOTEL_SUMMARY_NOT_SELECTED_ROOM_BOARD_QUOTE'
            if offer.currency!=claim.get('currency'):row['reason']='CURRENCY_DIFFERENT_OR_UNSTATED'
            if offer.nights!=(spec.check_out-spec.check_in).days:row['reason']='NIGHTS_NOT_COMPARABLE'
        elif claim['property']=='booking':row['reason']='PRIVATE_RESERVATION_NOT_PUBLICLY_VERIFIABLE'
        elif claim['property']=='service':row['reason']='ABSENCE_FROM_SOURCE_NOT_PROOF_OF_SERVICE_ABSENCE'
        else:row['reason']='NO_REVIEWED_COMPARISON_FOR_PROPERTY'
    return rows


def projection(retrieval,c):
    if not retrieval:return None
    names=' '.join(m.text for m in c.messages).casefold()
    selected=[o for o in retrieval.offers if o.hotel_name.casefold() in names][:3]
    if not selected:selected=list(retrieval.offers[:3])
    ids={i for o in selected for i in o.evidence_ids}
    fields={'hotel_name','amount','availability'}
    return {'data_status':retrieval.data_status,'origin':retrieval.origin,'fetched_at':retrieval.fetched_at.isoformat() if retrieval.fetched_at else None,
        'source_url':retrieval.source_url,'request_parameters':retrieval.request_parameters.model_dump(mode='json') if retrieval.request_parameters else None,
        'errors':retrieval.errors,'source_validation_issues':source_issues(retrieval),
        'hotels':[o.model_dump(mode='json',include={'offer_id','hotel_id','hotel_name','amount','currency','price_basis','nights','availability','room_type','board','detail_url','price_scope'}) for o in selected],
        'evidence':[e.model_dump(mode='json',include={'evidence_id','field','observed'}) for e in retrieval.evidence if e.evidence_id in ids and e.field in fields]}


def admission(payload,config):return old.admission(payload,config)


OFFER_KEYS=('hotel_name','room','room_type','board','total_price_text','price_text','amount','currency','availability','status')


def turn_metadata(records,c):
    """Model-facing view of the advisor turn records.

    The full records repeat every message text plus routing identifiers, which
    alone exceeded the local context budget on ordinary 10-turn chats. The
    transcript already carries the text, so only decision metadata is kept and
    each turn is linked to its advisor message_id. Nothing in the transcript is
    shortened; the complete records stay in the saved review.
    """
    advisors=[m for m in c.messages if m.role.value=='advisor']
    rows=[]
    for i,record in enumerate(records or []):
        if not isinstance(record,dict):continue
        response=record.get('advisor_response') or {};decision=record.get('advisor_decision') or {}
        row={'turn':i+1}
        if len(advisors)==len(records):row['advisor_message_id']=advisors[i].message_id
        if response.get('reason'):row['reason']=response['reason']
        if response.get('status') not in (None,'','completed'):row['status']=response['status']
        if response.get('verified') is not None:row['verified']=response['verified']
        if response.get('retryable'):row['retryable']=True
        for key in ('intent','confidence','handoff_required'):
            if decision.get(key) not in (None,''):row[key]=decision[key]
        if decision.get('tools_used'):row['tools_used']=decision['tools_used']
        trip={k:v for k,v in (decision.get('trip') or {}).items() if v not in (None,'',[],{})}
        if trip:row['trip']=trip
        offers=[{k:o[k] for k in OFFER_KEYS if o.get(k) not in (None,'')} for o in (decision.get('offers') or [])[:5] if isinstance(o,dict)]
        if offers:row['offers']=offers;row['offer_count']=len(decision.get('offers') or [])
        if decision.get('sources'):row['source_count']=len(decision['sources'])
        rows.append(row)
    return rows


def prepare(c,*,stage='context',context=None,retrieval=None,generation_config=None):
    if len(c.messages)>24:raise AdapterError('TURN_LIMIT_NO_TRUNCATION')
    checks=comparisons(c,context,retrieval) if stage=='analysis' else []
    model=models(c,[r['comparison_id'] for r in checks])[0 if stage=='context' else 1]
    config=generation_config or (CONTEXT if stage=='context' else ANALYSIS)
    prompt=(CONTEXT_PROMPT if stage=='context' else ANALYSIS_PROMPT).read_text(encoding='utf-8')
    data={'conversation':{'language':c.language.value,'complete':c.conversation_complete,
        'occurred_at':c.occurred_at.isoformat() if c.occurred_at else None,
        'messages':[m.model_dump(mode='json',include={'message_id','role','sequence','text'}) for m in c.messages]}}
    if stage=='analysis':data.update(travel_context={'fields':[r['field'] for r in (context or {}).get('fields',[]) if r['accepted']],
        'claims':(context or {}).get('claims',[]),'missing':(context or {}).get('missing_information',[])},
        user_confirmed_criteria=(context or {}).get('user_criteria'),ozone_snapshot=projection(retrieval,c),factual_comparisons=checks)
    if stage=='analysis' and (context or {}).get('shared_review_records'):
        # Advisor-provided metadata is useful for review but remains untrusted
        # source data. It cannot promote an AI hypothesis into a confirmed
        # finding or alter the deterministic scorecard.
        data['advisor_turn_records']=turn_metadata((context or {}).get('shared_review_records'),c)
    def build():
        payload={**config.settings(),'prompt':prompt+'\nUNTRUSTED_SOURCE_DATA\n'+compact(data),'format':base.schema(model)}
        if settings().hardware_mode=='cpu':
            # Ollama 0.40.4 must reject overflowing prompts/context instead of dropping text.
            payload.update(truncate=False,shift=False)
        return payload
    payload=build()
    try:return payload,admission(payload,config),checks
    except AdapterError as error:
        # Turn metadata is optional context. Drop it whole rather than refuse the
        # review; the conversation itself is still never truncated.
        if error.code!='CONTEXT_LIMIT_NO_TRUNCATION' or 'advisor_turn_records' not in data:raise
    del data['advisor_turn_records'];payload=build()
    return payload,{**admission(payload,config),'advisor_turn_records':'omitted_for_context_budget'},checks


PRICE=re.compile(r'\d{1,3}(?:[ .]\d{3})*,\d{2,3}|\d+(?:\.\d{1,3})?(?=\s*(?:TND|DT|dinars?)\b)',re.I)
CONFLICT=re.compile(r'inconsisten|contradict|varian|conflict|incoh|diff[eé]r|mismatch|changed? price|prix diff',re.I)


def same_list_prices(w,c):
    """A price "inconsistency" whose prices all sit in one advisor message.

    Advisor rate lists show one hotel with several room/board rows, so two
    prices for the same hotel are normally different options, not a
    contradiction. Deterministic, so a small model's misreading is caught.
    """
    if w.basis!='internal_inconsistency' and not CONFLICT.search(w.title+' '+w.description):return False
    prices={p.replace(' ','').replace('.','') for p in PRICE.findall(w.title+' '+w.description+' '+' '.join(q.quote for q in w.citations))}
    if len(prices)<2:return False
    return any(prices<={p.replace(' ','').replace('.','') for p in PRICE.findall(m.text)} for m in c.messages if m.role.value=='advisor')


def analysis_check(parsed,c,checks):
    check,quotes=old.quote_checker(c);by_id={r['comparison_id']:r for r in checks};rows=[]
    for i,w in enumerate(parsed.weaknesses):
        issues=[];exact=[check(q,'weakness/'+str(i)) for q in w.citations]
        if not all(exact):issues.append('CITATION_REJECTED')
        advisors={q.message_id for q in w.citations if q.role=='advisor'}
        if not advisors:issues.append('ADVISOR_QUOTE_REQUIRED')
        seq={m.message_id:m.sequence for m in c.messages};order=[seq[q.message_id] for q in w.citations]
        if order!=sorted(order):issues.append('QUOTE_CHRONOLOGY_REJECTED')
        if w.basis=='internal_inconsistency' and len(advisors)<2:issues.append('TWO_ADVISOR_QUOTES_REQUIRED')
        if same_list_prices(w,c):issues.append('PRICES_ARE_OPTIONS_FROM_ONE_RATE_LIST')
        if w.basis=='ozone_difference':
            selected=[by_id[k] for k in w.comparison_ids]
            if not selected or not all(r['status']=='source_differs' and r['evidence_ids'] for r in selected):issues.append('DATED_COMPARABLE_SOURCE_DIFFERENCE_REQUIRED')
        elif w.comparison_ids:issues.append('SOURCE_LINK_WITH_WRONG_BASIS')
        rows.append({'weakness':w.model_dump(mode='json'),'status':'rejected' if issues else 'hypothesis_for_review',
            'issues':issues,'verification_state':'not_confirmed_semantically',
            'source_state':'dated_display_difference_only' if w.basis=='ozone_difference' else 'conversation_quotes_exact_only',
            'confirmed_advisor_error':False})
    return {'weaknesses':rows,'checks':quotes,'factual_checks':checks,'semantic_verified':False,
        'no_weakness_is_not_satisfaction':True,'unsupported_accusations_cannot_be_automatically_semantically_detected':True}


def validate(envelope,c,*,stage='context',comparisons=()):
    base_result={'status':'invalid','proposal':None,'verification':None,'semantic_verified':False}
    config=CONTEXT if stage=='context' else ANALYSIS
    if envelope.get('model')!=config.model:return {**base_result,'reason':'MODEL_IDENTITY_MISMATCH'}
    if envelope.get('done_reason')!='stop' or envelope.get('done') is not True:return {**base_result,'reason':'INCOMPLETE_RESPONSE'}
    if any(type(envelope.get(k)) is not int or envelope[k]<0 for k in ['prompt_eval_count','eval_count']):return {**base_result,'reason':'TOKEN_COUNTS_MISSING'}
    if envelope['prompt_eval_count']>config.input_token_budget or envelope['eval_count']>config.num_predict:return {**base_result,'reason':'TOKEN_BUDGET_EXCEEDED'}
    return validate_output(envelope.get('response'),c,stage=stage,comparisons=comparisons)


def validate_output(text,c,*,stage='context',comparisons=()):
    """Provider-neutral schema, quote and source verification; no transport identity."""
    base_result={'status':'invalid','proposal':None,'verification':None,'semantic_verified':False}
    try:
        strict_json_loads(text);parsed=models(c,[r['comparison_id'] for r in comparisons])[0 if stage=='context' else 1].model_validate_json(text)
    except (ValueError,KeyError,TypeError):return {**base_result,'reason':'V02_SCHEMA_INVALID'}
    verification=check_context(parsed,c) if stage=='context' else analysis_check(parsed,c,comparisons)
    return {**base_result,'status':'context_checked' if stage=='context' else 'free_quality_checked',
        'proposal':parsed.model_dump(mode='json'),'verification':verification}
