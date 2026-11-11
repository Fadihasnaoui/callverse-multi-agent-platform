"""Conservative, transcript-only occupancy guard. No AI or manual-form evidence.

Supported declarative FR/EN counts, explicit zero children, child ages, labelled
rooms and literal Room JSON. Unsupported/ambiguous language fails closed.
"""
import json
import re
import unicodedata
from .ozone_live.contracts import Room

WORDS={'zero':0,'no':0,'without':0,'aucun':0,'aucune':0,'sans':0,'un':1,'une':1,'one':1,
    'deux':2,'two':2,'trois':3,'three':3,'quatre':4,'four':4,'cinq':5,'five':5,
    'six':6,'sept':7,'seven':7,'huit':8,'eight':8,'neuf':9,'nine':9,
    'dix':10,'ten':10,'onze':11,'eleven':11,'douze':12,'twelve':12}
NUMBER=r'(?:\d+|'+ '|'.join(WORDS)+r')'
TOKENS={'adults':r'adultes?|adults?', 'children':r'enfants?|children|child|kids?',
    'rooms':r'chambres?|rooms?'}
CORRECTION=re.compile(r'^(?:correction\b|je corrige\b|finalement\b|mise a jour\b|update\b|actually\b|instead\b|nous remplacons\b|we replace\b)')
UNCERTAIN=re.compile(r'\?|\b(?:si|if|maybe|perhaps|hypothetically|hypothetique|peut-etre|pourrait|would|environ|about|entre|between|ou|or|pas|not|ancien|previous|au moins|au plus|plus de|moins de|at least|at most|more than|less than|up to|maximum|minimum)\b')


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c))


def number(text):
    return int(text) if text.isdigit() else WORDS[text]


def valid_rooms(value):
    if not isinstance(value,list) or not 1<=len(value)<=4:
        raise ValueError('ROOM_LIST_REQUIRED')
    for room in value:
        if not isinstance(room,dict) or set(room)!={'adults','child_ages'}:
            raise ValueError('COMPLETE_ROOM_REQUIRED')
        if type(room['adults']) is not int or not isinstance(room['child_ages'],list) or any(type(a) is not int for a in room['child_ages']):
            raise ValueError('INTEGER_ROOM_VALUES_REQUIRED')
    result=[Room.model_validate(room).model_dump() for room in value]
    for room in result:room['child_ages']=sorted(room['child_ages'])
    return result


def flat(text):
    facts={};mentioned=set();ambiguous=False;starts=[]
    if re.search(r'(?:[+-]\d+(?:[.,]\d+)?|\d+[.,]\d+)\s+(?:adultes?|adults?|enfants?|children|child|chambres?|rooms?|ans|years?)\b',text):
        ambiguous=True
    for key,token in TOKENS.items():
        if re.search(r'\b(?:'+token+r')\b',text):mentioned.add(key)
        matches=list(re.finditer(r'\b('+NUMBER+r')\s+(?:'+token+r')\b',text))
        starts.extend(m.start() for m in matches)
        values={number(match[1]) for match in matches}
        if len(values)>1:ambiguous=True
        elif values:
            value=values.pop()
            limit=(1,6) if key=='adults' else (0,4) if key=='children' else (1,4)
            if not limit[0]<=value<=limit[1]:ambiguous=True
            else:facts[key]=value
    if re.search(r'\b(?:baby|babies|bebe|bebes)\b',text):
        mentioned.add('children');ambiguous=True
    ages=re.search(r'\b(?:enfants?|children|child|kids?)\s+(?:de|ages?|aged)\s+('+NUMBER+r'(?:\s*(?:,|et|and|&)\s*'+NUMBER+r')*)\s*(?:ans|years?(?:\s+old)?)\b',text)
    if ages:
        values=[number(x) for x in re.findall(NUMBER,ages.group(1))]
        if all(0<=age<=17 for age in values):facts['child_ages']=sorted(values)
        else:ambiguous=True
    if facts.get('children')==0:
        if ages:ambiguous=True
        facts['child_ages']=[]
    elif 'child_ages' in facts and len(facts['child_ages'])!=facts.get('children'):
        ambiguous=True
    if starts:
        allowed=set(WORDS)|{'adulte','adultes','adult','adults','enfant','enfants','child','children','kid','kids',
            'chambre','chambres','room','rooms','et','and','de','aged','age','ages','ans','year','years','old',
            'pour','for','avec','with','aux','dates','demandees','requested','these','same','stay'}
        # Do not interpret a recognised count inside an unsupported qualification.
        if set(re.findall(r'[a-z]+',text[min(starts):]))-allowed:ambiguous=True
    return facts,mentioned,ambiguous


def parse(text,*,customer_request=False):
    """Return partial explicit facts; never fill absent children with zero."""
    text=normalized(text)
    correction=bool(CORRECTION.match(text.strip()))
    # Narrow explicit replacement grammar: old values remain in the transcript.
    replacement=re.match(r'^(?:nous remplacons|we replace)\b.+?\b(?:par|with)\s+(.+)$',text.strip())
    if replacement:text=replacement.group(1)
    if not re.search(r'\b(?:adultes?|adults?|enfants?|children|child|kids?|chambres?|rooms?|bebe|baby)\b',text) and '[' not in text:
        return {},set(),False,correction
    if customer_request and not correction:
        leading=r'^(?:je\s+(?:cherche|souhaite|veux|demande|voyage|reserve)\b|nous\s+(?:sommes|serons|voyageons|souhaitons|demandons)\b|i\s+(?:am|need|want|request|travel)\b|we\s+(?:are|will|need|want|request|travel)\b|repartition\b|occupation\b|chambre\s*\d+\s*:|room\s*\d+\s*:|\[|'+NUMBER+r'\s+(?:adultes?|adults?|chambres?|rooms?|enfants?|children|child)\b)'
        if not re.match(leading,text.strip()) or re.search(r'\b(?:mon voisin|my neighbour|my neighbor|citation|quote|exemple|example|a dit|said)\b',text):
            return {},set(TOKENS),True,False
    if UNCERTAIN.search(text):return {},set(TOKENS),True,correction
    if '[' in text:
        try:
            fragment=text[text.index('['):]
            value,end=json.JSONDecoder().raw_decode(fragment)
            if not re.fullmatch(r'[\s.;]*',fragment[end:]):raise ValueError('AMBIGUOUS_ROOM_JSON_SUFFIX')
            if re.search(r'\b'+NUMBER+r'\s+(?:adultes?|adults?|enfants?|children|child)\b',text[:text.index('[')]):
                raise ValueError('AMBIGUOUS_ROOM_JSON_PREFIX')
            rooms=valid_rooms(value)
            return {'distribution':rooms,'rooms':len(rooms)},set(TOKENS),False,correction
        except (ValueError,TypeError,KeyError):
            return {},set(TOKENS),True,correction
    labels=list(re.finditer(r'\b(?:chambre|room)\s*(\d+)\s*:',text))
    if labels:
        if [int(m[1]) for m in labels]!=list(range(1,len(labels)+1)):
            return {},set(TOKENS),True,correction
        rooms=[]
        for i,match in enumerate(labels):
            chunk=text[match.end():labels[i+1].start() if i+1<len(labels) else len(text)]
            facts,_,bad=flat(chunk)
            if bad or not {'adults','children','child_ages'}<=facts.keys():
                return {},set(TOKENS),True,correction
            rooms.append({'adults':facts['adults'],'child_ages':facts['child_ages']})
        try:rooms=valid_rooms(rooms)
        except ValueError:return {},set(TOKENS),True,correction
        header,_,bad=flat(text[:labels[0].start()])
        if bad or ('rooms' in header and header['rooms']!=len(rooms)):
            return {},set(TOKENS),True,correction
        return {'distribution':rooms,'rooms':len(rooms)},set(TOKENS),False,correction
    facts,mentioned,bad=flat(text)
    return facts,mentioned,bad,correction


def compare_occupancy(conversation,search_rooms,claim=None):
    """Three states, with message provenance. Context/user_criteria is not read."""
    try:search=valid_rooms([room.model_dump() if hasattr(room,'model_dump') else room for room in search_rooms])
    except (ValueError,TypeError,KeyError):
        return {'status':'not_established','reason':'OCCUPANCY_NOT_ESTABLISHED',
            'conversation_rooms':None,'search_rooms':None,'evidence':[],
            'issues':['SEARCH_OCCUPANCY_INVALID'],'differences':[]}
    state={};conflicts=set();evidence=[];last_change=0
    for message in conversation.messages:
        if message.role.value!='customer':continue
        facts,mentioned,bad,correction=parse(message.text,customer_request=True)
        if not mentioned:continue
        evidence.append({'message_id':message.message_id,'role':'customer','sequence':message.sequence,
            'fields':sorted(mentioned),'explicit_correction':correction})
        if bad:
            conflicts.add('ambiguous_occupancy');last_change=message.sequence;continue
        complete='distribution' in facts or {'adults','children','child_ages','rooms'}<=facts.keys() and facts.get('rooms')==1
        if correction:
            if complete:
                conflicts.discard('ambiguous_occupancy')
            for key in mentioned:
                if key not in facts:
                    state.pop(key,None);conflicts.add(key)
            if complete:
                state={};conflicts=set()
            elif 'distribution' in state:
                state={};conflicts={'distribution'}
        if not correction and ('distribution' in state or 'distribution' in facts):
            distribution=state.get('distribution') or facts['distribution']
            flat_values={'adults':sum(r['adults'] for r in distribution),
                'children':sum(len(r['child_ages']) for r in distribution),
                'child_ages':sorted(a for r in distribution for a in r['child_ages']),
                'rooms':len(distribution)}
            other=facts if 'distribution' in state else state
            for key,value in flat_values.items():
                if key in other and other[key]!=value:conflicts.add(key)
        if 'distribution' not in facts:
            for key in mentioned-facts.keys():
                conflicts.add(key)
        for key,value in facts.items():
            if key in state and state[key]!=value and not correction:conflicts.add(key)
            if state.get(key)!=value:last_change=message.sequence
            state[key]=value
            if correction:conflicts.discard(key)
        # A changed child count invalidates a previous age list, never reuses it.
        if 'children' in facts and 'child_ages' not in facts:
            state.pop('child_ages',None)
    detail={'status':'not_established','reason':'OCCUPANCY_NOT_ESTABLISHED',
        'conversation_rooms':None,'conversation_partial':state,'search_rooms':search,
        'provenance':'explicit_customer_transcript_only_not_manual_search_form',
        'evidence':evidence,'issues':sorted(conflicts),'differences':[]}
    if conflicts:return detail
    claim_message=None
    if claim:
        cite=claim.get('citation') or {}
        claim_message=next((m for m in conversation.messages if m.message_id==cite.get('message_id')),None)
        if claim_message is None or claim_message.sequence<last_change:
            detail['issues']=['CLAIM_PREDATES_ACTIVE_OCCUPANCY_OR_MESSAGE_UNKNOWN'];return detail
        quoted,mentioned,bad,_=parse(cite.get('quote',''))
        if bad:
            detail['issues']=['CLAIM_OCCUPANCY_AMBIGUOUS'];return detail
        for key,value in quoted.items():
            if key in state and state[key]!=value:
                detail['issues']=['CLAIM_OCCUPANCY_CONFLICTS_WITH_CUSTOMER'];return detail
    differences=[]
    if 'rooms' in state and state['rooms']!=len(search):differences.append('room_count')
    distribution=state.get('distribution')
    if distribution is not None:
        detail['conversation_rooms']=distribution
        if distribution!=search:differences.append('per_room_distribution')
    else:
        if 'adults' in state and state['adults']!=sum(r['adults'] for r in search):differences.append('adult_count')
        if 'children' in state and state['children']!=sum(len(r['child_ages']) for r in search):differences.append('child_count')
        if 'child_ages' in state and state['child_ages']!=sorted(age for room in search for age in room['child_ages']):differences.append('child_ages')
        if state.get('rooms')==1 and {'adults','children','child_ages'}<=state.keys():
            distribution=[{'adults':state['adults'],'child_ages':state['child_ages']}]
            detail['conversation_rooms']=distribution
    if differences:
        detail.update(status='different',reason='OCCUPANCY_MISMATCH',differences=differences)
    elif distribution is not None:
        detail.update(status='compatible',reason='OCCUPANCY_CONFIRMED_COMPATIBLE')
    else:detail['issues']=['COMPLETE_PER_ROOM_OCCUPANCY_NOT_EXPLICIT']
    return detail
