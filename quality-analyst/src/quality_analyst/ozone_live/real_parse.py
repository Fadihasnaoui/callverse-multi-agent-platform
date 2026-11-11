"""Observed Ozone JSON/HTML summaries, Decimal amounts and individual provenance."""
from datetime import datetime
from typing import Annotated, Literal
from decimal import Decimal
import re
from urllib.parse import urljoin,urlsplit
from pydantic import Field
from bs4 import BeautifulSoup
from .contracts import Evidence
from .real_contracts import HotelSummary,SearchResult
from .access import AccessPolicy,safe_url,AccessError
from .parse import money
from quality_analyst.ollama import compact,sha256
from quality_analyst.contracts import strict_json_loads


def product_kind(row):
    explicit = row.get('data-product-type','').casefold()
    if explicit in {'hotel','accommodation'}: return 'hotel','explicit_product_type'
    labels = [n.get_text(' ',strip=True) for n in row.select('.type-room')]
    lodging = any(re.search(r'\b(chambre|double|single|twin|suite|bungalow|studio|appartement|room|triple|quadruple|standard|familiale)\b',s,re.I) for s in labels)
    activity = any(re.search(r'entr[ée]e simple|\bpass\b|carthage land|safari kids|cin[ée]ma|\bticket\b',s,re.I) for s in labels)
    if lodging and not activity:return 'hotel','lodging_room_labels'
    return ('non_hotel','activity_labels') if activity and not lodging else ('unknown','product_type_not_established')


def displayed_price(text):
    result = {'amount':None,'currency':None,'price_basis':'unknown','nights':None}
    matched = re.search(r'([\d\s.,]+)\s*(TND|EUR|USD)\b',text,re.I)
    if matched:
        result['amount'] = money(matched.group(1).strip());result['currency'] = matched.group(2).upper()
    total = re.search(r'Total pour\s+(\d+)\s+Nuits?',text,re.I)
    if re.search(r'\bpartir\b|\bfrom\b',text,re.I):result['price_basis']='starting_from'
    elif re.search(r'par nuit|per night',text,re.I):result['price_basis']='per_night'
    elif total:result.update(price_basis='stay_total',nights=int(total.group(1)))
    return result


def displayed_availability(text):
    available = re.search(r'\b([1-9]\d*)\s+offres?\s+Disponible\b',text,re.I)
    states = []
    if available:states.append('available')
    if re.search(r'\bSur demande\b|\bon request\b',text,re.I):states.append('on_request')
    if re.search(r'\bIndisponible\b|\bunavailable\b',text,re.I):states.append('unavailable')
    if re.search(r'\bstop[ -]sale\b',text,re.I):states.append('stop_sale')
    count = re.search(r'\b(\d+)\s+offres?\b',text,re.I)
    return (states[0] if len(states)==1 else 'unknown',int(count.group(1)) if count else None)


def parse_results(response,spec,*,context_verified,origin='live'):
    data = strict_json_loads(response.body)
    if not isinstance(data,dict) or not isinstance(data.get('html'),str):raise AccessError('RESULT_JSON_HTML_REQUIRED')
    if data.get('error') not in {None,False,0,'','0'}:raise AccessError('SOURCE_REPORTED_ERROR')
    soup = BeautifulSoup(data['html'],'lxml');offers=[];evidence=[];excluded=[]
    for index,row in enumerate(soup.select('.hotel_search_item')):
        title = row.select_one('.hotel_content_top_left h4')
        name = title.get_text(' ',strip=True) if title else ''
        kind,reason = product_kind(row)
        if kind!='hotel':
            if len(excluded)<12:excluded.append({'name':name[:160],'kind':kind,'reason':reason})
            continue
        if len(offers)>=12:break
        link = row.select_one('a[href*="/detail"]')
        if not name or len(name)>160 or link is None:continue
        detail = urljoin(response.url,link['href']);path = urlsplit(detail).path
        detail = safe_url(detail,AccessPolicy(approved_routes=(('GET',path),)))
        identity = 'path:'+path
        location = row.select_one('.hotel_content_top_left p')
        destination = location.get_text(' ',strip=True) if location else None
        top = row.select_one('.hotel_content_top_right')
        price_text = top.get_text(' ',strip=True) if top else ''
        summary = row.select_one('.hotel_item_inner_content')
        summary_text = summary.get_text(' ',strip=True) if summary else ''
        price = displayed_price(price_text)
        availability,count = displayed_availability(summary_text)
        stars = len(row.select('.hotel_content_top_left label i.fa-star')) or None
        if stars and stars>5:stars=None
        key = sha256((response.url+response.sha256+str(index)).encode())[:16]
        ids = []
        observed = {'hotel_name':name,'hotel_id':identity,'destination':destination,'stars':str(stars) if stars else None,
            'amount':price_text if price['amount'] is not None else None,
            'currency':price['currency'],'price_basis':price_text if price['price_basis']!='unknown' else None,
            'nights':price_text if price['nights'] else None,'availability':summary_text if availability!='unknown' else None,
            'offer_count':summary_text if count is not None else None,'detail_url':link['href'],
            'request_context':compact({'check_in':spec.check_in.isoformat(),'check_out':spec.check_out.isoformat(),'rooms':[r.model_dump() for r in spec.rooms]})}
        for field,text in observed.items():
            if text is None or not text:continue
            evidence_id='E-'+key+'-'+field;ids.append(evidence_id)
            evidence.append(Evidence(evidence_id=evidence_id,source_url=response.url,fetched_at=response.fetched_at,
                request_fingerprint=spec.fingerprint,field=field,observed=text[:2000],
                locator=f'JSON.html .hotel_search_item[{index}]; visible text normalization' if field!='request_context' else 'server-continuation-echo-and-sent-parameters',
                response_sha256=response.sha256,origin=origin))
        offers.append(HotelSummary(offer_id='H-'+key,hotel_id=identity,hotel_name=name,destination=destination,
            stars=stars,room_type=None,board=None,amount=price['amount'],currency=price['currency'],price_basis=price['price_basis'],
            nights=price['nights'],offer_count=count,availability=availability,bookable=None,conditions=['Displayed hotel summary; no room/board/provider offer selected.'],
            services=[],services_complete=False,detail_url=detail,search_fingerprint=spec.fingerprint,
            context_verified=context_verified and (price['nights'] is None or price['nights']==(spec.check_out-spec.check_in).days),
            evidence_ids=ids,price_scope='hotel_summary_not_selected_room_offer',product_type='hotel',
            identity_provenance='returned_detail_path_not_global_master_id'))
    count=data.get('NbResults')
    return offers,evidence,excluded,int(count) if isinstance(count,(str,int)) and str(count).isdigit() else None
