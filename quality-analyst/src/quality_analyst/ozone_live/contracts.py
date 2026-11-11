"""Exact money, complete occupancy and provenance; no historical QA categories."""
from datetime import date,datetime
from decimal import Decimal
from typing import Annotated,Literal
from pydantic import Field,model_validator,field_validator
from quality_analyst.contracts import ContractModel,Text,AwareDatetime
from quality_analyst.ollama import compact,sha256

Short=Annotated[Text,Field(max_length=160)]
Availability=Literal['available','unavailable','on_request','stop_sale','unknown']
DataStatus=Literal['fresh_live','cached','stale','unavailable','unverifiable']
TTL_SECONDS={'price':120,'availability':60,'bookable':60,'services':86400,'classification':86400,'identity':604800}


class Room(ContractModel):
    adults:Annotated[int,Field(ge=1,le=6)]
    child_ages:Annotated[list[Annotated[int,Field(ge=0,le=17)]],Field(max_length=4)]


class SearchSpec(ContractModel):
    destination:Short
    destination_id:Short|None
    check_in:date
    check_out:date
    rooms:Annotated[list[Room],Field(min_length=1,max_length=4)]
    travellers:Annotated[int,Field(ge=1,le=24)]
    hotel_id:Short|None
    hotel_name:Short|None
    room_type:Short|None
    board:Short|None

    @model_validator(mode='after')
    def consistent(self):
        if not 0<(self.check_out-self.check_in).days<=60:raise ValueError('LOCAL_STAY_LIMIT_1_TO_60_NIGHTS')
        if self.travellers!=sum(r.adults+len(r.child_ages) for r in self.rooms):raise ValueError('TRAVELLER_DISTRIBUTION_MISMATCH')
        return self

    @property
    def fingerprint(self):return sha256(compact(self.model_dump(mode='json')).encode())


class Evidence(ContractModel):
    evidence_id:Short
    source_url:Text
    fetched_at:AwareDatetime
    request_fingerprint:Annotated[str,Field(pattern='^[a-f0-9]{64}$')]
    field:Short
    observed:Annotated[Text,Field(max_length=2000)]
    locator:Short
    response_sha256:Annotated[str,Field(pattern='^[a-f0-9]{64}$')]
    origin:Literal['live','fixture']


class Offer(ContractModel):
    offer_id:Short
    hotel_id:Short|None
    hotel_name:Short
    destination:Short|None
    stars:Annotated[int,Field(ge=0,le=5)]|None
    room_type:Short|None
    board:Short|None
    amount:Annotated[Decimal,Field(ge=0,allow_inf_nan=False)]|None
    currency:Short|None
    price_basis:Literal['stay_total','per_night','starting_from','unknown']
    availability:Availability
    bookable:bool|None
    conditions:list[Text]
    services:list[Short]
    services_complete:bool
    detail_url:Text|None
    search_fingerprint:Annotated[str,Field(pattern='^[a-f0-9]{64}$')]
    context_verified:bool
    evidence_ids:list[Short]


class Retrieval(ContractModel):
    schema_version:Literal['qa.ozone-retrieval/0.1.0']
    data_status:DataStatus
    origin:Literal['live','fixture','none']
    fetched_at:AwareDatetime|None
    source_url:Text|None
    request_parameters:SearchSpec|None
    cache_hit:bool
    cache_age:Annotated[float,Field(ge=0,allow_inf_nan=False)]|None
    offers:Annotated[list[Offer],Field(max_length=12)]
    evidence:Annotated[list[Evidence],Field(max_length=160)]
    destination_candidates:list[dict]
    errors:list[Short]
    authorization_status:Short
    trace:Annotated[list[dict],Field(max_length=16)]

    @model_validator(mode='after')
    def source_consistency(self):
        ids=[e.evidence_id for e in self.evidence]
        if len(ids)!=len(set(ids)):raise ValueError('DUPLICATE_EVIDENCE')
        fp=self.request_parameters.fingerprint if self.request_parameters else None
        for offer in self.offers:
            if offer.search_fingerprint!=fp or any(e not in ids for e in offer.evidence_ids):raise ValueError('OFFER_PROVENANCE_MISMATCH')
        if any(e.request_fingerprint!=fp or e.origin!=self.origin for e in self.evidence):raise ValueError('EVIDENCE_PROVENANCE_MISMATCH')
        if self.data_status=='fresh_live' and (self.origin!='live' or not self.fetched_at):raise ValueError('LIVE_ORIGIN_REQUIRED')
        if self.origin=='fixture' and self.data_status=='fresh_live':raise ValueError('FIXTURE_IS_NEVER_LIVE')
        return self


def unavailable(spec=None,code='LIVE_ACCESS_NOT_AUTHORIZED',*,authorization='unestablished'):
    return Retrieval(schema_version='qa.ozone-retrieval/0.1.0',data_status='unverifiable',origin='none',fetched_at=None,
        source_url=None,request_parameters=spec,cache_hit=False,cache_age=None,offers=[],evidence=[],destination_candidates=[],
        errors=[code],authorization_status=authorization,trace=[])
