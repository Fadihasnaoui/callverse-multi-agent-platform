"""Live summary contracts load without requests, bs4, Streamlit or Ollama access."""
from typing import Annotated,Literal
from pydantic import Field,model_validator
from .contracts import Offer,Retrieval,Short

class HotelSummary(Offer):
    nights:Annotated[int,Field(ge=1,le=60)]|None
    offer_count:Annotated[int,Field(ge=0)]|None
    price_scope:Literal['hotel_summary_not_selected_room_offer']
    product_type:Literal['hotel']
    identity_provenance:Literal['returned_detail_path_not_global_master_id']

class SearchResult(Retrieval):
    offers:Annotated[list[HotelSummary],Field(max_length=12)]
    reported_results:int|None
    excluded_products:list[dict]
    duration_seconds:float
    contract_id:Literal['ozone.public-form.observed/0.2.0']
    search_session_id:Short|None
    nights:int|None

    @model_validator(mode='after')
    def check_snapshot_fields(self):
        by_id={e.evidence_id:e for e in self.evidence}
        if self.origin in {'live','fixture'} and self.request_parameters:
            expected=(self.request_parameters.check_out-self.request_parameters.check_in).days
            if self.nights!=expected:raise ValueError('SNAPSHOT_NIGHTS_MISMATCH')
            for offer in self.offers:
                if offer.context_verified and offer.nights is not None and offer.nights!=expected:
                    raise ValueError('VERIFIED_OFFER_NIGHTS_MISMATCH')
                for key in offer.evidence_ids:
                    e=by_id[key]
                    if e.source_url!=self.source_url or e.fetched_at!=self.fetched_at:
                        raise ValueError('SNAPSHOT_FIELD_SOURCE_OR_TIME_MISMATCH')
        return self
