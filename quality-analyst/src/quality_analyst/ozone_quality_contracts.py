"""Versioned source-bound context and free-form report; no ten-label classification."""
from typing import Annotated,Literal
from pydantic import Field,create_model
from quality_analyst.contracts import ContractModel,Text,MessageRole

Short=Annotated[Text,Field(max_length=180)]
Sentence=Annotated[Text,Field(max_length=420)]
FIELDS=('destination','destination_id','hotel','hotel_id','check_in','check_out','rooms','adults','children','child_ages','room_type','board','budget','services')


def bounded(item,n,minimum=0):return Annotated[list[item],Field(min_length=minimum,max_length=n)]
def make(name,**fields):return create_model(name,__base__=ContractModel,**{k:(v,...) for k,v in fields.items()})
def literals(ids):return Literal[tuple(ids)] if ids else str


def models(c,comparison_ids=()):
    ids=[m.message_id for m in c.messages]
    citation=make('ConversationQuote',message_id=literals(ids),role=Literal[tuple(sorted({m.role.value for m in c.messages}))],
        quote=Annotated[Text,Field(max_length=400)])
    slot=make('TravelField',field=Literal[FIELDS],value=Short,owner=Literal['customer_request','advisor_claim'],citation=citation)
    claim=make('AdvisorClaim',property=Literal['price_total','price_from','board','service','availability','booking','classification','other'],
        value=Short,hotel_id=Short|None,currency=Short|None,citation=citation)
    context=make('TravelContext',schema_version=Literal['qa.ozone-context/0.1.0'],
        fields=bounded(slot,16),advisor_claims=bounded(claim,6),missing_information=bounded(Short,8))
    weakness=make('Weakness',observation=Sentence,impact=Short,mode=Literal['observed','hypothesis'],
        basis=Literal['conversation','internal_inconsistency','external_comparison','unverifiable'],
        citations=bounded(citation,4,1),comparison_ids=bounded(literals(comparison_ids),4 if comparison_ids else 0),
        recommendation=Sentence)
    analysis=make('FreeFormQuality',schema_version=Literal['qa.ozone-free-quality/0.1.0'],needs_summary=Sentence,
        handling_summary=Sentence,reviewed_message_ids=bounded(literals(ids),len(ids)),weaknesses=bounded(weakness,6),
        factual_comparison_ids=bounded(literals(comparison_ids),len(comparison_ids)),limitations=bounded(Short,6))
    return context,analysis


def schema(model):
    def trim(x):
        if isinstance(x,dict):
            # Strip schema annotations, preserving business fields with the same
            # names (e.g. QualitativeHypothesis.title and .description).
            result={}
            for key,value in x.items():
                if key in {'title','default','description'}:continue
                if key in {'properties','$defs','definitions','patternProperties'}:
                    result[key]={name:trim(child) for name,child in value.items()}
                else:result[key]=trim(value)
            return result
        if isinstance(x,list):return [trim(v) for v in x]
        return x
    return trim(model.model_json_schema())
