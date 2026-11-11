from decimal import Decimal, InvalidOperation
import re
def money(text):
    if isinstance(text,Decimal):return text if text.is_finite() and text>=0 else None
    if isinstance(text,int) and not isinstance(text,bool):return Decimal(text) if text>=0 else None
    if not isinstance(text,str):return None  # A binary float is never silently accepted as exact money.
    raw=re.sub(r'[\s\u00a0\u202f]','',text)
    if not re.fullmatch(r'\d+(?:[.,]\d+)?',raw):return None
    try:return Decimal(raw.replace(',','.'))
    except InvalidOperation:return None
