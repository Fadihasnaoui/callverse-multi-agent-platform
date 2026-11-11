"""Name similarity suggests candidates, never identity or offer merging."""
import re,unicodedata
from difflib import SequenceMatcher


def normalize_name(text):
    text=''.join(c for c in unicodedata.normalize('NFKD',text) if not unicodedata.combining(c))
    return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',text.casefold())).strip()


def candidates(name,items):
    key=normalize_name(name)
    return sorted([{'candidate':item,'similarity':SequenceMatcher(None,key,normalize_name(item['name'])).ratio(),
        'identity_confirmed':False} for item in items],key=lambda r:r['similarity'],reverse=True)
