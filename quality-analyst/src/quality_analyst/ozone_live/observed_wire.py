"""Decode the observed Ozone form/JS subset as data, never execute JavaScript."""
import json
import re
from .access import AccessError


class LiteralReader:
    def __init__(self, text): self.text, self.i = text, 0
    def whitespace(self):
        while self.i < len(self.text) and self.text[self.i].isspace(): self.i += 1
    def take(self, token):
        self.whitespace()
        if not self.text.startswith(token, self.i): raise AccessError('CONTINUATION_LITERAL_EXPECTED')
        self.i += len(token)
    def string(self):
        self.whitespace(); quote = self.text[self.i]; self.i += 1; result = []
        while self.i < len(self.text):
            c = self.text[self.i]; self.i += 1
            if c == quote: return ''.join(result)
            if c == '\\':
                c = self.text[self.i]; self.i += 1
                if c == 'u':
                    raw = self.text[self.i:self.i+4]
                    if not re.fullmatch('[0-9a-fA-F]{4}', raw): raise AccessError('JS_ESCAPE_REFUSED')
                    result.append(chr(int(raw,16))); self.i += 4
                elif c in {'n','r','t','b','f','\\',"'",'"','/'}:
                    result.append({'n':'\n','r':'\r','t':'\t','b':'\b','f':'\f'}.get(c,c))
                else: raise AccessError('JS_ESCAPE_REFUSED')
            else: result.append(c)
        raise AccessError('UNTERMINATED_JS_STRING')
    def value(self):
        self.whitespace()
        if self.i >= len(self.text): raise AccessError('CONTINUATION_VALUE_MISSING')
        c = self.text[self.i]
        if c in {'"',"'"}: return self.string()
        if self.text.startswith('JSON.stringify', self.i):
            self.take('JSON.stringify'); self.take('('); value = self.value(); self.take(')')
            return json.dumps(value, ensure_ascii=False, separators=(',',':'))
        if c == '{':
            self.i += 1; result = {}; self.whitespace()
            while not self.text.startswith('}', self.i):
                self.whitespace()
                if self.text[self.i] in {'"',"'"}: key = self.string()
                else:
                    m = re.match(r'[A-Za-z_$][\w$]*', self.text[self.i:])
                    if not m: raise AccessError('JS_KEY_REFUSED')
                    key = m.group(); self.i += len(key)
                if key in result: raise AccessError('DUPLICATE_CONTINUATION_FIELD')
                self.take(':'); result[key] = self.value(); self.whitespace()
                if not self.text.startswith(',',self.i): break
                self.i += 1; self.whitespace()
            self.take('}'); return result
        if c == '[':
            self.i += 1; result = []; self.whitespace()
            while not self.text.startswith(']',self.i):
                result.append(self.value()); self.whitespace()
                if not self.text.startswith(',',self.i): break
                self.i += 1; self.whitespace()
            self.take(']'); return result
        for literal, value in [('true',True),('false',False),('null',None)]:
            if self.text.startswith(literal,self.i): self.i += len(literal); return value
        match = re.match(r'-?(?:0|[1-9]\d*)\b', self.text[self.i:])
        if match: self.i += len(match.group()); return int(match.group())
        raise AccessError('CONTINUATION_NON_LITERAL_VALUE')


def ajax_parameters(html, route='/recherchehotel/SearchResult'):
    """Only the exact list-view $.post; not map, pagination or arbitrary code."""
    matches = list(re.finditer(r'\$\.post\(\s*([\'"])'+re.escape(route)+r'\1\s*,',html))
    if len(matches) != 1: raise AccessError('UNIQUE_OBSERVED_RESULT_POST_REQUIRED')
    reader = LiteralReader(html[matches[0].end():]); value = reader.value()
    reader.whitespace()
    if not isinstance(value,dict) or not reader.text.startswith(',',reader.i):
        raise AccessError('OBSERVED_AJAX_OBJECT_REQUIRED')
    return value


def controls(form):
    result = {}
    for node in form.find_all(['input','select']):
        name = node.get('name')
        if not name or node.has_attr('disabled') or node.get('type') in {'button','submit','file','password'}: continue
        if node.get('type') in {'radio','checkbox'} and not node.has_attr('checked'): continue
        if name in result: raise AccessError('DUPLICATE_FORM_FIELD_REQUIRES_REVIEW')
        if node.name == 'select':
            selected = node.find('option',selected=True) or node.find('option')
            result[name] = selected.get('value','') if selected else ''
        else: result[name] = node.get('value','')
    return result


def occupation_echo(parameters, spec):
    try:
        if parameters['du'] != spec.check_in.isoformat() or parameters['au'] != spec.check_out.isoformat():
            raise AccessError('SERVER_DATES_DIFFER_FROM_REQUEST')
        if int(parameters['nbChambre']) != len(spec.rooms): raise AccessError('SERVER_ROOM_COUNT_MISMATCH')
        rooms = json.loads(parameters['params_chambre'])
        if set(rooms) != {str(i) for i in range(1,len(spec.rooms)+1)}: raise AccessError('SERVER_ROOMS_MISMATCH')
        for i, requested in enumerate(spec.rooms,1):
            echoed = rooms[str(i)]
            if int(echoed['nb_adlt']) != requested.adults or int(echoed['nb_enf']) != len(requested.child_ages):
                raise AccessError('SERVER_OCCUPANCY_MISMATCH')
            ages = []
            def flatten(value):
                if isinstance(value,dict):
                    for v in value.values(): flatten(v)
                elif isinstance(value,list):
                    for v in value: flatten(v)
                else: ages.append(int(value))
            for key,value in echoed.items():
                if 'age' in key.lower(): flatten(value)
            if ages != requested.child_ages: raise AccessError('SERVER_CHILD_AGE_ECHO_UNVERIFIED')
    except (ValueError,TypeError,KeyError): raise AccessError('SERVER_OCCUPANCY_ECHO_UNVERIFIED') from None
    return True
