import hashlib,json,re
from quality_analyst.contracts import load_conversation_json
from quality_analyst.ollama import compact
from quality_analyst import release_hardware as hardware
ROLE_NAMES = {"client":"customer", "customer":"customer", "c":"customer",
              "conseiller":"advisor", "advisor":"advisor", "a":"advisor"}

SECRET_KEYS = {"password", "passwd", "mot_de_passe", "api_key", "apikey", "access_token", "refresh_token", "authorization", "private_key", "secret", "gemini_api_key", "google_api_key"}

SECRET_PATTERN = re.compile(
    r"AIza[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b|"
    r"\bBearer\s+\S{12,}|"
    r"\b(?:password|passwd|mot[ _]de[ _]passe|api[ _-]?key|access[ _]token|secret)\s*[:=]\s*\S+",
    re.IGNORECASE)

class InputError(ValueError):
    """Only static error codes/messages, never raw input or validation exceptions."""

def ensure_no_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in SECRET_KEYS or SECRET_PATTERN.search(str(key)):
                raise InputError("SECRET_DETECTED")
            ensure_no_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value: ensure_no_secrets(item)
    elif isinstance(value, str) and SECRET_PATTERN.search(value):
        raise InputError("SECRET_DETECTED")

def stable_conversation(messages, language="fr", complete=True):
    if language not in ("fr", "en", "ar", "aeb", "mixed", "unknown") or type(complete) is not bool or not messages:
        raise InputError("INVALID_CONVERSATION")
    normalized=[]
    for row in messages:
        if not isinstance(row,dict) or set(row)!={"role","text"} or not isinstance(row["role"],str): raise InputError("INVALID_MESSAGE")
        role=ROLE_NAMES.get(row["role"].casefold().strip())
        text=row["text"]
        if role is None or not isinstance(text,str) or not text.strip(): raise InputError("INVALID_MESSAGE")
        ensure_no_secrets(text)
        normalized.append({"role":role,"text":text})
    digest=hashlib.sha256(compact({"messages":normalized,"language":language,"complete":complete}).encode()).hexdigest()[:16]
    cid="DEMO-"+digest
    payload={"schema_version":"qa.input/0.1.0","conversation_id":cid,"language":language,"conversation_complete":complete,
             "messages":[{"message_id":f"{cid}-M{i:03d}","sequence":i,"role":m["role"],"text":m["text"],
                          "speaker_id":None,"timestamp":None,"channel":"text","transcription_confidence":None} for i,m in enumerate(normalized,1)],
             "source":{"system":"interactive_quality_analyst","source_type":"user_provided_demo_input",
                       "provenance_id":None,"permitted_use":"local_demo_only_no_automatic_training"},
             "occurred_at":None,"policy_refs":[],"business_context":{"agency_id":None,"product_id":None,"currency":None,"travel_date":None}}
    return load_conversation_json(compact(payload))

def resources(snapshot):
    system=snapshot.get("system",{})
    return {"ram":system.get("ram",{}),"gpu":system.get("gpu",{})}

def write_json(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
