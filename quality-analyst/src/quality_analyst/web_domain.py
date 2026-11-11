from quality_analyst.contracts import load_conversation_json,strict_json_loads
from quality_analyst.interactive import InputError,ensure_no_secrets,stable_conversation
from quality_analyst.ollama import compact as canonical,sha256 as digest
def import_bytes(raw):
    if not isinstance(raw, bytes) or len(raw) > 1024 * 1024:
        raise InputError("JSON_TOO_LARGE")
    try:
        value = strict_json_loads(raw.decode("utf-8-sig"))
        ensure_no_secrets(value)
        if not isinstance(value, dict):
            raise ValueError()
        if "schema_version" in value:
            return load_conversation_json(canonical(value))
        if set(value) != {"language", "conversation_complete", "messages"}:
            raise ValueError()
        return stable_conversation(value["messages"], value["language"], value["conversation_complete"])
    except InputError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise InputError("INVALID_JSON_CONVERSATION") from None

def rebuild(messages, language, complete, original=None):
    """Preserve canonical IDs/roles/metadata when the imported input is unchanged."""
    short = [{"role": m["role"], "text": m["text"]} for m in messages]
    if original is None:
        return stable_conversation(short, language, complete)
    before = [{"role": m.role.value, "text": m.text} for m in original.messages]
    if short == before and language == original.language.value and complete == original.conversation_complete:
        return original
    if not short:
        raise InputError("INVALID_CONVERSATION")
    value = original.model_dump(mode="json")
    edited_id = "WEB-" + digest(canonical({"messages": short, "language": language, "complete": complete}).encode())[:16]
    value.update(conversation_id=edited_id, language=language, conversation_complete=complete,
                 source={"system": "quality_analyst_web", "source_type": "edited_demo_input",
                         "provenance_id": original.conversation_id, "permitted_use": "local_demo_only_no_automatic_training"},
                 messages=[{"message_id": f"{edited_id}-M{i:03d}", "sequence": i, "role": m["role"],
                            "text": m["text"], "speaker_id": None, "timestamp": None, "channel": "text",
                            "transcription_confidence": None} for i, m in enumerate(short, 1)])
    ensure_no_secrets(value)
    return load_conversation_json(canonical(value))
