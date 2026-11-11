"""Private HTTP adapter for the Quality Analyst service.

The adapter deliberately accepts a completed (or explicitly partial) transcript
instead of customer messages.  It is used by the Ozone dashboard today and by
the client-simulator controller later.  It never sends anything to a cloud
provider unless the caller records the required consent in the request.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quality_analyst.interactive import InputError, stable_conversation
from quality_analyst.ozone_service_v02 import history, run
from quality_analyst.web_domain import import_bytes

REVIEW_SCHEMA = Path(os.getenv("SHARED_REVIEW_SCHEMA", "/contracts/conversation_review_v1.schema.json"))


MAX_BODY_BYTES = 1024 * 1024
REVIEW_DIR = Path(os.getenv("APP_DATA_DIR", "/var/lib/quality-analyst")) / "saved_reviews"


def combined_scorecard(deterministic: dict | None, result: dict) -> dict | None:
    """Keep hard safety scoring authoritative while exposing bounded LLM coaching.

    Qwen observations are never silently treated as facts. Only citations that
    survive the analyst's exact-quote checks can lower the *provisional*
    composite, and a critical deterministic failure always blocks scoring.
    """
    if not deterministic:
        return None
    official = deterministic.get("score_100")
    if official is None:
        return {"status": "critical_review_required", "official_score_100": None,
                "llm_signal_score_100": None, "composite_score_100": None,
                "reason": "deterministic_critical_failure", "human_review_required": True}
    checked = result.get("checked") if isinstance(result.get("checked"), dict) else {}
    verification = checked.get("verification") if isinstance(checked.get("verification"), dict) else {}
    hypotheses = [item for item in verification.get("weaknesses", [])
                  if isinstance(item, dict) and item.get("status") == "hypothesis_for_review"]
    if result.get("status") != "free_quality_checked":
        return {"status": "deterministic_only", "official_score_100": official,
                "llm_signal_score_100": None, "composite_score_100": official,
                "reason": "llm_unavailable_or_unverified", "human_review_required": bool(deterministic.get("human_review_required"))}
    # One cited hypothesis is a small coaching signal, never an automatic
    # allegation. Cap its total influence at twenty points.
    penalty = min(20, 5 * len(hypotheses))
    return {"status": "provisional_with_llm" if hypotheses else "combined_scored",
            "official_score_100": official,
            "llm_signal_score_100": 100 - penalty,
            "composite_score_100": official - penalty,
            "llm_hypotheses": len(hypotheses), "max_llm_penalty": 20,
            "reason": "cited_llm_coaching_signal" if hypotheses else "no_cited_llm_coaching_signal",
            "human_review_required": bool(deterministic.get("human_review_required") or hypotheses)}


def save_review(conversation, payload: dict, scorecard: dict | None, result: dict, combined: dict | None) -> dict:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    review_id = f"review-{uuid4().hex}"
    messages = conversation.model_dump(mode="json").get("messages", [])
    record = {"review_id": review_id, "created_at": datetime.now(timezone.utc).isoformat(),
              "conversation": conversation.model_dump(mode="json"),
              "review_records": payload.get("review_records") or ([payload.get("review_record")] if payload.get("review_record") else []),
              "scorecard": scorecard, "combined_scorecard": combined, "result": result}
    path = REVIEW_DIR / f"{review_id}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, allow_nan=False)
    return {"review_id": review_id, "created_at": record["created_at"],
            "conversation_id": conversation.conversation_id, "message_count": len(messages),
            "scorecard": scorecard, "combined_scorecard": combined,
            "analysis_status": result.get("status")}


def saved_review_history() -> list[dict]:
    rows = []
    for path in sorted(REVIEW_DIR.glob("review-*.json"), reverse=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.append({key: value.get(key) for key in ("review_id", "created_at", "scorecard", "combined_scorecard")}
                        | {"conversation_id": (value.get("conversation") or {}).get("conversation_id"),
                           "message_count": len((value.get("conversation") or {}).get("messages") or []),
                           "analysis_status": (value.get("result") or {}).get("status")})
        except (OSError, ValueError):
            continue
    return rows[:50]


def shared_score(records: list[dict], *, authoritative_metadata: bool = True) -> dict:
    shared_dir = os.getenv("SHARED_REVIEW_CODE_DIR", "/shared")
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    import quality_review_contract as review_contract
    score = review_contract.score_records(records)
    score["coverage"] = "full_shared_turn_record" if authoritative_metadata else "transcript_safety_only"
    score["authority"] = "advisor_metadata" if authoritative_metadata else "transcript_without_offer_or_tool_metadata"
    return score


def transcript_scorecard(conversation) -> dict:
    """Score only text-safe failures when legacy chat lacks turn metadata."""
    records, pending_customer = [], None
    for message in conversation.messages:
        if message.role.value == "customer":
            pending_customer = message
        elif message.role.value == "advisor":
            records.append({"customer_message": {"text": pending_customer.text if pending_customer else ""},
                            "advisor_response": {"status": "completed", "text": message.text},
                            "advisor_decision": {"offers": [], "handoff_required": False}})
    return shared_score(records, authoritative_metadata=False)


def conversation_from_request(payload: dict):
    """Accept the analyst's canonical JSON or the dashboard's simple transcript."""
    conversation = payload.get("conversation")
    if isinstance(conversation, dict) and "schema_version" in conversation:
        return import_bytes(json.dumps(conversation, ensure_ascii=False).encode("utf-8"))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise InputError("INVALID_CONVERSATION")
    rows = []
    for item in messages:
        if not isinstance(item, dict):
            raise InputError("INVALID_MESSAGE")
        role = item.get("role")
        if role == "client":
            role = "customer"
        elif role == "agent":
            role = "advisor"
        rows.append({"role": role, "text": item.get("text")})
    return stable_conversation(
        rows,
        language=str(payload.get("language") or "fr"),
        complete=payload.get("conversation_complete", True) is True,
    )


def review_records_from_request(payload: dict) -> list[dict]:
    rows = payload.get("review_records")
    if rows is None and payload.get("review_record") is not None:
        rows = [payload["review_record"]]
    if not isinstance(rows, list) or not rows:
        return []
    try:
        import jsonschema
        schema = json.loads(REVIEW_SCHEMA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ValueError("SHARED_REVIEW_SCHEMA_UNAVAILABLE") from None
    validator = jsonschema.Draft202012Validator(schema)
    messages = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("INVALID_REVIEW_RECORD")
        errors = list(validator.iter_errors(row))
        if errors:
            raise ValueError("INVALID_REVIEW_RECORD")
        messages.extend((row["customer_message"], row["advisor_response"]))
    return messages


class QualityAnalystHandler(BaseHTTPRequestHandler):
    server_version = "OzoneQualityAnalyst/0.1"

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/health":
            self.send_json({"status": "ok", "service": "quality-analyst", "mode": "diagnostic_only"})
        elif route == "/v1/quality-analyst/history":
            # Metadata only: raw conversations and model outputs stay in the
            # analyst's private volume and are never returned by this index.
            rows = history()[:50]
            self.send_json({"items": saved_review_history(), "runs": [{key: row.get(key) for key in (
                "session_id", "stage", "status", "provider", "model", "started_at",
                "wall_seconds", "generation_calls", "errors", "worker_state_uncertain",
            )} for row in rows]})
        elif route.startswith("/v1/quality-analyst/reviews/"):
            review_id = route.rsplit("/", 1)[-1]
            if not review_id.startswith("review-") or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in review_id):
                self.send_json({"error": "not found"}, 404)
                return
            try:
                value = json.loads((REVIEW_DIR / f"{review_id}.json").read_text(encoding="utf-8"))
                self.send_json(value)
            except (OSError, ValueError):
                self.send_json({"error": "not found"}, 404)
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/v1/quality-analyst/analyze":
            self.send_json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("INVALID_REQUEST_SIZE")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("INVALID_REQUEST")
            review_messages = review_records_from_request(payload)
            if review_messages:
                payload = dict(payload, messages=review_messages,
                               language=payload.get("language") or "unknown")
            conversation = conversation_from_request(payload)
            provider = payload.get("provider")
            if provider not in {None, "gemini", "ollama"}:
                raise ValueError("AI_PROVIDER_REFUSED")
            scorecard = None
            if review_messages:
                # Mounted from Ozone's shared contract directory. The analyst
                # uses the same deterministic rubric, not a copied formula.
                records = payload.get("review_records") or [payload.get("review_record")]
                scorecard = shared_score(records)
            analysis_context = {"shared_review_records": records} if review_messages else None
            if scorecard is None:
                scorecard = transcript_scorecard(conversation)
            result = run(
                conversation,
                stage="analysis",
                context=analysis_context,
                provider=provider,
                cloud_consent=payload.get("cloud_consent") is True,
                free_tier_confirmed=payload.get("free_tier_confirmed") is True,
            )
            combined = combined_scorecard(scorecard, result)
            saved = save_review(conversation, payload, scorecard, result, combined)
            self.send_json({
                "api_version": "quality-analyst/v1",
                "mode": "diagnostic_only",
                "conversation_id": conversation.conversation_id,
                "conversation_complete": conversation.conversation_complete,
                "scorecard": scorecard,
                "combined_scorecard": combined,
                "saved_review": saved,
                "result": result,
            })
        except (InputError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error) or "INVALID_REQUEST"}, 400)
        except Exception:
            # Do not disclose transcript content, local paths, model output or
            # configuration in an HTTP response.
            self.send_json({"error": "QUALITY_ANALYST_UNAVAILABLE"}, 503)

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    port = int(os.getenv("QUALITY_ANALYST_API_PORT", "8010"))
    ThreadingHTTPServer(("0.0.0.0", port), QualityAnalystHandler).serve_forever()
