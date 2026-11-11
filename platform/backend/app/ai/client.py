"""The single entry point for every model call in the app.

    result = ai.run("content", system, user, ContentOut, guard=...)   # -> validated pydantic object

What it guarantees, in order:
  1. transport: timeouts / 429 / 5xx are retried with backoff;
  2. parse: the reply must be JSON (JSON mode requested; a balanced {...} is extracted if wrapped in prose);
  3. validate: the JSON must satisfy the task's pydantic schema;
  4. guard: deterministic business checks (e.g. no invented numbers) must pass;
  5. repair: a failure at 2-4 is sent back to the model with the exact error, up to MAX_REPAIRS times;
  6. fallback: if still failing, one try on the fallback model;
  7. otherwise AIError - callers decide (most degrade gracefully or surface a clear message).
Every call - success or failure - is logged to `ai_calls` (task, prompt version, model, latency, tokens,
attempts, error), so "why did the agent write that" is answerable later."""
import json
import logging
import time

import requests
from pydantic import BaseModel, ValidationError

from app import config, db
from app.services import costs

log = logging.getLogger("ai")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_REPAIRS = 2
TRANSPORT_TRIES = 3

# task -> temperature. Judging/classifying is near-deterministic; writing gets some freedom.
TEMPERATURE = {
    "intent": 0.0, "proofread": 0.1, "compact": 0.1, "learning": 0.2, "audit": 0.2,
    "schedule": 0.3, "weekly": 0.4, "content": 0.5, "spoken": 0.5,
}


class AIError(RuntimeError):
    """The model could not produce a valid answer. `.problems` lists what was wrong on the last attempt."""

    def __init__(self, message: str, problems: list[str] | None = None):
        super().__init__(message)
        self.problems = problems or []


_json_mode_unsupported: set[str] = set()


def _chat(messages: list[dict], model: str, temperature: float, timeout: int) -> tuple[str, dict]:
    """One HTTP exchange with transport-level retries. Returns (text, usage)."""
    if not config.OPENROUTER_API_KEY:
        raise AIError("OPENROUTER_API_KEY is not set")
    last = None
    for attempt in range(TRANSPORT_TRIES):
        body = {"model": model, "messages": messages, "temperature": temperature}
        if model not in _json_mode_unsupported:
            body["response_format"] = {"type": "json_object"}
        try:
            costs.dispatched()
            response = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json"},
                json=body, timeout=timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as error:
            last = f"network: {error}"
        else:
            if response.status_code == 400 and "response_format" in body:
                _json_mode_unsupported.add(model)  # this model rejects JSON mode; stop asking for it
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {response.status_code}"
            elif response.status_code != 200:
                raise AIError(f"model API error HTTP {response.status_code}: {response.text[:300]}")
            else:
                data = response.json()
                try:
                    return data["choices"][0]["message"]["content"] or "", data.get("usage") or {}
                except (KeyError, IndexError):
                    last = f"malformed API reply: {str(data)[:200]}"
        time.sleep(min(2 ** attempt, 8))
    raise AIError(f"model API unavailable after {TRANSPORT_TRIES} tries ({last})")


def _extract_json(raw: str):
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except ValueError:
        pass
    start = raw.find("{")
    while start != -1:  # first balanced {...} that actually parses
        depth = 0
        for i in range(start, len(raw)):
            depth += (raw[i] == "{") - (raw[i] == "}")
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except ValueError:
                    break
        start = raw.find("{", start + 1)
    raise ValueError("the reply did not contain a valid JSON object")


def _log(task, version, model, ok, attempts, latency_ms, usage, error):
    costs.account(usage.get('cost'))
    try:
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO ai_calls (task, prompt_version, model, provider, ok, attempts, latency_ms, prompt_tokens, "
                "completion_tokens, cost_usd, usage_details, error,reservation_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (task, version, model, "openrouter", ok, attempts, latency_ms, usage.get("prompt_tokens"),
                 usage.get("completion_tokens"), usage.get("cost"), json.dumps(usage),
                 (error or "")[:500] or None,costs.reservation_id()),
            )
            conn.commit()
    except Exception:  # noqa: BLE001 - observability must never break the task
        log.exception("could not log ai call")


@costs.budgeted('openrouter',0.25)
def run(task: str, system: str, user: str, schema: type[BaseModel], *, guard=None, model: str | None = None,
        version: str = "", timeout: int = 60, temperature: float | None = None) -> BaseModel:
    try:
        costs.ensure_budget_available()
    except RuntimeError as error:
        raise AIError(str(error)) from error
    models = [model or config.OPENROUTER_MODEL]
    if config.AI_FALLBACK_MODEL and config.AI_FALLBACK_MODEL not in models:
        models.append(config.AI_FALLBACK_MODEL)
    temp = TEMPERATURE.get(task, 0.3) if temperature is None else temperature
    started = time.time()
    attempts = 0
    usage_total: dict = {}
    problems: list[str] = []

    for current_model in models:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for repair in range(MAX_REPAIRS + 1):
            attempts += 1
            try:
                raw, usage = _chat(messages, current_model, temp, timeout)
            except AIError as error:
                problems = [str(error)]
                break  # transport is down for this model: go to the fallback, repairing cannot help
            for k, v in usage.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    usage_total[k] = usage_total.get(k, 0) + v
            try:
                output = schema.model_validate(_extract_json(raw))
                problems = guard(output) if guard else []
            except ValidationError as error:
                problems = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()][:6]
            except ValueError as error:
                problems = [str(error)]
            if not problems:
                _log(task, version, current_model, True, attempts, int((time.time() - started) * 1000), usage_total, None)
                return output
            log.warning("ai %s attempt %d on %s: %s", task, attempts, current_model, problems)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your reply was rejected: " + "; ".join(problems)
                 + ". Reply again with ONLY the corrected JSON object."},
            ]

    message = f"AI task '{task}' failed after {attempts} attempt(s): {'; '.join(problems)[:400]}"
    _log(task, version, models[-1], False, attempts, int((time.time() - started) * 1000), usage_total, message)
    raise AIError(message, problems)
