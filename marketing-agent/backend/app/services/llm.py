"""Thin OpenRouter client. Same provider-call shape as the Ozone project's
llm_chat(), trimmed to the one provider this service actually needs — no
fallback chain yet since OpenRouter alone has been reliable enough in
production use so far; add a chain here the same way if that changes."""
import json

import requests

from app import config

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def llm_chat(system_prompt: str, user_prompt: str, timeout: int = 60, model: str | None = None) -> str:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or config.OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def llm_json(system_prompt: str, user_prompt: str, timeout: int = 60, model: str | None = None) -> dict:
    """Same as llm_chat but extracts the first JSON object from the reply -
    mirrors extract_json_object() in the Ozone project, since LLMs reliably
    wrap JSON in prose/markdown fences even when told not to."""
    raw = llm_chat(system_prompt, user_prompt, timeout=timeout, model=model)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in LLM reply: {raw!r}")
    return json.loads(raw[start : end + 1])
