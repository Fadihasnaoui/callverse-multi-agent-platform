"""The AI layer: prompts (prompts.py), output contracts (schemas.py), code-level checks (guards.py),
and the one function every model call goes through (client.run)."""
from app.ai.client import AIError, run  # noqa: F401
