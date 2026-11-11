"""AI spend controls backed by the exact cost OpenRouter returns per request."""
import logging
from contextvars import ContextVar
from functools import wraps

import requests

from app import config, db

log = logging.getLogger("costs")
_reservation = ContextVar('spend_reservation', default=None)


class BudgetExceeded(RuntimeError):
    pass


def account(cost):
    current = _reservation.get()
    if current is not None and cost is not None:
        current['actual'] = float(cost)


def dispatched():
    current = _reservation.get()
    if current is not None: current['dispatched'] = True


def reservation_id():
    current = _reservation.get()
    return current['id'] if current else None


def budgeted(provider, estimate):
    """Reserve before concurrent work. Unknown provider charges retain the estimate."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            amount = float(estimate)
            with db.get_conn() as conn:
                conn.execute('SELECT pg_advisory_xact_lock(782015)')
                spent = conn.execute("SELECT COALESCE((SELECT sum(cost_usd) FROM ai_calls WHERE reservation_id IS NULL AND created_at>=date_trunc('month',now())),0)+COALESCE((SELECT sum(amount) FROM spend_reservations WHERE status IN ('reserved','estimated','settled') AND created_at>=date_trunc('month',now())),0)").fetchone()[0]
                if config.MONTHLY_AI_BUDGET_USD>0 and float(spent)+amount>config.MONTHLY_AI_BUDGET_USD:
                    raise BudgetExceeded('Monthly AI budget cannot authorize this operation including outstanding reservations')
                rid = conn.execute('INSERT INTO spend_reservations(provider,amount) VALUES (%s,%s) RETURNING id',(provider,amount)).fetchone()[0]
                conn.commit()
            state = {'id':rid,'actual':None,'dispatched':False}
            token = _reservation.set(state)
            try:
                return function(*args,**kwargs)
            finally:
                _reservation.reset(token)
                status = 'settled' if state['actual'] is not None else 'estimated' if state['dispatched'] else 'released'
                with db.get_conn() as conn:
                    conn.execute('UPDATE spend_reservations SET amount=%s,status=%s,finished_at=now() WHERE id=%s', (state['actual'] if state['actual'] is not None else amount,status,rid))
                    conn.commit()
        return wrapped
    return decorate


def current_month_spend() -> float:
    with db.get_conn() as conn:
        row = conn.execute(
            "SELECT COALESCE((SELECT sum(cost_usd) FROM ai_calls WHERE reservation_id IS NULL AND created_at>=date_trunc('month',now())),0)+COALESCE((SELECT sum(amount) FROM spend_reservations WHERE status IN ('reserved','estimated','settled') AND created_at>=date_trunc('month',now())),0)"
        ).fetchone()
    return float(row[0] or 0)


def openrouter_balance() -> dict | None:
    """Query OpenRouter /api/v1/auth/key for live credit balance."""
    if not config.OPENROUTER_API_KEY:
        return None
    try:
        resp = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        usage = data.get("usage", 0)
        limit = data.get("limit")
        return {
            "credits_used": usage,
            "credits_limit": limit,
            "credits_remaining": round(limit - usage, 6) if limit is not None else None,
        }
    except Exception as exc:
        log.warning("OpenRouter balance check failed: %s", exc)
        return None


def ensure_budget_available() -> None:
    if _reservation.get() is not None: return
    limit = config.MONTHLY_AI_BUDGET_USD
    if limit > 0 and current_month_spend() >= limit:
        raise RuntimeError(
            f"Monthly AI budget of ${limit:.2f} has been reached. Increase MONTHLY_AI_BUDGET_USD to continue."
        )
