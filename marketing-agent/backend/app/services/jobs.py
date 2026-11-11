"""Postgres-backed work queue. Paid/remote outcomes are never blindly replayed."""
import hashlib
import json
import logging
from contextvars import ContextVar

from app import db

log = logging.getLogger(__name__)
current_job = ContextVar('marketing_job', default=None)
REPLAY_SAFE = {'webhook','conversion_memory'}


class IdempotencyConflict(ValueError):
    pass


def enqueue(kind, payload, business_id=None, key=None, conn=None):
    encoded = json.dumps(payload, sort_keys=True, default=str)
    scope = f'{business_id}:{kind}:{key or hashlib.sha256(encoded.encode()).hexdigest()}'
    def insert(c):
        row = c.execute("INSERT INTO app_jobs(business_id,kind,payload,request_key) VALUES (%s,%s,%s,%s) "
                        "ON CONFLICT(request_key) DO UPDATE SET request_key=EXCLUDED.request_key RETURNING id,payload",
                        (business_id,kind,encoded,scope)).fetchone()
        if json.dumps(row[1],sort_keys=True,default=str) != encoded:
            raise IdempotencyConflict('idempotency key already used for different inputs')
        return {'job_id':str(row[0]),'status':'queued'}
    if conn is not None:
        return insert(conn)
    with db.get_conn() as c:
        result = insert(c); c.commit(); return result


def get(business_id, job_id):
    with db.get_conn() as conn:
        row = conn.execute("SELECT id,kind,status,result,error,progress,created_at,started_at,finished_at FROM app_jobs WHERE id=%s AND business_id=%s", (job_id,business_id)).fetchone()
    if not row:
        raise ValueError('job not found')
    return dict(zip(('id','kind','status','result','error','progress','created_at','started_at','finished_at'), row))


def step(key, operation):
    """Persist completed weekly-plan steps. Uncertain steps require explicit review."""
    job = current_job.get()
    if job is None:
        return operation()
    with db.get_conn() as conn:
        row = conn.execute("INSERT INTO job_steps(job_id,step_key) VALUES (%s,%s) ON CONFLICT DO NOTHING RETURNING status", (job,key)).fetchone()
        if row is None:
            existing = conn.execute('SELECT status,result FROM job_steps WHERE job_id=%s AND step_key=%s', (job,key)).fetchone()
            if existing[0]=='succeeded': return existing[1]
            raise RuntimeError('step outcome is uncertain; inspect existing artifacts before explicitly starting new work')
        conn.commit()
    result = operation()
    with db.get_conn() as conn:
        conn.execute("UPDATE job_steps SET status='succeeded',result=%s WHERE job_id=%s AND step_key=%s", (json.dumps(result,default=str),job,key))
        conn.execute('UPDATE app_jobs SET progress=progress || %s::jsonb,heartbeat_at=now() WHERE id=%s', (json.dumps({key:'succeeded'}),job))
        conn.commit()
    return result


def dispatch(kind, payload, business_id):
    from app.services import content, weekly, video, image_creative, music, avatar, leads, memory
    if kind=='webhook': return leads.ingest_event(payload)
    if kind=='conversion_memory':
        # Outbox retries create one deterministic memory record, not duplicate learn calls.
        source = 'conversion:' + payload['conversion_id']
        with db.get_conn() as conn:
            conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (source,))
            if not conn.execute('SELECT 1 FROM agent_memory WHERE business_id=%s AND source=%s', (business_id,source)).fetchone():
                conn.execute("INSERT INTO agent_memory(business_id,scope,kind,content,source) VALUES (%s,'leads','feedback',%s,%s)", (business_id,f"Confirmed sale: {payload.get('lead_text') or 'unattributed'}. Owner notes: {payload.get('notes') or ''}",source))
            conn.commit()
        return {'stored':True}
    if kind=='generate': return content.generate_content(business_id, **payload)
    if kind=='plan-week': return weekly.plan_week(business_id, **payload)
    if kind=='metrics-sync':
        from app.services import metrics
        return metrics.sync_business(business_id)
    cid = payload['content_id']
    handlers = {'render':video.render_reel,'generate-image':image_creative.generate_for_content,'music':music.generate_reel_music,
                'avatar':lambda cid: avatar.start_avatar_clip(cid,payload.get('language','fr'))}
    if kind not in handlers: raise ValueError('unsupported job type')
    return handlers[kind](cid)


def run_pending(limit=3, only_safe=None):
    for _ in range(limit):
        with db.get_conn() as conn:
            # An interrupted paid request may have been charged. Never automatically repeat it.
            conn.execute("UPDATE app_jobs SET status=CASE WHEN kind=ANY(%s) THEN 'queued' ELSE 'unknown' END, error='Worker lease expired; external outcome requires review',available_at=now() "
                         "WHERE status='running' AND COALESCE(heartbeat_at,started_at)<now()-interval '45 minutes'", (list(REPLAY_SAFE),))
            row = conn.execute("UPDATE app_jobs SET status='running',attempts=attempts+1,started_at=now(),heartbeat_at=now() "
                               "WHERE id=(SELECT id FROM app_jobs WHERE status='queued' AND available_at<=now() AND (%s::boolean IS NULL OR (kind=ANY(%s))=%s) ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1) "
                               "RETURNING id,kind,payload,business_id,attempts",(only_safe,list(REPLAY_SAFE),only_safe)).fetchone()
            conn.commit()
        if not row: return
        jid,kind,payload,bid,attempts = row
        token = current_job.set(str(jid))
        try:
            result = dispatch(kind,payload,str(bid) if bid else None)
            status,error = 'succeeded',None
        except Exception as exc:
            log.exception('job %s failed',jid)
            result,error = None,str(exc)[:1000]
            status = 'queued' if kind in REPLAY_SAFE and attempts<5 else 'failed' if kind in REPLAY_SAFE or isinstance(exc,ValueError) else 'unknown'
        finally:
            current_job.reset(token)
        with db.get_conn() as conn:
            conn.execute("UPDATE app_jobs SET status=%s,result=%s,error=%s,finished_at=CASE WHEN %s='queued' THEN NULL ELSE now() END,available_at=now()+interval '30 seconds' WHERE id=%s AND status='running' AND attempts=%s", (status,json.dumps(result,default=str),error,status,jid,attempts))
            conn.commit()
