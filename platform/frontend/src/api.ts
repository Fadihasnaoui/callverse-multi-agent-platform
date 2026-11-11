/** Thin wrapper around fetch that throws on non-2xx and returns JSON. */

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: string,
  ) {
    super(`API ${status}: ${body}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const slow = init?.method === 'POST' && (/\/content(?:\?|$|\/(?:plan-week|[^/]+\/(?:render|music|generate-image|avatar)(?:\?|$)))/.test(path) || /\/metrics\/sync$/.test(path));
  const identity = `marketing-operation:${path}:${init?.body || ''}`;
  let key: string | null = null;
  if (slow) {
    try { key = localStorage.getItem(identity); } catch {}
    key ||= crypto.randomUUID();
    try { localStorage.setItem(identity,key); } catch {}
  }
  const res = await fetch(path, {
    credentials: "include",
    ...init,
    headers: { "Content-Type": "application/json", ...(slow ? {'Prefer':'respond-async','Idempotency-Key':key!}:{}), ...init?.headers },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new ApiError(res.status, text);
  }
  if (res.status === 202) {
    const job = await res.json() as {job_id:string};
    const root = path.match(/^\/businesses\/[^/]+/)?.[0];
    for (;;) {
      await new Promise(resolve => setTimeout(resolve,1500));
      const response = await fetch(`${root}/content/jobs/${job.job_id}`,{credentials:'include'});
      if (!response.ok) throw new ApiError(response.status,'Job is saved. Reconnect and repeat the action to resume tracking it.');
      const state = await response.json();
      if (state.status === 'succeeded') {
        try { localStorage.removeItem(identity); } catch {}
        return state.result as T;
      }
      if (state.status === 'failed' || state.status === 'unknown') {
        // Keep the key: reconnecting must not silently create another paid operation.
        throw new ApiError(409,`${state.error || 'Job needs review'} (job ${job.job_id}, ${state.status}). Inspect before explicitly starting new work.`);
      }
    }
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined }),
  put: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  patch: <T>(path: string, body: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  del: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};
