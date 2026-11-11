import { useCallback, useEffect, useState } from "react";

const base = (import.meta.env.VITE_WORKFORCE_API_BASE || "/workforce-api").replace(/\/$/, "");

export async function workforceRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (!response.ok) {
    let detail = `Workforce API: ${response.status}`;
    try { const body = await response.json(); detail = typeof body.detail === "string" ? body.detail : detail; } catch { /* Proxy may return HTML. */ }
    throw new Error(detail);
  }
  if (!response.headers.get("content-type")?.includes("application/json")) throw new Error("Workforce API did not return JSON. Check the API proxy.");
  return response.json() as Promise<T>;
}

export function workforceAction(path: string, body?: unknown) {
  return workforceRequest<unknown>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
}

/** Poll only the mounted view; abort on navigation and never overlap requests. */
export function useWorkforceResource<T>(path: string | null, interval = 4000) {
  const [state, setState] = useState<{ path: string | null; data: T | null; error: string; loading: boolean }>({ path: null, data: null, error: "", loading: true });
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision(value => value + 1), []);
  useEffect(() => {
    if (!path) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const controller = new AbortController();
    async function load() {
      try {
        const data = await workforceRequest<T>(path!, { signal: controller.signal });
        if (!stopped) setState({ path, data, error: "", loading: false });
      } catch (error) {
        if (!stopped) setState(previous => ({ path, data: previous.path === path ? previous.data : null, error: error instanceof Error ? error.message : "Workforce API unavailable", loading: false }));
      } finally {
        if (!stopped && interval > 0) timer = setTimeout(load, interval);
      }
    }
    void load();
    return () => { stopped = true; controller.abort(); clearTimeout(timer); };
  }, [path, interval, revision]);
  return { ...(state.path === path && path ? state : { data: null, error: "", loading: !!path }), refresh };
}
