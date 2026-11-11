import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { WorkspaceHeader } from "@/pages/AgentHub";
import { api } from "@/api";
import { AiReview, CallRef, CallReview, Client, QualityBrief, ROUTE_LABEL, SOURCE_LABEL, TranscriptModal, clientLabel, clientName, findingLabel, humanize, shortDate } from "@/twin/shared";
import styles from "./DigitalTwin.module.css";

type Health = { services: Record<string, string> };
type BoardCall = CallRef & {
  call_id: string; messages: number; route: { queue?: string }; route_path: (string | null)[];
  last_message?: { role: string; content: string }; quality: QualityBrief; advisor_repeat_streak: number;
};
type Evaluation = {
  calls: number; finished: number; active: number; messages: number;
  advisor: { calls_with_repeat_loop: number; longest_repeat: number };
  routing: { decisions: number; by_queue: Record<string, number>; waiting_for_human: number; calls_escalated: number };
  quality: { reviewed: number; mean_score?: number | null; min_score?: number | null; below_80: number; human_review_required: number;
    missing_review: number; llm_checked: number; top_findings: Record<string, number> };
  clients?: Record<string, { calls: number; mean_score?: number | null }>;
  client_details?: { languages: Record<string, number>; scenarios: Record<string, number> };
  voice: { rooms: number; turns_spoken: number; turn_failures: number; audio_minutes: number };
  outcomes?: { completed: number; failed: number };
};
type Batch = {
  id: string; status: string; completed: number; created_at?: string; settings: { count: number; voice?: boolean };
  calls?: BoardCall[]; evaluation?: Evaluation;
  workforce?: { human_capacity: number; humans_available: number; active_human_assignments: number } | null;
};
type RunListItem = { id: string; calls: number; created_at: string; active: number };
type ReportSection = { key: string; title: string; text: string; warning?: string | null };
type WrittenReport = { batch_id: string; sections: ReportSection[]; provider?: string; model?: string; warnings: { section: string; warning: string }[]; call_reviews?: CallReview[]; created_at: string };
type HistoryRow = CallRef & { call_id?: string; batch_id?: string; created_at: string; messages: number; quality: QualityBrief };

const SERVICE_LABEL: Record<string, string> = { advisor: "Amira", client_simulator: "Jasmin", quality_analyst: "Quality", workforce: "Workforce", voice_rooms: "Voice rooms" };
const HISTORY_PAGE = 40;

export function DigitalTwin() {
  const [health, setHealth] = useState<Health>({ services: {} });
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [batch, setBatch] = useState<Batch | null>(null);
  const [history, setHistory] = useState<HistoryRow[]>([]);
  const [historyMore, setHistoryMore] = useState(false);
  const [open, setOpen] = useState<CallRef | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [callCount, setCallCount] = useState(12);
  const [parallel, setParallel] = useState(8);
  const [voice, setVoice] = useState(true);
  const [language, setLanguage] = useState("mixed");
  const [clientMix, setClientMix] = useState("random");

  async function loadRun(id: string) { setBatch(await api.get<Batch>(`/digital-twin/batches/${id}`)); }
  async function loadHistory(offset = 0) {
    const rows = await api.get<HistoryRow[]>(`/digital-twin/sessions?limit=${HISTORY_PAGE}&offset=${offset}`);
    setHistory(current => offset ? [...current, ...rows] : rows);
    setHistoryMore(rows.length === HISTORY_PAGE);
  }
  async function loadRuns() {
    const list = await api.get<RunListItem[]>("/digital-twin/batches?limit=20");
    setRuns(list);
    return list;
  }

  useEffect(() => {
    api.get<Health>("/digital-twin/health").then(setHealth).catch(() => undefined);
    loadRuns().then(list => list[0] && loadRun(list[0].id)).catch(() => setError("The Digital Twin services are not all reachable yet."));
    loadHistory().catch(() => undefined);
  }, []);
  // While a run is going: refresh its calls every 2 s and the history every 10 s.
  useEffect(() => {
    if (!batch || batch.status !== "running") return;
    const fast = window.setInterval(() => loadRun(batch.id).catch(() => undefined), 2000);
    const slow = window.setInterval(() => loadHistory().catch(() => undefined), 10000);
    return () => { window.clearInterval(fast); window.clearInterval(slow); };
  }, [batch?.id, batch?.status]);
  useEffect(() => { if (batch?.status === "completed") { loadRuns().catch(() => undefined); loadHistory().catch(() => undefined); } }, [batch?.status]);

  async function startRun() {
    setBusy(true); setError("");
    try {
      const started = await api.post<Batch>("/digital-twin/batches", { count: callCount, concurrency: Math.min(parallel, callCount), language, voice, client_mix: clientMix, max_turns: 10 });
      setBatch(started); loadRuns().catch(() => undefined);
    } catch { setError("The run could not start. Check that every agent is ready."); }
    finally { setBusy(false); }
  }
  // Keep the open window in sync with the run's latest data for that call.
  const openCall = open ? batch?.calls?.find(c => c.id === open.id) || open : null;
  const running = batch?.status === "running";

  return <div className={styles.page}><WorkspaceHeader />
    <main className={styles.shell}>
      <section className={styles.hero}><p>OZONE VOYAGES / DIGITAL TWIN</p><h1>Run the call centre. See every agent's result.</h1>
        <span>Simulated clients call Amira through the Customer Advisor API; Workforce routes every turn, each call has its own voice room, and Quality reviews the saved transcript.</span>
        <div className={styles.services}>{Object.entries(health.services).map(([name, state]) => <span key={name} className={state === "ready" ? styles.svcOk : styles.svcWarn}>{SERVICE_LABEL[name] || humanize(name)}</span>)}</div>
      </section>
      {error && <div className={styles.alert}>{error}</div>}

      <section className={`${styles.panel} ${styles.runBar}`}>
        <div><p className={styles.kicker}>NEW RUN</p><h2>Start a call-centre run</h2></div>
        <label>Calls<select value={callCount} onChange={e => setCallCount(Number(e.target.value))}>{[6, 12, 20, 30].map(n => <option key={n} value={n}>{n}</option>)}</select></label>
        <label>In parallel<select value={parallel} onChange={e => setParallel(Number(e.target.value))}>{[4, 8, 12].map(n => <option key={n} value={n}>{n}</option>)}</select></label>
        <label>Languages<select value={language} onChange={e => setLanguage(e.target.value)}><option value="mixed">French + English</option><option value="fr">French</option><option value="en">English</option></select></label>
        <label>Clients<select value={clientMix} onChange={e => setClientMix(e.target.value)}><option value="random">Mixed sources</option><option value="jasmin">Jasmin profiles only</option></select></label>
        <label className={styles.check}><input type="checkbox" checked={voice} onChange={e => setVoice(e.target.checked)} /> Voice rooms</label>
        <button className={styles.primary} disabled={busy || running} onClick={startRun}>{running ? "Run in progress…" : `Start ${callCount}-call run`}</button>
      </section>

      {batch && <RunReport batch={batch} runs={runs} select={id => loadRun(id).catch(() => setError("That run could not be loaded."))} />}
      {batch?.calls && <RunCalls batch={batch} onOpen={setOpen} />}
      <HistoryTable rows={history} more={historyMore} loadMore={() => loadHistory(history.length).catch(() => undefined)} onOpen={setOpen} />
      {openCall && <TranscriptModal call={openCall} close={() => setOpen(null)} />}
    </main>
  </div>;
}

function RunReport({ batch, runs, select }: { batch: Batch; runs: RunListItem[]; select: (id: string) => void }) {
  const ev = batch.evaluation;
  if (!ev) return null;
  const routing = ev.routing;
  const aiShare = routing.decisions ? Math.round(100 * (routing.by_queue.ai || 0) / routing.decisions) : 0;
  const runIndex = runs.findIndex(r => r.id === batch.id);
  return <section className={`${styles.panel} ${styles.report}`}>
    <div className={styles.panelHeader}>
      <div><p className={styles.kicker}>RUN REPORT · WHAT EACH AGENT RETURNED</p>
        <h2>{runIndex === 0 ? "Latest run" : "Run"} · {shortDate(batch.created_at || runs.find(r => r.id === batch.id)?.created_at)} · {ev.finished}/{ev.calls} calls finished{batch.status === "running" ? " · in progress" : ""}</h2></div>
      <label className={styles.runSelect}>Run<select value={batch.id} onChange={e => select(e.target.value)}>{runs.map(run => <option key={run.id} value={run.id}>{shortDate(run.created_at)} · {run.calls} calls{run.active ? " · running" : ""}</option>)}</select></label>
    </div>
    <div className={styles.agentGrid}>
      <article className={styles.agent}>
        <header><b>Amira</b><small>customer advisor</small></header>
        <dl>
          <div><dt>Calls completed / failed</dt><dd>{ev.outcomes?.completed ?? "—"} / {ev.outcomes?.failed ?? "—"}</dd></div>
          <div><dt>Messages exchanged</dt><dd>{ev.messages}</dd></div>
          <div><dt>Calls with a repetition loop</dt><dd className={ev.advisor.calls_with_repeat_loop ? styles.warnText : ""}>{ev.advisor.calls_with_repeat_loop} <small>longest {ev.advisor.longest_repeat}×</small></dd></div>
          <div><dt>Mean Quality score</dt><dd>{ev.quality.mean_score ?? "—"}<small>/100</small></dd></div>
        </dl>
      </article>
      <article className={styles.agent}>
        <header><b>Quality Analyst</b><small>evidence review</small></header>
        <dl>
          <div><dt>Calls reviewed</dt><dd>{ev.quality.reviewed} / {ev.finished}</dd></div>
          <div><dt>Mean · lowest score</dt><dd>{ev.quality.mean_score ?? "—"} · {ev.quality.min_score ?? "—"}</dd></div>
          <div><dt>Calls below 80</dt><dd className={ev.quality.below_80 ? styles.warnText : ""}>{ev.quality.below_80}</dd></div>
          <div><dt>AI check completed</dt><dd>{ev.quality.llm_checked} / {ev.quality.reviewed}</dd></div>
        </dl>
        {!!Object.keys(ev.quality.top_findings || {}).length && <div className={styles.chips}>{Object.entries(ev.quality.top_findings).map(([code, n]) => <span key={code}>{findingLabel(code)} <b>{n}</b></span>)}</div>}
        <Link className={styles.agentLink} to={`/quality-analyst?batch=${batch.id}`}>Open Quality Analyst reports →</Link>
      </article>
      <article className={styles.agent}>
        <header><b>Workforce Manager</b><small>routing</small></header>
        <dl>
          <div><dt>Routing decisions</dt><dd>{routing.decisions} <small>{aiShare}% to Amira</small></dd></div>
          <div><dt>Calls escalated to a human</dt><dd>{routing.calls_escalated}</dd></div>
          <div><dt>Turns waiting for an adviser</dt><dd className={routing.waiting_for_human ? styles.warnText : ""}>{routing.waiting_for_human}</dd></div>
          <div><dt>Advisers free now</dt><dd>{batch.workforce ? `${batch.workforce.humans_available} / ${batch.workforce.human_capacity}` : "—"}</dd></div>
        </dl>
        <div className={styles.chips}>{Object.entries(routing.by_queue).map(([queue, n]) => <span key={queue}>{ROUTE_LABEL[queue] || humanize(queue)} <b>{n}</b></span>)}</div>
        <Link className={styles.agentLink} to={`/workforce?batch=${batch.id}`}>Open Workforce Manager →</Link>
      </article>
      <article className={styles.agent}>
        <header><b>Client simulators</b><small>Jasmin + Voice Lab</small></header>
        <dl>{Object.entries(ev.clients || {}).map(([source, v]) => <div key={source}><dt>{SOURCE_LABEL[source] || source}</dt><dd>{v.calls} <small>calls · QA {v.mean_score ?? "—"}</small></dd></div>)}</dl>
        {ev.client_details && <div className={styles.chips}>{Object.entries(ev.client_details.languages).map(([lang, n]) => <span key={lang}>{lang.toUpperCase()} <b>{n}</b></span>)}{Object.entries(ev.client_details.scenarios).filter(([s]) => s !== "—").map(([s, n]) => <span key={s}>{humanize(s)} <b>{n}</b></span>)}</div>}
      </article>
      <article className={styles.agent}>
        <header><b>Voice rooms</b><small>LiveKit + Edge TTS</small></header>
        <dl>
          <div><dt>Rooms</dt><dd>{ev.voice.rooms}</dd></div>
          <div><dt>Turns spoken</dt><dd>{ev.voice.turns_spoken}</dd></div>
          <div><dt>Audio failures</dt><dd className={ev.voice.turn_failures ? styles.warnText : ""}>{ev.voice.turn_failures}</dd></div>
          <div><dt>Audio played</dt><dd>{ev.voice.audio_minutes} <small>min</small></dd></div>
        </dl>
      </article>
    </div>
    <WrittenReportPanel batch={batch} />
  </section>;
}

/** **Label** text → bold label; one paragraph per line. */
const REPORT_LABEL = /^\s*(?:\*\*)?(What it did|What went well|Problems seen|Recommendation)(?:\*\*)?\s*[:–—-]?\s*/;

function ReportText({ text }: { text: string }) {
  // Bold the four section labels however the model wrote them ("**What it did** –" or "What it did:").
  return <>{text.split(/\n+/).filter(line => line.trim()).map(line => line.replace(REPORT_LABEL, "**$1** ")).map((line, i) =>
    <p key={i}>{line.replace(/^[-*]\s+/, "").split(/(\*\*[^*]+\*\*)/).map((part, j) =>
      part.startsWith("**") ? <b key={j}>{part.slice(2, -2)}</b> : <span key={j}>{part}</span>)}</p>)}</>;
}

function WrittenReportPanel({ batch }: { batch: Batch }) {
  const [report, setReport] = useState<WrittenReport | null>(null);
  const [state, setState] = useState<"loading" | "none" | "writing" | "ready" | "error">("loading");
  useEffect(() => {
    setState("loading"); setReport(null);
    api.get<WrittenReport>(`/digital-twin/batches/${batch.id}/report`).then(r => { setReport(r); setState("ready"); }).catch(() => setState("none"));
  }, [batch.id]);
  async function generate() {
    setState("writing");
    try { setReport(await api.post<WrittenReport>(`/digital-twin/batches/${batch.id}/report`, {})); setState("ready"); }
    catch { setState("error"); }
  }
  const running = batch.status === "running";
  return <div className={styles.written}>
    <div className={styles.writtenHead}>
      <div><p className={styles.kicker}>WRITTEN REPORT</p>
        <h3>{report ? `Generated ${shortDate(report.created_at)} by ${report.model || "an LLM"}` : "Run report written by an LLM from this run's figures"}</h3></div>
      <button className={styles.secondary} disabled={running || state === "writing" || state === "loading"} onClick={generate}>
        {state === "writing" ? "Writing the report…" : report ? "Regenerate" : running ? "Available when the run ends" : "Generate report"}</button>
    </div>
    {state === "error" && <div className={styles.alert}>The report could not be written: no language model answered. Try again in a moment.</div>}
    {report && <div className={styles.writtenGrid}>{report.sections.map(section => <article key={section.key}>
      <h4>{section.title}</h4><ReportText text={section.text || "No text was produced for this section."} />
      {section.warning && <small className={styles.warnText}>Check: {section.warning}</small>}
    </article>)}</div>}
    {report?.call_reviews?.length ? <CallReviews reviews={report.call_reviews} /> : null}
    {report && <small className={styles.writtenNote}>Every number in this report is checked against the run's stored figures; sections where a number could not be matched are marked.</small>}
  </div>;
}

function CallReviews({ reviews }: { reviews: CallReview[] }) {
  const local = reviews.filter(r => r.checked_by?.startsWith("local")).length;
  const api = reviews.filter(r => r.checked_by === "API model").length;
  return <div className={styles.callReviews}>
    <h4>AI quality review of each call</h4>
    <small>{reviews.length} calls · {local} read by the Quality Analyst's local model · {api} read by an API model (the local check was skipped for lack of RAM)</small>
    {reviews.map(r => <details key={r.session_id} className={styles.callReview}>
      <summary>
        <strong>{r.client}</strong><span>#{r.call_id} · {(r.language || "").toUpperCase()}{r.scenario ? ` · ${humanize(r.scenario)}` : ""}</span>
        <span className={typeof r.rule_score === "number" && r.rule_score < 80 ? styles.scoreLow : styles.scoreOk}>{typeof r.rule_score === "number" ? `${r.rule_score}/100` : "—"}</span>
        <em>{r.checked_by ? `${r.checked_by}${r.model && r.model !== "local" ? ` · ${r.model}` : ""}` : "not reviewed"}</em>
      </summary>
      <AiReview review={r} />
    </details>)}
  </div>;
}

function RunCalls({ batch, onOpen }: { batch: Batch; onOpen: (call: CallRef) => void }) {
  const calls = batch.calls || [];
  const live = calls.filter(c => c.status === "active").length;
  return <section className={`${styles.panel} ${styles.board}`}>
    <div className={styles.panelHeader}><div><p className={styles.kicker}>CONVERSATIONS IN THIS RUN</p><h2>{calls.length} calls{live ? ` · ${live} live now` : ""}</h2></div></div>
    <div className={styles.cards}>{calls.map(call => {
      const score = call.quality.score;
      return <article key={call.id} className={styles.card}>
        <header><strong>{clientName(call.client)}</strong><span className={call.status === "active" ? styles.live : call.status === "failed" ? styles.failed : styles.done}>{call.status === "active" ? "live" : call.status}</span></header>
        <small>#{call.call_id} · {call.language.toUpperCase()} · {call.messages} messages</small>
        <small>{clientLabel(call.client)}</small>
        <div className={styles.path}>{(call.route_path || []).map((queue, i) => <i key={i} title={ROUTE_LABEL[queue || ""] || humanize(queue)} className={styles[`q_${queue}`] || ""} />)}</div>
        {call.advisor_repeat_streak >= 3 && <small className={styles.scoreLow}>Amira repeated the same step {call.advisor_repeat_streak}×</small>}
        {call.last_message && <p>{call.last_message.role === "advisor" ? "Amira" : "Client"}: {call.last_message.content}</p>}
        <footer>
          <span className={typeof score === "number" ? (score < 80 ? styles.scoreLow : styles.scoreOk) : ""}>{typeof score === "number" ? `QA ${score}/100` : call.status === "active" ? "QA after the call" : "QA pending"}</span>
          {call.voice?.status === "open" && call.voice.listener_url && <a href={call.voice.listener_url} target="_blank" rel="noreferrer">Listen</a>}
          <button onClick={() => onOpen(call)}>Transcript</button>
        </footer>
      </article>;
    })}</div>
  </section>;
}

function HistoryTable({ rows, more, loadMore, onOpen }: { rows: HistoryRow[]; more: boolean; loadMore: () => void; onOpen: (call: CallRef) => void }) {
  return <section className={styles.panel}>
    <div className={styles.panelHeader}><div><p className={styles.kicker}>CONVERSATION HISTORY</p><h2>Every Digital Twin conversation, from the database</h2></div></div>
    {rows.length ? <div className={styles.tableScroll}><table className={styles.historyTable}>
      <thead><tr><th>Started</th><th>Client</th><th>Source · situation</th><th>Language</th><th>Run</th><th>Status</th><th>Messages</th><th>Quality</th><th /></tr></thead>
      <tbody>{rows.map(row => <tr key={row.id}>
        <td>{shortDate(row.created_at)}</td>
        <td><b>{clientName(row.client as Client)}</b></td>
        <td>{clientLabel(row.client as Client) || "—"}</td>
        <td>{row.language.toUpperCase()}</td>
        <td>{row.batch_id ? row.batch_id.slice(0, 8) : "single"}</td>
        <td><span className={row.status === "active" ? styles.live : row.status === "failed" ? styles.failed : styles.done}>{row.status === "active" ? "live" : row.status}</span></td>
        <td>{row.messages}</td>
        <td className={typeof row.quality?.score === "number" ? (row.quality.score! < 80 ? styles.scoreLow : styles.scoreOk) : ""}>{typeof row.quality?.score === "number" ? `${row.quality.score}/100` : "—"}</td>
        <td><button onClick={() => onOpen(row)}>Transcript</button></td>
      </tr>)}</tbody></table></div> : <div className={styles.empty}>No conversation yet: start a run above.</div>}
    {more && <button className={styles.secondary} onClick={loadMore}>Show older conversations</button>}
  </section>;
}
