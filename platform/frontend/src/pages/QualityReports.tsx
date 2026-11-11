// Quality Analyst page: every Digital Twin call's Quality report, by run.
import { useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { WorkspaceHeader } from "@/pages/AgentHub";
import { api } from "@/api";
import { CallRef, CallReview, Client, QualityBrief, QualityReport, TranscriptModal, clientLabel, clientName, findingLabel, shortDate } from "@/twin/shared";
import styles from "./QualityReports.module.css";

type Report = { ai_review?: CallReview | null; ai_quality_report?: AiReport | null; id: string; call_id: string; batch_id?: string; language: string; status: string; client?: Client;
  created_at: string; completed_at?: string; messages: number; quality: QualityBrief; quality_review: Record<string, unknown> };
type AiReport = {
  outcome_label: string; outcome_note: string; ai_score: number | null; rule_score?: number | null; verdict: string; human_review: boolean;
  criteria: { key: string; label: string; score: number | null; reason: string }[];
  strengths: { point: string; turn: number; quote: string }[];
  problems: { title: string; severity: string; turn: number; quote: string; why: string; better_reply?: string }[];
  improvement: string; model?: string; created_at: string;
};
type RunListItem = { id: string; calls: number; created_at: string; active: number };
type Filter = "all" | "low" | "review";

export function QualityReports() {
  const [params, setParams] = useSearchParams();
  const runId = params.get("batch") || "";
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [open, setOpen] = useState<CallRef | null>(null);

  useEffect(() => { api.get<RunListItem[]>("/digital-twin/batches?limit=30").then(setRuns).catch(() => undefined); }, []);
  useEffect(() => {
    setLoading(true);
    api.get<Report[]>(`/digital-twin/quality/reports?limit=200${runId ? `&batch_id=${encodeURIComponent(runId)}` : ""}`)
      .then(rows => { setReports(rows); setSelected(rows[0]?.id || ""); })
      .catch(() => setReports([]))
      .finally(() => setLoading(false));
  }, [runId]);

  const scores = reports.map(r => r.quality.score).filter((s): s is number => typeof s === "number");
  const topFindings = useMemo(() => {
    const counts: Record<string, number> = {};
    reports.forEach(r => (r.quality.codes || []).forEach(code => { counts[code] = (counts[code] || 0) + 1; }));
    return Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 6);
  }, [reports]);
  const visible = reports.filter(r => filter === "all" || (filter === "low" ? (r.quality.score ?? 100) < 80 : r.quality.human_review_required));
  const current = reports.find(r => r.id === selected) || visible[0];

  return <div className={styles.page}><WorkspaceHeader />
    <main className={styles.shell}>
      <section className={styles.hero}>
        <div><p>OZONE VOYAGES / QUALITY ANALYST</p><h1>Every conversation, reviewed with evidence.</h1>
          <span>Rule checks score each Digital Twin call; an AI reads every call (the analyst's local model, or an API model when the PC lacks RAM) and lists issues whose quotes are verified word for word. Diagnostic only: it never changes a conversation.</span></div>
        <label>Run<select value={runId} onChange={e => setParams(e.target.value ? { batch: e.target.value } : {})}>
          <option value="">All runs</option>
          {runs.map(run => <option key={run.id} value={run.id}>{shortDate(run.created_at)} · {run.calls} calls{run.active ? " · running" : ""}</option>)}
        </select></label>
      </section>

      <section className={styles.summary}>
        <div><b>{reports.length}</b><small>calls reviewed</small></div>
        <div><b>{scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(1) : "—"}</b><small>mean score</small></div>
        <div><b>{scores.length ? Math.min(...scores) : "—"}</b><small>lowest score</small></div>
        <div className={scores.filter(s => s < 80).length ? styles.warn : ""}><b>{scores.filter(s => s < 80).length}</b><small>below 80</small></div>
        <div><b>{reports.filter(r => r.ai_quality_report).length}</b><small>AI quality reports</small></div>
        <div className={styles.top}><small>Most frequent findings</small><div>{topFindings.length ? topFindings.map(([code, n]) => <span key={code}>{findingLabel(code)} <b>{n}</b></span>) : <em>none</em>}</div></div>
      </section>

      <section className={styles.workspace}>
        <aside className={styles.list}>
          <div className={styles.filters}>{([["all", "All"], ["low", "Below 80"], ["review", "Needs review"]] as [Filter, string][]).map(([key, label]) =>
            <button key={key} className={filter === key ? styles.active : ""} onClick={() => setFilter(key)}>{label}</button>)}</div>
          {loading && <p className={styles.muted}>Loading reports…</p>}
          {!loading && !visible.length && <p className={styles.muted}>No report{runId ? " for this run" : ""} yet. Reports appear as soon as a call ends.</p>}
          {visible.map(report => <button key={report.id} className={`${styles.item} ${current?.id === report.id ? styles.selected : ""}`} onClick={() => setSelected(report.id)}>
            <span className={styles.itemTop}><b>{clientName(report.client)}</b><em className={(report.quality.score ?? 100) < 80 ? styles.low : styles.ok}>{report.quality.score ?? "—"}</em></span>
            <small>{report.language.toUpperCase()} · {shortDate(report.completed_at || report.created_at)} · {report.quality.findings || 0} finding(s){report.ai_quality_report ? ` · AI report ${report.ai_quality_report.ai_score ?? ""}` : ""}</small>
          </button>)}
        </aside>
        <article className={styles.detail}>
          {current ? <>
            <header>
              <div><p className={styles.kicker}>CALL #{current.call_id} · {current.status}</p><h2>{clientName(current.client)}</h2>
                <small>{clientLabel(current.client) || "Client"} · {current.language.toUpperCase()} · {current.messages} messages · ended {shortDate(current.completed_at)}</small></div>
              <button onClick={() => setOpen({ id: current.id, call_id: current.call_id, language: current.language, status: current.status, client: current.client, quality: current.quality })}>View conversation</button>
            </header>
            <AiReportPanel key={current.id} report={current}
              onSaved={saved => setReports(rows => rows.map(r => r.id === current.id ? { ...r, ai_quality_report: saved } : r))} />
            <QualityReport review={current.quality_review} />
          </> : <p className={styles.muted}>Select a conversation to see its report.</p>}
        </article>
      </section>
      <p className={styles.footnote}>The Customer Advisor workspace keeps its own Quality tool for its saved chats: <Link to="/quality-analyst/original">open it</Link>.</p>
      {open && <TranscriptModal call={open} close={() => setOpen(null)} />}
    </main>
  </div>;
}

const SCORE_CLASS = (score: number | null | undefined) => typeof score !== "number" ? "" : score >= 80 ? styles.ok : score >= 60 ? styles.mid : styles.low;

/** The button that writes (or shows) the call's AI quality report. */
function AiReportPanel({ report, onSaved }: { report: Report; onSaved: (saved: AiReport) => void }) {
  const stored = report.ai_quality_report || null;
  const [shown, setShown] = useState(false);
  const [state, setState] = useState<"idle" | "writing" | "error">("idle");
  const [error, setError] = useState("");
  async function generate() {
    setState("writing"); setError("");
    try {
      onSaved(await api.post<AiReport>(`/digital-twin/sessions/${report.id}/ai-quality-report`, {}));
      setShown(true); setState("idle");
    } catch (e) { setError(e instanceof Error ? e.message : "No language model answered."); setState("error"); }
  }
  const writing = state === "writing";
  return <section className={styles.aiBlock}>
    <div className={styles.aiHead}>
      <div><h3>AI quality report</h3>
        <small>{stored ? `Written ${shortDate(stored.created_at)} by ${stored.model || "an API model"}` : "Scores the call on six criteria, with quoted evidence for every point."}</small></div>
      <div className={styles.aiButtons}>
        {stored && <button className={styles.ghost} onClick={() => setShown(v => !v)} disabled={writing}>{shown ? "Hide report" : "View AI quality report"}</button>}
        <button onClick={generate} disabled={writing}>{writing ? "Writing the report…" : stored ? "Regenerate" : "Generate AI quality report"}</button>
      </div>
    </div>
    {state === "error" && <p className={styles.low}>The report could not be written: {error}</p>}
    {stored && shown && <AiReportView r={stored} />}
  </section>;
}

function AiReportView({ r }: { r: AiReport }) {
  return <div className={styles.aiReport}>
    <div className={styles.aiTop}>
      <div><small>Outcome</small><b>{r.outcome_label}</b><span>{r.outcome_note}</span></div>
      <div><small>AI score</small><b className={SCORE_CLASS(r.ai_score)}>{r.ai_score ?? "—"}<i>/100</i></b><span>mean of the six criteria</span></div>
      <div><small>Rule score</small><b className={SCORE_CLASS(r.rule_score)}>{r.rule_score ?? "—"}<i>/100</i></b><span>Quality Analyst rule checks</span></div>
      <div><small>Human review</small><b className={r.human_review ? styles.low : styles.ok}>{r.human_review ? "Needed" : "Not needed"}</b></div>
    </div>
    {r.verdict && <p className={styles.verdict}>{r.verdict}</p>}
    <h4>Scores by criterion</h4>
    <div className={styles.criteria}>{r.criteria.map(c => <div key={c.key}>
      <span>{c.label}</span>
      <span className={styles.dots} aria-label={`${c.score ?? 0} out of 5`}>{[1, 2, 3, 4, 5].map(n => <i key={n} className={c.score && n <= c.score ? SCORE_CLASS(c.score * 20) : ""} />)}</span>
      <b>{c.score ?? "—"}/5</b><small>{c.reason}</small>
    </div>)}</div>
    {!!r.strengths.length && <><h4>Strengths</h4><ul className={styles.evidence}>{r.strengths.map((s, i) => <li key={i}>
      {s.point} <small>(message {s.turn})</small><q>{s.quote}</q></li>)}</ul></>}
    <h4>Problems</h4>
    {r.problems.length ? <ul className={styles.evidence}>{r.problems.map((p, i) => <li key={i}>
      <span className={`${styles.sev} ${styles[`sev_${p.severity}`] || ""}`}>{p.severity}</span> <b>{p.title}</b> <small>(message {p.turn})</small>
      <q>{p.quote}</q><small>Why: {p.why}</small>{p.better_reply && <small>Better reply: {p.better_reply}</small>}</li>)}</ul>
      : <p className={styles.muted}>No problem proven by a quote from the conversation.</p>}
    {r.improvement && <div className={styles.improve}><small>For the team</small><p>{r.improvement}</p></div>}
    <small className={styles.muted}>Every quote is checked word for word against the conversation; points whose quote is not found are left out. The AI score is computed from the six criteria.</small>
  </div>;
}
