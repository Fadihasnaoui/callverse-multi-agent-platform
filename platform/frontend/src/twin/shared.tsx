// Pieces shared by the Digital Twin and Quality Analyst pages: the full
// conversation window and the Quality report of one call.
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { api } from "@/api";
import styles from "./twin.module.css";

export type Message = { id: string; sequence: number; role: "client" | "advisor" | "system"; content: string };
export type Client = { source?: string; name?: string; persona?: string; archetype?: string; scenario?: string; difficulty?: string };
export type QualityBrief = { score?: number | null; status?: string; findings?: number; codes?: string[]; llm_checked?: boolean; human_review_required?: boolean };
export type TwinSession = {
  id: string; call_id?: string; batch_id?: string; language: string; status: string; client?: Client;
  quality_review?: Record<string, unknown>; created_at: string; completed_at?: string; messages?: Message[];
};
/** What the window needs about a call; board cards and history rows both provide it. */
export type CallRef = {
  id: string; call_id?: string; language: string; status: string; client?: Client; route_path?: (string | null)[];
  voice?: { status?: string; listener_url?: string }; quality?: QualityBrief;
};

export function humanize(value?: string | null) { return (value || "").replace(/_/g, " "); }
export const SOURCE_LABEL: Record<string, string> = { jasmin_profile: "Jasmin profile", jasmin_generated: "New Jasmin client", scripted: "Voice Lab scripted" };
export function clientName(client?: Client) { return client?.name || humanize(client?.persona) || "Client"; }
export function clientLabel(client?: Client) {
  if (!client?.source) return "";
  return [SOURCE_LABEL[client.source] || client.source, humanize(client.archetype), humanize(client.scenario), client.difficulty].filter(Boolean).join(" · ");
}
export const ROUTE_LABEL: Record<string, string> = {
  ai: "Amira (AI)", customer_advisors: "Human adviser", reservations: "Reservations", waiting_human: "Waiting for an adviser",
  waiting: "Waiting", callbacks: "Callback",
};
export function shortDate(value?: string) {
  return value ? new Date(value).toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : "—";
}

/** The full conversation, opened over the page; refreshes every 3 s while the call is live. */
export function TranscriptModal({ call, close }: { call: CallRef; close: () => void }) {
  const [session, setSession] = useState<TwinSession | null>(null);
  const [closing, setClosing] = useState(false);
  const shut = () => { setClosing(true); window.setTimeout(close, 180); };
  const live = (session?.status || call.status) === "active";
  useEffect(() => {
    let alive = true;
    const load = () => api.get<TwinSession>(`/digital-twin/sessions/${call.id}`).then(s => { if (alive) setSession(s); }).catch(() => undefined);
    load();
    const timer = window.setInterval(() => { if (alive) load(); }, 3000);
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") shut(); };
    window.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { alive = false; window.clearInterval(timer); window.removeEventListener("keydown", onKey); document.body.style.overflow = overflow; };
  }, [call.id]);
  // Follow a live conversation down as new turns arrive.
  const count = session?.messages?.length || 0;
  useEffect(() => { if (live) document.getElementById("twin-transcript-end")?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [count, live]);
  const messages = session?.messages || [];
  const score = call.quality?.score;
  let clientTurn = -1;
  // Mounted on <body> so no page layout or scroll container can hide it.
  return createPortal(<div className={`${styles.backdrop} ${closing ? styles.closing : ""}`} onClick={shut} role="presentation">
    <div className={styles.modal} role="dialog" aria-modal="true" aria-label="Full conversation" onClick={event => event.stopPropagation()}>
      <header className={styles.modalHeader}>
        <div><p className={styles.kicker}>FULL CONVERSATION{call.call_id ? ` · #${call.call_id}` : ""}</p>
          <h2>{clientName(call.client || session?.client)}</h2>
          <small>{clientLabel(call.client || session?.client)}{clientLabel(call.client || session?.client) ? " · " : ""}{call.language.toUpperCase()} · {live ? "live" : session?.status || call.status} · {messages.length} messages{typeof score === "number" ? ` · Quality ${score}/100` : ""}</small></div>
        <div className={styles.modalActions}>
          {live && <span className={styles.livePill}>● Live</span>}
          {call.voice?.status === "open" && call.voice.listener_url && <a href={call.voice.listener_url} target="_blank" rel="noreferrer">Listen</a>}
          <button onClick={shut} aria-label="Close">✕</button>
        </div>
      </header>
      <div className={styles.modalBody}>
        {!session && <div className={styles.empty}>Loading the conversation…</div>}
        {session && !messages.length && <div className={styles.empty}>The call has just started; its first turn will appear here.</div>}
        {messages.map(message => {
          if (message.role === "client") clientTurn += 1;
          const route = message.role === "client" ? call.route_path?.[clientTurn] : null;
          return <article key={message.id} className={message.role === "advisor" ? styles.advisor : message.role === "system" ? styles.system : styles.client}>
            <small>{message.role === "advisor" ? "AMIRA / CUSTOMER ADVISOR" : message.role === "system" ? "OZONE OPERATIONS" : `CLIENT · ${clientName(call.client || session?.client)}`}
              {route && <span className={styles.routeChip}>Workforce → {ROUTE_LABEL[route] || humanize(route)}</span>}</small>
            <p>{message.content}</p>
          </article>;
        })}
        {live && session && <div className={styles.typing}>Conversation in progress…</div>}
        <span id="twin-transcript-end" />
      </div>
    </div>
  </div>, document.body);
}

export const FINDING_HELP: Record<string, string> = {
  accepted_offer_not_followed: "The advisor offered to do something, the customer said yes, and the advisor did not do it.",
  same_offers_after_new_criteria: "New criteria (e.g. a budget) returned exactly the same offers. Check the customer was told why.",
  possible_unsupported_booking_guarantee: "The advisor may have stated that a booking or payment is confirmed.",
  advisor_turn_not_completed: "An advisor turn did not complete.",
  offers_without_verified_evidence: "Offers were shown without verified Ozone evidence.",
  completed_turn_without_reply: "A turn completed without any reply.",
  turn_deadline_exceeded: "A reply exceeded the time budget.",
  policy_question_repeated_as_offer: "A cancellation or payment question repeated the previous offer list instead of answering the policy.",
  policy_question_unanswered: "A cancellation or payment question received an unavailable-answer fallback instead of the verified FAQ response.",
  advisor_repetition_loop: "The advisor gave the same decision three or more times in a row.",
  advisor_repetition_loop_extended: "The repetition loop went on for further turns.",
  same_reply_to_different_questions: "The same answer was pasted in reply to different customer questions.",
  provided_information_ignored: "The customer gave the information, and the advisor asked for it again.",
  reply_language_mismatch: "The reply was not in the customer's language.",
  offers_for_wrong_destination: "Offers were for a different destination than the customer asked for.",
  customer_request_unanswered: "The customer had to send the same message three or more times.",
  booking_request_unanswered: "The customer chose an offer and got a generic “cannot answer” reply.",
  scenario_expected_intent_missing: "The scenario's expected intent never appeared.",
};
export function findingLabel(code: string) {
  return { advisor_repetition_loop: "Repetition loop", advisor_repetition_loop_extended: "Long repetition", same_reply_to_different_questions: "Same reply, different questions",
    provided_information_ignored: "Answer ignored", reply_language_mismatch: "Wrong language", offers_for_wrong_destination: "Wrong destination",
    customer_request_unanswered: "Client left unanswered", booking_request_unanswered: "Booking not handled", turn_deadline_exceeded: "Slow reply",
    possible_unsupported_booking_guarantee: "Possible booking guarantee", policy_question_unanswered: "Policy question unanswered",
    policy_question_repeated_as_offer: "Policy answered with offers" }[code] || humanize(code);
}
const REJECTION_HELP: Record<string, string> = {
  CITATION_REJECTED: "its quotes are not word-for-word in the conversation",
  PRICES_ARE_OPTIONS_FROM_ONE_RATE_LIST: "the prices compared are different room/board options from the same rate list",
  ADVISOR_QUOTE_REQUIRED: "it does not quote the advisor",
  TWO_ADVISOR_QUOTES_REQUIRED: "an inconsistency needs two advisor quotes",
  QUOTE_CHRONOLOGY_REJECTED: "its quotes are out of order",
};
type Finding = { code: string; severity: string; turn?: number | null };
type Weakness = { status?: string; issues?: string[]; weakness?: { title?: string; description?: string; recommendation?: string } };

/** The Quality Analyst's report for one call, from the review saved with it. */
export type CallReview = {
  session_id: string; call_id?: string; client: string; language?: string; scenario?: string; rule_score?: number | null;
  rule_findings: string[]; checked_by?: string | null; model?: string; verdict?: string; strengths?: string[]; error?: string;
  weaknesses?: { title?: string; turn?: number; quote?: string; severity?: string; recommendation?: string }[];
};

/** One call's AI reading: the Quality Analyst's local model, or the API model when the local check was skipped. */
export function AiReview({ review, showRules = true }: { review: CallReview; showRules?: boolean }) {
  return <div className={styles.aiReview}>
    {review.error && <p className={styles.aiWarn}>{review.error}</p>}
    {review.verdict && <p>{review.verdict}</p>}
    {!!review.strengths?.length && <><b>Went well</b><ul>{review.strengths.map((s, i) => <li key={i}>{s}</li>)}</ul></>}
    {!!review.weaknesses?.length && <><b>Problems</b><ul>{review.weaknesses.map((w, i) => <li key={i}>
      <span className={styles.sev}>{w.severity || "issue"}</span> {w.title}{w.turn ? ` (message ${w.turn})` : ""}
      {w.quote && <q>{w.quote}</q>}{w.recommendation && <small>→ {w.recommendation}</small>}</li>)}</ul></>}
    {review.checked_by && !review.weaknesses?.length && <p>No problem found by the AI reading.</p>}
    {showRules && !!review.rule_findings.length && <small>Rule findings: {review.rule_findings.join(", ")}</small>}
  </div>;
}

export function QualityReport({ review }: { review: Record<string, unknown> }) {
  const scorecard = review.scorecard as { score_100?: number | null; turns_checked?: number; coverage?: string; findings?: Finding[] } | undefined;
  const combined = review.combined_scorecard as { status?: string; composite_score_100?: number | null; llm_hypotheses?: number; max_llm_penalty?: number } | undefined;
  const run = (review.result || {}) as { status?: string; provider?: string; wall_seconds?: number; errors?: string[];
    checked?: { proposal?: { needs?: string; handling?: string }; verification?: { weaknesses?: Weakness[] } } };
  const weaknesses = run.checked?.verification?.weaknesses || [];
  const open = weaknesses.filter(entry => entry.status !== "rejected");
  const rules = scorecard?.score_100;
  const final = combined?.composite_score_100 ?? rules;
  return <div className={styles.report}>
    <div className={styles.scores}>
      <div><small>Final score</small><b className={typeof final === "number" && final < 80 ? styles.low : styles.ok}>{final == null ? "Review" : `${final}`}<em>/100</em></b></div>
      <div><small>Rules score</small><b>{rules == null ? "—" : rules}<em>/100</em></b><span>{scorecard?.turns_checked || 0} turns checked</span></div>
      <div><small>AI check</small><b className={styles.small}>{run.status === "free_quality_checked" ? "Done" : "Not run"}</b><span>{run.status === "free_quality_checked" ? `${combined?.llm_hypotheses || 0} cited signal(s) · ${run.wall_seconds ? Number(run.wall_seconds).toFixed(0) : "—"} s` : run.errors?.join(", ") || "rules only"}</span></div>
    </div>
    {!!scorecard?.findings?.length && <section><h4>Rule findings</h4><ul className={styles.findings}>{scorecard.findings.map((finding, index) =>
      <li key={index}><span className={styles[`sev_${finding.severity}`] || styles.sev}>{finding.severity}</span><div><b>{findingLabel(finding.code)}{finding.turn ? ` · turn ${finding.turn}` : ""}</b><p>{FINDING_HELP[finding.code] || humanize(finding.code)}</p></div></li>)}</ul></section>}
    {!scorecard?.findings?.length && <p className={styles.clean}>No rule finding on this conversation.</p>}
    {run.checked?.proposal && <section><h4>AI reading of the call <em>· draft, not verified</em></h4>
      <div className={styles.reading}><div><small>Customer need</small><p>{run.checked.proposal.needs}</p></div><div><small>Advisor handling</small><p>{run.checked.proposal.handling}</p></div></div></section>}
    {!!weaknesses.length && <section><h4>Potential issues <em>· {open.length} to review{weaknesses.length > open.length ? `, ${weaknesses.length - open.length} rejected automatically` : ""}</em></h4>
      {weaknesses.map((entry, index) => <article key={entry.weakness?.title || index} className={entry.status === "rejected" ? styles.rejected : styles.issue}>
        <b>{entry.weakness?.title || "Potential issue"}</b><p>{entry.weakness?.description}</p>
        {entry.status === "rejected"
          ? <small>Rejected automatically: {(entry.issues || []).map(code => REJECTION_HELP[code] || humanize(code).toLowerCase()).join("; ")}.</small>
          : <small>Quotes verified word for word. Whether this is a real problem still needs a human decision.</small>}
        {entry.status !== "rejected" && entry.weakness?.recommendation && <p className={styles.suggestion}>Suggestion: {entry.weakness.recommendation}</p>}
      </article>)}</section>}
  </div>;
}
