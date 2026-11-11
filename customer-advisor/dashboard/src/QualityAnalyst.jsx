import { useEffect, useMemo, useState } from 'react'
import './QualityAnalyst.css'

const SESSION_KEY = 'ozone-advisor-sessions-v1'
const errorHelp = {
  OLLAMA_OUT_OF_MEMORY: 'Ollama could not load the Qwen model because this computer is short of free RAM. Close other applications or containers, then run the diagnostic again.',
  CONTEXT_LIMIT_NO_TRUNCATION: 'This conversation is too long for the local model. No text was cut and no AI call was made.',
  TURN_LIMIT_NO_TRUNCATION: 'This conversation has more than 24 messages, the most the local model can review in one pass.',
  RAM_GUARD_STOPPED_REQUEST: 'Free RAM dropped too low during the analysis, so it was stopped to protect the computer.',
  SERVER_UNAVAILABLE: 'Ollama is not reachable. Check that it is running.',
}
const findingHelp = {
  accepted_offer_not_followed: 'The advisor offered to do something, the customer said yes, and the advisor did not do it.',
  same_offers_after_new_criteria: 'New criteria (e.g. a budget) returned exactly the same offers. Check the customer was told why.',
  possible_unsupported_booking_guarantee: 'The advisor may have stated that a booking or payment is confirmed.',
  advisor_turn_not_completed: 'An advisor turn did not complete.',
  offers_without_verified_evidence: 'Offers were shown without verified Ozone evidence.',
  completed_turn_without_reply: 'A turn completed without any reply.',
  turn_deadline_exceeded: 'A reply exceeded the time budget.',
  policy_question_repeated_as_offer: 'A cancellation or payment question repeated the previous offer list instead of answering the policy.',
  policy_question_unanswered: 'A cancellation or payment question received an unavailable-answer fallback instead of the verified FAQ response.',
}
const rejectionHelp = {
  CITATION_REJECTED: 'its quotes are not word-for-word in the conversation',
  PRICES_ARE_OPTIONS_FROM_ONE_RATE_LIST: 'the prices compared are different room/board options from the same rate list',
  ADVISOR_QUOTE_REQUIRED: 'it does not quote the advisor',
  TWO_ADVISOR_QUOTES_REQUIRED: 'an inconsistency needs two advisor quotes',
  QUOTE_CHRONOLOGY_REJECTED: 'its quotes are out of order',
}

function loadSessions() {
  try {
    const value = JSON.parse(localStorage.getItem(SESSION_KEY) || '[]')
    return Array.isArray(value) ? value : []
  } catch { return [] }
}

function voiceRunToSession(run) {
  if (!run?.call_id || !Array.isArray(run.transcript) || !run.transcript.length) return null
  const messages = run.transcript.map((turn, index) => {
    const kind = turn.speaker === 'advisor' ? 'agent' : 'client'
    const message = {
      id: `voice-${run.call_id}-${index}`, kind, text: String(turn.text || ''),
      sender: kind === 'agent' ? 'Amira, travel assistant' : 'Simulation client',
      time: `Turn ${turn.turn || index + 1}`,
    }
    if (kind === 'agent') message.reviewRecord = {
      schema_version: 'ozone.conversation-review/v1',
      turn: { tenant_id: 'ozone-voice-simulation', agent_id: 'customer-advisor', customer_id: `sim-${run.call_id}`, session_id: `voice-${run.call_id}`, message_id: turn.message_id || message.id, trace_id: turn.trace_id || '', channel: 'client-simulator', occurred_at: run.finished_at || new Date().toISOString() },
      customer_message: { role: 'customer', text: String(run.transcript[index - 1]?.text || '') },
      advisor_response: { role: 'advisor', text: message.text, status: turn.status || 'completed', reply_language: turn.reply_language || 'unknown', verified: Boolean(turn.verified), reason: turn.reason || '', retryable: false },
      advisor_decision: { intent: turn.intent || 'unknown', confidence: 0, tools_used: turn.tools_used || [], trip: turn.trip || {}, offers: turn.offers || [], sources: [], handoff_required: Boolean(turn.handoff_required) },
      simulation: { call_id: run.call_id, persona: run.persona || 'voice scenario', outcome: run.outcome || 'unknown' },
    }
    return message
  })
  return {
    id: `voice-${run.call_id}`, title: `Voice simulation - ${run.persona || run.call_id}`,
    customer: { id: `sim-${run.call_id}`, name: 'Simulation client', language: 'en', status: 'Simulation complete' },
    messages, result: { reply_language: messages.find((item) => item.reviewRecord)?.reviewRecord?.advisor_response?.reply_language || 'unknown' },
    updatedAt: run.finished_at || new Date().toISOString(), simulation: true,
  }
}

function mergeVoiceRuns(sessions, runs) {
  const byId = new Map((sessions || []).map((session) => [session.id, session]))
  for (const run of runs || []) {
    const session = voiceRunToSession(run)
    if (session) byId.set(session.id, session)
  }
  return [...byId.values()]
}

function transcriptFor(session) {
  return (session?.messages || []).filter((message) => ['client', 'agent'].includes(message.kind)).map((message) => ({
    role: message.kind === 'client' ? 'customer' : 'advisor',
    text: String(message.text || '').trim(),
  })).filter((message) => message.text)
}

function reviewRecordsFor(session) {
  return (session?.messages || []).map((message) => message.reviewRecord).filter((record) => record?.schema_version === 'ozone.conversation-review/v1')
}

function AnalysisResult({ result }) {
  if (!result) return <div className="quality-empty">Choose a conversation, then request a diagnostic. The analyst never changes the advisor conversation.</div>
  const run = result.result || {}
  const combined = result.combined_scorecard || {}
  const checked = run.checked || {}
  const verification = checked.verification || {}
  const weaknesses = verification.weaknesses || []
  const open = weaknesses.filter((entry) => entry.status !== 'rejected')
  return <div className="quality-result">
    {result.scorecard && <article className="quality-score"><small>DETERMINISTIC QUALITY SCORE</small><strong>{result.scorecard.score_100 == null ? 'Review required' : `${result.scorecard.score_100}/100`}</strong><p>{result.scorecard.turns_checked} turn(s) checked · {result.scorecard.coverage.replaceAll('_', ' ')}</p>{result.scorecard.findings?.length > 0 && <ul>{result.scorecard.findings.map((finding, index) => <li key={index}><b>{finding.turn ? `Turn ${finding.turn}` : 'Conversation'} · {finding.severity}</b> — {findingHelp[finding.code] || finding.code.replaceAll('_', ' ')}</li>)}</ul>}</article>}
    {combined.status && <article className="quality-score"><small>COMBINED REVIEW SIGNAL</small><strong>{combined.composite_score_100 == null ? 'Review required' : `${combined.composite_score_100}/100`}</strong><p>{combined.status.replaceAll('_', ' ')}; the deterministic score remains authoritative.</p>{combined.llm_hypotheses > 0 && <p>{combined.llm_hypotheses} cited Qwen coaching signal(s), capped at {combined.max_llm_penalty} points.</p>}</article>}
    <div className="quality-result-status"><span className={run.status === 'completed' ? 'quality-ok' : 'quality-warn'}>{run.status || 'unknown'}</span><span>{run.provider || 'provider not selected'}</span><span>{run.wall_seconds ? `${Number(run.wall_seconds).toFixed(1)} s` : 'not started'}</span></div>
    {run.errors?.length > 0 && <div className="quality-error">{run.errors.map((code) => errorHelp[code] ? `${code}: ${errorHelp[code]}` : code).join(' · ')}</div>}
    {result.saved_review && <div className="quality-meta"><span>Saved server review</span><span>{result.saved_review.review_id}</span><span>{result.saved_review.message_count} message(s)</span></div>}
    {checked.proposal && <>
      <p className="quality-unverified">AI summary written by Qwen. It is not checked against the conversation; read it as a draft.</p>
      <article><small>CUSTOMER NEED · AI, UNVERIFIED</small><p>{checked.proposal.needs}</p></article>
      <article><small>ADVISOR HANDLING · AI, UNVERIFIED</small><p>{checked.proposal.handling}</p></article>
      <article><small>REVIEW STATUS</small><p>{open.length ? `${open.length} potential issue${open.length === 1 ? '' : 's'} require human review.` : 'No potential issue passed the checks. This is not proof that the conversation is error-free.'}{weaknesses.length > open.length && ` ${weaknesses.length - open.length} AI suggestion${weaknesses.length - open.length === 1 ? ' was' : 's were'} rejected automatically.`}</p></article>
      {weaknesses.map((entry, index) => <article className={`quality-finding${entry.status === 'rejected' ? ' quality-rejected' : ''}`} key={entry.weakness?.title || index}><b>{entry.weakness?.title || 'Potential issue'}</b><p>{entry.weakness?.description}</p>{entry.status === 'rejected'
        ? <small>Rejected automatically: {(entry.issues || []).map((code) => rejectionHelp[code] || code.replaceAll('_', ' ').toLowerCase()).join('; ')}.</small>
        : <small>Quotes verified word-for-word. Whether this is really a problem still needs a human decision.</small>}{entry.status !== 'rejected' && entry.weakness?.recommendation && <p><i>Suggestion: {entry.weakness.recommendation}</i></p>}</article>)}
    </>}
    {!checked.proposal && !run.errors?.length && <div className="quality-empty">The diagnostic did not return a usable structured report. No conclusion has been saved.</div>}
  </div>
}

export default function QualityAnalyst() {
  const [sessions, setSessions] = useState(loadSessions)
  const sorted = useMemo(() => [...sessions].sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt))), [sessions])
  const [sessionId, setSessionId] = useState(() => sorted[0]?.id || '')
  const [provider, setProvider] = useState('ollama')
  const [complete, setComplete] = useState(true)
  const [consent, setConsent] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)
  useEffect(() => {
    let cancelled = false
    async function syncVoiceHistory() {
      try {
        const response = await fetch('/voice-simulator-api/runs')
        const data = await response.json()
        if (!response.ok || cancelled) return
        setSessions((current) => {
          const merged = mergeVoiceRuns(current, data.runs)
          try { localStorage.setItem(SESSION_KEY, JSON.stringify(merged)) } catch {}
          return merged
        })
      } catch { /* Voice Lab can be offline without blocking normal reviews. */ }
    }
    syncVoiceHistory()
    const timer = window.setInterval(syncVoiceHistory, 5000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [])
  useEffect(() => {
    if (!sessionId && sorted[0]?.id) setSessionId(sorted[0].id)
  }, [sessionId, sorted])
  const session = sorted.find((item) => item.id === sessionId)
  const transcript = transcriptFor(session)
  const reviewRecords = reviewRecordsFor(session)

  async function analyze() {
    if (!transcript.length || loading) return
    setLoading(true); setError(''); setResult(null)
    try {
      const response = await fetch('/quality-analyst/analyze', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...(reviewRecords.length ? { review_records: reviewRecords } : { messages: transcript }),
          language: session?.result?.reply_language === 'en' ? 'en' : session?.result?.reply_language === 'ar' ? 'ar' : session?.result?.reply_language?.startsWith('aeb') ? 'aeb' : session?.result?.reply_language === 'fr' ? 'fr' : 'unknown',
          conversation_complete: complete,
          provider,
          cloud_consent: provider === 'gemini' && consent,
          free_tier_confirmed: provider === 'gemini' && consent,
        }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Quality Analyst is unavailable')
      setResult(data)
    } catch (reason) { setError(reason.message || 'Quality Analyst is unavailable') } finally { setLoading(false) }
  }

  return <div className="quality-shell">
    <header className="quality-header"><div><span className="crumb">OZONE VOYAGES / QUALITY REVIEW</span><h1>Conversation quality, with evidence.</h1><p>Review Amira conversations without changing the customer record or making autonomous decisions.</p></div><div className="quality-actions"><a href="/">Customer workspace</a><a href="/monitoring">Operations</a></div></header>
    <div className="quality-notice"><b>Diagnostic only</b><span>Findings are review aids. Exact citations validate text attribution, not an AI interpretation or a historical booking claim.</span></div>
    <main className="quality-grid">
      <section className="quality-card quality-setup"><header><small>1. SOURCE CONVERSATION</small><h2>Choose a saved advisor chat</h2></header>
        <label>Conversation<select value={sessionId} onChange={(event) => { setSessionId(event.target.value); setResult(null); setError('') }}><option value="">Choose a conversation</option>{sorted.map((item) => <option key={item.id} value={item.id}>{item.customer?.name || 'Walk-in client'} — {item.title || 'New conversation'}</option>)}</select></label>
        <div className="quality-meta"><span>{transcript.length} analyzable message{transcript.length === 1 ? '' : 's'}</span><span>{reviewRecords.length ? `${reviewRecords.length} shared advisor record(s)` : 'Legacy transcript: no advisor metadata'}</span><span>{session?.updatedAt ? new Date(session.updatedAt).toLocaleString() : 'No conversation selected'}</span></div>
        <label>Analysis engine<select value={provider} onChange={(event) => { setProvider(event.target.value); setConsent(false) }}><option value="ollama">Qwen local (Advisor model)</option></select></label>
        <label className="quality-check"><input type="checkbox" checked={complete} onChange={(event) => setComplete(event.target.checked)} /> This transcript is complete</label>
        {provider === 'gemini' && <label className="quality-check quality-consent"><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} /> I authorize this diagnostic transcript to be sent to Gemini and confirm the project has no billing enabled.</label>}
        <button className="quality-primary" type="button" onClick={analyze} disabled={!transcript.length || loading || (provider === 'gemini' && !consent)}>{loading ? 'Analysing conversation…' : 'Run quality diagnostic'}</button>
        {error && <div className="quality-error">{error}</div>}
      </section>
      <section className="quality-card quality-transcript"><header><small>2. IMMUTABLE REVIEW INPUT</small><h2>Advisor transcript</h2></header><div className="quality-messages">{transcript.length ? transcript.map((message, index) => <article className={message.role} key={index}><small>{message.role === 'customer' ? 'CUSTOMER' : 'AMIRA / ADVISOR'}</small><p dir="auto">{message.text}</p></article>) : <div className="quality-empty">Saved conversations from the customer workspace will appear here.</div>}</div></section>
      <section className="quality-card quality-report"><header><small>3. HUMAN REVIEW OUTPUT</small><h2>Quality Analyst report</h2></header><AnalysisResult result={result} /></section>
    </main>
  </div>
}
