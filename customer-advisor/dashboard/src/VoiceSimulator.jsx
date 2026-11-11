import { useEffect, useState } from 'react'
import './VoiceSimulator.css'

const api = '/voice-simulator-api'
const SESSION_KEY = 'ozone-advisor-sessions-v1'

function saveRunForQuality(run) {
  if (!run?.call_id || !Array.isArray(run.transcript)) return false
  const messages = run.transcript.map((turn, index) => {
    const kind = turn.speaker === 'advisor' ? 'agent' : 'client'
    const message = { id: `voice-${run.call_id}-${index}`, kind, text: String(turn.text || ''), sender: kind === 'agent' ? 'Amira, travel assistant' : 'Simulation client', time: `Turn ${turn.turn || index + 1}` }
    if (kind === 'agent') message.reviewRecord = {
      schema_version: 'ozone.conversation-review/v1',
      turn: { tenant_id: 'ozone-voice-simulation', agent_id: 'customer-advisor', customer_id: `sim-${run.call_id}`, session_id: `voice-${run.call_id}`, message_id: turn.message_id || message.id, trace_id: turn.trace_id || '', channel: 'client-simulator', occurred_at: new Date().toISOString() },
      customer_message: { role: 'customer', text: String(run.transcript[index - 1]?.text || '') },
      advisor_response: { role: 'advisor', text: message.text, status: turn.status || 'completed', reply_language: turn.reply_language || 'unknown', verified: Boolean(turn.verified), reason: turn.reason || '', retryable: false },
      advisor_decision: { intent: turn.intent || 'unknown', confidence: 0, tools_used: turn.tools_used || [], trip: turn.trip || {}, offers: turn.offers || [], sources: [], handoff_required: Boolean(turn.handoff_required) },
      simulation: { call_id: run.call_id, persona: run.persona || 'voice scenario' },
    }
    return message
  })
  try {
    const existing = JSON.parse(localStorage.getItem(SESSION_KEY) || '[]')
    const sessions = Array.isArray(existing) ? existing.filter((item) => item.id !== `voice-${run.call_id}`) : []
    sessions.unshift({ id: `voice-${run.call_id}`, title: `Voice lab - ${run.persona || run.call_id}`, customer: { id: `sim-${run.call_id}`, name: 'Simulation client', language: 'en', status: 'Simulation' }, messages, result: { reply_language: messages.find((item) => item.reviewRecord)?.reviewRecord?.advisor_response?.reply_language || 'unknown' }, updatedAt: new Date().toISOString(), simulation: true })
    localStorage.setItem(SESSION_KEY, JSON.stringify(sessions))
    return true
  } catch { return false }
}

export default function VoiceSimulator() {
  const [personas, setPersonas] = useState([])
  const [selected, setSelected] = useState('')
  const [active, setActive] = useState(null)
  const [runs, setRuns] = useState([])
  const [error, setError] = useState('')
  const [starting, setStarting] = useState(false)
  async function refresh() {
    try {
      const [scenarioResponse, runResponse] = await Promise.all([fetch(`${api}/personas`), fetch(`${api}/runs`)])
      const scenarioData = await scenarioResponse.json(); const runData = await runResponse.json()
      if (!scenarioResponse.ok || !runResponse.ok) throw new Error(runData.error || scenarioData.error || 'Voice lab is unavailable')
      setPersonas(scenarioData.personas || []); setSelected((value) => value || scenarioData.personas?.[0]?.id || '')
      setActive(runData.active); setRuns(runData.runs || []); setError('')
    } catch (reason) { setError(reason.message || 'Voice lab is unavailable') }
  }
  useEffect(() => { refresh(); const id = window.setInterval(refresh, 4000); return () => window.clearInterval(id) }, [])
  async function start() {
    setStarting(true); setError('')
    try {
      const response = await fetch(`${api}/runs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ persona: selected, audio: 'livekit' }) })
      const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Could not start the call'); setActive(data.active)
    } catch (reason) { setError(reason.message || 'Could not start the call') } finally { setStarting(false) }
  }
  function openQuality(run) { if (!saveRunForQuality(run)) { setError('The voice transcript could not be saved for Quality Review'); return }; window.location.assign('/quality') }
  const latest = runs[0]
  return <main className="voice-shell">
    <header className="voice-header"><div><span>OZONE VOYAGES / DIGITAL TWIN</span><h1>Voice simulation lab</h1><p>Run one scripted client scenario through Amira&apos;s isolated API, then hear and inspect the exact conversation.</p></div><div className="voice-actions"><a href="/">Customer advisor</a><a href="/simulator">Jasmin scenarios</a><a href="/quality">Quality review</a></div></header>
    <section className="voice-safe"><b>Safe test boundary</b><span>Scenario data uses a synthetic customer, isolated Advisor port 8011 and a dedicated simulation tenant. It does not contact or overwrite live customer conversations.</span></section>
    {error && <div className="voice-error">{error} <button onClick={refresh}>Retry</button></div>}
    <section className="voice-grid">
      <article className="voice-card launch-card"><small>CLIENT SCENARIO</small><h2>Start a LiveKit call</h2><p>The scripted digital-twin client speaks first. Amira receives the same text through her public API; LiveKit only voices and captions those authoritative turns.</p><label htmlFor="scenario">Scenario</label><select id="scenario" value={selected} onChange={(event) => setSelected(event.target.value)} disabled={active?.running}>{personas.map((item) => <option key={item.id} value={item.id}>{item.name} ({item.language})</option>)}</select><button className="voice-primary" onClick={start} disabled={!selected || starting || active?.running}>{starting ? 'Starting...' : active?.running ? 'Call in progress' : 'Start voice simulation'}</button><p className="voice-note">One call at a time. It waits up to three minutes for a listener to join.</p></article>
      <article className="voice-card listener-card"><small>LIVEKIT ROOM</small><h2>{active?.running ? 'Ready to listen' : 'No active call'}</h2>{active?.running ? <><p>Join the room now; the call begins after your listener connects.</p><a className="voice-primary link" href="/voice-listener/" target="voice-listener">Join and listen</a><p className="voice-note">Scenario: {active.persona}. The page closes its room when the call ends.</p><iframe title="LiveKit call listener" src="/voice-listener/" /></> : <p>Choose a scenario and start it. The listener will appear here when the local LiveKit service is available.</p>}</article>
    </section>
    <section className="voice-card results-card"><header><div><small>RECENT VERIFIED RUNS</small><h2>Call outcomes</h2></div><button onClick={refresh}>Refresh</button></header>{runs.length ? <div className="run-list">{runs.slice(0, 5).map((run) => <article className="run" key={run.call_id}><div><b>{run.persona || run.call_id}</b><small>{run.advisor_turns || 0} Amira turn(s) - {run.duration_s || 0}s</small></div><em className={run.outcome === 'completed' ? 'ok' : 'warn'}>{run.outcome}</em><span>{run.offers_returned ? 'Offers returned' : 'No offer returned'}</span><button type="button" onClick={() => openQuality(run)}>Review in Quality</button></article>)}</div> : <p className="empty">No completed voice call has been recorded yet.</p>}{latest?.transcript?.length ? <details><summary>Latest transcript ({latest.persona})</summary>{latest.transcript.map((turn, index) => <p className={`transcript ${turn.speaker}`} key={index}><b>{turn.speaker === 'advisor' ? 'Amira' : 'Client'}:</b> {turn.text}</p>)}</details> : null}</section>
  </main>
}
