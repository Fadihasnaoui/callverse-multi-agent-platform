import { useEffect, useMemo, useRef, useState } from 'react'
import './AdvisorDesign.css'
import './Monitoring.css'
import ManagerInbox, { ManagerInboxLink } from './ManagerInbox'
import QualityAnalyst from './QualityAnalyst'
import ClientSimulator from './ClientSimulator'
import VoiceSimulator from './VoiceSimulator'
import { sendAdvisorRequest } from './advisorTransport'

const sampleConversation = [
  { sender: 'Amira, travel assistant', kind: 'agent', text: 'Welcome to Ozone Voyages. Tell me where and when you would like to travel, and I will help you find the best available hotel offers.', time: 'Now' },
]

const STORAGE_KEY = 'ozone-advisor-sessions-v1'
const CUSTOMER_KEY = 'ozone-advisor-customer-id-v1'
const TENANT_ID = 'ozone-voyages'
const AGENT_ID = 'customer-advisor'
// Allow the server's 30-second network budget plus workflow/persistence overhead.
const ADVISOR_REQUEST_TIMEOUT_MS = 45000
const connectionUnavailable = {
  fr: 'La connexion avec Amira est momentanément indisponible. Vos informations sont conservées. Un conseiller Ozone Voyages peut vous aider à poursuivre votre demande.',
  en: 'The connection to Amira is temporarily unavailable. Your details are saved. An Ozone Voyages adviser can help you continue your request.',
  ar: 'الاتصال بأميرة غير متاح مؤقتاً. احتفظنا بمعلوماتك. يمكن لمستشار من Ozone Voyages مساعدتك في متابعة طلبك.',
  'aeb-Arab': 'الاتصال بأميرة موش متوفر توا. معلوماتك محفوظة. مستشار Ozone Voyages ينجم يعاونك باش تكمل طلبك.',
  'aeb-Latn': 'El connexion m3a Amira moch disponible taw. Ma3loumetek ma7foudha. Conseiller Ozone Voyages ynajem y3awnek tkammel talabek.',
}
const emptyResult = { reason: 'Ready for a request', tools: [], parsed: {}, memory: {}, trace_saved: false }

const assistanceShortcuts = [
  { icon: 'hotel', title: 'Find hotel offers', detail: 'Explore stays for your dates', message: 'Pouvez-vous me recommander des hotels disponibles pour mon sejour ?' },
  { icon: 'spark', title: 'Compare my options', detail: 'Rooms, board and prices', message: 'Pouvez-vous comparer les hotels et les offres que vous m’avez proposes ?' },
  { icon: 'chat', title: 'Ask a travel question', detail: 'Get help before you book', message: 'J’ai une question concernant mon sejour : ' },
  { icon: 'headphones', title: 'Speak with an adviser', detail: 'Get a little personal help', message: 'Je souhaite parler a un conseiller Ozone Voyages.' },
  { icon: 'shield', title: 'Make a complaint', detail: 'Tell us what went wrong', message: 'Je souhaite faire une réclamation : ' },
]

function Icon({ name = 'spark', ...props }) {
  const paths = {
    spark: 'm12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z',
    chat: 'M21 11.5a8.5 8.5 0 0 1-8.5 8.5H4l-1 1v-9.5A8.5 8.5 0 0 1 11.5 3h1A8.5 8.5 0 0 1 21 11.5Z',
    plane: 'm22 2-7 20-4-9-9-4L22 2Zm0 0L11 13',
    pin: 'M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0Z M15 10a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z',
    calendar: 'M8 2v4m8-4v4M3 10h18M5 4h14a2 2 0 0 1 2 2v14H3V6a2 2 0 0 1 2-2Z',
    users: 'M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2m20 0v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75M13 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
    hotel: 'M3 21V3h12v18M15 9h6v12M7 7h4M7 11h4M7 15h4M8 21v-3h3v3M18 13v1m0 3v1',
    shield: 'M12 22s9-4 9-12V5l-9-3-9 3v5c0 8 9 12 9 12Zm-4-10 3 3 5-6',
    arrow: 'M5 12h14m-6-6 6 6-6 6',
    headphones: 'M3 14v-3a9 9 0 0 1 18 0v3M3 13h4v8H3v-8Zm14 0h4v8h-4v-8Z',
  }
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}><path d={paths[name] || paths.spark} /></svg>
}

// Some replies saved by older builds were decoded with a Chinese legacy
// codepage instead of UTF-8. Repair only assistant text; customer messages are
// never changed. New API responses are already valid UTF-8.
function repairDisplayText(value) {
  return String(value || '')
    .replace(/\u8119\u5784\u8292\u9225\u6bec\ue0de\u2469\u5053\u807a/g, ' - ')
    .replace(/h\u4e48tels/gi, 'h\u00f4tels')
    .replace(/h\u4e48tel/gi, 'h\u00f4tel')
    .replace(/\u813f/g, '\u00e0')
    .replace(/\u8305/g, '\u00e9')
    .replace(/\u732b/g, '\u00e8')
    .replace(/\u83bd/g, '\u00e7')
    .replace(/\u951a/g, '\u00ea')
    .replace(/\u9709/g, '\u00f9')
    .replace(/\u00c3\u00b4/g, '\u00f4')
    .replace(/\u00c3\u00a9/g, '\u00e9')
    .replace(/\u00c3\u00a8/g, '\u00e8')
    .replace(/\u00c3\u00a0/g, '\u00e0')
}
function repairSavedSession(session) {
  return {
    ...session,
    messages: (session.messages || []).map((message) => message.kind === 'client'
      ? message
      : { ...message, text: repairDisplayText(message.text) }),
  }
}
function createId(prefix) {
  const value = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`
  return `${prefix}-${value}`
}
function browserCustomerId() {
  const existing = localStorage.getItem(CUSTOMER_KEY)
  if (existing) return existing
  const value = createId('customer')
  localStorage.setItem(CUSTOMER_KEY, value)
  return value
}
function makeSession() {
  return {
    id: createId('session'),
    title: 'New conversation',
    customer: { id: browserCustomerId(), name: 'Walk-in client', language: 'French', status: 'New' },
    messages: sampleConversation,
    result: emptyResult,
    updatedAt: new Date().toISOString(),
  }
}
function loadSessions() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    if (Array.isArray(saved) && saved.length) {
      const repaired = saved.map(repairSavedSession)
      if (!localStorage.getItem(CUSTOMER_KEY) && repaired[0]?.customer?.id) localStorage.setItem(CUSTOMER_KEY, repaired[0].customer.id)
      return repaired
    }
  } catch {}
  return [makeSession()]
}
const displayBoard = { DP: 'Half board', PC: 'Full board', AI: 'All inclusive', BB: 'Bed & breakfast' }
const MONITOR_TOKEN_KEY = 'ozone-operations-token-v1'

function sessionDate(value) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return 'Recent conversation'
  return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

function offerStatus(offer) {
  const value = String(offer.availability || '').toLowerCase().replaceAll('_', ' ')
  if (value.includes('sur demande') || value.includes('on request')) return 'On request'
  if (value.includes('stop sale')) return 'Stop sale'
  if (offer.is_bookable === false) return 'Confirmation required'
  return 'Bookable'
}

function liveOffers(result) {
  if (Array.isArray(result?.offers)) {
    return result.offers.map((offer, index) => ({
      hotel: offer.hotel_name || '',
      label: offer.label || `Offer ${index + 1}`,
      price: offer.total_price_text || '',
      board: offer.board || '',
      status: offerStatus(offer),
    })).filter((offer) => offer.price && offer.status !== 'Stop sale').slice(0, 3)
  }
  const quote = result?.tool_results?.quote
  if (!quote || typeof quote !== 'object') return []
  const groups = quote.status === 'destination_offers_found'
    ? [{ hotel: '', offers: quote.offers || [] }]
    : Array.isArray(quote.quotes)
      ? quote.quotes.map((entry) => ({ hotel: entry.hotel_name || entry.requested_hotel_name || '', offers: entry.diagnostic_offers || entry.offers || entry.offer_sets || [] }))
      : [{ hotel: quote.hotel_name || quote.requested_hotel_name || '', offers: quote.diagnostic_offers || quote.offers || quote.offer_sets || [] }]
  const seen = new Set()
  return groups.flatMap(({ hotel, offers }) => (offers || []).map((offer, index) => {
    if (!offer || typeof offer !== 'object') return null
    const label = offer.label || offer.room || offer.view_type || offer.room_group || `Offer ${index + 1}`
    const price = offer.total_price_text || offer.price_text || offer.price || offer.total || ''
    if (!price) return null
    const board = offer.board || ''
    const status = offerStatus(offer)
    const key = [hotel, label, board, price, status].join('|').toLowerCase()
    if (seen.has(key)) return null
    seen.add(key)
    return { hotel: offer.hotel_name || hotel, label: String(label), price: String(price), board: String(board), status }
  })).filter(Boolean).filter((offer) => offer.status !== 'Stop sale').slice(0, 3)
}
function MonitoringDashboard() {
  const [snapshot, setSnapshot] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [token, setToken] = useState(() => localStorage.getItem(MONITOR_TOKEN_KEY) || '')

  async function load(force = false, suppliedToken = token) {
    setLoading(true)
    setError('')
    try {
      const response = await fetch(`/admin-operations${force ? '?refresh=1' : ''}`, {
        headers: suppliedToken ? { Authorization: `Bearer ${suppliedToken}` } : {},
      })
      if (response.status === 401) throw new Error('Admin access token required')
      const data = await response.json()
      if (!response.ok) throw new Error(data.error || 'Operations data is unavailable')
      setSnapshot(data)
    } catch (reason) {
      setError(reason.message || 'Operations data is unavailable')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load(false)
    const timer = window.setInterval(() => load(false), 15000)
    return () => window.clearInterval(timer)
  }, [])

  function saveToken() {
    const value = window.prompt('Operations access token', token) ?? token
    localStorage.setItem(MONITOR_TOKEN_KEY, value)
    setToken(value)
    load(true, value)
  }

  const metrics = snapshot?.metrics || {}
  const services = Object.values(snapshot?.services || {})
  const providers = snapshot?.providers || []
  const generated = snapshot?.generated_at ? new Date(snapshot.generated_at).toLocaleTimeString() : 'Not loaded'
  const statusClass = (status) => ['healthy', 'ready'].includes(status) ? 'healthy' : status === 'not_configured' ? 'muted' : 'degraded'

  return <div className="operations-shell">
    <header className="operations-header">
      <div><span className="crumb">OZONE VOYAGES / ADMIN OPERATIONS</span><h1>Amira Operations Center</h1><p>Service health, customer-advisor performance, provider resilience, and recent failures.</p></div>
      <div className="operations-actions"><a href="/">Open customer chat</a><button onClick={saveToken}>Access token</button><button className="refresh" disabled={loading} onClick={() => load(true)}>{loading ? 'Refreshing...' : 'Refresh now'}</button></div>
    </header>
    {error && <div className="operations-error"><b>Monitoring unavailable</b><span>{error}</span></div>}
    <div className="operations-meta"><span className={`status-pill ${statusClass(snapshot?.status)}`}>{snapshot?.status || 'loading'}</span><span>Updated {generated}</span><span>Auto-refresh every 15 seconds</span></div>

    <section className="operations-kpis">
      <article><small>SUCCESS RATE / 24H</small><strong>{metrics.success_rate ?? 0}%</strong><span>{metrics.successful_turns ?? 0} of {metrics.total_turns ?? 0} turns</span></article>
      <article><small>AVERAGE LATENCY</small><strong>{metrics.average_latency_seconds ?? 0}s</strong><span>Target: conversation under 10s</span></article>
      <article><small>P95 LATENCY</small><strong>{metrics.p95_latency_seconds ?? 0}s</strong><span>Maximum {metrics.maximum_latency_seconds ?? 0}s</span></article>
      <article><small>ACTIVE SESSIONS / 24H</small><strong>{metrics.active_sessions ?? 0}</strong><span>{metrics.handoffs ?? 0} human handoff(s)</span></article>
    </section>

    <section className="operations-grid">
      <article className="operations-panel services-panel"><header><div><small>INFRASTRUCTURE</small><h2>Service health</h2></div><span>{services.filter((item) => item.status === 'healthy').length}/{services.length} healthy</span></header><div className="service-list">{services.map((service) => <div className="service-item" key={service.name}><i className={statusClass(service.status)} /><div><b>{service.name}</b><small>{service.error || `HTTP ${service.http_status || 'OK'}`}</small></div><strong>{service.latency_ms ?? 0} ms</strong></div>)}</div></article>
      <article className="operations-panel providers-panel"><header><div><small>AI RESILIENCE</small><h2>LLM providers</h2></div><span>Circuit breaker enabled</span></header><div className="provider-table"><div className="provider-row heading"><span>Provider</span><span>Model</span><span>Status</span><span>Latency</span></div>{providers.map((provider) => <div className="provider-row" key={provider.name}><b>{provider.name}</b><span title={provider.model}>{provider.model}</span><em className={statusClass(provider.status)}>{provider.status.replaceAll('_', ' ')}</em><span>{provider.last_latency_seconds == null ? '—' : `${provider.last_latency_seconds}s`}</span>{provider.last_error && <small>{provider.last_error}</small>}</div>)}</div></article>
      <article className="operations-panel"><header><div><small>REQUIRES ATTENTION</small><h2>Recent failures</h2></div><span>{metrics.failed_turns ?? 0} in 24h</span></header><div className="event-list">{(metrics.recent_failures || []).length ? metrics.recent_failures.map((item) => <div className="event" key={item.trace_id}><i className="degraded" /><div><b>{item.reason || item.status}</b><small>{item.trace_id}</small></div><time>{new Date(item.created_at).toLocaleTimeString()}</time></div>) : <div className="empty-monitor">No recent agent failures.</div>}</div></article>
      <article className="operations-panel"><header><div><small>PERFORMANCE</small><h2>Slowest turns</h2></div><span>Last 24 hours</span></header><div className="event-list">{(metrics.slow_turns || []).length ? metrics.slow_turns.map((item) => <div className="event" key={item.trace_id}><i className={item.latency_seconds > 30 ? 'degraded' : 'healthy'} /><div><b>{item.reason || 'Agent turn'}</b><small>{item.trace_id}</small></div><time>{item.latency_seconds}s</time></div>) : <div className="empty-monitor">No completed turns yet.</div>}</div></article>
    </section>
  </div>
}

function CustomerApp() {
  const [chatFocused, setChatFocused] = useState(false)
  const [sessions, setSessions] = useState(loadSessions)
  const [activeId, setActiveId] = useState(() => sessions[0].id)
  const initialSession = sessions.find((session) => session.id === activeId) || sessions[0]
  const [messages, setMessages] = useState(initialSession.messages || sampleConversation)
  const [text, setText] = useState('')
  const [sending, setSending] = useState(false)
  const [waitSeconds, setWaitSeconds] = useState(0)
  const [listening, setListening] = useState(false)
  const [voiceError, setVoiceError] = useState('')
  const [voiceStatus, setVoiceStatus] = useState('')
  const [audioInputs, setAudioInputs] = useState([])
  const [selectedAudioInput, setSelectedAudioInput] = useState('')
  const [voiceLevel, setVoiceLevel] = useState(0)
  const mediaRecorderRef = useRef(null)
  const messagesEndRef = useRef(null)
  const [customer, setCustomer] = useState(initialSession.customer)
  const [result, setResult] = useState(initialSession.result || emptyResult)
  const conversationLanguage = { fr: 'Français', en: 'English', ar: 'العربية', 'aeb-Arab': 'تونسي', 'aeb-Latn': 'Tounsi / Arabizi' }[result.reply_language] || customer.language
  useEffect(() => { setSessions((items) => { const next = items.map((session) => session.id === activeId ? { ...session, customer, messages, result, title: messages.find((item) => item.kind === 'client')?.text?.slice(0, 38) || session.title, updatedAt: new Date().toISOString() } : session); localStorage.setItem(STORAGE_KEY, JSON.stringify(next)); return next }) }, [activeId, customer, messages, result])
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages, sending])
  useEffect(() => {
    if (!sending) return undefined
    const timer = window.setInterval(() => setWaitSeconds((seconds) => seconds + 1), 1000)
    return () => window.clearInterval(timer)
  }, [sending])
  useEffect(() => {
    if (!navigator.mediaDevices?.enumerateDevices) return undefined
    const loadAudioInputs = async () => {
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === 'audioinput')
      setAudioInputs(devices)
      setSelectedAudioInput((current) => current || devices[0]?.deviceId || '')
    }
    loadAudioInputs().catch(() => {})
    navigator.mediaDevices.addEventListener?.('devicechange', loadAudioInputs)
    return () => navigator.mediaDevices.removeEventListener?.('devicechange', loadAudioInputs)
  }, [])
  function openSession(session) { if (sending || session.id === activeId) return; setActiveId(session.id); setCustomer(session.customer); setMessages(session.messages || sampleConversation); setResult(session.result || emptyResult); setText('') }
  function newSession() { if (sending) return; const session = makeSession(); setSessions((items) => [session, ...items]); setActiveId(session.id); setCustomer(session.customer); setMessages(session.messages); setResult(session.result); setText('') }
  function deleteSession(event, session) {
    event.stopPropagation()
    if (sending) return
    const label = session.title && session.title !== 'New conversation' ? `“${session.title}”` : 'this conversation'
    if (!window.confirm(`Delete ${label}? This removes the conversation from this browser and cannot be undone.`)) return

    const remaining = sessions.filter((item) => item.id !== session.id)
    const nextSessions = remaining.length ? remaining : [makeSession()]
    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextSessions))
    setSessions(nextSessions)

    if (session.id !== activeId) return
    const replacement = [...nextSessions].sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt)))[0]
    setActiveId(replacement.id)
    setCustomer(replacement.customer)
    setMessages(replacement.messages || sampleConversation)
    setResult(replacement.result || emptyResult)
    setText('')
  }
  const offers = useMemo(() => liveOffers(result), [result])
  const parsed = result.parsed || {}
  const memory = result.memory || {}
  const trip = result.trip || {}
  const request = {
    destination: trip.destination || parsed.destination || memory.destination || 'Not selected',
    checkin: trip.checkin || parsed.checkin || memory.checkin,
    checkout: trip.checkout || parsed.checkout || memory.checkout,
    adults: trip.adults ?? parsed.adults ?? memory.adults,
    children: Array.isArray(trip.children) ? trip.children.length : Array.isArray(parsed.children) ? parsed.children.length : Array.isArray(memory.children) ? memory.children.length : 0,
    board: trip.board || parsed.board || memory.board,
    hotel: trip.hotel_name || parsed.hotel_name || memory.selected_hotel,
  }

  function speakReply(reply, language = result.reply_language) {
    if (!reply || !('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(reply)
    utterance.lang = { fr: 'fr-FR', en: 'en-US', ar: 'ar', 'aeb-Arab': 'ar-TN', 'aeb-Latn': 'fr-FR' }[language] || 'fr-FR'
    utterance.rate = 0.96
    window.speechSynthesis.speak(utterance)
  }

  async function submitMessage(rawMessage, speakResponse = false) {
    const message = rawMessage.trim()
    if (!message || sending) return
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const request = {
      tenant_id: TENANT_ID, agent_id: AGENT_ID, customer_id: customer.id,
      session_id: activeId, message_id: createId('message'), channel: 'advisor_dashboard',
      language: customer.language === 'French' ? 'fr' : 'en', message,
    }
    setMessages((items) => [...items, { sender: customer.name, kind: 'client', text: message, time }])
    setText('')
    setSending(true)
    setWaitSeconds(0)
    try {
      const data = await sendAdvisorRequest(request, { timeoutMs: ADVISOR_REQUEST_TIMEOUT_MS })
      setResult(data)
      setMessages((items) => [...items, { sender: 'Amira, travel assistant', kind: 'agent', text: repairDisplayText(data.reply), time, reviewRecord: data.quality_review || null,
      }])
      if (speakResponse) speakReply(data.reply, data.reply_language)
    } catch (error) {
      const detail = error.customerReply || connectionUnavailable[result.reply_language] || connectionUnavailable.fr
      setMessages((items) => [...items, { sender: 'Ozone Voyages', kind: 'error', text: detail, time }])
    } finally { setSending(false) }
  }

  function sendMessage(event) {
    event.preventDefault()
    submitMessage(text)
  }

  async function toggleVoiceInput() {
    if (sending) return
    if (listening) { mediaRecorderRef.current?.stop(); return }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { setVoiceError('Audio recording is not supported by this browser.'); return }
    setVoiceError('')
    setVoiceStatus('Requesting microphone access...')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: selectedAudioInput ? { exact: selectedAudioInput } : undefined, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
      const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === 'audioinput')
      setAudioInputs(devices)
      setSelectedAudioInput(stream.getAudioTracks()[0]?.getSettings().deviceId || selectedAudioInput)
      const startedAt = Date.now()
      const AudioContext = window.AudioContext || window.webkitAudioContext
      const audioContext = AudioContext ? new AudioContext() : null
      const analyser = audioContext?.createAnalyser()
      let peakLevel = 0
      let levelTimer = null
      if (analyser) {
        analyser.fftSize = 512
        audioContext.createMediaStreamSource(stream).connect(analyser)
        const samples = new Uint8Array(analyser.fftSize)
        levelTimer = window.setInterval(() => {
          analyser.getByteTimeDomainData(samples)
          const rms = Math.sqrt(samples.reduce((sum, value) => sum + ((value - 128) / 128) ** 2, 0) / samples.length)
          peakLevel = Math.max(peakLevel, rms)
          setVoiceLevel(Math.min(100, Math.round(rms * 900)))
        }, 100)
      }
      const preferredType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find((type) => MediaRecorder.isTypeSupported(type)) || ''
      const recorder = new MediaRecorder(stream, preferredType ? { mimeType: preferredType } : undefined)
      const chunks = []
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data) }
      recorder.onstart = () => { setListening(true); setVoiceStatus('Recording... speak clearly, then click the red microphone to send.') }
      recorder.onerror = () => setVoiceError('The microphone recording failed. Please try again.')
      recorder.onstop = async () => {
        setListening(false)
        setVoiceStatus('')
        setVoiceLevel(0)
        if (levelTimer) window.clearInterval(levelTimer)
        if (audioContext) audioContext.close()
        stream.getTracks().forEach((track) => track.stop())
        mediaRecorderRef.current = null
        const audio = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
        if (!audio.size || Date.now() - startedAt < 900) { setVoiceError('The recording was too short. Hold the microphone for at least one second and speak clearly.'); return }
        if (analyser && peakLevel < 0.0015) { setVoiceError('No voice was detected. Choose another microphone below and verify it is not muted in Windows.'); return }
        setVoiceStatus('Transcribing your voice message...')
        try {
          const response = await fetch('/transcribe-audio', { method: 'POST', headers: { 'Content-Type': audio.type, 'X-Audio-Language': 'auto' }, body: audio })
          const data = await response.json()
          if (!response.ok || !data.text) throw new Error(data.error || 'Transcription failed')
          setText(data.text)
          setVoiceError(''); setVoiceStatus('')
          await submitMessage(data.text, true)
        } catch (error) { setVoiceStatus(''); setVoiceError('We could not understand the recording. Please try again and speak clearly.') }
      }
      mediaRecorderRef.current = recorder
      recorder.start()
    } catch (error) {
      setVoiceStatus('')
      setVoiceError(error.name === 'NotAllowedError' ? 'Microphone permission was refused. Click the padlock in the address bar and allow microphone access.' : 'The microphone could not start. Check that it is connected and selected, then try again.')
    }
  }

  return <div className={`app-shell${chatFocused ? ' chat-focused' : ''}`}>
    <aside className="sidebar">
      <div className="logo"><span><Icon name="plane" /></span><b>Ozone</b><em>Desk</em></div>
      <div className="agency-switch"><small>YOUR TRAVEL AGENCY</small><strong>Ozone Voyages</strong><span>TN</span></div>
      <div className="sidebar-current"><Icon name="headphones" /><div><b>Customer advisor</b><small>Your travel companion</small></div><i /></div>
      <div className="history-heading"><b>Conversations</b><button onClick={newSession}>+ New</button></div>
      <div className="session-list">{[...sessions].sort((a, b) => String(b.updatedAt).localeCompare(String(a.updatedAt))).map((session) => <div className={session.id === activeId ? 'session-row active' : 'session-row'} key={session.id}><button className="session" onClick={() => openSession(session)}><span>{session.customer?.name || 'Walk-in client'}</span><b>{session.title || 'New conversation'}</b><small>{sessionDate(session.updatedAt)}</small></button><button className="delete-session" type="button" onClick={(event) => deleteSession(event, session)} aria-label={`Delete ${session.title || 'conversation'}`} title="Delete conversation"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 3h6l1 2h4v2H4V5h4l1-2Zm-3 6h12l-1 12H7L6 9Zm4 2v8h2v-8h-2Zm4 0v8h2v-8h-2Z" /></svg></button></div>)}</div>
      <div className="sidebar-footer"><Icon name="plane" /><small>A LITTLE INSPIRATION</small><b>Your next chapter<br />starts with a trip.</b><p>Discover Tunisia, one beautiful stay at a time.</p><button onClick={newSession} disabled={sending}>Plan a new stay <Icon name="arrow" /></button></div>
    </aside>

    <main className="dashboard">
      <div className="workspace-bar"><div><span className="workspace-mark"><Icon name="headphones" /></span><b>Customer workspace</b><span className="workspace-divider">/</span><span>Ozone Voyages</span></div><div><a className="workspace-link" href="/simulator">Client simulator</a><a className="workspace-link" href="/voice-lab">Voice lab</a><a className="workspace-link" href="/quality">Quality review</a><ManagerInboxLink /><span className="workspace-secure"><Icon name="shield" /> Travel with confidence</span><span className="workspace-avatar">{customer.name.slice(0, 2).toUpperCase()}</span></div></div>
      <header className="topbar">
        <div><span className="crumb">YOUR STAY, THOUGHTFULLY PLANNED</span><h1>Good trips start with a conversation<span>.</span></h1><p>Meet Amira. Your personal guide to finding the right stay in Tunisia.</p></div>
        <div className="top-actions"><span className="workflow-status"><i /> {sending ? 'Amira is working on it' : 'Amira is here for you'}</span></div>
      </header>

      <section className="stats">
        <div><small>YOUR TRIP</small><b>{request.hotel || (request.destination !== 'Not selected' ? request.destination : 'Not selected yet')}</b><span>Share a destination or hotel to begin</span></div>
        <div><small>CONVERSATION</small><b>{messages.some((item) => item.kind === 'client') ? 'In progress' : 'Ready to begin'}</b><span>You can return and continue this conversation</span></div>
        <div><small>TRAVEL ASSISTANCE</small><b>{sending ? 'Searching for you...' : 'Amira is ready'}</b><span>Current offers and clear reservation guidance</span></div>
      </section>

      <div className="dashboard-grid">
        <section className="panel conversation-panel">
          <div className="panel-title"><div><span className="panel-icon"><Icon name="chat" /></span><b>Let’s talk travel</b></div><div className="chat-panel-actions"><span className="live-badge"><i /> Live chat</span><button className="chat-expand" type="button" aria-pressed={chatFocused} onClick={() => setChatFocused((focused) => !focused)} title={chatFocused ? 'Show trip details' : 'Expand the conversation'}>{chatFocused ? 'Show trip details' : 'Expand chat'}<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.7"><path d={chatFocused ? 'M9 3v6H3m18 6h-6v6M3 9l6-6m6 18 6-6' : 'M15 3h6v6M3 15v6h6M21 3l-7 7M3 21l7-7'} /></svg></button></div></div>
          <div className="client-profile"><div className="profile-avatar">{customer.name.slice(0, 2).toUpperCase()}</div><div><b>{customer.name}</b><small>{conversationLanguage} / Web dashboard</small></div><button onClick={() => { const name = window.prompt('Client name', customer.name); if (name?.trim()) setCustomer((old) => ({ ...old, name: name.trim(), status: 'Known' })) }}>Edit profile</button></div>
          <div className="messages" role="log" aria-label="Conversation with Amira" aria-relevant="additions">
            <div className="chat-date"><span>Your travel story starts here</span></div>
            {messages.map((message, index) => <article className={`message ${message.kind}`} key={`${index}-${message.time}`}><span className="message-avatar">{message.kind === 'client' ? customer.name.slice(0, 2).toUpperCase() : message.kind === 'error' ? '!' : <Icon />}</span><div><header><b>{message.sender}</b><small>{message.time}</small></header><p dir="auto">{message.kind === 'client' ? message.text : repairDisplayText(message.text)}</p></div></article>)}
            {sending && <article className="message agent"><span className="message-avatar"><Icon /></span><div><header><b>Amira</b><small>{waitSeconds} s</small></header><p className="thinking" role="status"><span className="typing-dots"><i /><i /><i /></span><span>{waitSeconds >= 20 ? 'La vérification prend plus de temps. Vos informations sont conservées.' : waitSeconds >= 8 ? 'Recherche en cours, merci de patienter…' : 'Amira prépare votre réponse…'}</span></p></div></article>}
            <div ref={messagesEndRef} />
          </div>
          <form className="composer" onSubmit={sendMessage}><input value={text} onChange={(event) => setText(event.target.value)} placeholder={listening ? 'ListeningÃ¢â‚¬Â¦ speak now' : 'Write or say a customer request...'} /><button className={`mic-button${listening ? ' listening' : ''}`} disabled={sending} type="button" onClick={toggleVoiceInput} aria-label={listening ? 'Stop recording' : 'Send a voice message'} title={listening ? 'Stop and send' : 'Speak to Amira'}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a4 4 0 0 0 4-4V5a4 4 0 0 0-8 0v6a4 4 0 0 0 4 4Zm7-4a1 1 0 0 0-2 0 5 5 0 0 1-10 0 1 1 0 0 0-2 0 7 7 0 0 0 6 6.92V21H8a1 1 0 0 0 0 2h8a1 1 0 0 0 0-2h-3v-3.08A7 7 0 0 0 19 11Z" /></svg></button><button className="send-button" disabled={sending || listening} type="submit">{sending ? '...' : 'Send'}</button></form>
          {audioInputs.length > 0 && <div className="mic-settings"><label htmlFor="microphone-device">Microphone</label><select id="microphone-device" value={selectedAudioInput} disabled={listening || sending} onChange={(event) => setSelectedAudioInput(event.target.value)}>{audioInputs.map((device, index) => <option value={device.deviceId} key={device.deviceId || index}>{device.label || `Microphone ${index + 1}`}</option>)}</select><div className="voice-meter"><i style={{ width: `${voiceLevel}%` }} /><span>{listening ? `${voiceLevel}%` : 'Ready'}</span></div></div>}
          {voiceError && <p className="voice-error">{voiceError}</p>}
          {voiceStatus && <p className="voice-status">{voiceStatus}</p>}
          <p className="trace-note">You can return to this conversation later and continue where you left off.</p>
        </section>

        <section className="panel intelligence-panel">
          <div className="panel-title"><div><span className="panel-icon purple"><Icon /></span><b>Your trip, taking shape</b></div><span className="source-badge">{offers.length ? 'Offers received' : 'Made for you'}</span></div>
          <div className="agent-summary"><div className="spark"><Icon /></div><div><strong>A stay that feels like you.</strong><p>Share your destination, dates and guests. We’ll bring the details together right here.</p></div></div>
          <details className="amira-help" open>
            <summary><span className="panel-icon purple"><Icon name="headphones" /></span><span><b>How Amira can help</b><small>A little guidance for your next step</small></span><svg className="help-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="m6 9 6 6 6-6" /></svg></summary>
            <div className="help-shortcuts">{assistanceShortcuts.map((shortcut) => <button key={shortcut.title} type="button" disabled={sending || listening} onClick={() => { setText(shortcut.message); document.querySelector('.composer input')?.focus(); document.querySelector('.composer')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' }) }}><span className="help-icon"><Icon name={shortcut.icon} /></span><span><b>{shortcut.title}</b><small>{shortcut.detail}</small></span><Icon name="arrow" /></button>)}</div>
            <p className="help-hint">Choose a shortcut to draft a message. Review it, then send.</p>
          </details>
          <div className="request-card"><small>YOUR CURRENT REQUEST</small><h2>{request.hotel || (request.destination !== 'Not selected' ? `${request.destination} stay` : 'Tell us about your trip')}</h2><div className="request-details"><span>Location <b>{request.destination}</b></span><span>Dates <b>{request.checkin ? `${request.checkin} - ${request.checkout}` : 'To confirm'}</b></span><span>Guests <b>{request.adults ? `${request.adults} adult(s)${request.children ? ` + ${request.children} child` : ''}` : 'To confirm'}</b></span><span>Board <b>{displayBoard[request.board] || request.board || 'To confirm'}</b></span></div></div>
          <div className="offer-heading"><div><small>AVAILABLE OFFERS</small><h3>{offers.length ? `${offers.length} result${offers.length > 1 ? 's' : ''} found` : 'Your offers will appear here'}</h3></div><span>{offers.length ? 'Current offers' : 'Search not started'}</span></div>
          <div className="offers">{offers.length ? offers.map((offer, index) => <article className="offer" key={`${offer.hotel}-${offer.label}-${index}`}><div className="hotel-image">{index + 1}</div><div className="offer-details"><b>{offer.label}</b><small>{offer.hotel || request.hotel || 'Ozone Voyages'}</small><span>{offer.board || displayBoard[request.board] || 'Formula to confirm'}</span></div><div className="offer-value"><strong>{offer.price}</strong><em className={offer.status === 'On request' ? 'on-request' : ''}>{offer.status}</em></div></article>) : <div className="empty-offer">No priced live offer is available yet. Complete the request to search Ozone.</div>}</div>
          <div className="safety-message"><Icon name="shield" /><span>Prices come from Ozone Voyages. Your adviser confirms final availability before booking.</span></div>
        </section>

        <aside className="right-rail">
          <section className="panel customer-card"><div className="panel-title"><div><span className="panel-icon green">ME</span><b>Your profile</b></div><span className="state">{customer.status === 'Known' ? 'Updated' : 'Guest'}</span></div><div className="identity"><div className="large-avatar">{customer.name.slice(0, 2).toUpperCase()}</div><div><b>{customer.name}</b><small>Ozone Voyages customer</small></div></div><dl><div><dt>Contact</dt><dd>Online chat</dd></div><div><dt>Language</dt><dd>{conversationLanguage}</dd></div><div><dt>Messages</dt><dd>{messages.filter((item) => item.kind === 'client').length}</dd></div><div><dt>Destination</dt><dd>{request.destination}</dd></div></dl></section>
          <section className="panel action-card"><span className="panel-icon purple"><Icon name="headphones" /></span><b>A little extra help?</b><p>Let’s find your next step together.</p><button className="primary-action" disabled={sending || listening} onClick={() => submitMessage('Je souhaite un devis pour mon sejour.')}>Ask Amira for a quote <Icon name="arrow" /></button><button disabled={sending || listening} onClick={() => submitMessage('Je souhaite parler a un conseiller Ozone Voyages.')}>Speak with an adviser <Icon name="arrow" /></button></section>
        </aside>
      </div>
    </main>
  </div>
}

function App() {
  if (window.location.pathname.startsWith('/manager')) return <ManagerInbox />
  if (window.location.pathname.startsWith('/quality')) return <QualityAnalyst />
  if (window.location.pathname.startsWith('/simulator')) return <ClientSimulator />
  if (window.location.pathname.startsWith('/voice-lab')) return <VoiceSimulator />
  return window.location.pathname.startsWith('/monitoring') ? <MonitoringDashboard /> : <CustomerApp />
}

export default App
