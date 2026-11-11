import { useCallback, useEffect, useRef, useState } from 'react'
import './ManagerInbox.css'

const KEY = 'ozone-manager-access-key'
const labels = { new: 'New', contacted: 'Contacted', availability_confirmed: 'Availability confirmed', awaiting_payment: 'Awaiting payment', paid: 'Paid', completed: 'Completed', cancelled: 'Cancelled' }
const complaintLabels = { new: 'New', contacted: 'Contacted', in_progress: 'In progress', resolved: 'Resolved', closed: 'Closed' }
const time = (value) => value ? new Date(value).toLocaleString() : '—'

export function ManagerInboxLink() {
  const [unread, setUnread] = useState(null)
  useEffect(() => {
    const token = sessionStorage.getItem(KEY)
    if (!token) return undefined
    let alive = true
    const check = async () => {
      try {
        const response = await fetch('/manager-api/notifications', { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' })
        if (!response.ok) return
        const data = await response.json()
        if (alive) setUnread(data.unread)
      } catch { /* Keep the chat usable if the manager service is unavailable. */ }
    }
    check()
    const timer = window.setInterval(check, 15000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [])
  return <a className="manager-link" href="/manager">Manager inbox{unread > 0 && <span className="manager-link-count" aria-label={`${unread} unread requests`}>{unread}</span>}</a>
}

function RequestDetails({ item, onSave, busy, history }) {
  const [status, setStatus] = useState(item.status)
  const [notes, setNotes] = useState(item.notes)
  const [assigned, setAssigned] = useState(item.assigned_to)
  const trip = item.trip || {}
  const selection = item.selection || {}
  const dates = trip.dates || {}
  const party = trip.party || {}
  function save(event) {
    event.preventDefault()
    if (status === 'paid' && item.status !== 'paid' && !window.confirm('Confirm that the agency has received this payment. This records payment status; it does not charge the customer.')) return
    if (status === 'cancelled' && !window.confirm('Mark this customer request as cancelled?')) return
    onSave({ lead_id: item.lead_id, version: item.version, status, notes, assigned_to: assigned })
  }
  return <section className="manager-detail">
    <div className="manager-detail-heading"><div><small>CUSTOMER REQUEST</small><h2>{item.customer_name}</h2><a href={`tel:${item.phone}`}>{item.phone}</a></div><span className={`request-status status-${item.status}`}>{labels[item.status]}</span></div>
    <p className="manager-reference">Reference: {item.lead_id}</p>
    <div className="manager-trip">
      <div><small>Hotel</small><b>{selection.hotel_name || trip.selected_hotel || 'Callback / hotel to confirm'}</b></div>
      <div><small>Destination</small><b>{trip.destination || 'To confirm'}</b></div>
      <div><small>Dates</small><b>{dates.checkin ? `${dates.checkin} → ${dates.checkout || 'To confirm'}` : trip.travel_dates || 'To confirm'}</b></div>
      <div><small>Guests</small><b>{party.adults ? `${party.adults} adults${party.children?.length ? ` · ${party.children.length} children (${party.children.join(', ')} years)` : ''}` : trip.num_travelers ? `${trip.num_travelers} guests` : 'To confirm'}</b></div>
      <div><small>Board</small><b>{selection.board || trip.board || 'To confirm'}</b></div>
      <div><small>Selected quote total</small><b className="manager-price">{selection.total_price_text || 'Not quoted'}</b></div>
    </div>
    {(selection.rooms || []).length > 0 && <div className="manager-rooms"><h3>Selected rooms</h3>{selection.rooms.map((room, index) => <div key={index}><span>Room {room.room_index || index + 1} · {room.room || 'Type to confirm'}</span><b>{room.price_text || 'To confirm'}</b></div>)}</div>}
    <p className="manager-disclaimer">This is a customer request and saved quote. Confirm hotel availability and payment with the agency before completing the reservation.</p>
    <form onSubmit={save} className="manager-treatment">
      <h3>Treatment of this request</h3>
      <div className="manager-form-row"><label>Status<select value={status} onChange={(event) => setStatus(event.target.value)}><option value={item.status}>{labels[item.status]}</option>{item.allowed_statuses.map((value) => <option key={value} value={value}>{labels[value]}</option>)}</select></label><label>Assigned adviser<input maxLength={100} value={assigned} placeholder="Adviser name" onChange={(event) => setAssigned(event.target.value)} /></label></div>
      <label>Internal notes<textarea maxLength={4000} rows={4} value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Call outcome, confirmation details, agency payment reference…" /></label>
      <small className="manager-note">Record payment references only. Do not store card details here.</small>
      <button className="manager-primary" disabled={busy}>{busy ? 'Saving…' : 'Save treatment & mark reviewed'}</button>
    </form>
    <div className="manager-history"><h3>Request history</h3><p>Received {time(item.created_at)} · Updated {time(item.updated_at)}</p>{history.map((entry, index) => <div key={index}><span>{entry.event === 'created' ? 'Customer request received' : entry.event === 'status_changed' ? `${labels[entry.previous_status]} → ${labels[entry.status]}` : 'Reviewed / notes updated'}</span><small>{time(entry.created_at)} · {entry.actor}</small></div>)}</div>
  </section>
}

function ComplaintDetails({ item, onSave, busy, history }) {
  const [status, setStatus] = useState(item.status)
  const [notes, setNotes] = useState(item.notes)
  const [assigned, setAssigned] = useState(item.assigned_to)
  function save(event) {
    event.preventDefault()
    if (status === 'closed' && !window.confirm('Close this complaint? Closed complaints cannot be reopened.')) return
    onSave({ id: item.id, version: item.version, status, notes, assigned_to: assigned })
  }
  return <section className="manager-detail">
    <div className="manager-detail-heading"><div><small>CUSTOMER COMPLAINT</small><h2>{item.customer_name}</h2><a href={`tel:${item.phone}`}>{item.phone}</a></div><span className={`request-status complaint-status-${item.status}`}>{complaintLabels[item.status]}</span></div>
    <p className="manager-reference">Reference: {item.reference} · {item.id}</p>
    <div className="manager-trip">
      <div><small>Hotel</small><b>{item.hotel_name || 'Not specified'}</b></div>
      <div><small>Incident date</small><b>{item.incident_date || 'Not specified'}</b></div>
      <div><small>Received</small><b>{time(item.created_at)}</b></div>
      <div><small>Channel</small><b>{item.channel || 'Chat'}</b></div>
    </div>
    <div className="manager-complaint-text"><h3>What the customer reported</h3><p dir="auto">{item.description}</p></div>
    <p className="manager-disclaimer">The customer was told that an Ozone Voyages adviser will contact them about this complaint. Call them back and record the outcome below.</p>
    <form onSubmit={save} className="manager-treatment">
      <h3>Treatment of this complaint</h3>
      <div className="manager-form-row"><label>Status<select value={status} onChange={(event) => setStatus(event.target.value)}><option value={item.status}>{complaintLabels[item.status]}</option>{item.allowed_statuses.map((value) => <option key={value} value={value}>{complaintLabels[value]}</option>)}</select></label><label>Assigned adviser<input maxLength={100} value={assigned} placeholder="Adviser name" onChange={(event) => setAssigned(event.target.value)} /></label></div>
      <label>Internal notes<textarea maxLength={4000} rows={4} value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Call outcome, hotel response, compensation agreed…" /></label>
      <button className="manager-primary" disabled={busy}>{busy ? 'Saving…' : 'Save treatment & mark reviewed'}</button>
    </form>
    <div className="manager-history"><h3>Complaint history</h3><p>Received {time(item.created_at)} · Updated {time(item.updated_at)}</p>{history.map((entry, index) => <div key={index}><span>{entry.event === 'created' ? 'Complaint received from customer' : entry.event === 'status_changed' ? `${complaintLabels[entry.previous_status]} → ${complaintLabels[entry.status]}` : 'Reviewed / notes updated'}</span><small>{time(entry.created_at)} · {entry.actor}</small></div>)}</div>
  </section>
}

export default function ManagerInbox() {
  const [token, setToken] = useState(() => sessionStorage.getItem(KEY) || '')
  const [draftToken, setDraftToken] = useState('')
  const [view, setView] = useState('reservations')
  const [snapshot, setSnapshot] = useState(null)
  const [complaintSnapshot, setComplaintSnapshot] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState('')
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState('all')
  const [history, setHistory] = useState([])
  const [notice, setNotice] = useState('')
  const latest = useRef(null)
  const latestComplaint = useRef(null)
  const activeToken = useRef(token)
  useEffect(() => { activeToken.current = token }, [token])
  const selectionRequest = useRef(0)
  const api = useCallback(async (path = '', options = {}) => {
    const response = await fetch(`/manager-api${path}`, { ...options, headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` }, cache: 'no-store' })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error || 'The manager inbox is unavailable.')
    return data
  }, [token])
  const refresh = useCallback(async () => {
    if (!token) return
    setLoading(true)
    try {
      const [data, complaintData] = await Promise.all([api(), api('/complaints')])
      if (activeToken.current !== token) return
      const first = data.items[0]?.lead_id
      if (latest.current && first && latest.current !== first && data.unread > 0) setNotice('A new customer request has arrived. Review the highlighted requests below.')
      latest.current = first
      const firstComplaint = complaintData.items[0]?.id
      if (latestComplaint.current && firstComplaint && latestComplaint.current !== firstComplaint && complaintData.unread > 0) setNotice('A new customer complaint has arrived. Open Complaints to treat it.')
      latestComplaint.current = firstComplaint
      setSnapshot(data)
      setComplaintSnapshot(complaintData)
      setError('')
    } catch (reason) { setError(reason.message) }
    finally { setLoading(false) }
  }, [api, token])
  useEffect(() => {
    if (!token) return undefined
    let alive = true
    const tick = () => { if (alive) refresh() }
    tick()
    const timer = window.setInterval(tick, 15000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [token, refresh])
  const complaints = view === 'complaints'
  const idOf = (item) => complaints ? item.id : item.lead_id
  function switchView(next) {
    setView(next)
    setSelected('')
    setHistory([])
    setFilter('all')
  }
  async function select(item) {
    const id = idOf(item)
    setSelected(id)
    setHistory([])
    const requestId = ++selectionRequest.current
    try { const data = await api(complaints ? `/complaints/${id}` : `/${id}`); if (requestId === selectionRequest.current) setHistory(data.items || []) } catch (reason) { setError(reason.message) }
  }
  async function save(data) {
    setBusy(true)
    try {
      const result = await api(complaints ? '/complaints/update' : '/update', { method: 'POST', body: JSON.stringify(data) })
      await refresh()
      await select(result.item)
      setNotice(complaints ? 'Treatment saved. The complaint status is up to date.' : 'Treatment saved. The request status is up to date.')
    } catch (reason) { setError(reason.message) }
    finally { setBusy(false) }
  }
  const active = complaints ? complaintSnapshot : snapshot
  const statusLabels = complaints ? complaintLabels : labels
  const unreadTotal = (snapshot?.unread || 0) + (complaintSnapshot?.unread || 0)
  const searchText = (item) => complaints ? `${item.customer_name} ${item.phone} ${item.reference} ${item.hotel_name} ${item.description}` : `${item.customer_name} ${item.phone} ${item.lead_id} ${item.trip?.destination || ''} ${item.selection?.hotel_name || ''}`
  const items = (active?.items || []).filter((item) => (filter === 'all' || (filter === 'unread' ? !item.read_at : item.status === filter)) && searchText(item).toLowerCase().includes(query.toLowerCase()))
  const current = active?.items.find((item) => idOf(item) === selected)
  return <main className="manager-shell">
    <header className="manager-header"><div><a className="manager-back" href="/">← Back to customer chat</a><small>OZONE VOYAGES · AGENCY WORKSPACE</small><h1>{complaints ? 'Customer complaints' : 'Customer requests'}<span>.</span></h1><p>{complaints ? 'Call customers back about what went wrong and track each complaint to resolution.' : 'Follow up with customers, confirm their stay, and track agency payment.'}</p></div><div className="manager-header-actions">{token && <><span className="manager-notifications" aria-label={`${unreadTotal} unread customer requests and complaints`}>Notifications <b>{unreadTotal}</b></span><button disabled={loading} onClick={refresh}>{loading ? 'Refreshing…' : 'Refresh'}</button><button onClick={() => { sessionStorage.removeItem(KEY); setToken(''); setSnapshot(null); setComplaintSnapshot(null); setSelected(''); setHistory([]); latest.current = null; latestComplaint.current = null }}>Lock inbox</button></>}</div></header>
    {!token ? <form className="manager-login" onSubmit={(event) => { event.preventDefault(); sessionStorage.setItem(KEY, draftToken.trim()); setToken(draftToken.trim()); setDraftToken('') }}><span className="manager-login-icon">✦</span><h2>Manager access</h2><p>Customer contacts are private. Enter the agency access key to open the inbox.</p><label>Access key<input required type="password" autoComplete="off" value={draftToken} onChange={(event) => setDraftToken(event.target.value)} /></label><button className="manager-primary">Open manager inbox</button><small>For this local installation, the key is in <code>data/manager-access-token.txt</code>. A configured administrator key takes priority.</small></form> : <>
      {error && <div className="manager-error" role="alert">{error} <button onClick={() => { sessionStorage.removeItem(KEY); setToken(''); setSnapshot(null); setComplaintSnapshot(null) }}>Change access key</button></div>}
      {notice && <div className="manager-notice" role="status">{notice}<button aria-label="Dismiss notification" onClick={() => setNotice('')}>×</button></div>}
      <nav className="manager-tabs" aria-label="Inbox sections"><button className={!complaints ? 'active' : ''} aria-pressed={!complaints} onClick={() => switchView('reservations')}>Reservation requests{snapshot?.unread > 0 && <b>{snapshot.unread}</b>}</button><button className={complaints ? 'active' : ''} aria-pressed={complaints} onClick={() => switchView('complaints')}>Complaints{complaintSnapshot?.unread > 0 && <b>{complaintSnapshot.unread}</b>}</button></nav>
      <section className="manager-kpis"><article><small>{complaints ? 'ALL COMPLAINTS' : 'ALL CUSTOMER REQUESTS'}</small><b>{active?.total ?? '—'}</b><span>Saved in the agency database</span></article><article><small>NEEDS REVIEW</small><b>{active?.unread ?? '—'}</b><span>{complaints ? 'New complaints to read' : 'New customer contacts'}</span></article><article><small>{complaints ? 'OPEN' : 'IN PROGRESS'}</small><b>{active?.active ?? '—'}</b><span>{complaints ? 'Not yet resolved' : 'Awaiting agency treatment'}</span></article></section>
      <div className="manager-layout"><section className="manager-list"><div className="manager-filters"><input aria-label={complaints ? 'Search complaints' : 'Search customer requests'} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search name, phone, hotel or reference…" /><select aria-label="Filter status" value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">All statuses</option><option value="unread">Unread</option>{Object.entries(statusLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></div><small className="manager-list-count">{items.length} shown · Latest 200 {complaints ? 'complaints' : 'requests'} · Updates every 15 seconds</small>{items.map((item) => complaints
        ? <button className={`manager-request${item.id === selected ? ' selected' : ''}${!item.read_at ? ' unread' : ''}`} key={item.id} onClick={() => select(item)}><div><b>{!item.read_at && <i aria-label="Unread" />}{item.customer_name}</b><span className={`request-status complaint-status-${item.status}`}>{complaintLabels[item.status]}</span></div><span>{item.phone}</span><p className="manager-complaint-excerpt">{item.description}</p><footer><small>{time(item.created_at)}</small><b>{item.hotel_name || item.reference}</b></footer></button>
        : <button className={`manager-request${item.lead_id === selected ? ' selected' : ''}${!item.read_at ? ' unread' : ''}`} key={item.lead_id} onClick={() => select(item)}><div><b>{!item.read_at && <i aria-label="Unread" />}{item.customer_name}</b><span className={`request-status status-${item.status}`}>{labels[item.status]}</span></div><span>{item.phone}</span><p>{item.selection?.hotel_name || item.trip?.destination || 'Callback request'}</p><footer><small>{time(item.created_at)}</small><b>{item.selection?.total_price_text || 'Quote to confirm'}</b></footer></button>)}{!items.length && <div className="manager-empty">{loading ? 'Loading…' : complaints ? 'No complaints match this view. Complaints appear here when a customer reports a problem to Amira.' : 'No requests match this view. Customer contacts will appear here when submitted to Amira.'}</div>}</section>{current ? (complaints
        ? <ComplaintDetails key={`${current.id}-${current.version}`} item={current} onSave={save} busy={busy} history={history} />
        : <RequestDetails key={`${current.lead_id}-${current.version}`} item={current} onSave={save} busy={busy} history={history} />) : <section className="manager-empty manager-detail-empty"><span>✦</span><h2>{complaints ? 'Every complaint, answered.' : 'Every request, clearly followed.'}</h2><p>{complaints ? 'Select a complaint to read what happened, call the customer back and record the outcome.' : 'Select a customer to review their contact details, reservation and treatment history.'}</p></section>}</div>
    </>}
  </main>
}
