import './ClientSimulator.css'

export default function ClientSimulator() {
  return <main className="simulator-shell">
    <header className="simulator-header">
      <div>
        <span>OZONE VOYAGES / TRAINING</span>
        <h1>Client simulator</h1>
        <p>Train Amira against Jasmin Voyages personas, then review the outcome and coaching signals.</p>
      </div>
      <div className="simulator-actions">
        <a href="/">Customer advisor</a>
        <a href="/voice-lab">Voice lab</a>
        <a href="/quality">Quality review</a>
      </div>
    </header>
    <section className="simulator-notice">
      <strong>Connected safely</strong>
      <span>Jasmin calls Amira through the public customer-advisor API. Its personas, simulation history and configuration remain isolated from live customer data.</span>
    </section>
    <section className="simulator-frame-wrap">
      <iframe title="Jasmin Voyages client simulator" src="/simulator-workbench/" className="simulator-frame" />
    </section>
  </main>
}
