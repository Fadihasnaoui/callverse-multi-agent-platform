import { Link, NavLink } from "react-router-dom";
import { useAuth } from "@/auth";
import { LanguageSwitch, useI18n } from "@/i18n";
import { ThemeToggle } from "@/components/ThemeToggle";
import styles from "./AgentHub.module.css";

function AgentIcon({ workforce = false }: { workforce?: boolean }) {
  return <svg viewBox="0 0 32 32" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    {workforce ? <><rect x="11" y="3" width="10" height="8" rx="2" /><rect x="2" y="21" width="10" height="8" rx="2" /><rect x="20" y="21" width="10" height="8" rx="2" /><path d="M16 11v5M7 21v-5h18v5" /></> : <><path d="m6 12 20-6v20L6 20zM6 12H3v8h3M10 22l2 6h5l-2-5M26 12l3-1M26 20l3 1" /><path d="M10 11v10" /></>}
  </svg>;
}

function OpenArrow() {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6" /></svg>;
}

export function WorkspaceHeader() {
  const { t } = useI18n();
  const { logout } = useAuth();
  return <header className={styles.header}>
    <Link to="/" className={styles.brand}><span className={styles.mark}>O<span /></span><span>Ozone Voyages<small>{t("Agent workspace")}</small></span></Link>
    <nav className={styles.navigation} aria-label={t("Agent navigation")}>
      <NavLink to="/" end>{t("Overview")}</NavLink>
      <NavLink to="/marketing">{t("Marketing")}</NavLink>
      <NavLink to="/customer-advisor">Amira</NavLink>
      <NavLink to="/client-simulator">Jasmin</NavLink>
      <NavLink to="/quality-analyst">Quality</NavLink>
      <NavLink to="/digital-twin">Digital Twin</NavLink>
      <NavLink to="/workforce">{t("Workforce")}</NavLink>
    </nav>
    <div className={styles.actions}><ThemeToggle /><LanguageSwitch /><button className="btn-ghost" onClick={logout}>{t("Log out")}</button></div>
  </header>;
}

export function AgentHub() {
  const { t } = useI18n();
  return <div className={styles.workspace}>
    <WorkspaceHeader />
    <main className={styles.hub}>
      <section className={styles.hero} aria-labelledby="workspace-title">
      <div className={styles.intro}><p className={styles.eyebrow}><span />{t("OZONE OPERATIONS")}</p>
        <h1 id="workspace-title">{t("Your agents. One workspace.")}</h1>
        <p>{t("Choose an agent to manage your marketing or oversee your customer contact center.")}</p>
        <a href="#agents" className={styles.explore}>{t("Explore your agents")} <OpenArrow /></a>
      </div>
      <div className={styles.workspaceMap} aria-hidden="true">
        <div className={styles.mapOrbit} /><div className={styles.mapOrbitInner} />
        <div className={styles.mapCenter}><span className={styles.mapLogo}>O</span><span>OZONE<small>{t("Agent workspace")}</small></span></div>
        <div className={styles.mapAgent}><span><AgentIcon /></span><div>{t("Marketing")}<small>{t("GROWTH & CAMPAIGNS")}</small></div></div>
        <div className={`${styles.mapAgent} ${styles.mapWorkforce}`}><span><AgentIcon workforce /></span><div>{t("Workforce")}<small>{t("CUSTOMER OPERATIONS")}</small></div></div>
        <span className={styles.orbitDot} />
      </div>
      </section>
      <div className={styles.sectionHeading}><div><p className={styles.sectionEyebrow}>{t("YOUR WORKSPACE")}</p><h2>{t("Where would you like to start?")}</h2></div><span className={styles.agentCount}>06 <span>{t("specialized agents")}</span></span></div>
      <div className={styles.grid} id="agents">
        <Link to="/marketing" className={styles.card}>
          <div className={styles.cardTop}><span className={styles.agentIcon}><AgentIcon /></span><span className={styles.cardNumber}>01</span></div>
          <span className={styles.category}>{t("GROWTH & CAMPAIGNS")}</span>
          <h2>{t("Digital Marketing Agent")}</h2>
          <p>{t("Plan content, review campaigns, manage your publishing calendar and follow your leads.")}</p>
          <div className={styles.tags}><span>{t("Content")}</span><span>{t("Calendar")}</span><span>{t("Leads")}</span></div>
          <div className={styles.open}>{t("Open marketing dashboard")} <span><OpenArrow /></span></div>
        </Link>
        <Link to="/workforce" className={`${styles.card} ${styles.workforce}`}>
          <div className={styles.cardTop}><span className={styles.agentIcon}><AgentIcon workforce /></span><span className={styles.cardNumber}>02</span></div>
          <span className={styles.category}>{t("CUSTOMER OPERATIONS")}</span>
          <h2>{t("Workforce Manager")}</h2>
          <p>{t("Follow customer requests, advisor availability, routing decisions and contact center performance.")}</p>
          <div className={styles.tags}><span>{t("Routing")}</span><span>{t("Supervision")}</span><span>{t("Performance")}</span></div>
          <div className={styles.open}>{t("Open workforce dashboard")} <span><OpenArrow /></span></div>
        </Link>
        <Link to="/digital-twin" className={`${styles.card} ${styles.digitalTwin}`}>
          <div className={styles.cardTop}><span className={styles.agentIcon}><AgentIcon workforce /></span><span className={styles.cardNumber}>03</span></div>
          <span className={styles.category}>SIMULATION &amp; QUALITY</span>
          <h2>Ozone Digital Twin</h2>
          <p>Run a controlled Jasmin and Amira conversation, inspect each turn, then send the saved evidence to quality review and routing.</p>
          <div className={styles.tags}><span>Amira</span><span>Jasmin</span><span>Quality</span></div>
          <div className={styles.open}>Open simulation control room <span><OpenArrow /></span></div>
        </Link>
        <Link to="/customer-advisor" className={`${styles.card} ${styles.advisor}`}><div className={styles.cardTop}><span className={styles.agentIcon}><AgentIcon /></span><span className={styles.cardNumber}>04</span></div><span className={styles.category}>CUSTOMER CONVERSATION</span><h2>Amira</h2><p>Handle hotel requests, FAQ answers, reservations and human handoffs in the adviser workspace.</p><div className={styles.tags}><span>Offers</span><span>Memory</span><span>Handoff</span></div><div className={styles.open}>Open Amira <span><OpenArrow /></span></div></Link>
        <Link to="/client-simulator" className={`${styles.card} ${styles.simulator}`}><div className={styles.cardTop}><span className={styles.agentIcon}><AgentIcon workforce /></span><span className={styles.cardNumber}>05</span></div><span className={styles.category}>SIMULATED CLIENTS</span><h2>Jasmin</h2><p>Create client personas, objections and scenarios before running the Digital Twin.</p><div className={styles.tags}><span>Personas</span><span>Scenarios</span><span>Voice</span></div><div className={styles.open}>Open Jasmin <span><OpenArrow /></span></div></Link>
        <Link to="/quality-analyst" className={`${styles.card} ${styles.quality}`}><div className={styles.cardTop}><span className={styles.agentIcon}><AgentIcon /></span><span className={styles.cardNumber}>06</span></div><span className={styles.category}>EVIDENCE REVIEW</span><h2>Quality Analyst</h2><p>Inspect saved conversations, evidence, quality findings and improvement opportunities.</p><div className={styles.tags}><span>Evidence</span><span>Scoring</span><span>Findings</span></div><div className={styles.open}>Open Quality <span><OpenArrow /></span></div></Link>
      </div>
      <footer className={styles.footer}><span>Ozone Voyages <span aria-hidden="true">/</span> {t("Agent workspace")}</span><span>{t("Built around your daily operations.")}</span></footer>
    </main>
  </div>;
}

