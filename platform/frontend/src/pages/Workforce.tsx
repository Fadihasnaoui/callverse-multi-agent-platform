import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { WorkspaceHeader } from "./AgentHub";
import { useWorkforceResource, workforceAction } from "@/workforce/api";
import type { Metric } from "@/workforce/types";
import { ClientPanel, ContactPanel, ContactTable, duration, Empty, ErrorNotice, ExportCsv, Facts, MetricCard, Modal, number, SeriesChart } from "@/workforce/components";
import type { AiAlert, Attention, Charts, ClientSummary, Contact, DaySummary, History, HistoryDay, ModelStatus, Overview, PendingApproval, Scenario, Supervision } from "@/workforce/types";
import styles from "./Workforce.module.css";

const tabs = ["Vue d'ensemble", "Demandes en direct", "Historique clients", "Activité et qualité", "Superviseur"] as const;
type Tab = typeof tabs[number];
const phases = { pret: "Prête · en attente de démarrage", en_cours: "En cours", pause: "En pause", terminee: "Journée terminée", echec: "Simulation arrêtée · erreur" };
type Selection = { date: string; client: string };

function Approval({ pending, busy, answer }: { pending: PendingApproval; busy: boolean; answer: (decision: "approve" | "refuse") => void }) {
  const alert = pending.alert;
  const estimate = alert?.whatif;
  return <section className={`${styles.panel} ${styles.approval}`}>
    <h2>Validation demandée · {pending.sent}</h2>
    {alert && <p>Situation détectée : {({ AI_FAILURE_RATE: "taux d'échec IA", ABANDONMENT: "taux d'abandon", QUEUE_GROWTH: "hausse de la file" } as Record<string, string>)[alert.trigger] || alert.trigger} · {number(alert.observed, true)} (seuil {number(alert.threshold, true)}).</p>}
    <p>{pending.headline} · durée : {pending.duration} minutes.</p>
    {pending.restriction && <p>L'IA reste autorisée uniquement pour les demandes à faible risque : score IA ≥ 80 %, sensibilité faible et urgence non élevée. PPO V3 continue de choisir parmi les actions autorisées.</p>}
    {estimate && <p>Sur {estimate.k} évolutions simulées des 2 prochaines heures, l'intervention fait mieux dans {number(estimate.share_positive, true)} des cas. Effet estimé : {estimate.estimated_gain.toFixed(1)} points.</p>}
    <p>Sans réponse avant {pending.expires_at}, la proposition expire sans appliquer de restriction.</p>
    <div className={styles.controls}><button className="btn-primary" disabled={busy} onClick={() => answer("approve")}>Approuver</button><button disabled={busy} onClick={() => answer("refuse")}>Refuser</button></div>
  </section>;
}

type TwinDecision = { at: string; call_id: string; client: string; turn: string; request_type: string; channel: string; language: string; ai_score: number; decision: string; queue: string; reason: string; humans_available: number };
type TwinCall = { conversation_id: string; call_id: string; client: string; started_at: string; ended_at?: string | null; status: string;
  turns?: number | null; quality_score?: number | null; decisions: number; to_ai: number; to_human: number; escalated: boolean; waited_minutes?: number | null; path: string[] };
type TwinSummary = {
  human_capacity: number; humans_available: number; active_human_assignments: number; conversations: number;
  decisions: number; to_ai: number; to_human: number; waiting_for_human: number;
  by_queue: Record<string, number>; by_reason: Record<string, number>; by_request_type: Record<string, number>;
  tasks: Record<string, number>; recent: TwinDecision[]; calls?: TwinCall[]; calls_ended?: number;
};
const TWIN_QUEUE: Record<string, string> = { ai: "Amira (IA)", customer_advisors: "Conseillers", reservations: "Réservations", waiting_human: "En attente d'un conseiller", waiting: "Attente", callbacks: "Rappels" };
const TWIN_REASON: Record<string, string> = {
  ppo_v3_observed_contact: "Politique PPO", reservation_requires_human_confirmation: "Réservation à confirmer",
  customer_requested_human: "Client demande un humain", ai_not_resolving_contact: "IA ne résout pas (réponses répétées)",
  no_human_adviser_available: "Aucun conseiller libre",
};

function TwinOperations({ summary }: { summary: TwinSummary }) {
  const metric = (value: number, note: string) => ({ value, available: true, unit: "", source: "digital_twin", note });
  const share = (part: number) => summary.decisions ? `${Math.round((100 * part) / summary.decisions)} %` : "—";
  return <>
    <div className={styles.metrics}>
      <MetricCard label="Appels Digital Twin" metric={metric(summary.conversations, "Conversations reçues, conservées après redémarrage")} />
      <MetricCard label="Décisions de routage" metric={metric(summary.decisions, "Une décision par tour client")} />
      <MetricCard label="Vers Amira (IA)" metric={metric(summary.to_ai, `${share(summary.to_ai)} des décisions`)} />
      <MetricCard label="Vers un conseiller" metric={metric(summary.to_human, `${share(summary.to_human)} des décisions`)} />
      <MetricCard label="En attente d'un conseiller" metric={metric(summary.waiting_for_human, "Conseillers tous occupés à ce moment")} />
      <MetricCard label="Conseillers disponibles" metric={metric(summary.humans_available, `Sur ${summary.human_capacity} · ${summary.active_human_assignments} affectation(s) active(s)`)} />
    </div>
    {!!summary.decisions && <p className={styles.sectionIntro}>
      Files : {Object.entries(summary.by_queue).map(([queue, n]) => `${TWIN_QUEUE[queue] || queue} ${n}`).join(" · ")}
      {" — "}Motifs : {Object.entries(summary.by_reason).map(([reason, n]) => `${TWIN_REASON[reason] || reason} ${n}`).join(" · ")}
    </p>}
    {!!summary.calls?.length && <><h3>Historique des appels</h3><div className={styles.tableScroll}><table><thead><tr>{["Début", "Appel", "Client", "Statut", "Tours", "Décisions", "IA / conseiller", "Escaladé", "Attente conseiller", "Score Quality", "Parcours"].map(label => <th key={label}>{label}</th>)}</tr></thead>
      <tbody>{summary.calls.map(call => <tr key={call.conversation_id}><td>{new Date(call.started_at).toLocaleString()}</td><td>{call.call_id}</td><td>{call.client || "—"}</td><td>{call.status === "active" ? "En cours" : call.status === "failed" ? "Interrompu" : "Terminé"}</td><td>{call.turns ?? "—"}</td><td>{call.decisions}</td><td>{call.to_ai} / {call.to_human}</td><td>{call.escalated ? "Oui" : "Non"}</td><td>{call.waited_minutes == null ? "—" : duration(call.waited_minutes)}</td><td>{call.quality_score ?? "—"}</td><td>{call.path.map(q => TWIN_QUEUE[q] || q).join(" → ")}</td></tr>)}</tbody></table></div>
      <h3>Décisions récentes</h3></>}
    {summary.recent.length ? <div className={styles.tableScroll}><table><thead><tr>{["Heure", "Appel", "Client", "Tour", "Demande", "Langue", "Score IA", "Décision", "File", "Motif", "Conseillers libres"].map(label => <th key={label}>{label}</th>)}</tr></thead>
      <tbody>{summary.recent.map((row, index) => <tr key={`${row.call_id}-${row.turn}-${index}`}><td>{row.at ? new Date(row.at).toLocaleTimeString() : "—"}</td><td>{row.call_id}</td><td>{row.client || "—"}</td><td>{row.turn}</td><td>{row.request_type}</td><td>{row.language}</td><td>{row.ai_score}</td><td>{row.decision}</td><td>{TWIN_QUEUE[row.queue] || row.queue}</td><td>{TWIN_REASON[row.reason] || row.reason}</td><td>{row.humans_available}</td></tr>)}</tbody></table></div>
      : <Empty>Aucune décision Digital Twin enregistrée pour le moment.</Empty>}
  </>;
}

function OverviewTab({ overview }: { overview: Overview }) {
  const runs = useWorkforceResource<Supervision>("/biz/supervision", 5000);
  // Opened from a Digital Twin run report: show only that run.
  const [params] = useSearchParams();
  const runId = params.get("batch") || "";
  const runQuery = runId ? `batch_id=${encodeURIComponent(runId)}` : "";
  const twinOperations = useWorkforceResource<TwinSummary>(`/external/summary?limit=60${runQuery ? `&${runQuery}` : ""}`, 5000);
  // KPIs from the real Digital Twin calls (journaled by Workforce), refreshed every 5 s.
  const twinKpis = useWorkforceResource<{ kpis: Record<string, Metric> }>(`/external/kpis${runQuery ? `?${runQuery}` : ""}`, 5000);
  const hasTwinCalls = !!twinKpis.data?.kpis.contacts_recus?.value;
  const groups: [string, [string, string][]][] = [
    ["Demandes et résultats", [["contacts_recus", "Contacts reçus"], ["contacts_actifs", "Contacts actifs"], ["resolution", "Taux de résolution observé"], ["abandon", "Taux d'abandon"]]],
    ["Routage et fiabilité", [["decisions_ia", "Décisions vers l'IA"], ["echec_ia", "Échecs IA · dernier contrôle"], ["resolutions_confirmees", "Résolutions confirmées"], ["resolutions_ia_apparentes", "Résolutions IA apparentes"]]],
    ["Ressources et attente", [["conseillers_disponibles", "Conseillers disponibles"], ["occupation_conseillers", "Occupation des conseillers"], ["charge_humaine", "Charge humaine cumulée"], ["en_ligne", "Demandes en attente en ligne"], ["attente_moyenne", "Attente moyenne"], ["attente_p95", "Attente au 95e centile"], ["attente_plus_longue", "Attente la plus longue"]]],
  ];
  return <>
    <p className={styles.sectionIntro}>{hasTwinCalls ? "Indicateurs calculés sur les appels Digital Twin réels, enregistrés par Workforce Manager et actualisés toutes les 5 secondes." : "Aucun appel Digital Twin enregistré pour le moment : les indicateurs se rempliront au premier appel."}</p>
    <section className={styles.panel} aria-label="Digital Twin operations">
      <div className={styles.sectionHeader}><h2>Opérations Digital Twin</h2><small>{runId ? <>Run du Digital Twin <b>{runId.slice(0, 8)}</b> · <Link to="/workforce">voir tous les runs</Link></> : "Demandes reçues depuis les conversations Amira / Jasmin (tous les runs)"}</small></div>
      {twinOperations.loading ? <p>Chargement des décisions Digital Twin…</p> : twinOperations.data ? <TwinOperations summary={twinOperations.data} /> : <ErrorNotice error={twinOperations.error} retry={twinOperations.refresh} />}
    </section>
    {!!runs.data?.journees?.length && <section className={styles.panel} aria-label="Résultats des scénarios exécutés">
      <div className={styles.sectionHeader}><h2>Résultats des scénarios exécutés</h2><ExportCsv rows={runs.data.journees} filename="resultats-scenarios-workforce.csv" /></div>
      <p>Journées simulées exécutées dans cette session. Les indicateurs détaillés ci-dessous concernent la journée affichée en haut de page ; les parcours des journées précédentes sont dans Historique clients.</p>
      <div className={styles.tableScroll}><table><thead><tr>{["Journée", "Scénario", "État", "Contacts reçus", "Résolution observée", "Abandon", "Décisions vers l'IA", "Attente moyenne", "Échecs IA"].map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{runs.data.journees.map((day, index) => <tr key={`${day.jour}-${index}`}><td>{day.jour}</td><td>{day.scenario}</td><td>{day.terminee ? "Terminée" : "En cours"}</td><td>{number(day.contacts_recus)}</td><td>{number(day.resolution, true)}</td><td>{number(day.abandon, true)}</td><td>{number(day.decisions_ia, true)}</td><td>{duration(day.attente_moyenne)}</td><td>{number(day.echecs_ia)}</td></tr>)}</tbody></table></div>
      <p>La résolution observée inclut les résolutions IA apparentes, encore non confirmées. Ces scénarios utilisent des journées et charges différentes ; ils ne mesurent pas un gain causal entre politiques.</p>
    </section>}
    <ErrorNotice error={runs.error} retry={runs.refresh} />
    {groups.map(([heading, metrics]) => <section key={heading}><h2>{heading}</h2><div className={styles.metrics}>{metrics.map(([key, label]) => <MetricCard key={key} label={label} metric={twinKpis.data?.kpis[key]} />)}</div></section>)}
    <ErrorNotice error={twinKpis.error} retry={twinKpis.refresh} />
    <details className={styles.panel}><summary>Journée simulée de démonstration (données synthétiques Workforce)</summary>
      <p>Ces chiffres viennent de la journée simulée interne de Workforce Manager (« Préparer une démonstration »), pas des appels Digital Twin.</p>
      {groups.map(([heading, metrics]) => <section key={heading}><h3>{heading} · démonstration</h3><div className={styles.metrics}>{metrics.map(([key, label]) => <MetricCard key={key} label={label} metric={overview.kpis[key]} />)}</div></section>)}
    </details>
    <details className={styles.panel}><summary>Guide de lecture des indicateurs</summary><p>Une résolution IA apparente n'est pas confirmée : une erreur peut être découverte plus tard. Les taux indisponibles restent affichés « — ».</p><p>Les décisions de routage peuvent dépasser le nombre de contacts : une mise en attente peut être suivie d'une nouvelle décision. Les ressources reflètent la dernière observation du centre, et les échecs IA le dernier contrôle du superviseur.</p></details>
  </>;
}

function LiveContacts({ overview, select }: { overview: Overview; select: (contact: Contact) => void }) {
  const contacts = useWorkforceResource<Contact[]>("/biz/contacts?jour_courant=true", 3000);
  return <section className={styles.panel}><h2>Suivi des demandes clients</h2><p>Cliquez sur un contact pour consulter son analyse, les décisions enregistrées et son parcours.</p>
    <ErrorNotice error={contacts.error} retry={contacts.refresh} />
    {contacts.loading ? <p role="status">Chargement des demandes…</p> : overview.phase === "pret" ? <Empty>Préparez une démonstration pour recevoir les premières demandes.</Empty> : contacts.data && <ContactTable rows={contacts.data} select={select} />}
  </section>;
}

function HistoryTab({ selected, setSelected }: { selected: Selection | null; setSelected: (selection: Selection | null) => void }) {
  const days = useWorkforceResource<HistoryDay[]>("/biz/historique/jours", 5000);
  const [chosenDate, setChosenDate] = useState("");
  const date = selected?.date || chosenDate || (days.data?.length ? days.data[days.data.length - 1].date : "");
  const history = useWorkforceResource<History>(date ? `/biz/historique/${encodeURIComponent(date)}` : null, 5000);
  const [view, setView] = useState("clients");
  const [query, setQuery] = useState("");
  const [channel, setChannel] = useState("");
  const [handler, setHandler] = useState("");
  const [returnsOnly, setReturnsOnly] = useState(false);
  const data = history.data;
  const returning = new Set(data?.clients.filter(client => client.retours > 0).map(client => client.client));
  const visible = (data?.contacts || []).filter(row => `${row.client} ${row.identifiant}`.toLowerCase().includes(query.trim().toLowerCase()) && (!channel || row.canal === channel) && (!handler || row.traite_par === handler) && (!returnsOnly || returning.has(row.client)));
  const ids = new Set(visible.map(row => row.client));
  const clients = data?.clients.filter(client => ids.has(client.client)) || [];
  const perHour = new Map<string, Record<string, number>>();
  for (const contact of visible) { const hour = `${contact.heure.slice(0, 2)}h`; const row = perHour.get(hour) || {}; row[contact.traite_par] = (row[contact.traite_par] || 0) + 1; perHour.set(hour, row); }
  const handlers = ["Assistant IA", "Conseiller humain", "IA puis conseiller", "Aucune prise en charge"];
  const selectClient = (client: ClientSummary) => setSelected({ date, client: client.client });
  return <>
    <section className={styles.panel}><h2>Historique des clients et des contacts</h2><ErrorNotice error={days.error || history.error} retry={() => { days.refresh(); history.refresh(); }} />
      {days.loading ? <p role="status">Chargement des journées…</p> : !days.data?.length ? <Empty>Aucun historique disponible. Les journées archivées apparaîtront ici.</Empty> : <>
        <label>Journée <select aria-label="Journée historique" value={date} onChange={event => { setChosenDate(event.target.value); setSelected(null); }}>
          {[...days.data].reverse().map(day => <option key={day.date} value={day.date}>{day.jour} · {day.scenario}{day.archivee ? " · archivée" : ""}</option>)}
        </select></label>
        <div className={styles.filters}><input aria-label="Rechercher un client" placeholder="Client ou contact…" value={query} onChange={event => setQuery(event.target.value)} />
          <select aria-label="Canal historique" value={channel} onChange={event => setChannel(event.target.value)}><option value="">Tous les canaux</option>{[...new Set(data?.contacts.map(row => row.canal))].map(value => <option key={value}>{value}</option>)}</select>
          <select aria-label="Traitement historique" value={handler} onChange={event => setHandler(event.target.value)}><option value="">Tous les traitements</option>{handlers.map(value => <option key={value}>{value}</option>)}</select>
          <label><input type="checkbox" checked={returnsOnly} onChange={event => setReturnsOnly(event.target.checked)} /> Clients revenus</label>
          <button onClick={() => { setQuery(""); setChannel(""); setHandler(""); setReturnsOnly(false); }}>Réinitialiser</button>
        </div>
        <div className={styles.controls}>{[["clients", "Par client"], ["contacts", "Tous les contacts"], ["heures", "Prise en charge par heure"]].map(([key, label]) => <button key={key} className={view === key ? "btn-primary" : "btn-ghost"} aria-pressed={view === key} onClick={() => setView(key)}>{label}</button>)}</div>
        {history.loading && <p role="status">Chargement de l'historique…</p>}
        {data && <>
          <p>{clients.length} client(s) · {visible.length} contact(s) affiché(s) sur {data.contacts.length}. Un client regroupe son contact et ses éventuels retours dans la même journée.</p>
          {view === "clients" && <><ExportCsv rows={clients} filename={`clients-${date}.csv`} /><div className={styles.tableScroll}><table><thead><tr>{["Client", "Premier contact", "Dernier contact", "Contacts", "Retours", "Motif du retour", "Demande", "Canal", "Traités par l'IA", "Par un conseiller", "Dernier résultat", "Statut"].map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{clients.map(client => <tr key={client.client}><td><button className={styles.contactLink} onClick={() => selectClient(client)}>{client.client}</button></td><td>{client.premier}</td><td>{client.dernier}</td><td>{client.contacts}</td><td>{client.retours}</td><td>{client.motif_retour || "—"}</td><td>{client.type_demande}</td><td>{client.canal}</td><td>{client.par_ia}</td><td>{client.par_conseiller}</td><td>{client.dernier_resultat}</td><td>{client.etat}</td></tr>)}</tbody></table></div>{!clients.length && <Empty>Aucun client ne correspond à la recherche.</Empty>}</>}
          {view === "contacts" && <ContactTable rows={visible} select={contact => setSelected({ date, client: contact.client })} />}
          {view === "heures" && <SeriesChart title="Prise en charge par heure d'arrivée" rows={[...perHour].sort(([a], [b]) => a.localeCompare(b)).map(([heure, counts]) => ({ heure, ...Object.fromEntries(handlers.map(key => [key, counts[key] || 0])) }))} series={handlers} />}
        </>}
      </>}
    </section>
    {selected && <ClientPanel key={`${selected.date}-${selected.client}`} date={selected.date} client={selected.client} close={() => setSelected(null)} />}
  </>;
}

function ActivityTab() {
  const resource = useWorkforceResource<Charts>("/biz/charts");
  const charts = resource.data;
  return <><ErrorNotice error={resource.error} retry={resource.refresh} />{resource.loading && <p role="status">Chargement des graphiques…</p>}{charts && <>
    <div className={styles.columns}>
      <SeriesChart title="Contacts reçus par tranche de 15 minutes" rows={charts.demandes} series={["Téléphone", "Chat", "WhatsApp", "Email"]} />
      <section className={styles.panel}><h3>Décisions de routage depuis 08:00</h3>{charts.routage.map(row => <div className={styles.barRow} key={row.action}><span>{row.libelle}</span><div className={styles.barTrack}><span style={{ width: `${(row.part ?? 0) * 100}%` }} /></div><span>{row.decisions} · {number(row.part, true)}</span></div>)}</section>
      <SeriesChart title="Occupation des conseillers" rows={charts.charge} series={["Taux d'occupation"]} percent />
      <SeriesChart title="Demandes en attente" rows={charts.charge} series={["Demandes en ligne"]} />
    </div>
    <SeriesChart title="Qualité de service · événements par tranche de 15 minutes" rows={charts.qualite} series={["Résolutions", "Abandons", "Échecs IA"]} />
    <p className={styles.sectionIntro}>Les résolutions incluent les résolutions IA apparentes. Les échecs IA incluent les échecs visibles et les fausses résolutions découvertes. Les ressources et l'attente sont observées toutes les 5 minutes.</p>
  </>}</>;
}

function SupervisorTab({ overview }: { overview: Overview }) {
  const resource = useWorkforceResource<Supervision>("/biz/supervision");
  const alert = useWorkforceResource<AiAlert>("/biz/alerte-ia");
  const supervision = resource.data;
  return <><ErrorNotice error={resource.error || alert.error} retry={() => { resource.refresh(); alert.refresh(); }} />
    <div className={styles.columns}>
      <section className={styles.panel}><h2>État du centre et dernier contrôle</h2><p><strong>{overview.etat.libelle}</strong> · {overview.politique}</p><p>{overview.etat.texte}</p>
        {overview.controle && <Facts data={Object.fromEntries(Object.entries(overview.controle).filter(([key]) => ["heure", "debut", "traitements", "echecs", "visibles", "fausses", "arrivees", "abandons", "file"].includes(key)).map(([key, value]) => [({ heure: "Heure", debut: "Début de fenêtre", traitements: "Traitements IA", echecs: "Échecs IA", visibles: "Échecs visibles", fausses: "Fausses résolutions", arrivees: "Contacts reçus", abandons: "Abandons", file: "File conseiller" } as Record<string, string>)[key], value]))} />}
        {alert.data && <><p>Échecs IA : {number(alert.data.taux, true)} · seuil {number(alert.data.seuil, true)}</p><p>{alert.data.texte_echecs}</p><p className={styles.notice}>{alert.data.etat}</p><small>{alert.data.regle}</small></>}
      </section>
      <section className={styles.panel}><h2>Alertes et recommandations</h2>{resource.loading && <p role="status">Chargement du superviseur…</p>}{supervision && <>
        {!supervision.alertes.some(row => row.jour === overview.jour) && <Empty>Aucune alerte du superviseur aujourd'hui.</Empty>}
        {supervision.alertes.filter(row => row.jour === overview.jour).slice().reverse().map((row, index) => <article key={index} className={styles.notice}><strong>{row.heure} · {row.alerte} {number(row.valeur, true)} (seuil {number(row.seuil, true)})</strong><p>{row.decision}</p><p>{row.suite}</p></article>)}
      </>}</section>
    </div>
    {supervision && <section className={styles.panel}><h2>Journal du superviseur</h2><ExportCsv rows={supervision.transitions.filter(row => row.jour === overview.jour)} filename="supervision.csv" />
      {supervision.transitions.filter(row => row.jour === overview.jour).slice().reverse().map((row, index) => <p key={index}>{row.heure} · {row.de} → {row.vers} · {row.motif}</p>)}
      {supervision.interventions.filter(row => row.jour === overview.jour).map((row, index) => <p key={`i${index}`}>{row.envoyee} · {row.proposition} · {row.reponse}</p>)}
      {supervision.secours.filter(row => row.jour === overview.jour).map((row, index) => <p key={`s${index}`}>{row.heure} · {row.evenement}</p>)}
      {supervision.historique?.filter(row => row.jour === overview.jour).map((row, index) => <p key={`h${index}`}>{row.heure} · {row.evenement} · {row.detail}</p>)}
    </section>}
    <p className={styles.sectionIntro}>Le superviseur contrôle le centre à la première demande reçue après chaque échéance de 15 minutes. La restriction temporaire de l'IA nécessite la validation du responsable. Sans réponse, elle n'est pas appliquée.</p>
  </>;
}

function SummaryModal({ close, prepare }: { close: () => void; prepare: () => void }) {
  const resource = useWorkforceResource<DaySummary>("/biz/bilan");
  const data = resource.data;
  return <Modal title="Bilan de la journée" close={close}><ErrorNotice error={resource.error} retry={resource.refresh} />{resource.loading && <p role="status">Chargement du bilan…</p>}{data && <>
    <p>{data.jour} · {data.scenario} · {data.terminee ? "Journée terminée" : `En cours à ${data.heure}`}</p>
    <Facts data={{ "Contacts reçus": data.demandes_recues, "Contacts résolus": data.resolues, "Dont IA apparentes": data.dont_ia_apparentes, "Contacts abandonnés": data.abandonnees, "Contacts non terminés": data.non_termines, "Autres échecs": data.autres_echecs, "Décisions de routage": data.decisions, "Décisions vers l'IA": number(data.decisions_ia, true), "Charge humaine": duration(data.charge_humaine), "Attente moyenne": duration(data.attente_moyenne), "Attente maximale": duration(data.attente_max), "Échecs IA observés": data.echecs_ia, "Alertes du superviseur": data.alertes }} />
    <p className={styles.notice}>Les résolutions IA apparentes restent non confirmées. Les fausses résolutions découvertes sont exclues des contacts résolus.</p>
    {data.liste_alertes.map((row, index) => <p key={index}>{row.heure} · {row.alerte} · {row.decision} {row.suite}</p>)}
    {data.interventions.map((row, index) => <p key={`i${index}`}>{row.envoyee} · {row.proposition} · {row.reponse}</p>)}
    <button className="btn-primary" onClick={prepare}>Préparer une nouvelle démonstration</button>
  </>}</Modal>;
}

function PrepareModal({ overview, close, run, busy }: { overview: Overview; close: () => void; run: (path: string, body?: unknown) => Promise<boolean>; busy: boolean }) {
  const scenarios = useWorkforceResource<Scenario[]>("/biz/scenarios", 0);
  const [code, setCode] = useState("");
  const [speed, setSpeed] = useState(overview.vitesse || 1);
  const [interrupt, setInterrupt] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const scenario = scenarios.data?.find(row => row.code === code) || scenarios.data?.[0];
  const active = overview.phase === "en_cours" || overview.phase === "pause";
  async function start() {
    if (!scenario || starting || busy || (active && !interrupt)) return;
    setStarting(true); setError("");
    try {
      if (!await run(`/live/speed?x=${speed}`)) { setError("La vitesse n'a pas pu être appliquée. La journée n'a pas été démarrée."); return; }
      if (await run("/live/start", { family: scenario.code })) close();
      else setError("La journée n'a pas pu être démarrée. Vérifiez la connexion puis réessayez.");
    } finally { setStarting(false); }
  }
  return <Modal title="Préparer une démonstration" close={close}>
    <ErrorNotice error={scenarios.error || error} retry={scenarios.refresh} />{scenarios.loading && <p role="status">Chargement des scénarios…</p>}
    {scenario && <><label>Scénario de la journée<select value={scenario.code} onChange={event => setCode(event.target.value)} disabled={starting}>{scenarios.data?.map(row => <option key={row.code} value={row.code}>{row.libelle}</option>)}</select></label>
      <p>{scenario.description}</p><p>Journée sélectionnée automatiquement : <strong>{scenario.journee}</strong>. L'ordre des journées est fixé à l'avance.</p>
      <label>Vitesse de la démonstration<select value={speed} onChange={event => setSpeed(Number(event.target.value))} disabled={starting}><option value={.5}>Lente</option><option value={1}>Normale</option><option value={2}>Rapide</option></select></label>
      <p>La vitesse ne change pas les décisions de routage.</p>
      {active && <label><input type="checkbox" checked={interrupt} onChange={event => setInterrupt(event.target.checked)} disabled={starting} /> Interrompre la journée en cours ({overview.scenario}, {overview.heure})</label>}
      <div className={styles.controls}><button className="btn-primary" disabled={busy || starting || !!scenarios.error || (active && !interrupt)} onClick={() => void start()}>{starting ? "Démarrage…" : "Démarrer"}</button>
        {active && <button disabled={busy || starting} onClick={async () => { if (await run(`/live/speed?x=${speed}`)) close(); }}>Appliquer seulement la vitesse</button>}
      </div>
    </>}
  </Modal>;
}

export function WorkforceWorkspace() {
  const [tab, setTab] = useState<Tab>(tabs[0]);
  const [modal, setModal] = useState<"prepare" | "summary" | null>(null);
  const [contact, setContact] = useState<Contact | null>(null);
  const [client, setClient] = useState<Selection | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState("");
  const model = useWorkforceResource<ModelStatus>("/biz/model", 10000);
  const ready = model.data?.ok && !model.error;
  const overview = useWorkforceResource<Overview>(ready ? "/biz/overview" : null, 2000);
  const attention = useWorkforceResource<Attention>(ready ? "/biz/attention" : null, 3000);
  const data = overview.data;
  async function run(path: string, body?: unknown) {
    if (busy || !ready || !!overview.error) return false;
    setBusy(true); setActionError("");
    try {
      await workforceAction(path, body);
      if (path === "/live/start") { setContact(null); setClient(null); }
      overview.refresh(); attention.refresh();
      return true;
    } catch (error) { setActionError(error instanceof Error ? error.message : "Action impossible"); overview.refresh(); attention.refresh(); return false; }
    finally { setBusy(false); }
  }
  const pending = attention.data?.validation || data?.demande;
  const disabled = busy || !ready || !!overview.error;
  return <div className={styles.workspace}>
    <WorkspaceHeader />
    <div className={styles.layout}>
      <aside className={styles.sidebar}><strong>Workforce Manager</strong><small>Centre de relation client</small><nav aria-label="Navigation Workforce">{tabs.map(label => <button key={label} aria-current={tab === label ? "page" : undefined} onClick={() => setTab(label)}>{label}</button>)}</nav></aside>
      <main className={styles.main}>
        <div className={styles.heading}><div><p className={styles.eyebrow}>CUSTOMER OPERATIONS</p><h1>Centre de relation client</h1><p>Workforce Manager · Pilotage des demandes clients</p></div><div className={styles.controls}><span className={styles.badge}>Simulation · Données synthétiques</span><button className="btn-primary" disabled={!data || disabled} onClick={() => setModal("prepare")}>Préparer une démonstration</button></div></div>
        <ErrorNotice error={model.error || overview.error} retry={() => { model.refresh(); overview.refresh(); }} />
        {model.loading && <p role="status">Connexion à Workforce Manager…</p>}
        {model.data && !model.data.ok && <div className={styles.error} role="alert"><strong>PPO V3 Risk-Aware n'a pas pu être chargé : la démonstration est bloquée.</strong><p>{model.data.error}</p><p>Aucune autre politique n'est utilisée à sa place.</p></div>}
        {ready && overview.loading && <p role="status">Chargement du centre de relation client…</p>}
        {ready && data && <>
          <section className={styles.statusBar} aria-label="État de la simulation">
            <div><small>Heure simulée</small><strong className={styles.clock}>{data.heure}</strong></div><div><small>Journée</small><strong>{data.jour}</strong></div><div><small>Scénario</small><strong>{data.scenario}</strong></div><div><small>Simulation</small><strong className={data.phase === "en_cours" ? styles.live : ""}>{phases[data.phase]}</strong></div><div><small>Politique active</small><strong>{data.politique}</strong></div><div><small>Superviseur</small><strong className={data.etat.niveau === "critical" ? styles.critical : data.etat.niveau === "warning" ? styles.warning : styles.live}>{data.etat.libelle}</strong></div>
            <div className={styles.controls}>{data.phase === "en_cours" && <button disabled={disabled} onClick={() => void run("/live/pause")}>Pause</button>}{data.phase === "pause" && <button className="btn-primary" disabled={disabled} onClick={() => void run("/live/resume")}>Reprendre</button>}{data.phase !== "pret" && <button disabled={disabled} onClick={() => setModal("summary")}>Consulter le bilan</button>}</div>
          </section>
          {actionError && <div role="alert" className={styles.error}>{actionError}</div>}
          <ErrorNotice error={attention.error} retry={attention.refresh} />
          {data.phase === "pret" && <Empty>Aucune journée en cours. Préparez une démonstration pour choisir un scénario et recevoir les premières demandes.</Empty>}
          {data.phase === "terminee" && <p className={styles.notice}>Journée terminée. Consultez le bilan ou préparez une nouvelle démonstration.</p>}
          {data.sante?.etat === "FAILED" && <div role="alert" className={styles.error}>La simulation s'est arrêtée : {data.sante.erreur}. Préparez une nouvelle démonstration pour reprendre.</div>}
          {data.sante?.progression_bloquee && <div role="alert" className={styles.error}>L'horloge de simulation ne progresse plus. Actualisez l'état avant de continuer.</div>}
          {data.sante?.superviseur_degrade && <p role="alert" className={styles.notice}>Le superviseur est dégradé. Les décisions de routage continuent ; les nouvelles analyses peuvent être indisponibles.</p>}
          {pending && <Approval pending={pending} busy={disabled || !!attention.error || !pending.run_id || !Number.isInteger(pending.version)} answer={decision => void run("/live/manager", { run_id: pending.run_id, request_id: pending.request_id, expected_version: pending.version, decision })} />}
          {data.intervention && <section className={`${styles.panel} ${styles.approval}`}><h2>Intervention approuvée en cours</h2><p>IA réservée aux demandes à faible risque · de {data.intervention.debut} à {data.intervention.fin} · {duration(data.intervention.restant)} restantes.</p><p>{data.intervention.affected_decisions} décision(s) avec restriction IA sur {data.intervention.decisions} décision(s) pendant l'intervention.</p><button disabled={disabled} onClick={() => void run("/live/intervention/cancel", { run_id: data.intervention!.run_id, intervention_id: data.intervention!.id })}>Annuler l'intervention</button></section>}
          {attention.data && data.phase !== "pret" && <div className={styles.attention}><strong>À surveiller</strong><button className="btn-ghost" onClick={() => setTab(tabs[1])}>{attention.data.en_attente.length} demande(s) en attente · {attention.data.rappels.length} rappel(s)</button>{attention.data.incidents.map((incident, index) => <span className={styles.badge} key={index}>{incident.texte}</span>)}</div>}
          {tab === tabs[0] && <OverviewTab overview={data} />}
          {tab === tabs[1] && <><LiveContacts overview={data} select={setContact} />{contact && contact.date === data.date && <ContactPanel key={`${contact.date}-${contact.identifiant}`} contact={contact} close={() => setContact(null)} history={() => { setClient({ date: contact.date, client: contact.client }); setTab(tabs[2]); }} />}</>}
          {tab === tabs[2] && <HistoryTab selected={client} setSelected={setClient} />}
          {tab === tabs[3] && <ActivityTab />}
          {tab === tabs[4] && <SupervisorTab overview={data} />}
          {modal === "prepare" && <PrepareModal overview={data} close={() => setModal(null)} run={run} busy={disabled} />}
          {modal === "summary" && <SummaryModal close={() => setModal(null)} prepare={() => setModal("prepare")} />}
        </>}
      </main>
    </div>
  </div>;
}
