import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { useWorkforceResource } from "./api";
import type { ChartRow, ClientHistory, Contact, ContactDetail, Metric } from "./types";
import styles from "../pages/Workforce.module.css";

export function number(value: number | null | undefined, percent = false) {
  return value == null ? "—" : new Intl.NumberFormat("fr-TN", { maximumFractionDigits: percent ? 1 : 0, ...(percent ? { style: "percent" as const } : {}) }).format(value);
}
export function duration(value: number | null | undefined) { return value == null ? "—" : `${new Intl.NumberFormat("fr-TN", { maximumFractionDigits: 1 }).format(value)} min`; }

export function ErrorNotice({ error, retry }: { error: string; retry?: () => void }) {
  return error ? <div role="alert" className={styles.error}><strong>Connexion Workforce Manager interrompue</strong><p>{error}</p><p>Vérifiez que son API est démarrée. Les dernières données affichées peuvent être anciennes.</p>{retry && <button onClick={retry}>Réessayer</button>}</div> : null;
}
export function Empty({ children }: { children: ReactNode }) { return <div className={styles.empty}>{children}</div>; }
export function MetricCard({ label, metric }: { label: string; metric?: Metric }) {
  return <article className={styles.metric}><p>{label}</p><strong>{metric?.available ? metric.unit === "min" ? duration(metric.value) : number(metric.value, metric.unit === "%") : "—"}</strong><small>{metric?.note || (metric?.available ? metric.source : "Indisponible pour l'instant")}</small></article>;
}
export function Facts({ data }: { data: Record<string, string | number | null> }) {
  return <dl className={styles.facts}>{Object.entries(data).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value ?? "—"}</dd></div>)}</dl>;
}

export function Modal({ title, children, close }: { title: string; children: ReactNode; close: () => void }) {
  const id = useId();
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const focused = document.activeElement as HTMLElement | null;
    const original = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    dialog.current?.querySelector<HTMLButtonElement>("button")?.focus();
    return () => { document.body.style.overflow = original; focused?.focus(); };
  }, []);
  return <div className={styles.overlay} onClick={event => { if (event.target === event.currentTarget) close(); }}>
    <dialog ref={dialog} open aria-modal="true" aria-labelledby={id} className={styles.modal} onKeyDown={event => {
      if (event.key === "Escape") close();
      if (event.key === "Tab") {
        const items = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), input:not(:disabled), select:not(:disabled), a[href], [tabindex="0"]'));
        const first = items[0], last = items[items.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    }}><div className={styles.sectionHeader}><h2 id={id}>{title}</h2><button onClick={close} aria-label="Fermer">×</button></div>{children}</dialog>
  </div>;
}

export function csvText(rows: Record<string, unknown>[]): string {
  const columns = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const cell = (value: unknown) => {
    const text = Array.isArray(value) ? value.join(" → ") : value == null ? "" : String(value);
    return `"${(/^[=+\-@\t\r]/.test(text) ? "'" : "") + text.replace(/"/g, '""')}"`;
  };
  return "\uFEFF" + [columns.map(cell).join(","), ...rows.map(row => columns.map(key => cell(row[key])).join(","))].join("\r\n");
}
export function ExportCsv({ rows, filename }: { rows: object[]; filename: string }) {
  return <button className="btn-ghost" disabled={!rows.length} onClick={() => {
    const url = URL.createObjectURL(new Blob([csvText(rows as Record<string, unknown>[])], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a"); link.href = url; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }}>Exporter CSV</button>;
}

export function ContactTable({ rows, select }: { rows: Contact[]; select: (contact: Contact) => void }) {
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [page, setPage] = useState(0);
  const filterFields = [["canal", "Canal"], ["affectation", "Routage"], ["traite_par", "Traité par"], ["etat", "Statut"], ["resultat", "Résultat"]] as const;
  const filtered = rows.filter(row => `${row.identifiant} ${row.client} ${row.type_demande}`.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()) && filterFields.every(([key]) => !filters[key] || row[key] === filters[key]));
  const pages = Math.max(1, Math.ceil(filtered.length / 40));
  const current = Math.min(page, pages - 1);
  return <>
    <div className={styles.filters}>
      <input aria-label="Rechercher un contact" placeholder="Client ou contact…" value={query} onChange={event => { setQuery(event.target.value); setPage(0); }} />
      {filterFields.map(([key, label]) => <select key={key} aria-label={label} value={filters[key] || ""} onChange={event => { setFilters(previous => ({ ...previous, [key]: event.target.value })); setPage(0); }}><option value="">{label} : tous</option>{[...new Set(rows.map(row => row[key]))].sort().map(value => <option key={value}>{value}</option>)}</select>)}
      <button className="btn-ghost" onClick={() => { setQuery(""); setFilters({}); setPage(0); }}>Réinitialiser</button>
    </div>
    <div className={styles.sectionHeader}><small>{filtered.length} contact(s) affiché(s) sur {rows.length}</small><ExportCsv rows={filtered} filename="contacts-workforce.csv" /></div>
    {!filtered.length ? <Empty>Aucun contact ne correspond à la recherche.</Empty> : <div className={styles.tableScroll}><table><thead><tr>{["Contact", "Heure", "Canal", "Demande", "Priorité", "Score IA", "Attente", "Routage", "Traité par", "Résultat", "Statut"].map(label => <th key={label}>{label}</th>)}</tr></thead><tbody>{filtered.slice(current * 40, current * 40 + 40).map(row => <tr key={`${row.date}-${row.identifiant}`}>
      <td><button className={styles.contactLink} onClick={() => select(row)}>{row.identifiant}</button></td><td>{row.heure}</td><td>{row.canal}</td><td>{row.type_demande}</td><td>{row.priorite}</td><td>{number(row.score_ia, true)}</td><td>{duration(row.attente_min)}</td><td>{row.routages.join(" → ")}</td><td>{row.traite_par}</td><td>{row.resultat}</td><td><span className={styles.badge}>{row.etat}</span></td>
    </tr>)}</tbody></table></div>}
    {pages > 1 && <div className={styles.pagination}><button disabled={current === 0} onClick={() => setPage(current - 1)}>Précédent</button><span>{current + 1} / {pages}</span><button disabled={current + 1 >= pages} onClick={() => setPage(current + 1)}>Suivant</button></div>}
  </>;
}

export function Journey({ detail }: { detail: ContactDetail }) {
  return <>
    <Facts data={detail.info} />
    <p className={styles.notice}>{detail.etat_actuel || detail.resultat || "En cours"}{detail.apparente ? " · Résolution IA apparente, non confirmée." : ""}</p>
    {detail.verite && <p className={styles.notice}>Bilan rétrospectif après fermeture : {detail.verite}</p>}
    <h3>Parcours du contact</h3>
    <ol className={styles.timeline}>{(detail.timeline || detail.evenements.map(event => ({ heure: event.heure, texte: `${event.evenement}${event.tentative ? ` · tentative ${event.tentative}` : ""}` }))).map((event, index) => <li key={index}><time>{event.heure}</time><span>{event.texte}</span></li>)}</ol>
    <h3>Décisions de routage</h3>
    {detail.decisions.map((decision, index) => <details className={styles.decision} key={index}><summary>{decision.heure} · {decision.action} · {decision.decideur}</summary>
      <p>{decision.explication}</p><p>Score IA estimé : {number(decision.confiance_ia, true)}{decision.proposition ? ` · Proposition PPO : ${decision.proposition}` : ""}</p>
      <Facts data={decision.etat} />
      <ul>{decision.autorisees.map(action => <li key={action.action}>{action.action} : {action.autorisee ? "autorisée" : `bloquée · ${action.motif}`}{decision.probas?.[action.action] != null ? ` · probabilité PPO ${number(decision.probas[action.action], true)}` : ""}</li>)}</ul>
    </details>)}
    {detail.note && <small>{detail.note}</small>}
  </>;
}
export function ContactPanel({ contact, close, history }: { contact: Contact; close: () => void; history: () => void }) {
  const resource = useWorkforceResource<ContactDetail>(`/biz/contact/${encodeURIComponent(contact.date)}/${encodeURIComponent(contact.identifiant)}`);
  return <section className={styles.panel} aria-label="Parcours du contact"><div className={styles.sectionHeader}><h2>{contact.identifiant} · {contact.heure}</h2><button onClick={close}>Fermer</button></div>
    <ErrorNotice error={resource.error} retry={resource.refresh} />
    {resource.loading && <p role="status">Chargement du parcours…</p>}
    {resource.data && <><Journey detail={resource.data} /><button className="btn-primary" onClick={history}>Historique du client {contact.client}</button></>}
  </section>;
}
export function ClientPanel({ date, client, close }: { date: string; client: string; close: () => void }) {
  const resource = useWorkforceResource<ClientHistory>(`/biz/historique/${encodeURIComponent(date)}/${encodeURIComponent(client)}`);
  return <section className={styles.panel}><div className={styles.sectionHeader}><h2>Historique du client {client}</h2><button onClick={close}>Fermer</button></div>
    <ErrorNotice error={resource.error} retry={resource.refresh} />{resource.loading && <p role="status">Chargement de l'historique…</p>}
    {resource.data && <><p>{resource.data.contacts} contact(s) · {resource.data.retours} retour(s) · {resource.data.par_ia} traité(s) par l'IA · {resource.data.par_conseiller} par un conseiller</p>
      <ExportCsv rows={resource.data.appels.flatMap(call => call.evenements.map(event => ({ client, contact: call.identifiant, heure: event.heure, evenement: event.evenement })))} filename={`historique-${client}.csv`} />
      {resource.data.appels.map(call => <details key={call.identifiant} className={styles.decision} open><summary>{call.identifiant} · {call.heure} · {call.resultat}</summary><Journey detail={call} /></details>)}</>}
  </section>;
}

const colors = ["#4361ee", "#159b89", "#f49b31", "#dd5277", "#8b65dc"];
export function SeriesChart({ title, rows, series, percent = false }: { title: string; rows: ChartRow[]; series: string[]; percent?: boolean }) {
  const [hidden, setHidden] = useState<string[]>([]);
  const id = useId();
  const visible = series.filter(key => !hidden.includes(key));
  const values = rows.flatMap(row => visible.map(key => row[key]).filter((value): value is number => typeof value === "number"));
  const max = percent ? 1 : Math.max(1, ...values);
  const x = (index: number) => 55 + index * 600 / Math.max(1, rows.length - 1);
  const y = (value: number) => 195 - value / max * 165;
  return <article className={styles.panel}><h3>{title}</h3>
    <div className={styles.legend}>{series.map((key, index) => <button key={key} aria-pressed={!hidden.includes(key)} onClick={() => setHidden(previous => previous.includes(key) ? previous.filter(value => value !== key) : [...previous, key])}><i style={{ background: colors[index % colors.length] }} />{key}</button>)}</div>
    {!rows.length ? <Empty>Aucune donnée enregistrée.</Empty> : <svg viewBox="0 0 690 230" role="img" aria-labelledby={id} className={styles.chart}><title id={id}>{title}</title>
      {[0, .5, 1].map(fraction => <g key={fraction}><line x1="55" x2="655" y1={y(max * fraction)} y2={y(max * fraction)} stroke="var(--border-light)" /><text x="48" y={y(max * fraction) + 4} textAnchor="end">{number(max * fraction, percent)}</text></g>)}
      {visible.map(key => {
        const segments: string[][] = [[]];
        rows.forEach((row, index) => { if (typeof row[key] === "number") segments[segments.length - 1].push(`${x(index)},${y(row[key] as number)}`); else segments.push([]); });
        const color = colors[series.indexOf(key) % colors.length];
        return <g key={key}>{segments.filter(segment => segment.length).map((segment, index) => <polyline key={index} points={segment.join(" ")} fill="none" stroke={color} strokeWidth="2.5" />)}{rows.map((row, index) => typeof row[key] === "number" ? <circle key={index} cx={x(index)} cy={y(row[key] as number)} r="3" fill={color}><title>{row.heure} · {key} : {number(row[key] as number, percent)}</title></circle> : null)}</g>;
      })}
      {rows.filter((_, index) => index === 0 || index === rows.length - 1 || index % Math.max(1, Math.ceil(rows.length / 6)) === 0).map(row => <text key={row.heure} x={x(rows.indexOf(row))} y="220" textAnchor="middle">{row.heure}</text>)}
    </svg>}
    {!!rows.length && <details><summary>Voir les données</summary><div className={styles.tableScroll}><table><thead><tr><th>Heure</th>{series.map(key => <th key={key}>{key}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={index}><td>{row.heure}</td>{series.map(key => <td key={key}>{typeof row[key] === "number" ? number(row[key] as number, percent) : "—"}</td>)}</tr>)}</tbody></table></div></details>}
  </article>;
}
