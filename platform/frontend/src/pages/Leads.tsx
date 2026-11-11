import { useI18n } from "@/i18n";
import { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "react-router-dom";
import { api, ApiError } from "@/api";
import styles from "./Leads.module.css";

interface Lead {
  id: string;
  source: string;
  platform: string;
  source_content_item_id: string | null;
  author_name: string;
  message_text: string;
  intent_confidence: number;
  status: string;
  created_at: string;
  source_caption: string | null;
  converted: boolean;
  demo?: boolean;
}

const DEMO_LEADS: Lead[] = [
  { id: "demo-1", source: "comment", platform: "facebook", source_content_item_id: null, author_name: "Amira B.", message_text: "Bonsoir, le séjour Djerba est disponible pour deux adultes et deux enfants en novembre ?", intent_confidence: 94, status: "new", created_at: new Date().toISOString(), source_caption: "Djerba en famille", converted: false, demo: true },
  { id: "demo-2", source: "dm", platform: "instagram", source_content_item_id: null, author_name: "Youssef K.", message_text: "Je peux avoir le prix et les dates pour Istanbul au départ de Tunis ?", intent_confidence: 91, status: "new", created_at: new Date(Date.now() - 3600000).toISOString(), source_caption: "Escapade Istanbul", converted: false, demo: true },
  { id: "demo-3", source: "comment", platform: "facebook", source_content_item_id: null, author_name: "Sarra M.", message_text: "Très belles photos, merci pour le partage", intent_confidence: 18, status: "ignored", created_at: new Date(Date.now() - 7200000).toISOString(), source_caption: "Week-end à Hammamet", converted: false, demo: true },
  { id: "demo-4", source: "dm", platform: "instagram", source_content_item_id: null, author_name: "Mehdi A.", message_text: "Je confirme la réservation familiale dont on a parlé hier.", intent_confidence: 98, status: "new", created_at: new Date(Date.now() - 10800000).toISOString(), source_caption: "Vacances en famille", converted: true, demo: true },
];

type Filter = "all" | "new" | "ignored" | "converted";

export function Leads() {
  const { t, locale } = useI18n();
  const { businessId } = useParams();
  const [leads, setLeads] = useState<Lead[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [saleLeadId, setSaleLeadId] = useState<string | null>(null);
  const [saleAmount, setSaleAmount] = useState("");
  const [saleNotes, setSaleNotes] = useState("");
  const saleRequest = useRef<{fingerprint:string;key:string}|null>(null);
  const [mode, setMode] = useState<"live" | "demo">("live");

  const load = useCallback(() => {
    setLoading(true);
    api.get<Lead[]>(`/businesses/${businessId}/leads`)
      .then((rows) => { setLeads(rows); if (!rows.length) setMode("demo"); })
      .finally(() => setLoading(false));
  }, [businessId]);

  useEffect(load, [load]);

  const visibleLeads = mode === "demo" ? DEMO_LEADS : leads;
  const filtered = filter === "all"
    ? visibleLeads
    : filter === "converted"
      ? visibleLeads.filter((l) => l.converted)
      : visibleLeads.filter((l) => l.status === filter && !l.converted);

  async function feedback(leadId: string, isLead: boolean) {
    try {
      await api.post(`/businesses/${businessId}/leads/${leadId}/feedback`, { is_lead: isLead });
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Feedback failed");
      setTimeout(() => setError(""), 4000);
    }
  }

  async function recordSale() {
    if (!saleLeadId) return;
    const fingerprint = JSON.stringify([saleLeadId,saleAmount,saleNotes]);
    if (saleRequest.current?.fingerprint !== fingerprint) saleRequest.current={fingerprint,key:crypto.randomUUID()};
    try {
      await api.post(`/businesses/${businessId}/conversions`, {
        lead_id: saleLeadId,
        value_amount: saleAmount ? Number(saleAmount) : null,
        value_currency: "TND",
        notes: saleNotes || null,
        idempotency_key: saleRequest.current!.key,
      });
      setSaleLeadId(null);
      saleRequest.current=null;
      setSaleAmount("");
      setSaleNotes("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Failed to record sale");
      setTimeout(() => setError(""), 4000);
    }
  }

  const counts = {
    all: visibleLeads.length,
    new: visibleLeads.filter((l) => l.status === "new" && !l.converted).length,
    ignored: visibleLeads.filter((l) => l.status === "ignored" && !l.converted).length,
    converted: visibleLeads.filter((l) => l.converted).length,
  };

  return (
    <div>
      <div className={styles.header}>
        <h2>{t("Leads")}</h2>
        <div className={styles.filters}>
          {(["all", "new", "ignored", "converted"] as Filter[]).map((f) => (
            <button
              key={f}
              className={`btn-ghost ${filter === f ? styles.activeFilter : ""}`}
              onClick={() => setFilter(f)}
            >
              {t(f)} ({counts[f]})
            </button>
          ))}
        </div>
      </div>

      <div className={styles.modeBar}>
        <div><button className={mode === "demo" ? styles.activeMode : ""} onClick={() => setMode("demo")}>{t("Demo pipeline")}</button><button className={mode === "live" ? styles.activeMode : ""} onClick={() => setMode("live")}>{t("Live webhooks")}</button></div>
        <span>{mode === "demo" ? t("Simulated examples — excluded from reports and learning") : t(`${leads.length} real leads received`)}</span>
      </div>

      {error && <p className={styles.error}>{t(error)}</p>}

      {loading ? <p>{t("Loading…")}</p> : filtered.length === 0 ? (
        <p className={styles.muted}>{t("No leads")}{filter !== "all" ? t(` with filter "${t(filter)}"`) : ""}.</p>
      ) : (
        <div className={styles.list}>
          {filtered.map((lead) => (
            <div key={lead.id} className={`card ${styles.card}`}>
              <div className={styles.cardTop}>
                <strong className={styles.author}>{lead.author_name || t("Anonymous")}</strong>
                <span className={styles.meta}>{t(lead.platform)} · {t(lead.source)}</span>
                <span className={styles.confidence} title={t("AI confidence")}>
                  {lead.intent_confidence}%
                </span>
                <span className={styles.time}>
                  {new Date(lead.created_at).toLocaleDateString(locale, {
                    day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
                  })}
                </span>
              </div>

              <p className={styles.message}>{lead.message_text}</p>

              {lead.source_caption && (
                <p className={styles.sourceCaption}>
                  <span className={styles.sourceLabel}>{t("On post:")}</span> {lead.source_caption}
                </p>
              )}

              <div className={styles.actions}>
                {!lead.converted && !lead.demo && (
                  <>
                    {lead.status !== "new" && (
                      <button
                        className={`btn-ghost ${styles.leadBtn}`}
                        onClick={() => feedback(lead.id, true)}
                      >
                        {t("✅ Real lead")}</button>
                    )}
                    {lead.status !== "ignored" && (
                      <button
                        className={`btn-ghost ${styles.ignoreBtn}`}
                        onClick={() => feedback(lead.id, false)}
                      >
                        {t("❌ Not a lead")}</button>
                    )}
                    {lead.status === "new" && (
                      <button
                        className="btn-primary"
                        onClick={() => setSaleLeadId(lead.id)}
                      >
                        {t("💰 Record sale")}</button>
                    )}
                  </>
                )}
                {lead.converted && (
                  <span className={`badge badge-published`}>{t("Converted")}</span>
                )}
                {lead.demo && <span className={styles.demoBadge}>{t("SIMULATION")}</span>}
              </div>

              {/* Inline sale form */}
              {saleLeadId === lead.id && (
                <div className={styles.saleForm}>
                  <input
                    type="number"
                    placeholder={t("Amount (TND)")}
                    value={saleAmount}
                    onChange={(e) => setSaleAmount(e.target.value)}
                    style={{ width: 120 }}
                  />
                  <input
                    placeholder={t("Notes (optional)")}
                    value={saleNotes}
                    onChange={(e) => setSaleNotes(e.target.value)}
                    style={{ flex: 1 }}
                  />
                  <button className="btn-primary" onClick={recordSale}>{t("Save")}</button>
                  <button className="btn-ghost" onClick={() => setSaleLeadId(null)}>{t("Cancel")}</button>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
