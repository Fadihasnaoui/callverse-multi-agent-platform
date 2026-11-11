import { useI18n } from "@/i18n";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, ApiError } from "@/api";
import { WorkspaceHeader } from "@/pages/AgentHub";
import type { Business } from "@/types";
import styles from "./Customers.module.css";

export function Customers() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const [businesses, setBusinesses] = useState<Business[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [industry, setIndustry] = useState("");
  const [error, setError] = useState("");
  const bootstrapping = useRef(false);

  function load() {
    setLoading(true);
    api.get<Business[]>("/businesses")
      .then(async (items) => {
        setBusinesses(items);
        const ozone = items.find((business) => business.name.toLowerCase().includes("ozone"));
        if (ozone) { navigate(`/c/${ozone.id}`, { replace: true }); return; }
        if (!bootstrapping.current) {
          bootstrapping.current = true;
          const created = await api.post<Business>("/businesses", {
            name: "Ozone Voyages", industry: "Travel agency",
            brand_voice: "Warm, aspirational and trustworthy. Speak to Tunisian families looking for memorable travel experiences.",
          });
          navigate(`/c/${created.id}`, { replace: true });
        }
      })
      .catch((err) => setError(err instanceof ApiError ? err.body : "Could not prepare the Ozone workspace"))
      .finally(() => setLoading(false));
  }

  useEffect(load, []);

  async function create() {
    if (!name.trim()) return;
    setError("");
    try {
      const biz = await api.post<Business>("/businesses", { name: name.trim(), industry: industry.trim() });
      setCreating(false); setName(""); setIndustry(""); navigate(`/c/${biz.id}/settings`);
    } catch (err) { setError(err instanceof ApiError ? err.body : "Create failed"); }
  }

  if (loading) return <><WorkspaceHeader /><div className={styles.page}><p>{t("Loading…")}</p></div></>;

  return <><WorkspaceHeader /><div className={styles.page}>
    <div className={styles.header}><h1>{t("Customers")}</h1><div className={styles.headerActions}>
      <button className="btn-primary" onClick={() => setCreating(!creating)}>{creating ? t("Cancel") : t("+ New customer")}</button>
      <Link to="/ai-stats" className="btn-ghost">{t("AI Stats")}</Link>
    </div></div>
    {creating && <div className={`card ${styles.createForm}`}>
      <input placeholder={t("Business name")} value={name} onChange={(e) => setName(e.target.value)} autoFocus />
      <input placeholder={t("Industry (optional)")} value={industry} onChange={(e) => setIndustry(e.target.value)} />
      <button className="btn-primary" onClick={create} disabled={!name.trim()}>{t("Create")}</button>
      {error && <p className={styles.error}>{t(error)}</p>}
    </div>}
    {businesses.length === 0 && !creating ? <p className={styles.empty}>{t('No customers yet. Click "+ New customer" to add one.')}</p> :
      <div className={styles.grid}>{businesses.map((b) => <Link key={b.id} to={`/c/${b.id}`} className={`card ${styles.card}`}>
        <div className={styles.cardHeader}><span className={styles.name}>{b.name}</span><span className={`badge badge-${b.status === "active" ? "published" : "pending"}`}>{t(b.status)}</span></div>
        <p className={styles.industry}>{b.industry || "—"}</p>
        {b.connected.length > 0 && <div className={styles.platforms}>{b.connected.map((p) => <span key={p} className={styles.platform}>{p}</span>)}</div>}
      </Link>)}</div>}
  </div></>;
}
