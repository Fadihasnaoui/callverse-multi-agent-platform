import { useI18n } from "@/i18n";
import { useEffect, useState, useRef } from "react";
import { useParams } from "react-router-dom";
import { api, ApiError } from "@/api";
import styles from "./Audit.module.css";

interface AuditResult {
  audit_id: string;
  posts_analyzed: number;
  report_text?: string;
  created_at: string;
  platforms_audited: string[];
  cadence: Record<string, number>;
  format_mix: Record<string, { count: number; avg_engagement: number; pct_of_total: number }>;
  category_performance: Record<string, { count: number; avg_engagement: number; pct_of_total: number }>;
  strategy_mix: Record<string, { count: number; pct: number }>;
  health_score: { consistency: number; format_diversity: number; strategy_balance: number; overall: number };
  time_distribution: { by_hour: Record<string, number>; by_day: Record<string, number>; peak_hour: number; peak_day: string };
  best_and_worst: { best: { message: string; media_type: string; engagement_score: number; platform: string }[]; worst: { message: string; media_type: string; engagement_score: number; platform: string }[] };
  platform_breakdown: Record<string, { posts: number; cadence: Record<string, number>; format_mix: Record<string, { count: number; avg_engagement: number }>; page_metrics?: Record<string, number | string | null>; profile?: Record<string, number | string | null> }>;
}

export function Audit() {
  const { t, locale, lang } = useI18n();
  const { businessId } = useParams();
  const [latest, setLatest] = useState<AuditResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const reportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.get<AuditResult>(`/businesses/${businessId}/audit/latest`)
      .then(setLatest)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [businessId]);

  async function runAudit() {
    setRunning(true);
    setError("");
    try {
      const result = await api.post<AuditResult>(`/businesses/${businessId}/audit`);
      setLatest(result);
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Audit failed");
    } finally {
      setRunning(false);
    }
  }

  function downloadPdf() {
    if (!reportRef.current) return;
    const printWindow = window.open("", "_blank");
    if (!printWindow) return;
    printWindow.document.write(`<!DOCTYPE html><html lang="${lang}" dir="${lang === "ar" ? "rtl" : "ltr"}"><head><meta charset="UTF-8"><title>Ozone Voyages - ${t("Social Media Health Report")}</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #172033; padding: 2rem; max-width: 900px; margin: 0 auto; }
  .brand { display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 3px solid #6c5ce7; }
  .brand-mark { width: 48px; height: 48px; display: grid; place-items: center; border-radius: 12px; background: linear-gradient(135deg, #7c6cf2, #46c6b2); color: #fff; font-weight: 900; font-size: 1.5rem; }
  .brand h1 { font-size: 1.5rem; } .brand small { color: #6c5ce7; font-size: .75rem; font-weight: 700; letter-spacing: .1em; }
  h2 { font-size: 1.1rem; margin: 1.5rem 0 .75rem; color: #6c5ce7; border-bottom: 1px solid #eee; padding-bottom: .4rem; }
  .score-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin: .75rem 0; }
  .score-box { text-align: center; padding: .75rem; border: 1px solid #eee; border-radius: 10px; }
  .score-box strong { display: block; font-size: 1.8rem; } .score-box span { font-size: .7rem; color: #666; }
  .overall { border-color: #6c5ce7; background: #f8f7ff; }
  table { width: 100%; border-collapse: collapse; margin: .75rem 0; font-size: .8rem; }
  th, td { padding: .5rem .6rem; text-align: start; border-bottom: 1px solid #eee; }
  th { color: #666; font-size: .7rem; text-transform: uppercase; letter-spacing: .05em; }
  .footer { margin-top: 2rem; padding-top: 1rem; border-top: 2px solid #eee; color: #888; font-size: .7rem; text-align: center; }
  @media print { body { padding: 0; } }
</style></head><body>`);
    printWindow.document.write(`<div class="brand"><div class="brand-mark">O</div><div><h1>Ozone Voyages</h1><small>${t("DIGITAL MARKETING AUDIT REPORT")}</small></div></div>`);
    printWindow.document.write(reportRef.current.innerHTML);
    const date = latest ? new Date(latest.created_at).toLocaleDateString(locale, { day: "numeric", month: "long", year: "numeric" }) : "";
    printWindow.document.write(`<div class="footer">${t("Generated")} ${date} &middot; Ozone Voyages &middot; ${t("Powered by AI Marketing Agent")}</div>`);
    printWindow.document.write(`</body></html>`);
    printWindow.document.close();
    setTimeout(() => { printWindow.print(); }, 400);
  }

  if (loading) return <p>{t("Loading...")}</p>;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <p className={styles.eyebrow}>{t("DIGITAL MARKETING AUDIT")}</p>
          <h1>{t("Social Media Health Report")}</h1>
          <p className={styles.subtitle}>{t("CMO-level analysis across all connected platforms")}</p>
        </div>
        <div className={styles.headerActions}>
          {latest && <button className={styles.pdfBtn} onClick={downloadPdf}>{t("Download PDF")}</button>}
          <button className="btn-primary" onClick={runAudit} disabled={running}>
            {running ? t("Running audit...") : t("Run new audit")}
          </button>
        </div>
      </div>

      {error && <p className={styles.error}>{t(error)}</p>}

      {!latest ? (
        <div className={styles.emptyState}>
          <span className={styles.emptyIcon}>🔍</span>
          <p>{t("No audit has been run yet.")}</p>
          <p>{t("Click \"Run new audit\" to analyze your social media presence across Facebook and Instagram.")}</p>
        </div>
      ) : (
        <div ref={reportRef}>
          <div className={styles.metaBar}>
            <span>{latest.posts_analyzed} {t(" posts analyzed")}</span>
            <span>{(latest.platforms_audited || []).map(p => p.charAt(0).toUpperCase() + p.slice(1)).join(" + ")}</span>
            <span>{new Date(latest.created_at).toLocaleDateString(locale, { day: "numeric", month: "long", year: "numeric" })}</span>
          </div>

          {latest.health_score && (
            <section className={styles.healthSection}>
              <h2>{t("Content Health Score")}</h2>
              <div className={styles.scoreGrid}>
                <div className={`${styles.scoreBox} ${styles.overall}`}>
                  <strong>{latest.health_score.overall}</strong>
                  <span>{t("/100 overall")}</span>
                  <div className={styles.scoreBar}><div style={{ width: `${latest.health_score.overall}%`, background: scoreColor(latest.health_score.overall) }} /></div>
                </div>
                <ScoreCard label={t("Consistency")} value={latest.health_score.consistency} />
                <ScoreCard label={t("Format diversity")} value={latest.health_score.format_diversity} />
                <ScoreCard label={t("Strategy balance")} value={latest.health_score.strategy_balance} />
              </div>
            </section>
          )}

          {latest.platform_breakdown && Object.keys(latest.platform_breakdown).length > 0 && (
            <section>
              <h2>{t("Platform Breakdown")}</h2>
              <div className={styles.platformGrid}>
                {Object.entries(latest.platform_breakdown).map(([platform, data]) => (
                  <article key={platform} className={styles.platformCard}>
                    <div className={styles.platformHeader}>
                      <strong>{t(platform)}</strong>
                      <span>{data.posts} {t(" posts")}</span>
                    </div>
                    <div className={styles.platformMeta}>
                      <div><span>{t("Posts/week")}</span><strong>{data.cadence?.posts_per_week?.toFixed(1) ?? "—"}</strong></div>
                      <div><span>{t("Avg gap")}</span><strong>{data.cadence?.avg_gap_days?.toFixed(1) ?? "—"}d</strong></div>
                      <div><span>{t("Longest gap")}</span><strong>{data.cadence?.longest_gap_days?.toFixed(0) ?? "—"}d</strong></div>
                      {data.page_metrics?.followers_count != null && <div><span>{t("Followers")}</span><strong>{Number(data.page_metrics.followers_count).toLocaleString()}</strong></div>}
                      {data.profile?.followers_count != null && <div><span>{t("Followers")}</span><strong>{Number(data.profile.followers_count).toLocaleString()}</strong></div>}
                    </div>
                    {data.format_mix && (
                      <div className={styles.miniFormats}>
                        {Object.entries(data.format_mix).map(([fmt, d]) => (
                          <div key={t(fmt)}><span>{t(fmt)}</span><strong>{d.count}</strong><small>{d.avg_engagement?.toFixed(1)} {t(" avg eng")}</small></div>
                        ))}
                      </div>
                    )}
                  </article>
                ))}
              </div>
            </section>
          )}

          {latest.category_performance && Object.keys(latest.category_performance).length > 0 && (
            <section>
              <h2>{t("Category Performance")}</h2>
              <table className={styles.dataTable}>
                <thead><tr><th>{t("Category")}</th><th>{t("Posts")}</th><th>{t("% of total")}</th><th>{t("Avg engagement")}</th></tr></thead>
                <tbody>
                  {Object.entries(latest.category_performance)
                    .sort(([, a], [, b]) => b.count - a.count)
                    .map(([cat, d]) => (
                      <tr key={cat}>
                        <td className={styles.catName}>{t(cat)}</td>
                        <td>{d.count}</td>
                        <td>{d.pct_of_total?.toFixed(1)}%</td>
                        <td><strong>{d.avg_engagement?.toFixed(1)}</strong></td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </section>
          )}

          {latest.strategy_mix && Object.keys(latest.strategy_mix).length > 0 && (
            <section>
              <h2>{t("Strategy Layer Mix")}</h2>
              <div className={styles.strategyRow}>
                {Object.entries(latest.strategy_mix).map(([layer, d]) => (
                  <div key={layer} className={styles.strategyTile}>
                    <strong>{d.pct}%</strong>
                    <span>{t(layer)}</span>
                    <small>{d.count} {t(" posts")}</small>
                  </div>
                ))}
              </div>
              <p className={styles.targetNote}>{t("Target: 20% pillar / 50% distribution / 30% engagement")}</p>
            </section>
          )}

          {latest.format_mix && Object.keys(latest.format_mix).length > 0 && (
            <section>
              <h2>{t("Format Performance")}</h2>
              <table className={styles.dataTable}>
                <thead><tr><th>{t("Format")}</th><th>{t("Posts")}</th><th>{t("% of total")}</th><th>{t("Avg engagement")}</th></tr></thead>
                <tbody>
                  {Object.entries(latest.format_mix)
                    .sort(([, a], [, b]) => b.avg_engagement - a.avg_engagement)
                    .map(([fmt, d]) => (
                      <tr key={t(fmt)}>
                        <td className={styles.catName}>{t(fmt)}</td>
                        <td>{d.count}</td>
                        <td>{d.pct_of_total?.toFixed(1)}%</td>
                        <td><strong>{d.avg_engagement?.toFixed(1)}</strong></td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </section>
          )}

          {latest.time_distribution && (
            <section>
              <h2>{t("Posting Patterns")}</h2>
              <div className={styles.timeGrid}>
                <div className={styles.timeTile}><span>{t("Peak hour")}</span><strong>{latest.time_distribution.peak_hour}:00</strong></div>
                <div className={styles.timeTile}><span>{t("Peak day")}</span><strong>{t(latest.time_distribution.peak_day)}</strong></div>
                {latest.time_distribution.by_day && Object.entries(latest.time_distribution.by_day)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 3)
                  .map(([day, count]) => (
                    <div key={day} className={styles.timeTile}><span>{t(day)}</span><strong>{count} {t(" posts")}</strong></div>
                  ))}
              </div>
            </section>
          )}

          {latest.best_and_worst?.best && (
            <section>
              <h2>{t("Top Performing Posts")}</h2>
              <div className={styles.postList}>
                {latest.best_and_worst.best.slice(0, 3).map((post, i) => (
                  <div key={i} className={styles.postItem}>
                    <div className={styles.postRank}>{i + 1}</div>
                    <div className={styles.postContent}>
                      <div className={styles.postMeta}>
                        <span className={styles[post.platform]}>{t(post.platform)}</span>
                        <span>{post.media_type}</span>
                        <strong className={styles.engScore}>{post.engagement_score} {t(" eng")}</strong>
                      </div>
                      <p>{(post.message || "").slice(0, 150)}{(post.message || "").length > 150 ? "..." : ""}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {latest.cadence && (
            <section>
              <h2>{t("Posting Cadence")}</h2>
              <div className={styles.cadenceGrid}>
                <div className={styles.cadenceTile}><strong>{latest.cadence.posts_per_week?.toFixed(1)}</strong><span>{t("posts/week")}</span></div>
                <div className={styles.cadenceTile}><strong>{latest.cadence.avg_gap_days?.toFixed(1)}</strong><span>{t("avg gap (days)")}</span></div>
                <div className={`${styles.cadenceTile} ${(latest.cadence.longest_gap_days ?? 0) > 14 ? styles.danger : ""}`}>
                  <strong>{latest.cadence.longest_gap_days?.toFixed(0)}</strong><span>{t("longest gap (days)")}</span>
                </div>
                <div className={styles.cadenceTile}><strong>{latest.cadence.posts_count}</strong><span>{t("total posts")}</span></div>
              </div>
            </section>
          )}

          {latest.report_text && (
            <section>
              <h2>{t("Full Analysis")}</h2>
              <pre className={styles.reportText}>{latest.report_text}</pre>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

function ScoreCard({ label, value }: { label: string; value: number }) {
  const { t } = useI18n();
  return (
    <div className={styles.scoreBox}>
      <strong>{value}</strong>
      <span>{t(label)}</span>
      <div className={styles.scoreBar}><div style={{ width: `${value}%`, background: scoreColor(value) }} /></div>
    </div>
  );
}

function scoreColor(v: number): string {
  if (v >= 70) return "#159f77";
  if (v >= 40) return "#d89113";
  return "#e76f51";
}
