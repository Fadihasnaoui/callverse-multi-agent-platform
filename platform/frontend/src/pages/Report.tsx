import { useI18n } from "@/i18n";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { api, ApiError } from "@/api";
import styles from "./Report.module.css";

interface Strategist {
  platform: string;
  suggested_category: string;
  suggested_strategy_layer: string;
  suggested_format: string;
  suggested_funnel_stage: string;
  reasoning: string;
  platform_playbook: Record<string, string>;
}

interface AuditDetail {
  health_score: { consistency: number; format_diversity: number; strategy_balance: number; overall: number };
  strategy_mix: Record<string, { count: number; pct: number }>;
  format_mix: Record<string, { count: number; avg_engagement: number; pct_of_total: number }>;
  category_performance: Record<string, { count: number; avg_engagement: number }>;
  best_and_worst: { best: { message: string; media_type: string; engagement_score: number; platform: string }[] };
  time_distribution: { peak_hour: number; peak_day: string };
  platform_breakdown: Record<string, { posts: number; cadence: Record<string, number> }>;
}

interface SentimentContext {
  overall_sentiment: number;
  avg_buying_intent: number;
  high_intent_comments: number;
  top_topics: Record<string, number>;
  by_category: Record<string, { avg_sentiment: number; avg_intent: number; comments: number }>;
  total_comments: number;
}

interface SelfBenchmark {
  period_days: number;
  current: { start: string; end: string; posts: number; reels: number; statics: number; reel_pct: number; avg_engagement: number; total_likes: number; total_comments: number };
  previous: { start: string; end: string; posts: number; reels: number; statics: number; reel_pct: number; avg_engagement: number; total_likes: number; total_comments: number };
  changes?: { posts_delta_pct: number | null; engagement_delta_pct: number | null; likes_delta_pct: number | null; comments_delta_pct: number | null; reel_pct_shift: number };
  insights?: string[];
}

interface ReportData {
  period_days: number;
  content: { published: number; awaiting_approval: number; failed: number };
  leads: { total: number; traced_to_a_post: number };
  sales: { count: number; revenue: number };
  performance: { reach: number; impressions: number; engagement: number };
  top_content: { id: string; type: string; platform: string; caption_start: string; leads: number; sales: number; revenue: number }[];
  latest_insight: { summary: string; recommendations: string[]; as_of: string } | null;
  audit: { posts_analyzed: number; cadence: Record<string, number>; engagement_available: boolean; as_of: string; recommendations: string[] } | null;
  learning: { stage: string; real_published_posts: number; confidence_target: number; progress_percent: number; data_quality: string };
  baseline: { label: string; numbers: Record<string, number>; weekly: { label: string; facebook: number; instagram: number }[]; learning: { summary: string; recommendations: string[] } };
  strategist: Strategist | null;
  strategic_brief: string;
  audit_detail: AuditDetail | null;
  sentiment: SentimentContext | null;
  industry_benchmarks: Record<string, unknown> | null;
  self_benchmark: SelfBenchmark | null;
}

export function Report() {
  const { t, locale, lang } = useI18n();
  const { businessId } = useParams();
  const [data, setData] = useState<ReportData | null>(null);
  const [days, setDays] = useState(30);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState("");
  const reportRef = useRef<HTMLDivElement>(null);

  const load = () => api.get<ReportData>(`/businesses/${businessId}/report?days=${days}`).then(setData);
  useEffect(() => { load(); }, [businessId, days]);

  async function runLearning() {
    setRunning(true); setMessage("");
    try { await api.post(`/businesses/${businessId}/learning/run`); await load(); setMessage("Learning analysis refreshed"); }
    catch (error) { setMessage(error instanceof ApiError ? error.body : "Learning failed"); }
    finally { setRunning(false); }
  }

  function downloadPdf() {
    if (!reportRef.current) return;
    const w = window.open("", "_blank");
    if (!w) return;
    w.document.write(`<!DOCTYPE html><html lang="${lang}" dir="${lang === "ar" ? "rtl" : "ltr"}"><head><meta charset="UTF-8"><title>Ozone Voyages - ${t("Performance & Strategy Report")}</title>
<style>
  * { box-sizing: border-box; margin: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #172033; padding: 2rem; max-width: 900px; margin: 0 auto; }
  .brand { display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 3px solid #6c5ce7; }
  .brand-mark { width: 48px; height: 48px; display: grid; place-items: center; border-radius: 12px; background: linear-gradient(135deg, #7c6cf2, #46c6b2); color: #fff; font-weight: 900; font-size: 1.5rem; }
  .brand h1 { font-size: 1.5rem; } .brand small { color: #6c5ce7; font-size: .75rem; font-weight: 700; letter-spacing: .1em; }
  h2 { font-size: 1.1rem; margin: 1.5rem 0 .75rem; color: #6c5ce7; border-bottom: 1px solid #eee; padding-bottom: .4rem; }
  .kpis { display: grid; grid-template-columns: repeat(5,1fr); gap: .5rem; margin: .75rem 0; }
  .kpi { text-align: center; padding: .6rem; border: 1px solid #eee; border-radius: 8px; }
  .kpi strong { display: block; font-size: 1.4rem; } .kpi span { font-size: .65rem; color: #666; }
  table { width: 100%; border-collapse: collapse; margin: .75rem 0; font-size: .8rem; }
  th, td { padding: .4rem .5rem; text-align: start; border-bottom: 1px solid #eee; }
  th { color: #666; font-size: .7rem; text-transform: uppercase; }
  .brief { white-space: pre-wrap; font-size: .78rem; line-height: 1.6; padding: .8rem; background: #f8f7ff; border-radius: 8px; border-left: 3px solid #6c5ce7; }
  .footer { margin-top: 2rem; padding-top: 1rem; border-top: 2px solid #eee; color: #888; font-size: .7rem; text-align: center; }
  @media print { body { padding: 0; } }
</style></head><body>`);
    w.document.write(`<div class="brand"><div class="brand-mark">O</div><div><h1>Ozone Voyages</h1><small>${t("STRATEGIC PERFORMANCE REPORT")} &middot; ${t(`${days}-DAY PERIOD`)}</small></div></div>`);
    w.document.write(reportRef.current.innerHTML);
    w.document.write(`<div class="footer">${t("Generated")} ${new Date().toLocaleDateString(locale, { day: "numeric", month: "long", year: "numeric" })} &middot; Ozone Voyages &middot; ${t("AI Marketing Agent")}</div>`);
    w.document.write(`</body></html>`);
    w.document.close();
    setTimeout(() => w.print(), 400);
  }

  if (!data) return <p>{t("Loading report...")}</p>;

  const health = data.audit_detail?.health_score;
  const strat = data.strategist;
  const brief = data.strategic_brief;

  return <div className={styles.page}>
    <header className={styles.header}>
      <div>
        <p className={styles.eyebrow}>{t("STRATEGIC INTELLIGENCE")}</p>
        <h1>{t("Performance & Strategy Report")}</h1>
        <p className={styles.subtitle}>{t("Data-driven marketing intelligence for Ozone Voyages")}</p>
      </div>
      <div className={styles.controls}>
        <select value={days} onChange={(e) => setDays(Number(e.target.value))}><option value={30}>{t("30 days")}</option><option value={90}>{t("90 days")}</option></select>
        <button className={styles.pdfBtn} onClick={downloadPdf}>{t("Download PDF")}</button>
        <button className="btn-primary" onClick={runLearning} disabled={running}>{running ? t("Analysing...") : t("Run learning")}</button>
      </div>
    </header>
    {message && <p className={styles.message}>{t(message)}</p>}

    <div ref={reportRef}>
      {/* Health score banner */}
      {health && (
        <section className={styles.healthBanner}>
          <div className={styles.healthMain}>
            <div className={styles.healthRing} data-score={health.overall}>
              <strong>{health.overall}</strong><span>/100</span>
            </div>
            <div>
              <h2 className={styles.healthTitle}>{t("Content Health")}</h2>
              <p className={styles.healthVerdict}>
                {health.overall >= 70 ? t("Strong foundation — optimize and scale.") :
                 health.overall >= 40 ? t("Needs improvement — address the gaps below.") :
                 t("Critical gaps — strategy overhaul required.")}
              </p>
            </div>
          </div>
          <div className={styles.healthBreakdown}>
            <HealthBar label={t("Consistency")} value={health.consistency} />
            <HealthBar label={t("Format diversity")} value={health.format_diversity} />
            <HealthBar label={t("Strategy balance")} value={health.strategy_balance} />
          </div>
        </section>
      )}

      {/* KPIs */}
      <section className={styles.kpis}>
        <KpiTile label={t("Published")} value={String(data.content.published)} sub={`${data.content.awaiting_approval} pending`} />
        <KpiTile label={t("Reach")} value={fmt(data.performance.reach)} sub={data.performance.reach > 0 ? "measured" : "no post metrics yet"} />
        <KpiTile label={t("Engagement")} value={fmt(data.performance.engagement)} sub={data.performance.engagement > 0 ? "interactions" : "no post metrics yet"} />
        <KpiTile label={t("Leads")} value={String(data.leads.total)} sub={`${data.leads.traced_to_a_post} traced to post`} />
        <KpiTile label={t("Revenue")} value={`${data.sales.revenue.toLocaleString()} TND`} sub={`${data.sales.count} sales`} />
      </section>

      {/* Strategist recommendation */}
      {strat && (
        <section className={styles.stratSection}>
          <h2>{t("AI Strategist Recommendation")}</h2>
          <div className={styles.stratCard}>
            <p className={styles.stratReasoning}>{strat.reasoning}</p>
            <div className={styles.stratTiles}>
              <div><span>{t("Platform")}</span><strong>{t(strat.platform)}</strong></div>
              <div><span>{t("Category")}</span><strong>{t(strat.suggested_category)}</strong></div>
              <div><span>{t("Strategy layer")}</span><strong>{t(strat.suggested_strategy_layer)}</strong></div>
              <div><span>{t("Format")}</span><strong>{t(strat.suggested_format)}</strong></div>
              <div><span>{t("Funnel stage")}</span><strong>{t(strat.suggested_funnel_stage)}</strong></div>
            </div>
            {strat.platform_playbook && (
              <div className={styles.playbook}>
                <strong>{t("Platform playbook:")}</strong>
                {Object.entries(strat.platform_playbook).map(([k, v]) => (
                  <p key={k}><span>{k.replace(/_/g, " ")}:</span> {v}</p>
                ))}
              </div>
            )}
          </div>
        </section>
      )}

      {/* Strategic brief */}
      {brief && (
        <section>
          <h2>{t("Strategic Brief")}</h2>
          <pre className={styles.briefText}>{brief}</pre>
        </section>
      )}

      {/* Strategy gap */}
      {data.audit_detail?.strategy_mix && Object.keys(data.audit_detail.strategy_mix).length > 0 && (
        <section>
          <h2>{t("Strategy Layer Gap Analysis")}</h2>
          <div className={styles.gapGrid}>
            {Object.entries(data.audit_detail.strategy_mix).map(([layer, d]) => {
              const target = layer === "pillar" ? 20 : layer === "distribution" ? 50 : 30;
              const diff = d.pct - target;
              return (
                <div key={layer} className={styles.gapTile}>
                  <div className={styles.gapHeader}>
                    <strong>{t(layer)}</strong>
                    <span className={diff > 5 ? styles.excess : diff < -5 ? styles.deficit : styles.onTrack}>
                      {d.pct}% {diff > 5 ? t("(excess)") : diff < -5 ? t("(needs more)") : t("(on target)")}
                    </span>
                  </div>
                  <div className={styles.gapBar}>
                    <div className={styles.gapFill} style={{ width: `${Math.min(d.pct, 100)}%` }} />
                    <div className={styles.gapTarget} style={{ left: `${target}%` }} />
                  </div>
                  <p>{d.count} {t(" posts &middot; target ")}{target}%</p>
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Top content */}
      {data.top_content && data.top_content.length > 0 && (
        <section>
          <h2>{t("Top Performing Content")}</h2>
          <table className={styles.dataTable}>
            <thead><tr><th>{t("Platform")}</th><th>{t("Type")}</th><th>{t("Caption")}</th><th>{t("Leads")}</th><th>{t("Sales")}</th><th>{t("Revenue")}</th></tr></thead>
            <tbody>
              {data.top_content.map((c) => (
                <tr key={c.id}>
                  <td className={styles.platform}>{t(c.platform)}</td>
                  <td>{t(c.type)}</td>
                  <td className={styles.captionCell}>{c.caption_start}</td>
                  <td><strong>{c.leads}</strong></td>
                  <td><strong>{c.sales}</strong></td>
                  <td>{c.revenue.toLocaleString()} {t(" TND")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {/* Comment sentiment */}
      {data.sentiment && data.sentiment.total_comments > 0 && (
        <section>
          <h2>{t("Comment Intelligence")}</h2>
          <div className={styles.sentimentGrid}>
            <div className={styles.sentimentTile}>
              <span>{t("Sentiment")}</span>
              <strong className={data.sentiment.overall_sentiment >= 0 ? styles.sentPositive : styles.sentNegative}>
                {data.sentiment.overall_sentiment > 0 ? "+" : ""}{data.sentiment.overall_sentiment}%
              </strong>
            </div>
            <div className={styles.sentimentTile}>
              <span>{t("Buying intent")}</span>
              <strong>{data.sentiment.avg_buying_intent}/100</strong>
            </div>
            <div className={styles.sentimentTile}>
              <span>{t("High-intent")}</span>
              <strong>{data.sentiment.high_intent_comments}</strong>
            </div>
            <div className={styles.sentimentTile}>
              <span>{t("Comments analyzed")}</span>
              <strong>{data.sentiment.total_comments}</strong>
            </div>
          </div>
          {Object.keys(data.sentiment.top_topics).length > 0 && (
            <div className={styles.topicRow}>
              <strong>{t("Top topics:")}</strong>
              {Object.entries(data.sentiment.top_topics).slice(0, 8).map(([topic, count]) => (
                <span key={topic} className={styles.topicChip}>{topic} ({count})</span>
              ))}
            </div>
          )}
          {Object.keys(data.sentiment.by_category || {}).length > 0 && (
            <table className={styles.dataTable}>
              <thead><tr><th>{t("Category")}</th><th>{t("Sentiment")}</th><th>{t("Intent")}</th><th>{t("Comments")}</th></tr></thead>
              <tbody>
                {Object.entries(data.sentiment.by_category).slice(0, 6).map(([cat, d]) => (
                  <tr key={cat}>
                    <td className={styles.platform}>{t(cat)}</td>
                    <td className={d.avg_sentiment >= 0 ? styles.sentPositive : styles.sentNegative}>
                      {d.avg_sentiment > 0 ? "+" : ""}{d.avg_sentiment}%
                    </td>
                    <td>{d.avg_intent}/100</td>
                    <td>{d.comments}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      )}

      {/* Self-benchmark */}
      {data.self_benchmark && data.self_benchmark.changes && (
        <section>
          <h2>{t("Period-over-Period Benchmark")}</h2>
          <p className={styles.benchPeriodLabel}>
            {t("Current:")}{data.self_benchmark.current.start} {t(" to ")}{data.self_benchmark.current.end} {t("vs Previous:")}{data.self_benchmark.previous.start} {t(" to ")}{data.self_benchmark.previous.end}
          </p>
          <div className={styles.benchCompare}>
            {([
              ["Posts", data.self_benchmark.current.posts, data.self_benchmark.previous.posts, data.self_benchmark.changes.posts_delta_pct, "%"],
              ["Avg engagement", data.self_benchmark.current.avg_engagement, data.self_benchmark.previous.avg_engagement, data.self_benchmark.changes.engagement_delta_pct, "%"],
              ["Likes", data.self_benchmark.current.total_likes, data.self_benchmark.previous.total_likes, data.self_benchmark.changes.likes_delta_pct, "%"],
              ["Comments", data.self_benchmark.current.total_comments, data.self_benchmark.previous.total_comments, data.self_benchmark.changes.comments_delta_pct, "%"],
              ["Reel %", data.self_benchmark.current.reel_pct, data.self_benchmark.previous.reel_pct, data.self_benchmark.changes.reel_pct_shift, "pp"],
            ] as [string, number, number, number | null, string][]).map(([label, curr, prev, delta, suffix]) => (
              <div key={label} className={styles.benchCompareTile}>
                <span>{t(label)}</span>
                <strong>{curr}</strong>
                <small>{t("was ")}{prev}</small>
                {delta !== null && (
                  <span className={delta > 0 ? styles.sentPositive : delta < 0 ? styles.sentNegative : ""}>
                    {delta > 0 ? "+" : ""}{delta}{t(suffix)}
                  </span>
                )}
              </div>
            ))}
          </div>
          {data.self_benchmark.insights && data.self_benchmark.insights.length > 0 && (
            <div className={styles.benchInsightBox}>
              {data.self_benchmark.insights.map((ins) => (
                <p key={ins}>{ins}</p>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Learning readiness */}
      <section className={styles.learningSection}>
        <h2>{t("Learning Engine Status")}</h2>
        <div className={styles.learningCard}>
          <div className={styles.learningHeader}>
            <div>
              <strong>{data.learning.stage === "baseline" ? t("Building baseline") : t("Descriptive signals available")}</strong>
              <p>{data.learning.real_published_posts} {t(" of ")}{data.learning.confidence_target} {t(" published posts needed")}</p>
            </div>
            <span className={styles.learningPct}>{data.learning.progress_percent}%</span>
          </div>
          <div className={styles.progressBar}><div style={{ width: `${data.learning.progress_percent}%` }} /></div>
          <p className={styles.learningNote}>
            {data.learning.stage === "baseline"
              ? t("Recommendations are hypotheses until enough posts have real engagement, lead and sales data.")
              : t("Published count is not proof of improvement. Compare measured outcomes on later posts.")}
          </p>
        </div>
      </section>

      {/* Recommendations */}
      <section>
        <h2>{t("Next Actions")}</h2>
        <div className={styles.recGrid}>
          {(data.latest_insight?.recommendations || data.audit?.recommendations || []).map((r, i) => (
            <div key={r} className={styles.recItem}>
              <span className={styles.recNum}>0{i + 1}</span>
              <p>{r}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Insight */}
      {data.latest_insight?.summary && (
        <section className={styles.insightBox}>
          <span className={styles.insightLabel}>{t("AGENT INTERPRETATION")}</span>
          <p>{data.latest_insight.summary}</p>
        </section>
      )}
    </div>
  </div>;
}

function KpiTile({ label, value, sub }: { label: string; value: string; sub: string }) {
  const { t } = useI18n();
  return <article className={styles.kpiTile}><span>{t(label)}</span><strong>{t(value)}</strong><small>{t(sub)}</small></article>;
}

function HealthBar({ label, value }: { label: string; value: number }) {
  const { t } = useI18n();
  const color = value >= 70 ? "#159f77" : value >= 40 ? "#d89113" : "#e76f51";
  return (
    <div className={styles.healthBarRow}>
      <span>{t(label)}</span>
      <div className={styles.healthBarTrack}><div style={{ width: `${value}%`, background: color }} /></div>
      <strong>{value}</strong>
    </div>
  );
}

function fmt(n: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}
