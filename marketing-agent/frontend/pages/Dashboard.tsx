import { useI18n } from "@/i18n";
import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "@/api";
import type { Business, CampaignSimulation, Overview, UpcomingItem } from "@/types";
import styles from "./Dashboard.module.css";

type DashboardMode = "demo" | "live";

interface CampaignNumbers {
  reach: number;
  impressions: number;
  engagementRate: number;
  leads: number;
  sales: number;
  revenue: number;
  planned: number;
  approved: number;
  scheduled: number;
  published: number;
}

interface SentimentData {
  overall_sentiment: number;
  avg_buying_intent: number;
  high_intent_comments: number;
  top_topics: Record<string, number>;
  by_category: Record<string, { avg_sentiment: number; avg_intent: number; comments: number }>;
  total_comments: number;
}

interface CompetitorEntry { name: string; followers: number; talking_about: number; is_us: boolean }
interface CompetitivePosition {
  status: string;
  our_followers: number;
  our_talking_about: number;
  our_posts_30d: number;
  our_avg_engagement: number;
  rank: number;
  total_tracked: number;
  leaderboard: CompetitorEntry[];
}

interface SelfBenchmark {
  status?: string;
  period_days: number;
  current: { start: string; end: string; posts: number; reels: number; statics: number; reel_pct: number; avg_engagement: number; total_likes: number; total_comments: number };
  previous: { start: string; end: string; posts: number; reels: number; statics: number; reel_pct: number; avg_engagement: number; total_likes: number; total_comments: number };
  changes?: { posts_delta_pct: number | null; engagement_delta_pct: number | null; likes_delta_pct: number | null; comments_delta_pct: number | null; reel_pct_shift: number };
  insights?: string[];
}

export function Dashboard() {
  const { t, locale } = useI18n();
  const { businessId } = useParams();
  const [overview, setOverview] = useState<Overview | null>(null);
  const [business, setBusiness] = useState<Business | null>(null);
  const [simulation, setSimulation] = useState<CampaignSimulation | null>(null);
  const [mode, setMode] = useState<DashboardMode>("demo");
  const [error, setError] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [sentiment, setSentiment] = useState<SentimentData | null>(null);
  const [directives, setDirectives] = useState<string | null>(null);
  const [selfBench, setSelfBench] = useState<SelfBenchmark | null>(null);
  const [compPos, setCompPos] = useState<CompetitivePosition | null>(null);
  const [refreshingComp, setRefreshingComp] = useState(false);
  const [compRefreshMsg, setCompRefreshMsg] = useState("");

  async function refreshCompetitors() {
    setRefreshingComp(true);
    setCompRefreshMsg("");
    try {
      const res = await api.post<{ status: string; updated: number; total: number }>(
        `/businesses/${businessId}/competitors/refresh-auto`,
      );
      setCompRefreshMsg(`Updated ${res.updated}/${res.total} competitors`);
      const fresh = await api.get<CompetitivePosition>(`/businesses/${businessId}/competitive-position`);
      if (fresh.status === "ok") setCompPos(fresh);
    } catch {
      setCompRefreshMsg("Refresh failed — Facebook may require login for some pages");
    } finally {
      setRefreshingComp(false);
    }
  }

  useEffect(() => {
    Promise.all([
      api.get<Overview>(`/businesses/${businessId}/overview`),
      api.get<Business>(`/businesses/${businessId}`),
      api.get<CampaignSimulation>(`/businesses/${businessId}/learning/simulation`),
    ])
      .then(([overviewData, businessData, simulationData]) => {
        setOverview(overviewData);
        setBusiness(businessData);
        setSimulation(simulationData);
      })
      .catch(() => setError("Failed to load the campaign dashboard"));
    // Load intelligence data (non-blocking)
    api.get<{ status: string } & SentimentData>(`/businesses/${businessId}/sentiment`)
      .then((d) => { if (d.status === "ok") setSentiment(d); })
      .catch(() => {});
    api.get<{ directives: string }>(`/businesses/${businessId}/content-directives`)
      .then((d) => setDirectives(d.directives || null))
      .catch(() => {});
    api.get<SelfBenchmark>(`/businesses/${businessId}/self-benchmark`)
      .then((d) => { if (d.status !== "insufficient_data") setSelfBench(d); })
      .catch(() => {});
    api.get<CompetitivePosition>(`/businesses/${businessId}/competitive-position`)
      .then((d) => { if (d.status === "ok") setCompPos(d); })
      .catch(() => {});
  }, [businessId]);

  const liveNumbers = useMemo<CampaignNumbers>(() => {
    const content = overview?.content ?? {};
    const total = Object.values(content).reduce((sum, value) => sum + value, 0);
    const perf = overview?.performance_30d;
    const kpis = overview?.page_kpis ?? {};
    const fbFollowers = Number(kpis.facebook?.followers_count ?? 0);
    const igFollowers = Number(kpis.instagram?.followers_count ?? 0);
    const totalFollowers = fbFollowers + igFollowers;
    return {
      reach: (perf?.posts_with_metrics ?? 0)>0 ? perf!.reach : totalFollowers,
      impressions: (perf?.posts_with_metrics ?? 0)>0 ? perf!.impressions : totalFollowers,
      engagementRate: perf?.engagement_rate ?? (Number(kpis.facebook?.talking_about_count ?? 0) > 0
        ? Number(kpis.facebook.talking_about_count) / fbFollowers * 100 : 0),
      leads: overview?.leads_new_30d ?? 0,
      sales: overview?.sales_30d.count ?? 0,
      revenue: overview?.sales_30d.revenue ?? 0,
      planned: total,
      approved: content.approved ?? 0,
      scheduled: overview?.scheduled_total ?? overview?.upcoming.length ?? 0,
      published: content.published ?? 0,
    };
  }, [overview]);

  async function syncMetaMetrics() {
    setSyncing(true);
    try {
      await api.post(`/businesses/${businessId}/metrics/sync`);
      setOverview(await api.get<Overview>(`/businesses/${businessId}/overview`));
    } catch {
      setError("Meta metric refresh failed. Existing dashboard data was kept.");
    } finally {
      setSyncing(false);
    }
  }

  if (error) return <div className={styles.errorState}>{t(error)}</div>;
  if (!overview || !business) return <div className={styles.loading}>{t("Preparing campaign workspace…")}</div>;

  if (!simulation) return <div className={styles.loading}>{t("Preparing campaign simulation...")}</div>;
  const numbers: CampaignNumbers = mode === "demo"
    ? { ...simulation.numbers, engagementRate: simulation.numbers.engagement_rate }
    : liveNumbers;
  const isDemo = mode === "demo";
  const upcoming = isDemo && overview.upcoming.length === 0 ? demoUpcoming() : overview.upcoming;
  const campaignName = business.industry
    ? `${business.name} ${business.industry} Growth Campaign`
    : `${business.name} Growth Campaign`;

  return (
    <div className={styles.page}>
      <header className={styles.hero}>
        <div>
          <div className={styles.eyebrowRow}>
            <span className={styles.eyebrow}>{t("Campaign command center")}</span>
            <span className={isDemo ? styles.demoBadge : styles.liveBadge}>
              <span className={styles.statusDot} />
              {isDemo ? t("Demo preview") : t("Live workspace")}
            </span>
          </div>
          <h1>{t(campaignName)}</h1>
          <p className={styles.heroText}>
            {t("One coordinated campaign across Instagram and Facebook—from strategy and content production to leads, sales, and AI-powered learning.")}</p>
          <div className={styles.heroMeta}>
            <span>{t("30-day campaign")}</span>
            <span>{t("Awareness → Engagement → Conversion")}</span>
            <span>{business.status === "active" ? t("Active") : t("Presentation ready")}</span>
          </div>
        </div>
        <div className={styles.heroActions}>
          <div className={styles.modeSwitch} aria-label={t("Dashboard data mode")}>
            <button className={isDemo ? styles.modeActive : ""} onClick={() => setMode("demo")}>{t("Demo preview")}</button>
            <button className={!isDemo ? styles.modeActive : ""} onClick={() => setMode("live")}>{t("Live data")}</button>
          </div>
          {isDemo
            ? <Link className={styles.primaryAction} to={`/c/${businessId}/calendar`}>{t("Open campaign calendar")}</Link>
            : <button className={styles.primaryAction} onClick={syncMetaMetrics} disabled={syncing}>{syncing ? t("Syncing Meta...") : t("Refresh Meta data")}</button>}
        </div>
      </header>

      {isDemo && (
        <div className={styles.demoNotice}>
          <strong>{t("Learning sandbox:")}</strong> {t("figures are simulated from Ozone's current content mix. They never enter production learning or real reports. Switch to Live data for stored Meta, lead, and sales results.")}</div>
      )}
      {!isDemo && (
        <div className={styles.demoNotice}>
          <strong>{t("Live sources:")}</strong>
          {overview.connected.includes("facebook") ? ` Facebook ${formatCompact(Number(overview.page_kpis?.facebook?.followers_count ?? 0))} followers` : " Facebook not connected"}
          {overview.connected.includes("instagram") ? ` · Instagram ${formatCompact(Number(overview.page_kpis?.instagram?.followers_count ?? 0))} followers` : ""}
          {` · ${(overview.platform_posts?.facebook?.total ?? 0) + (overview.platform_posts?.instagram?.total ?? 0)} posts tracked`}
          {overview.performance_30d.last_synced_at
            ? ` · synced ${new Date(overview.performance_30d.last_synced_at).toLocaleString(locale)}`
            : ""}.
        </div>
      )}
      {!isDemo && overview.connected.length === 0 && (
        <section className={styles.connectionEmpty} aria-label="Live marketing metrics setup">
          <div><span>LIVE METRICS</span><h2>Connect Meta to populate this dashboard</h2><p>No Facebook or Instagram credential is connected yet, so Ozone has no authorised source for followers, reach, impressions, or post engagement. Demo preview remains available for the presentation forecast.</p></div>
          <Link className={styles.connectionAction} to={`/c/${businessId}/settings`}>Open connection settings →</Link>
        </section>
      )}

      <section className={styles.kpiGrid} aria-label={t("Campaign performance")}>
        {(() => {
          const hasPostMetrics = !isDemo && overview.performance_30d.posts_with_metrics > 0;
          const kpiSource = isDemo ? "Sandbox forecast" : hasPostMetrics ? realMetricStatus(overview, locale) : "Page-level (live)";
          return <>
            <KpiCard label={hasPostMetrics || isDemo ? t("Summed post reach") : t("Total followers")} value={!isDemo && hasPostMetrics && overview.performance_30d.availability?.reach===false ? '—' : formatCompact(numbers.reach)} detail={hasPostMetrics || isDemo ? t("Lifetime results for recent posts; audience may overlap") : t("Facebook + Instagram combined")} trend={t(kpiSource)} tone="violet" />
            <KpiCard label={hasPostMetrics || isDemo ? t("Impressions") : t("Audience size")} value={formatCompact(numbers.impressions)} detail={hasPostMetrics || isDemo ? t("Across both channels") : t("Total page followers")} trend={t(kpiSource)} tone="blue" />
            <KpiCard label={t("Engagement rate")} value={`${numbers.engagementRate.toFixed(1)}%`} detail={hasPostMetrics || isDemo ? t("Measured interactions / reach") : t("Talking about / followers")} trend={t(kpiSource)} tone="coral" />
            <KpiCard label={t("Qualified leads")} value={String(numbers.leads)} detail={t("Comments and direct messages")} trend={isDemo ? t("+9 this week") : t("Live workflow data")} tone="amber" />
            <KpiCard label={t("Confirmed sales")} value={String(numbers.sales)} detail={`${numbers.revenue.toLocaleString(locale)} TND revenue`} trend={isDemo ? t("29% lead conversion") : t("Live workflow data")} tone="green" />
          </>;
        })()}
      </section>

      {!isDemo && (
        <section className={styles.platformGrid} aria-label={t("Per-platform breakdown")}>
          {overview.connected.includes("facebook") && (
            <article className={`card ${styles.platformCard}`}>
              <SectionHeading kicker={t("Facebook")} title={t(`${formatCompact(Number(overview.page_kpis?.facebook?.followers_count ?? 0))} followers`)} />
              <div className={styles.platformStats}>
                <div><span>{t("Posts tracked")}</span><strong>{overview.platform_posts?.facebook?.total ?? 0}</strong></div>
                <div><span>{t("Avg engagement")}</span><strong>{overview.platform_performance?.facebook?.avg_engagement_per_post?.toFixed(1) ?? "—"}</strong></div>
                <div><span>{t("Engagement rate")}</span><strong>{overview.platform_performance?.facebook?.engagement_rate != null ? `${overview.platform_performance.facebook.engagement_rate}%` : "—"}</strong></div>
                <div><span>{t("Posts/week")}</span><strong>{overview.posting_cadence?.facebook?.per_week?.toFixed(1) ?? "—"}</strong></div>
                <div><span>{t("Talking about")}</span><strong>{formatCompact(Number(overview.page_kpis?.facebook?.talking_about_count ?? 0))}</strong></div>
              </div>
            </article>
          )}
          {overview.connected.includes("instagram") && (
            <article className={`card ${styles.platformCard}`}>
              <SectionHeading kicker={t("Instagram")} title={t(`${formatCompact(Number(overview.page_kpis?.instagram?.followers_count ?? 0))} followers`)} />
              <div className={styles.platformStats}>
                <div><span>{t("Posts tracked")}</span><strong>{overview.platform_posts?.instagram?.total ?? 0}</strong></div>
                <div><span>{t("Avg engagement")}</span><strong>{overview.platform_performance?.instagram?.avg_engagement_per_post?.toFixed(1) ?? "—"}</strong></div>
                <div><span>{t("Engagement rate")}</span><strong>{overview.platform_performance?.instagram?.engagement_rate != null ? `${overview.platform_performance.instagram.engagement_rate}%` : "—"}</strong></div>
                <div><span>{t("Posts/week")}</span><strong>{overview.posting_cadence?.instagram?.per_week?.toFixed(1) ?? "—"}</strong></div>
                <div><span>{t("Media count")}</span><strong>{formatCompact(Number(overview.page_kpis?.instagram?.media_count ?? 0))}</strong></div>
              </div>
            </article>
          )}
        </section>
      )}

      {!isDemo && Object.keys(overview.category_distribution ?? {}).length > 0 && (
        <section className={styles.analyticsGrid} aria-label={t("Content analytics")}>
          <article className={`card ${styles.categoryCard}`}>
            <SectionHeading kicker={t("Learning signals")} title={t("Content categories")} />
            <div className={styles.categoryBars}>
              {Object.entries(overview.category_distribution)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 8)
                .map(([cat, count]) => {
                  const max = Math.max(...Object.values(overview.category_distribution));
                  const perf = overview.category_performance?.[cat];
                  return (
                    <div className={styles.categoryRow} key={cat}>
                      <span className={styles.categoryLabel}>{t(cat)}</span>
                      <div className={styles.categoryBarTrack}>
                        <div className={styles.categoryBarFill} style={{ width: `${(count / max) * 100}%` }} />
                      </div>
                      <span className={styles.categoryCount}>{count}</span>
                      {perf && <span className={styles.categoryEngagement} title={t("Avg engagement")}>{perf.avg_engagement.toFixed(0)}</span>}
                    </div>
                  );
                })}
            </div>
          </article>

          <article className={`card ${styles.strategyCard}`}>
            <SectionHeading kicker={t("Strategy")} title={t("Content layer mix")} />
            <div className={styles.strategyMix}>
              {(() => {
                const mix = overview.strategy_mix ?? {};
                const total = Object.values(mix).reduce((s, v) => s + v, 0) || 1;
                const pillar = mix.pillar ?? 0;
                const distribution = mix.distribution ?? 0;
                const engagement = mix.engagement ?? 0;
                return <>
                  <div className={styles.strategyBar}>
                    {pillar > 0 && <div className={styles.pillarSegment} style={{ width: `${(pillar / total) * 100}%` }} title={t(`Pillar: ${pillar}`)} />}
                    {distribution > 0 && <div className={styles.distributionSegment} style={{ width: `${(distribution / total) * 100}%` }} title={t(`Distribution: ${distribution}`)} />}
                    {engagement > 0 && <div className={styles.engagementSegment} style={{ width: `${(engagement / total) * 100}%` }} title={t(`Engagement: ${engagement}`)} />}
                  </div>
                  <div className={styles.strategyLegend}>
                    <Legend color="var(--pillar)" label={t("Pillar")} value={`${Math.round((pillar / total) * 100)}%`} />
                    <Legend color="var(--distribution)" label={t("Distribution")} value={`${Math.round((distribution / total) * 100)}%`} />
                    <Legend color="var(--engagement-color)" label={t("Engagement")} value={`${Math.round((engagement / total) * 100)}%`} />
                  </div>
                  <p className={styles.strategyTarget}>{t("Target: 20% pillar / 50% distribution / 30% engagement")}</p>
                </>;
              })()}
            </div>
          </article>
        </section>
      )}

      <section className={styles.mainGrid}>
        <article className={`card ${styles.funnelCard}`}>
          <SectionHeading kicker={t("Workflow")} title={t("Campaign funnel")} action={<Link to={`/c/${businessId}/content`}>{t("Manage content")}</Link>} />
          <div className={styles.funnel}>
            <FunnelStep label={t("Planned")} value={numbers.planned} width={100} tone="planned" />
            <FunnelStep label={t("Approved")} value={numbers.approved} width={82} tone="approved" />
            <FunnelStep label={t("Scheduled")} value={numbers.scheduled} width={65} tone="scheduled" />
            <FunnelStep label={t("Published")} value={numbers.published} width={50} tone="published" />
            <FunnelStep label={t("Leads")} value={numbers.leads} width={36} tone="leads" />
            <FunnelStep label={t("Sales")} value={numbers.sales} width={23} tone="sales" />
          </div>
          <p className={styles.funnelNote}>{t("Every post stays traceable from idea to revenue.")}</p>
        </article>

        <article className={`card ${styles.mixCard}`}>
          <SectionHeading kicker={t("Creative strategy")} title={t("Content mix")} />
          {Object.keys(isDemo ? simulation.content_mix : overview.content_mix).length > 0 ? (
            <>
              <div className={styles.mixVisual}>
                <div className={styles.donut}>
                  <div><strong>{sumValues(isDemo ? simulation.content_mix : overview.content_mix)}</strong><span>{t("assets")}</span></div>
                </div>
                <div className={styles.legend}>
                  <Legend color="#6c5ce7" label={t("Reels")} value={mixPercent(isDemo ? simulation.content_mix : overview.content_mix, "reel")} />
                  <Legend color="#00b894" label={t("Static posts")} value={mixPercent(isDemo ? simulation.content_mix : overview.content_mix, "static_post")} />
                  <Legend color="#fdcb6e" label={t("Avatar clips")} value={mixPercent(isDemo ? simulation.content_mix : overview.content_mix, "avatar_clip")} />
                </div>
              </div>
              <div className={styles.platformSplit}>
                <div><span>{t("Instagram")}</span><strong>{isDemo ? mixPercent(simulation.platform_mix, "instagram") : t("Live")}</strong></div>
                <div><span>{t("Facebook")}</span><strong>{isDemo ? mixPercent(simulation.platform_mix, "facebook") : t("Live")}</strong></div>
              </div>
            </>
          ) : <LiveEmpty label={t("Content mix appears after campaign posts are created.")} />}
        </article>

        <article className={`card ${styles.momentumCard}`}>
          <SectionHeading kicker={t("Momentum")} title={t("Weekly channel performance")} />
          {(isDemo ? simulation.weekly.length : overview.weekly_performance.length) > 0 ? (
            <>
              <div className={styles.chartLegend}>
                <span><i className={styles.instagramKey} /> {t(" Instagram")}</span>
                <span><i className={styles.facebookKey} /> {t(" Facebook")}</span>
              </div>
              <div className={styles.barChart}>
                {weeklySeries(isDemo, simulation, overview, locale).map((week) => (
                  <div className={styles.week} key={week.label}>
                    <div className={styles.barPair}>
                      <span className={styles.facebookBar} style={{ height: `${week.facebook}%` }} title={t(`${week.facebookRaw} interactions`)} />
                      <span className={styles.instagramBar} style={{ height: `${week.instagram}%` }} title={t(`${week.instagramRaw} interactions`)} />
                    </div>
                    <span>{t(week.label)}</span>
                  </div>
                ))}
              </div>
            </>
          ) : <LiveEmpty label={t("Weekly reach and engagement appear after Meta is connected.")} />}
        </article>
      </section>

      <section className={styles.lowerGrid}>
        <article className={`card ${styles.scheduleCard}`}>
          <SectionHeading kicker={t("Next up")} title={t("Campaign schedule")} action={<Link to={`/c/${businessId}/calendar`}>{t("View calendar")}</Link>} />
          <div className={styles.scheduleList}>
            {upcoming.slice(0, 4).map((item, index) => (
              <ScheduleItem key={item.id} item={item} index={index} />
            ))}
            {upcoming.length === 0 && <p className={styles.empty}>{t("No posts scheduled yet. Build the first campaign week in Calendar.")}</p>}
          </div>
        </article>

        <article className={`card ${styles.insightCard}`}>
          <SectionHeading kicker={t("Senior strategist")} title={t("Next move recommendation")} action={<Link to={`/c/${businessId}/report`}>{t("Full report")}</Link>} />
          {overview.strategist ? (
            <>
              <div className={styles.insightLead}>
                <span className={styles.spark}>✦</span>
                <p>{overview.strategist.reasoning}</p>
              </div>
              <div className={styles.strategistRec}>
                <div><span>{t("Platform")}</span><strong>{t(overview.strategist.platform)}</strong></div>
                <div><span>{t("Category")}</span><strong>{t(overview.strategist.suggested_category)}</strong></div>
                <div><span>{t("Layer")}</span><strong>{t(overview.strategist.suggested_strategy_layer)}</strong></div>
                <div><span>{t("Format")}</span><strong>{t(overview.strategist.suggested_format)}</strong></div>
                <div><span>{t("Funnel")}</span><strong>{t(overview.strategist.suggested_funnel_stage)}</strong></div>
              </div>
              {overview.strategist.platform_playbook && (
                <div className={styles.playbookHints}>
                  {overview.strategist.platform_playbook.hook_window && (
                    <p>{t("Hook: ")}{overview.strategist.platform_playbook.hook_window}</p>
                  )}
                  {overview.strategist.platform_playbook.caption_length && (
                    <p>{t("Caption: ")}{overview.strategist.platform_playbook.caption_length}</p>
                  )}
                </div>
              )}
            </>
          ) : (
            <>
              <div className={styles.insightLead}>
                <span className={styles.spark}>✦</span>
                <p>{isDemo ? simulation.learning.summary : overview.latest_insight?.summary
                  || t("AI recommendations will appear after published posts receive real leads or sales.")}</p>
              </div>
              <ol className={styles.recommendations}>
                {recommendationsFor(isDemo, simulation, overview).map((recommendation, index) => (
                  <li key={recommendation}>
                    <span>0{index + 1}</span>
                    <p><strong>{isDemo ? t("Sandbox experiment") : t("Evidence-based action")}</strong>{recommendation}</p>
                  </li>
                ))}
              </ol>
            </>
          )}
        </article>
      </section>

      {/* ── Competitive position ── */}
      {!isDemo && compPos && compPos.leaderboard.length > 0 && (
        <section className={styles.competitorSection} aria-label={t("Competitive position")}>
          <SectionHeading
            kicker={t("Competitive intelligence")}
            title={t(`Rank #${compPos.rank} of ${compPos.total_tracked}`)}
            action={
              <button className={styles.refreshBtn} onClick={refreshCompetitors} disabled={refreshingComp}>
                {refreshingComp ? t("Refreshing...") : t("Refresh data")}
              </button>
            }
          />
          {compRefreshMsg && <p className={styles.compRefreshMsg}>{t(compRefreshMsg)}</p>}
          <div className={styles.compInsights}>
            {(() => {
              const us = compPos.leaderboard.find(e => e.is_us);
              const leader = compPos.leaderboard[0];
              const ourIdx = compPos.leaderboard.findIndex(e => e.is_us);
              const closestAbove = ourIdx > 0 ? compPos.leaderboard[ourIdx - 1] : null;
              const mostActive = [...compPos.leaderboard]
                .filter(e => e.talking_about && !e.is_us)
                .sort((a, b) => b.talking_about - a.talking_about)[0];
              return <>
                {leader && !leader.is_us && us && (
                  <div className={styles.compInsight}>
                    <strong>{t("Gap to #1")}</strong>
                    <span>{formatCompact(leader.followers - us.followers)} {t(" followers behind ")}{leader.name}</span>
                  </div>
                )}
                {closestAbove && us && (
                  <div className={styles.compInsight}>
                    <strong>{t("Next target")}</strong>
                    <span>{closestAbove.name} — {formatCompact(closestAbove.followers - us.followers)} {t(" ahead")}</span>
                  </div>
                )}
                {mostActive && (
                  <div className={styles.compInsight}>
                    <strong>{t("Most buzz")}</strong>
                    <span>{mostActive.name} — {formatCompact(mostActive.talking_about)} {t(" people talking")}</span>
                  </div>
                )}
              </>;
            })()}
          </div>
          <div className={styles.leaderboard}>
            {compPos.leaderboard.map((entry, i) => (
              <div key={entry.name} className={`${styles.lbRow} ${entry.is_us ? styles.lbUs : ""}`}>
                <span className={styles.lbRank}>#{i + 1}</span>
                <span className={styles.lbName}>{entry.name}{entry.is_us ? t(" (you)") : ""}</span>
                <strong className={styles.lbFollowers}>{formatCompact(entry.followers)}</strong>
                <span className={styles.lbTalking}>{entry.talking_about ? t(`${formatCompact(entry.talking_about)} active`) : "—"}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Intelligence panels (live data only) ── */}
      {!isDemo && (sentiment || directives || selfBench) && (
        <section className={styles.intelligenceGrid} aria-label={t("AI Intelligence")}>
          {sentiment && sentiment.total_comments > 0 && (
            <article className={`card ${styles.intelligenceCard}`}>
              <SectionHeading kicker={t("Comment intelligence")} title={t("What your audience says")} />
              <div className={styles.sentimentRow}>
                <div className={styles.sentimentStat}>
                  <strong className={sentiment.overall_sentiment >= 0 ? styles.positive : styles.negative}>
                    {sentiment.overall_sentiment > 0 ? "+" : ""}{sentiment.overall_sentiment}%
                  </strong>
                  <span>{t("Sentiment")}</span>
                </div>
                <div className={styles.sentimentStat}>
                  <strong>{sentiment.avg_buying_intent}/100</strong>
                  <span>{t("Buying intent")}</span>
                </div>
                <div className={styles.sentimentStat}>
                  <strong>{sentiment.high_intent_comments}</strong>
                  <span>{t("High-intent")}</span>
                </div>
                <div className={styles.sentimentStat}>
                  <strong>{sentiment.total_comments}</strong>
                  <span>{t("Comments")}</span>
                </div>
              </div>
              {Object.keys(sentiment.top_topics).length > 0 && (
                <div className={styles.topicTags}>
                  <span className={styles.topicLabel}>{t("People ask about:")}</span>
                  {Object.entries(sentiment.top_topics).slice(0, 6).map(([topic, count]) => (
                    <span key={topic} className={styles.topicTag}>{topic} <small>({count})</small></span>
                  ))}
                </div>
              )}
              {Object.keys(sentiment.by_category || {}).length > 0 && (
                <div className={styles.sentimentCats}>
                  {Object.entries(sentiment.by_category).slice(0, 4).map(([cat, d]) => (
                    <div key={cat} className={styles.sentimentCatRow}>
                      <span>{t(cat)}</span>
                      <span className={d.avg_sentiment >= 0 ? styles.positive : styles.negative}>
                        {d.avg_sentiment > 0 ? "+" : ""}{d.avg_sentiment}%
                      </span>
                      <span>{t("intent ")}{d.avg_intent}/100</span>
                      <small>{d.comments} {t(" comments")}</small>
                    </div>
                  ))}
                </div>
              )}
            </article>
          )}

          {selfBench && selfBench.changes && (
            <article className={`card ${styles.intelligenceCard}`}>
              <SectionHeading kicker={t("Self-benchmark")} title={t("Period-over-period")} />
              <p className={styles.benchPeriod}>
                {selfBench.current.start} → {selfBench.current.end} {t(" vs ")}{selfBench.previous.start} → {selfBench.previous.end}
              </p>
              <div className={styles.benchGrid}>
                <BenchDelta label={t("Posts")} current={selfBench.current.posts} previous={selfBench.previous.posts} delta={selfBench.changes.posts_delta_pct} />
                <BenchDelta label={t("Avg engagement")} current={selfBench.current.avg_engagement} previous={selfBench.previous.avg_engagement} delta={selfBench.changes.engagement_delta_pct} />
                <BenchDelta label={t("Total likes")} current={selfBench.current.total_likes} previous={selfBench.previous.total_likes} delta={selfBench.changes.likes_delta_pct} />
                <BenchDelta label={t("Comments")} current={selfBench.current.total_comments} previous={selfBench.previous.total_comments} delta={selfBench.changes.comments_delta_pct} />
                <BenchDelta label={t("Reel %")} current={selfBench.current.reel_pct} previous={selfBench.previous.reel_pct} delta={selfBench.changes.reel_pct_shift} suffix="pp" />
              </div>
              {selfBench.insights && selfBench.insights.length > 0 && (
                <div className={styles.benchInsights}>
                  {selfBench.insights.map((insight) => (
                    <p key={insight}>{insight}</p>
                  ))}
                </div>
              )}
            </article>
          )}

          {directives && (
            <article className={`card ${styles.intelligenceCard} ${styles.directivesCard}`}>
              <SectionHeading kicker={t("Active rules")} title={t("Content directives")} />
              <div className={styles.directivesList}>
                {directives.replace("--- CONTENT DIRECTIVES (apply these to every piece of content) ---\n", "")
                  .split("\n")
                  .filter(Boolean)
                  .map((line) => (
                    <p key={line} className={styles.directiveLine}>{line}</p>
                  ))}
              </div>
            </article>
          )}
        </section>
      )}

      <section className={styles.readiness}>
        <div>
          <span className={styles.readinessIcon}>M</span>
          <div>
            <p className={styles.readinessTitle}>{t("Meta publishing readiness")}</p>
            <p>{t("Campaign planning, approvals, scheduling, leads and reporting are ready for the demo.")}</p>
          </div>
        </div>
        <div className={styles.connectionSteps}>
          <ConnectionStep label={t("Campaign system")} status="Ready" ready />
          <ConnectionStep label={t("Facebook")} status={overview.connected.includes("facebook") ? "Connected" : "Pending access"} ready={overview.connected.includes("facebook")} />
          <ConnectionStep label={t("Instagram")} status={overview.connected.includes("instagram") ? "Connected" : "Pending access"} ready={overview.connected.includes("instagram")} />
        </div>
        <Link className={styles.settingsLink} to={`/c/${businessId}/settings`}>{t("Connection settings")}</Link>
      </section>
    </div>
  );
}

function KpiCard({ label, value, detail, trend, tone }: { label: string; value: string; detail: string; trend: string; tone: string }) {
  const { t } = useI18n();
  return (
    <article className={`${styles.kpiCard} ${styles[tone]}`}>
      <p>{t(label)}</p>
      <strong>{value}</strong>
      <span>{t(detail)}</span>
      <small>{t(trend)}</small>
    </article>
  );
}

function SectionHeading({ kicker, title, action }: { kicker: string; title: string; action?: React.ReactNode }) {
  const { t } = useI18n();
  return (
    <div className={styles.sectionHeading}>
      <div><span>{t(kicker)}</span><h2>{t(title)}</h2></div>
      {action}
    </div>
  );
}

function FunnelStep({ label, value, width, tone }: { label: string; value: number; width: number; tone: string }) {
  const { t } = useI18n();
  return (
    <div className={styles.funnelRow}>
      <span>{t(label)}</span>
      <div className={`${styles.funnelBar} ${styles[tone]}`} style={{ width: `${width}%` }}><strong>{value}</strong></div>
    </div>
  );
}

function Legend({ color, label, value }: { color: string; label: string; value: string }) {
  const { t } = useI18n();
  return <div><span><i style={{ background: color }} />{t(label)}</span><strong>{value}</strong></div>;
}

function ScheduleItem({ item, index }: { item: UpcomingItem; index: number }) {
  const { t, locale } = useI18n();
  const date = new Date(item.scheduled_for);
  return (
    <div className={styles.scheduleItem}>
      <div className={styles.dateTile}>
        <strong>{date.toLocaleDateString(locale, { day: "2-digit" })}</strong>
        <span>{date.toLocaleDateString(locale, { month: "short" })}</span>
      </div>
      <div className={styles.scheduleCopy}>
        <div><span className={styles[item.platform]}>{t(item.platform)}</span><span>{t(item.type)}</span></div>
        <p dir="auto">{item.id.startsWith("demo-") ? t(item.caption) : item.caption}</p>
      </div>
      <time>{date.toLocaleTimeString(locale, { hour: "2-digit", minute: "2-digit" })}</time>
      <span className={styles.queueNumber}>0{index + 1}</span>
    </div>
  );
}

function ConnectionStep({ label, status, ready }: { label: string; status: string; ready: boolean }) {
  const { t } = useI18n();
  return <div><span className={ready ? styles.readyDot : styles.pendingDot} /><p><strong>{t(label)}</strong><small>{t(status)}</small></p></div>;
}

function LiveEmpty({ label }: { label: string }) {
  const { t } = useI18n();
  return <div className={styles.liveEmpty}><span>↗</span><p>{t(label)}</p></div>;
}

function demoUpcoming(): UpcomingItem[] {
  const captions = [
    ["instagram", "reel", "Your next escape starts here: sun, comfort and unforgettable moments."],
    ["facebook", "static_post", "A complete stay designed for families who want an easy, memorable holiday."],
    ["instagram", "avatar_clip", "Three reasons this destination belongs on your travel list."],
    ["facebook", "reel", "Ready to turn your next weekend into a real experience?"],
  ];
  return captions.map(([platform, type, caption], index) => {
    const date = new Date();
    date.setDate(date.getDate() + index + 1);
    date.setHours(index % 2 ? 12 : 19, index % 2 ? 30 : 0, 0, 0);
    return { id: `demo-${index}`, platform, type, caption, scheduled_for: date.toISOString() };
  });
}

function formatCompact(value: number): string {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function sumValues(values: Record<string, number>): number {
  return Object.values(values).reduce((total, value) => total + value, 0);
}

function mixPercent(values: Record<string, number>, key: string): string {
  const total = sumValues(values);
  return total ? `${Math.round(((values[key] ?? 0) / total) * 100)}%` : "0%";
}

function realMetricStatus(overview: Overview, locale: string): string {
  if (!overview.performance_30d.last_synced_at) return "No Meta metric snapshot yet";
  return `Synced ${new Date(overview.performance_30d.last_synced_at).toLocaleDateString(locale)}`;
}

function weeklySeries(isDemo: boolean, simulation: CampaignSimulation, overview: Overview, locale: string) {
  const raw = isDemo
    ? simulation.weekly.map((week) => ({ ...week }))
    : Array.from({ length: 4 }, (_, index) => ({ label: `Week ${index + 1}`, facebook: 0, instagram: 0 }));
  if (!isDemo) {
    const weeks = [...new Set(overview.weekly_performance.map((item) => item.week))].slice(-4);
    weeks.forEach((week, index) => {
      raw[index].label = new Date(week).toLocaleDateString(locale, { month: "short", day: "numeric" });
      overview.weekly_performance.filter((item) => item.week === week).forEach((item) => {
        if (item.platform === "facebook" || item.platform === "instagram") raw[index][item.platform] = item.engagement;
      });
    });
  }
  const maximum = Math.max(1, ...raw.flatMap((week) => [week.facebook, week.instagram]));
  return raw.map((week) => ({ ...week, facebookRaw: week.facebook, instagramRaw: week.instagram,
    facebook: Math.max(4, Math.round((week.facebook / maximum) * 100)),
    instagram: Math.max(4, Math.round((week.instagram / maximum) * 100)) }));
}

function BenchDelta({ label, current, previous, delta, suffix = "%" }: { label: string; current: number; previous: number; delta: number | null; suffix?: string }) {
  const { t } = useI18n();
  const isUp = delta !== null && delta > 0;
  const isDown = delta !== null && delta < 0;
  return (
    <div className={styles.benchCell}>
      <span className={styles.benchLabel}>{t(label)}</span>
      <strong>{current}</strong>
      <small>{t("was ")}{previous}</small>
      {delta !== null && (
        <span className={isUp ? styles.deltaUp : isDown ? styles.deltaDown : styles.deltaNeutral}>
          {isUp ? "+" : ""}{delta}{t(suffix)}
        </span>
      )}
    </div>
  );
}

function recommendationsFor(isDemo: boolean, simulation: CampaignSimulation, overview: Overview): string[] {
  const live = overview.latest_insight?.recommendations ?? [];
  return (isDemo ? simulation.learning.recommendations : live.length ? live : [
    "Publish the first approved post, then refresh Meta metrics after it gathers activity.",
    "Review incoming comments and messages so qualified leads are labeled correctly.",
    "Record confirmed sales to connect each content decision to business outcomes.",
  ]).slice(0, 3);
}
