import { useI18n } from "@/i18n";
import { useEffect, useState } from "react";
import { api } from "@/api";
import styles from "./Costs.module.css";

interface CostBreakdown {
  task: string; model: string; calls: number; succeeded: number;
  cost_usd: number; average_cost_usd: number;
}

interface CostCall {
  at: string; task: string; model: string; ok: boolean; cost_usd: number | null;
  prompt_tokens: number | null; completion_tokens: number | null; latency_ms: number | null;
}

interface CostData {
  days: number; period_spend: number; today_spend: number; priced_calls: number;
  average_cost: number; month_spend: number; all_time_spend: number; monthly_budget: number;
  remaining_budget: number | null; budget_percent: number | null;
  budget_committed?: number;
  reservations?: {provider:string;status:string;amount_usd:number;operations:number}[];
  total_credits: number; credits_remaining: number | null;
  breakdown: CostBreakdown[]; recent: CostCall[];
}

const money = (value: number | null) => {
  if (value === null) return "Pending";
  const abs = Math.abs(value);
  const formatted = `$${abs.toFixed(abs < 0.01 && abs > 0 ? 4 : 2)}`;
  return value < 0 ? `-${formatted}` : formatted;
};

export function Costs() {
  const { t, locale } = useI18n();
  const [data, setData] = useState<CostData | null>(null);
  const [days, setDays] = useState(30);
  const [error, setError] = useState("");

  useEffect(() => {
    setError("");
    api.get<CostData>(`/ai/costs?days=${days}`).then(setData).catch((reason) => setError(String(reason)));
  }, [days]);

  if (error) return <div className={styles.page}><p className={styles.error}>{t(error)}</p></div>;
  if (!data) return <div className={styles.page}><p>{t("Loading costs...")}</p></div>;

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>{t("OPENROUTER CONTROL CENTER")}</p>
          <h1>{t("AI Costs")}</h1>
          <p>{t("Exact provider charges for every text and image generation.")}</p>
        </div>
        <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
          <option value={7}>{t("Last 7 days")}</option><option value={30}>{t("Last 30 days")}</option><option value={90}>{t("Last 90 days")}</option>
        </select>
      </header>

      <section className={styles.cards}>
        <article className={styles.heroCard}>
          <span>{t("Credits Remaining")}</span>
          <strong>{data.credits_remaining != null ? money(data.credits_remaining) : money(data.all_time_spend)}</strong>
          {data.total_credits > 0 && (
            <div className={styles.progress}>
              <i style={{ width: `${Math.min(100, (data.all_time_spend / data.total_credits) * 100)}%` }} />
            </div>
          )}
          <small>{data.total_credits > 0
            ? t(`${money(data.all_time_spend)} spent of ${money(data.total_credits)} deposited`)
            : t("Set OPENROUTER_TOTAL_CREDITS to track balance")}</small>
        </article>
        <article><span>{t("All time")}</span><strong>{money(data.all_time_spend)}</strong><small>{t("Total recorded spend")}</small></article>
        <article><span>{t("This month")}</span><strong>{money(data.month_spend)}</strong>
          <small>{data.monthly_budget > 0
            ? (data.remaining_budget !== null && data.remaining_budget < 0
              ? t(`${money(Math.abs(data.remaining_budget))} over budget`)
              : t(`${money(data.remaining_budget)} of $${data.monthly_budget.toFixed(2)} left`))
            : t("No monthly limit")}</small>
        </article>
        <article><span>{t("Today")}</span><strong>{money(data.today_spend)}</strong><small>{data.priced_calls} {t(" priced calls")}</small></article>
      </section>

      {!!data.reservations?.length && <section className="card"><h2>{t('Reserved and estimated charges')}</h2><p>{t('These amounts reduce the available budget. Estimates are retained when the provider does not return a verified charge.')}</p>{data.reservations.map(row=><p key={`${row.provider}-${row.status}`}>{row.provider} · {row.status} · {money(row.amount_usd)} · {row.operations} operations</p>)}</section>}

      <section className={styles.section}>
        <div className={styles.sectionTitle}><h2>{t("Spend by tool")}</h2><span>{t("Text + images")}</span></div>
        {data.breakdown.length === 0 ? <p className={styles.empty}>{t("Costs appear after the next AI generation.")}</p> : (
          <div className={styles.tableWrap}><table>
            <thead><tr><th>{t("Tool")}</th><th>{t("Model")}</th><th>{t("Success")}</th><th>{t("Average")}</th><th>{t("Total")}</th></tr></thead>
            <tbody>{data.breakdown.map((row) => (
              <tr key={`${row.task}-${row.model}`}>
                <td><b>{row.task}</b><small>{row.calls} {t(" calls")}</small></td>
                <td className={styles.model}>{row.model}</td><td>{row.succeeded}/{row.calls}</td>
                <td>{money(row.average_cost_usd)}</td><td className={styles.total}>{money(row.cost_usd)}</td>
              </tr>
            ))}</tbody>
          </table></div>
        )}
      </section>

      <section className={styles.section}>
        <div className={styles.sectionTitle}><h2>{t("Generation ledger")}</h2><span>{t("Latest 50 calls")}</span></div>
        <div className={styles.ledger}>
          {data.recent.map((call, index) => (
            <div className={styles.call} key={`${call.at}-${index}`}>
              <span className={`${styles.dot} ${call.ok ? styles.ok : styles.failed}`} />
              <div><b>{call.task}</b><small>{call.model}</small></div>
              <div className={styles.tokens}>{(call.prompt_tokens || 0) + (call.completion_tokens || 0)} {t(" tokens")}</div>
              <time>{new Date(call.at).toLocaleString(locale)}</time><strong>{money(call.cost_usd)}</strong>
            </div>
          ))}
          {!data.recent.length && <p className={styles.empty}>{t("No AI calls in this period.")}</p>}
        </div>
      </section>
      <p className={styles.note}>{t("Budget changes use ")}<code>MONTHLY_AI_BUDGET_USD</code>{t(". The agent blocks new calls once the limit is reached.")}</p>
    </div>
  );
}
