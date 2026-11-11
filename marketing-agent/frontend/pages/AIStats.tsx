import { useI18n } from "@/i18n";
import { useEffect, useState } from "react";
import { api } from "@/api";
import styles from "./AIStats.module.css";

interface TaskStat {
  task: string;
  calls: number;
  succeeded: number;
  success_rate: number;
  needed_repair: number;
  avg_attempts: number;
  avg_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  prompt_version: string;
}

interface RecentFailure {
  task: string;
  at: string;
  error: string;
}

interface StatsData {
  days: number;
  tracking_since: string | null;
  tasks: TaskStat[];
  recent_failures: RecentFailure[];
}

export function AIStats() {
  const { t, locale } = useI18n();
  const [data, setData] = useState<StatsData | null>(null);
  const [days, setDays] = useState(7);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.get<StatsData>(`/ai/stats?days=${days}`).then(setData).finally(() => setLoading(false));
  }, [days]);

  if (loading) return <p>{t("Loading…")}</p>;
  if (!data) return <p>{t("No data")}</p>;

  const totalCalls = data.tasks.reduce((s, t) => s + t.calls, 0);
  const totalOk = data.tasks.reduce((s, t) => s + t.succeeded, 0);
  const totalRepairs = data.tasks.reduce((s, t) => s + t.needed_repair, 0);
  const isResolved = (failure: RecentFailure) => failure.task === "image" && failure.error.includes("output pixels");
  const activeFailures = data.recent_failures.filter((failure) => !isResolved(failure));
  const resolvedFailures = data.recent_failures.length - activeFailures.length;

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div><p className={styles.eyebrow}>{t("MODEL OPERATIONS")}</p><h2>{t("AI Stats")}</h2></div>
        <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
          <option value={1}>{t("Last 24h")}</option>
          <option value={7}>{t("Last 7 days")}</option>
          <option value={30}>{t("Last 30 days")}</option>
        </select>
      </div>

      <div className={styles.explainer}>
        <strong>{t("How to read this")}</strong>
        <span><b>{t("Calls")}</b> {t(" are complete agent tasks, not individual posts. ")}<b>{t("Repairs")}</b> {t(" mean the guardrail rejected an answer and the model corrected it automatically. A failure means every retry was rejected.")}</span>
        {data.tracking_since && <small>{t("Operational view starts ")}{new Date(data.tracking_since).toLocaleString(locale)}{t("; earlier development/test calls are excluded.")}</small>}
      </div>

      {/* Summary */}
      <div className={styles.summary}>
        <div className="card">
          <p className={styles.statLabel}>{t("Total calls")}</p>
          <p className={styles.statValue}>{totalCalls}</p>
        </div>
        <div className="card">
          <p className={styles.statLabel}>{t("Success rate")}</p>
          <p className={styles.statValue} style={{ color: totalCalls ? undefined : "var(--text-muted)" }}>
            {totalCalls ? `${((totalOk / totalCalls) * 100).toFixed(1)}%` : "—"}
          </p>
        </div>
        <div className="card">
          <p className={styles.statLabel}>{t("Needed repair")}</p>
          <p className={styles.statValue}>{totalRepairs}</p>
        </div>
      </div>

      {/* Per-task table */}
      {data.tasks.length > 0 && (
        <table className={styles.table}>
          <thead>
            <tr>
              <th>{t("Task")}</th>
              <th>{t("Calls")}</th>
              <th>{t("Success")}</th>
              <th>{t("Rate")}</th>
              <th>{t("Repairs")}</th>
              <th>{t("Avg attempts")}</th>
              <th>{t("Avg latency")}</th>
              <th>{t("Prompt ver")}</th>
              <th>{t("Tokens (in/out)")}</th>
            </tr>
          </thead>
          <tbody>
            {data.tasks.map((task) => (
              <tr key={task.task}>
                <td className={styles.taskName}>{t(task.task)}</td>
                <td>{task.calls}</td>
                <td>{task.succeeded}</td>
                <td className={task.success_rate < 0.9 ? styles.warn : styles.good}>
                  {(task.success_rate * 100).toFixed(1)}%
                </td>
                <td>{task.needed_repair}</td>
                <td>{task.avg_attempts}</td>
                <td>{task.avg_latency_ms} {t("ms")}</td>
                <td>v{task.prompt_version}</td>
                <td>{(task.prompt_tokens || 0).toLocaleString(locale)} / {(task.completion_tokens || 0).toLocaleString(locale)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Recent failures */}
      {activeFailures.length > 0 ? (
        <section className={styles.failures}>
          <h3>{t("Active failures")}</h3>
          {activeFailures.map((f, i) => (
            <div key={i} className={`card ${styles.failCard}`}>
              <div className={styles.failHeader}>
                <span className={styles.taskName}>{f.task}</span>
                <span className={styles.failTime}>
                  {new Date(f.at).toLocaleString(locale)}
                </span>
              </div>
              <p className={styles.failError}>{f.error}</p>
            </div>
          ))}
        </section>
      ) : <div className={styles.healthy}>{t("No active AI failures.")}</div>}
      {resolvedFailures > 0 && <p className={styles.resolvedSummary}>{resolvedFailures} {t(" historical incident resolved and retained in the audit log.")}</p>}
    </div>
  );
}
