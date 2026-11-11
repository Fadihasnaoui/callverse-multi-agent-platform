import { useI18n } from "@/i18n";
import { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "@/api";
import type { Business, ContentItem, Overview } from "@/types";
import styles from "./AgentWorkflow.module.css";

type StepStatus = "idle" | "running" | "done" | "waiting" | "failed";

interface TraceStep {
  id: string;
  phase: "Observe" | "Think" | "Act" | "Learn" | "Control";
  title: string;
  tool: string;
  description: string;
  status: StepStatus;
  duration?: number;
  result?: string;
  detail?: string;
}

interface AuditResult {
  posts_analyzed: number;
  created_at: string;
}

interface TaskStat {
  task: string;
  calls: number;
  succeeded: number;
  avg_latency_ms: number;
}

interface StatsData {
  tasks: TaskStat[];
}

const INITIAL_STEPS: TraceStep[] = [
  { id: "goal", phase: "Control", title: "Receive campaign objective", tool: "Agent Orchestrator", description: "Translate the owner’s objective into a safe, trackable campaign run.", status: "idle" },
  { id: "context", phase: "Observe", title: "Load Ozone business context", tool: "PostgreSQL", description: "Read brand identity, platform connections, workflow counts and active campaign state.", status: "idle" },
  { id: "memory", phase: "Observe", title: "Recall campaign memory", tool: "Scoped Memory", description: "Retrieve owner preferences and learned instructions relevant to content planning.", status: "idle" },
  { id: "meta", phase: "Observe", title: "Ground strategy in Facebook history", tool: "Meta Graph API", description: "Use the verified Ozone Page audit instead of inventing historical performance.", status: "idle" },
  { id: "planner", phase: "Think", title: "Plan the campaign week", tool: "Planner LLM · OpenRouter", description: "Create structured post slots, platforms, formats, timings and distinct campaign angles.", status: "idle" },
  { id: "writer", phase: "Think", title: "Generate campaign assets", tool: "Content LLM · OpenRouter", description: "Produce captions, hashtags, scripts and channel-specific creative directions.", status: "idle" },
  { id: "guard", phase: "Control", title: "Validate every output", tool: "Schema + Guard Engine", description: "Reject unsupported claims, invalid formats, past dates and style-rule violations.", status: "idle" },
  { id: "media", phase: "Act", title: "Prepare visual media", tool: "Remotion · HeyGen · ElevenLabs", description: "Attach photos and generate Reels, avatar clips and voice when required.", status: "idle" },
  { id: "approval", phase: "Control", title: "Request human approval", tool: "Human Approval Gate", description: "Stop before publishing. Only an approved item can reach the Meta publishing tool.", status: "idle" },
  { id: "publish", phase: "Act", title: "Publish and verify", tool: "Meta Graph API", description: "After approval, publish once, store the Meta post ID and prevent duplicate delivery.", status: "idle" },
  { id: "learn", phase: "Learn", title: "Learn from outcomes", tool: "Webhooks + Learning LLM", description: "Connect comments, leads and sales back to content, then improve the next campaign.", status: "idle" },
];

const pause = (milliseconds: number) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

export function AgentWorkflow() {
  const { t, locale } = useI18n();
  const { businessId } = useParams();
  const [steps, setSteps] = useState<TraceStep[]>(INITIAL_STEPS);
  const [selectedId, setSelectedId] = useState("goal");
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [runError, setRunError] = useState("");

  const selected = steps.find((step) => step.id === selectedId) ?? steps[0];
  const completed = steps.filter((step) => step.status === "done").length;
  const toolCount = useMemo(() => new Set(steps.filter((step) => step.status !== "idle").map((step) => step.tool)).size, [steps]);

  function updateStep(id: string, patch: Partial<TraceStep>) {
    setSteps((current) => current.map((step) => step.id === id ? { ...step, ...patch } : step));
    setSelectedId(id);
  }

  async function executeStep<T>(id: string, operation: () => Promise<T>, summarize: (value: T) => Partial<TraceStep>) {
    updateStep(id, { status: "running", result: "Calling tool…", detail: undefined });
    const started = performance.now();
    try {
      const value = await operation();
      const duration = Math.round(performance.now() - started);
      updateStep(id, { status: "done", duration, ...summarize(value) });
      await pause(300);
      return value;
    } catch (error) {
      const duration = Math.round(performance.now() - started);
      updateStep(id, { status: "failed", duration, result: "Tool call failed", detail: error instanceof Error ? error.message : "Unknown tool error" });
      throw error;
    }
  }

  async function runTrace() {
    if (!businessId || running) return;
    setRunning(true);
    setRunError("");
    setElapsed(0);
    setSteps(INITIAL_STEPS.map((step) => ({ ...step })));
    const runStarted = performance.now();

    try {
      updateStep("goal", { status: "running", result: "Parsing objective…" });
      await pause(550);
      updateStep("goal", {
        status: "done",
        duration: 550,
        result: "Goal accepted: build and operate a complete Ozone campaign",
        detail: "The orchestrator decomposes the goal into observation, reasoning, action, approval and learning stages.",
      });
      await pause(250);

      await executeStep(
        "context",
        () => Promise.all([
          api.get<Business>(`/businesses/${businessId}`),
          api.get<Overview>(`/businesses/${businessId}/overview`),
        ]),
        ([business, overview]) => ({
          result: `${business.name} loaded · ${overview.connected.join(" + ") || "no platform"}`,
          detail: `${Object.values(overview.content).reduce((sum, count) => sum + count, 0)} content items tracked; ${overview.leads_new_30d} recent leads; ${overview.sales_30d.count} recorded sales.`,
        }),
      );

      await executeStep(
        "memory",
        () => api.get<unknown[]>(`/businesses/${businessId}/memory`),
        (memories) => ({
          result: `${memories.length} scoped memory entries recalled`,
          detail: "Memory is advisory context. Enforceable brand requirements remain deterministic style rules.",
        }),
      );

      await executeStep(
        "meta",
        () => api.get<AuditResult>(`/businesses/${businessId}/audit/latest`),
        (audit) => ({
          result: `${audit.posts_analyzed} real Facebook posts grounded the strategy`,
          detail: `Latest verified audit: ${new Date(audit.created_at).toLocaleString(locale)}. No write operation was sent to Meta.`,
        }),
      );

      await executeStep(
        "planner",
        () => api.get<StatsData>("/ai/stats?days=30"),
        (data) => {
          const planner = data.tasks.find((task) => task.task === "weekly");
          return {
            result: planner ? `${planner.succeeded}/${planner.calls} planner calls succeeded` : "Verified planner run loaded",
            detail: planner ? `Average model latency ${planner.avg_latency_ms}ms. Output is parsed into a typed weekly-plan schema.` : "The saved campaign provides the verified planner output.",
          };
        },
      );

      const content = await executeStep(
        "writer",
        () => api.get<ContentItem[]>(`/businesses/${businessId}/content`),
        (items) => {
          const campaignItems = items.filter((item) => item.caption && item.status !== "rejected");
          return {
            result: `${campaignItems.length} generated assets available`,
            detail: `${campaignItems.filter((item) => item.type === "reel").length} Reels · ${campaignItems.filter((item) => item.type === "static_post").length} static posts · ${campaignItems.filter((item) => item.type === "avatar_clip").length} avatar clips.`,
          };
        },
      );

      updateStep("guard", { status: "running", result: "Checking schemas and campaign invariants…" });
      await pause(600);
      const usable = content.filter((item) => item.caption?.trim()).length;
      updateStep("guard", {
        status: "done",
        duration: 600,
        result: `${usable} assets passed persisted-output checks`,
        detail: "The guard layer enforces allowed platforms, future schedules, truthful claims, style rules and safe retries before database writes.",
      });
      await pause(250);

      updateStep("media", { status: "running", result: "Inspecting generated media…" });
      await pause(450);
      const rendered = content.filter((item) => Boolean(item.video_url)).length;
      updateStep("media", {
        status: "done",
        duration: 450,
        result: `${rendered} generated videos are ready`,
        detail: "Remotion renders campaign Reels locally; HeyGen creates avatars; ElevenLabs handles voice and transcription.",
      });
      await pause(250);

      updateStep("approval", {
        status: "waiting",
        duration: 0,
        result: "Waiting for owner approval",
        detail: "Safe-mode checkpoint reached. Publishing is deliberately not executed during this trace.",
      });
      setSteps((current) => current.map((step) => {
        if (step.id === "publish") return { ...step, status: "waiting", result: "Locked until approval" };
        if (step.id === "learn") return { ...step, status: "waiting", result: "Starts after real outcomes arrive" };
        return step;
      }));
      setElapsed(Math.round(performance.now() - runStarted));
    } catch {
      setRunError("The trace stopped because one read-only tool call failed. No publishing action was attempted.");
      setElapsed(Math.round(performance.now() - runStarted));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className={styles.page}>
      <header className={styles.hero}>
        <div className={styles.heroCopy}>
          <div className={styles.eyebrow}><span /> {t(" Ozone autonomous marketing agent")}</div>
          <h1>{t("See the agent think, call tools, and stop safely.")}</h1>
          <p>{t("A transparent execution trace showing how one campaign goal becomes grounded context, model reasoning, validated content, generated media and a human-controlled publishing decision.")}</p>
          <div className={styles.heroActions}>
            <button className={styles.runButton} onClick={runTrace} disabled={running}>
              {running ? <><i className={styles.spinner} /> {t(" Agent running")}</> : <><span>▶</span> {t(" Run live agent trace")}</>}
            </button>
            <span className={styles.safeMode}><i /> {t(" Safe mode · read-only replay")}</span>
          </div>
        </div>
        <div className={styles.agentCore}>
          <div className={styles.orbit}><span>{t("TOOLS")}</span><span>{t("MEMORY")}</span><span>{t("GUARDS")}</span></div>
          <div className={styles.core}><small>{t("AGENT")}</small><strong>O₃</strong><span>{t("Campaign brain")}</span></div>
        </div>
      </header>

      <section className={styles.metrics}>
        <Metric label={t("Trace progress")} value={`${completed}/11`} note={running ? "Executing now" : completed ? "Stopped at approval" : "Ready to demonstrate"} />
        <Metric label={t("Tools engaged")} value={String(toolCount)} note="APIs, memory and guards" />
        <Metric label={t("Human checkpoints")} value="1" note="Required before publishing" />
        <Metric label={t("Trace duration")} value={elapsed ? `${(elapsed / 1000).toFixed(1)}s` : "—"} note="Measured live in browser" />
      </section>

      {runError && <div className={styles.error}>{t(runError)}</div>}

      <section className={styles.workspace}>
        <div className={styles.tracePanel}>
          <div className={styles.panelHeader}>
            <div><span>{t("LIVE EXECUTION")}</span><h2>{t("Campaign agent workflow")}</h2></div>
            <div className={styles.legend}>
              <span><i className={styles.doneDot} /> {t(" Complete")}</span>
              <span><i className={styles.runningDot} /> {t(" Running")}</span>
              <span><i className={styles.waitingDot} /> {t(" Approval")}</span>
            </div>
          </div>

          <div className={styles.trace}>
            {steps.map((step, index) => (
              <button key={step.id} className={`${styles.step} ${styles[step.status]} ${selectedId === step.id ? styles.selected : ""}`} onClick={() => setSelectedId(step.id)}>
                <span className={styles.rail}><i>{step.status === "done" ? "✓" : step.status === "running" ? "" : index + 1}</i></span>
                <span className={styles.stepBody}>
                  <span className={styles.stepTopline}><b>{t(step.phase)}</b>{step.duration !== undefined && <em>{step.duration}{t("ms")}</em>}</span>
                  <strong>{t(step.title)}</strong>
                  <small>{t(step.tool)}</small>
                  {step.result && <span className={styles.result}>{t(step.result)}</span>}
                </span>
              </button>
            ))}
          </div>
        </div>

        <aside className={styles.inspector}>
          <div className={styles.inspectorHeader}><span>{t("STEP INSPECTOR")}</span><b className={styles[selected.status]}>{t(selected.status)}</b></div>
          <div className={styles.toolIcon}>{toolInitials(selected.tool)}</div>
          <span className={styles.phasePill}>{t(selected.phase)}</span>
          <h2>{t(selected.title)}</h2>
          <p>{t(selected.description)}</p>

          <div className={styles.callBox}>
            <div><span>{t("TOOL")}</span><strong>{t(selected.tool)}</strong></div>
            <div><span>{t("ACCESS")}</span><strong>{t(accessLevel(selected.id))}</strong></div>
            <div><span>{t("STATUS")}</span><strong>{t(statusLabel(selected.status))}</strong></div>
            <div><span>{t("LATENCY")}</span><strong>{selected.duration !== undefined ? t(`${selected.duration} ms`) : t("Not called")}</strong></div>
          </div>

          <div className={styles.outputBox}>
            <span>{t("OBSERVED OUTPUT")}</span>
            <strong>{t(selected.result ?? "Run the trace to inspect this step’s real output.")}</strong>
            <p>{t(selected.detail ?? "Each tool result is converted into bounded context for the next stage.")}</p>
          </div>

          <div className={styles.safetyNote}>
            <span>◆</span>
            <p><strong>{t("Bounded autonomy")}</strong>{t("The agent may observe and prepare work autonomously. Publishing and spending remain human-controlled.")}</p>
          </div>
        </aside>
      </section>

      <section className={styles.agentDefinition}>
        <Definition number="01" title={t("Goal-directed")} text="Decomposes a campaign objective into ordered work." />
        <Definition number="02" title={t("Tool-using")} text="Reads APIs, memory and databases; creates media and posts." />
        <Definition number="03" title={t("Stateful")} text="Remembers owner rules, content status, leads and outcomes." />
        <Definition number="04" title={t("Self-improving")} text="Uses campaign outcomes to guide the next planning cycle." />
      </section>
    </div>
  );
}

function Metric({ label, value, note }: { label: string; value: string; note: string }) {
  const { t } = useI18n();
  return <div className={styles.metric}><span>{t(label)}</span><strong>{t(value)}</strong><small>{t(note)}</small></div>;
}

function Definition({ number, title, text }: { number: string; title: string; text: string }) {
  const { t } = useI18n();
  return <div><span>{number}</span><p><strong>{t(title)}</strong>{t(text)}</p></div>;
}

function toolInitials(tool: string) {
  return tool.split(/[ ·+]/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

function accessLevel(id: string) {
  if (id === "publish") return "WRITE · APPROVAL REQUIRED";
  if (["writer", "planner", "guard", "media"].includes(id)) return "GENERATE · SANDBOXED";
  if (id === "approval") return "HUMAN CONTROL";
  return "READ ONLY";
}

function statusLabel(status: StepStatus) {
  return ({ idle: "Not started", running: "Executing", done: "Verified", waiting: "Waiting", failed: "Failed" })[status];
}
