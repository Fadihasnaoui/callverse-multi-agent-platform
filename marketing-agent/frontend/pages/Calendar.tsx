import { useI18n } from "@/i18n";
import { useEffect, useState, useCallback, type DragEvent } from "react";
import { useParams } from "react-router-dom";
import { api, ApiError } from "@/api";
import type { ContentItem } from "@/types";
import styles from "./Calendar.module.css";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const HOURS = Array.from({ length: 14 }, (_, i) => i + 8); // 08:00 – 21:00

function weekDates(offset: number): Date[] {
  const now = new Date();
  const day = now.getDay();
  const mon = new Date(now);
  mon.setDate(now.getDate() - ((day + 6) % 7) + offset * 7);
  mon.setHours(0, 0, 0, 0);
  return Array.from({ length: 7 }, (_, i) => {
    const d = new Date(mon);
    d.setDate(mon.getDate() + i);
    return d;
  });
}

function fmt(d: Date) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

interface PlanWeekPayload {
  main_idea: string;
  week_start: string;
  posts: number;
  platforms: string[];
  content_types: string[];
  generate_images: boolean;
}

interface PlanWeekResult {
  created: { id: string; image_status: string }[];
  failed: { topic: string; error: string }[];
  media_failed: { id: string; topic: string; error: string }[];
}

export function Calendar() {
  const { t, locale } = useI18n();
  const { businessId } = useParams();
  const [items, setItems] = useState<ContentItem[]>([]);
  const [weekOffset, setWeekOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [dragging, setDragging] = useState<string | null>(null);
  const [planOpen, setPlanOpen] = useState(false);
  const [planIdea, setPlanIdea] = useState("");
  const [planPosts, setPlanPosts] = useState(5);
  const [planning, setPlanning] = useState(false);
  const [error, setError] = useState("");
  const [planSummary, setPlanSummary] = useState("");
  const [selected, setSelected] = useState<ContentItem | null>(null);
  const [editing, setEditing] = useState(false);
  const [editCaption, setEditCaption] = useState("");

  const dates = weekDates(weekOffset);

  const load = useCallback(() => {
    setLoading(true);
    api.get<ContentItem[]>(`/businesses/${businessId}/content`)
      .then(setItems)
      .finally(() => setLoading(false));
  }, [businessId]);

  useEffect(load, [load]);

  // Shelf: items without a schedule (pending, no scheduled_for)
  const shelf = items.filter((i) => !i.scheduled_for && i.status === "pending");

  // Calendar slots: items scheduled in this week

  function slotItems(dayIdx: number, hour: number) {
    const d = dates[dayIdx];
    return items.filter((i) => {
      if (!i.scheduled_for) return false;
      const s = new Date(i.scheduled_for);
      return s.getFullYear() === d.getFullYear() &&
        s.getMonth() === d.getMonth() &&
        s.getDate() === d.getDate() &&
        s.getHours() === hour;
    });
  }

  function onDragStart(e: DragEvent, id: string) {
    e.dataTransfer.setData("text/plain", id);
    setDragging(id);
  }

  function onDragOver(e: DragEvent) {
    e.preventDefault();
  }

  async function onDrop(e: DragEvent, dayIdx: number, hour: number) {
    e.preventDefault();
    const id = e.dataTransfer.getData("text/plain");
    setDragging(null);
    if (!id) return;

    const d = dates[dayIdx];
    const local = `${fmt(d)}T${String(hour).padStart(2, "0")}:00`;

    try {
      await api.post(`/businesses/${businessId}/content/${id}/schedule`, { local_datetime: local });
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Schedule failed");
      setTimeout(() => setError(""), 4000);
    }
  }

  async function unschedule(id: string) {
    try {
      await api.post(`/businesses/${businessId}/content/${id}/schedule`, { local_datetime: null });
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Unschedule failed");
      setTimeout(() => setError(""), 4000);
    }
  }

  async function deleteItem(id: string) {
    try {
      await api.del(`/businesses/${businessId}/content/${id}`);
      setSelected(null);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Delete failed");
      setTimeout(() => setError(""), 4000);
    }
  }

  async function saveEdit() {
    if (!selected) return;
    try {
      await api.patch(`/businesses/${businessId}/content/${selected.id}`, { caption: editCaption });
      setEditing(false);
      setSelected({ ...selected, caption: editCaption });
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Save failed");
      setTimeout(() => setError(""), 4000);
    }
  }

  async function planWeek() {
    if (!planIdea.trim()) return;
    setPlanning(true);
    setError("");
    try {
      const payload: PlanWeekPayload = {
        main_idea: planIdea.trim(),
        week_start: fmt(dates[0]),
        posts: planPosts,
        platforms: ["facebook", "instagram"],
        content_types: ["static_post", "reel"],
        generate_images: false,
      };
      const result = await api.post<PlanWeekResult>(`/businesses/${businessId}/content/plan-week`, payload);
      setPlanSummary(`${result.created.length} posts planned. All posts are waiting for approval.`);
      setPlanOpen(false);
      setPlanIdea("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Plan failed");
    } finally {
      setPlanning(false);
    }
  }

  return (
    <div>
      <div className={styles.header}>
        <h2>{t("Calendar")}</h2>
        <div className={styles.weekNav}>
          <button className="btn-ghost" onClick={() => setWeekOffset((w) => w - 1)}>{t("← Prev")}</button>
          <span className={styles.weekLabel}>
            {dates[0].toLocaleDateString(locale, { day: "numeric", month: "short" })}
            {" – "}
            {dates[6].toLocaleDateString(locale, { day: "numeric", month: "short", year: "numeric" })}
          </span>
          <button className="btn-ghost" onClick={() => setWeekOffset((w) => w + 1)}>{t("Next →")}</button>
          {weekOffset !== 0 && (
            <button className="btn-ghost" onClick={() => setWeekOffset(0)}>{t("Today")}</button>
          )}
        </div>
        <button className="btn-primary" onClick={() => setPlanOpen(!planOpen)}>
          {planOpen ? t("Cancel") : t("AI Plan Week")}
        </button>
      </div>

      {error && <p className={styles.error}>{t(error)}</p>}
      {planSummary && <p className={styles.success}>{t(planSummary)}</p>}

      {planOpen && (
        <div className={`card ${styles.planForm}`}>
          <label>
            {t("Main idea")}<input
              value={planIdea}
              onChange={(e) => setPlanIdea(e.target.value)}
              placeholder={t("e.g. Autumn in Tunisia — best hotels with heated pools")}
              style={{ width: "100%" }}
            />
          </label>
          <label>
            {t("Posts")}<select value={planPosts} onChange={(e) => setPlanPosts(Number(e.target.value))}>
              {[3, 4, 5, 6, 7].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <button className="btn-primary" onClick={planWeek} disabled={planning || !planIdea.trim()}>
            {planning ? t("Planning…") : t("Generate week")}
          </button>
          <small className={styles.planCost}>{t("Posts are generated without images. Use \"Generate AI visual\" on the Content page to add visuals individually.")}</small>
        </div>
      )}

      {/* Video shelf */}
      <section className={styles.shelf}>
        <h3 className={styles.shelfTitle}>{t("Unscheduled (")}{shelf.length})</h3>
        <div className={styles.shelfItems}>
          {shelf.length === 0 && <p className={styles.muted}>{t("No unscheduled content. Use \"AI Plan Week\" to generate some.")}</p>}
          {shelf.map((item) => (
            <div
              key={item.id}
              className={`card ${styles.shelfCard} ${dragging === item.id ? styles.dragging : ""}`}
              draggable
              onDragStart={(e) => onDragStart(e, item.id)}
            >
              <span className={styles.itemMeta}>{t(item.platform)} · {t(item.type)}</span>
              <p className={styles.itemCaption}>{(item.caption || "").slice(0, 80)}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Week grid */}
      {loading ? <p>{t("Loading…")}</p> : (
        <div className={styles.grid}>
          {/* Header row */}
          <div className={styles.corner} />
          {DAYS.map((d, i) => (
            <div key={d} className={styles.dayHeader}>
              {t(d)} {dates[i].getDate()}
            </div>
          ))}

          {/* Hour rows */}
          {HOURS.map((h) => (
            <>
              <div key={`h-${h}`} className={styles.hourLabel}>{String(h).padStart(2, "0")}:00</div>
              {DAYS.map((_, dayIdx) => {
                const inSlot = slotItems(dayIdx, h);
                return (
                  <div
                    key={`${dayIdx}-${h}`}
                    className={styles.cell}
                    onDragOver={onDragOver}
                    onDrop={(e) => onDrop(e, dayIdx, h)}
                  >
                    {inSlot.map((item) => (
                      <div
                        key={item.id}
                        className={`${styles.slotCard} ${styles[`slot-${item.status}`] || ""}`}
                        draggable
                        onDragStart={(e) => onDragStart(e, item.id)}
                        onClick={() => setSelected(item)}
                        title={t("Open post preview")}
                      >
                        {item.photo_urls?.[0] && <img className={styles.slotThumb} src={item.photo_urls[0]} alt="" />}
                        <span className={styles.slotMeta}>{t(item.platform)}</span>
                        <p className={styles.slotCaption}>{(item.caption || "").slice(0, 40)}</p>
                        {item.status !== "published" && item.status !== "publishing" && (
                          <button className={styles.unscheduleBtn} onClick={(event) => { event.stopPropagation(); unschedule(item.id); }} title={t("Remove from calendar")}>×</button>
                        )}
                      </div>
                    ))}
                  </div>
                );
              })}
            </>
          ))}
        </div>
      )}

      {selected && (
        <div className={styles.modalBackdrop} onClick={() => setSelected(null)}>
          <article className={styles.preview} onClick={(event) => event.stopPropagation()}>
            <button className={styles.closeBtn} onClick={() => setSelected(null)} aria-label={t("Close preview")}>×</button>
            <div className={styles.previewMedia}>
              {selected.video_url ? <video src={selected.video_url} controls /> : selected.photo_urls?.length ? (
                <img src={selected.photo_urls[0]} alt={t("Planned campaign visual")} />
              ) : <div className={styles.mediaPlaceholder}>{t("Visual generation pending")}</div>}
            </div>
            <div className={styles.previewBody}>
              <div className={styles.previewTopline}>
                <span>{t(selected.platform)}</span><span>{t(selected.type)}</span><span>{t(selected.status)}</span>
              </div>
              <h3>{t("Planned post")}</h3>
              {editing ? (
                <div>
                  <textarea className={styles.editArea} value={editCaption} onChange={(e) => setEditCaption(e.target.value)} />
                  <div className={styles.editActions}>
                    <button className="btn-primary" onClick={saveEdit}>{t("Save")}</button>
                    <button className="btn-ghost" onClick={() => setEditing(false)}>{t("Cancel")}</button>
                  </div>
                </div>
              ) : (
                <p className={styles.fullCaption}>{selected.caption}</p>
              )}
              {selected.hashtags?.length > 0 && <p className={styles.hashtags}>{selected.hashtags.map((tag) => `#${tag}`).join(" ")}</p>}
              <div className={styles.previewDetails}>
                <span>{t("Scheduled")}</span>
                <strong>{selected.scheduled_for ? new Date(selected.scheduled_for).toLocaleString(locale) : t("Not scheduled")}</strong>
              </div>
              <p className={styles.approvalNote}>{t("Human approval is still required before this post can be published.")}</p>
              {selected.status !== "published" && selected.status !== "publishing" && (
                <div className={styles.previewActions}>
                  <button className={styles.editBtn} onClick={() => { setEditCaption(selected.caption || ""); setEditing(true); }}>{t("Edit caption")}</button>
                  {selected.scheduled_for && (
                    <button className="btn-ghost" onClick={() => unschedule(selected.id)}>{t("Remove from calendar")}</button>
                  )}
                  <button className={styles.deleteBtn} onClick={() => { if (confirm(t("Delete this post?"))) deleteItem(selected.id); }}>{t("Delete post")}</button>
                </div>
              )}
            </div>
          </article>
        </div>
      )}
    </div>
  );
}
