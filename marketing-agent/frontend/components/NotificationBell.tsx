import { useI18n } from "@/i18n";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useParams, useNavigate } from "react-router-dom";
import { api } from "@/api";
import styles from "./NotificationBell.module.css";

interface Notification {
  id: string;
  event_type: string;
  title: string;
  body: string;
  meta: Record<string, unknown>;
  read: boolean;
  created_at: string;
}

const EVENT_ICON: Record<string, string> = {
  lead_new: "🎯",
  lead_converted: "💰",
  content_published: "✓",
  content_failed: "✗",
  content_generated: "📝",
  week_planned: "📅",
  learning_updated: "🧠",
  audit_completed: "🔍",
  competitor_refreshed: "📊",
};

function eventRoute(type: string): string {
  if (type.startsWith("lead")) return "leads";
  if (type.startsWith("content") || type === "week_planned") return "content";
  if (type === "learning_updated") return "report";
  if (type === "audit_completed") return "";
  if (type === "competitor_refreshed") return "";
  return "";
}

function timeAgo(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "now";
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  return `${Math.floor(hrs / 24)}d`;
}

export function NotificationBell() {
  const { t } = useI18n();
  const { businessId } = useParams();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unread, setUnread] = useState(0);

  useEffect(() => {
    if (!businessId) return;
    const load = () => {
      api.get<{ unread: number }>(`/businesses/${businessId}/notifications/count`)
        .then((d) => setUnread(d.unread))
        .catch(() => {});
    };
    load();
    const interval = setInterval(load, 30000);
    return () => clearInterval(interval);
  }, [businessId]);

  useEffect(() => {
    if (open && businessId) {
      api.get<Notification[]>(`/businesses/${businessId}/notifications?limit=30`)
        .then(setNotifications)
        .catch(() => {});
    }
  }, [open, businessId]);

  async function markAllRead() {
    if (!businessId) return;
    await api.post(`/businesses/${businessId}/notifications/read-all`);
    setUnread(0);
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  }

  async function handleClick(n: Notification) {
    if (!businessId) return;
    if (!n.read) {
      api.post(`/businesses/${businessId}/notifications/${n.id}/read`).catch(() => {});
      setUnread((c) => Math.max(0, c - 1));
      setNotifications((prev) => prev.map((x) => x.id === n.id ? { ...x, read: true } : x));
    }
    const route = eventRoute(n.event_type);
    setOpen(false);
    navigate(`/c/${businessId}/${route}`);
  }

  return (
    <>
      <button className={styles.bellBtn} onClick={() => setOpen(!open)} title={t("Notifications")}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
          <path d="M13.73 21a2 2 0 0 1-3.46 0" />
        </svg>
        {unread > 0 && <span className={styles.badge}>{unread > 99 ? "99+" : unread}</span>}
      </button>

      {open && createPortal(
        <>
          <div className={styles.overlay} onClick={() => setOpen(false)} />
          <div className={styles.panel}>
            <div className={styles.panelHeader}>
              <h3>{t("Notifications")}</h3>
              <div className={styles.headerActions}>
                {unread > 0 && (
                  <button className={styles.markAllBtn} onClick={markAllRead}>{t("Mark all read")}</button>
                )}
                <button className={styles.closeBtn} onClick={() => setOpen(false)}>×</button>
              </div>
            </div>
            <div className={styles.panelList}>
              {notifications.length === 0 && (
                <p className={styles.emptyState}>{t("No notifications yet")}</p>
              )}
              {notifications.map((n) => (
                <div
                  key={n.id}
                  className={`${styles.notifItem} ${!n.read ? styles.unread : ""}`}
                  onClick={() => handleClick(n)}
                >
                  <span className={styles.notifIcon}>{EVENT_ICON[n.event_type] || "•"}</span>
                  <div className={styles.notifContent}>
                    <p className={styles.notifTitle}>{t(n.title)}</p>
                    {n.body && <p className={styles.notifBody}>{t(n.body)}</p>}
                  </div>
                  <span className={styles.notifTime}>{t(timeAgo(n.created_at))}</span>
                </div>
              ))}
            </div>
          </div>
        </>,
        document.body,
      )}
    </>
  );
}
