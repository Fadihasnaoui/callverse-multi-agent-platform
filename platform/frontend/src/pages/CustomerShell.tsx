import { useI18n } from "@/i18n";
import { NavLink, Outlet, useParams } from "react-router-dom";
import { NotificationBell } from "@/components/NotificationBell";
import { WorkspaceHeader } from "@/pages/AgentHub";
import styles from "./CustomerShell.module.css";

const NAV = [
  { to: "", label: "Dashboard", icon: "D" },
  { to: "content", label: "Content", icon: "C" },
  { to: "calendar", label: "Calendar", icon: "K" },
  { to: "leads", label: "Leads", icon: "L" },
  { to: "report", label: "Report", icon: "R" },
  { to: "audit", label: "Audit", icon: "A" },
  { to: "catalogue", label: "Catalogue", icon: "H" },
  { to: "costs", label: "AI Costs", icon: "$" },
  { to: "settings", label: "Settings", icon: "S" },
];

export function CustomerShell() {
  const { t } = useI18n();
  const { businessId } = useParams();

  return <><WorkspaceHeader />
    <div className={styles.layout}>
      <aside className={styles.sidebar}>
        <NavLink to={`/c/${businessId}`} className={styles.logo}>
          <span className={styles.logoMark}>O</span>
          <span><strong>Ozone Voyages</strong><small>{t("Campaign OS")}</small></span>
        </NavLink>
        <nav className={styles.nav} aria-label="Marketing workspace navigation">
          {[...NAV, { to: "agent-workflow", label: "Agent Workflow", icon: "AG" }].map((n) => (
            <NavLink key={n.to} to={`/c/${businessId}/${n.to}`} end={n.to === ""}
              className={({ isActive }) => `${styles.link} ${isActive ? styles.active : ""}`}>
              <span>{n.icon}</span> {t(n.label)}
            </NavLink>
          ))}
        </nav>
        <div className={styles.sidebarFooter}><NotificationBell /></div>
      </aside>
      <main className={styles.main}><Outlet /></main>
    </div>
  </>;
}
