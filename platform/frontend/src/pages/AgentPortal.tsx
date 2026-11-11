import { WorkspaceHeader } from "@/pages/AgentHub";
import styles from "./AgentPortal.module.css";

type PortalProps = { title: string; description: string; source: string; label: string };

export function AgentPortal({ title, description, source, label }: PortalProps) {
  return <div className={styles.page}><WorkspaceHeader />
    <main className={styles.shell}>
      <section className={styles.hero}>
        <div className={styles.heroCopy}>
          <p>OZONE DIGITAL TWIN / INDIVIDUAL AGENT</p>
          <h1>{title}</h1>
          <span>{description}</span>
        </div>
        <div className={styles.heroStatus} aria-label="Workspace status"><i /> Original workspace available</div>
      </section>
      <section className={styles.frame}>
        <div className={styles.frameHead}>
          <div><strong>{label}</strong><small>Original workspace · preserved as built</small></div>
          <span>Embedded in Ozone Digital Twin</span>
        </div>
        <iframe title={title} src={source} />
      </section>
    </main>
  </div>;
}
