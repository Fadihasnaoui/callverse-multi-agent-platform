import { LanguageSwitch, useI18n } from "@/i18n";
import { useState, type FormEvent } from "react";
import { useAuth } from "@/auth";
import styles from "./Login.module.css";

export function Login() {
  const { t } = useI18n();
  const { login } = useAuth();
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.page}>
      <form className={styles.form} onSubmit={handleSubmit}>
        <LanguageSwitch />
        <h1 className={styles.title}>{t("Ozone Campaign OS")}</h1>
        <p className={styles.subtitle}>{t("Sign in to manage the Ozone Voyages growth campaign")}</p>
        <input
          type="password"
          placeholder={t("Password")}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoFocus
        />
        {error && <p className={styles.error}>{t(error)}</p>}
        <button className="btn-primary" type="submit" disabled={loading || !password}>
          {loading ? t("Signing in…") : t("Sign in")}
        </button>
      </form>
    </div>
  );
}
