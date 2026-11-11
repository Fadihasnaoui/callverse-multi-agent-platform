import { useEffect, useState } from "react";

type Theme = "light" | "dark";

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const saved = localStorage.getItem("ozone-theme");
      if (saved === "light" || saved === "dark") return saved;
      return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    catch { return "light"; }
  });
  useEffect(() => { document.documentElement.dataset.theme = theme; try { localStorage.setItem("ozone-theme", theme); } catch {} }, [theme]);
  const next = theme === "dark" ? "light" : "dark";
  return <button className="theme-toggle" onClick={() => setTheme(next)} title={`Switch to ${next} mode`} aria-label={`Switch to ${next} mode`}>
    <span aria-hidden="true">{theme === "dark" ? "☀" : "☾"}</span>{theme === "dark" ? "Light" : "Dark"}
  </button>;
}
