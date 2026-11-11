import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { ar } from "./ar";

export type Language = "fr" | "ar";
const normalize = (value: string) => value.replace(/\s+/g, " ").trim();
const dictionary = new Map(Object.entries(ar).map(([key, value]) => [normalize(key), value]));
const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const patterns = [...dictionary].filter(([key]) => /\{\d+\}/.test(key)).map(([key, value]) => {
  const indices: string[] = [];
  const pattern = key.split(/(\{\d+\})/).map(part => {
    if (/^\{\d+\}$/.test(part)) {
      indices.push(part);
      return key === "{0}m" || key === "{0}h" || key === "{0}d" || key === "{0}s" ? "(\\d+(?:\\.\\d+)?)" : "(.*?)";
    }
    return escapeRegex(part);
  }).join("");
  return { regex: new RegExp(`^${pattern}$`), value, indices };
});

export function translate(text: string, language: Language): string {
  if (language !== "ar" || !text) return text;
  const key = normalize(text);
  const exact = dictionary.get(key);
  let result = exact;
  if (result === undefined) {
    for (const pattern of patterns) {
      const match = pattern.regex.exec(key);
      if (match) {
        result = pattern.value.replace(/\{\d+\}/g, placeholder => match[pattern.indices.indexOf(placeholder) + 1] ?? placeholder);
        break;
      }
    }
  }
  if (result === undefined) return text;
  return `${text.match(/^\s*/)?.[0] ?? ""}${result}${text.match(/\s*$/)?.[0] ?? ""}`;
}

function savedLanguage(): Language {
  try { return localStorage.getItem("lang") === "ar" ? "ar" : "fr"; }
  catch { return "fr"; }
}

type I18n = { lang: Language; locale: string; setLang: (lang: Language) => void; t: (text: string) => string };
const Context = createContext<I18n | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Language>(savedLanguage);
  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
    try { localStorage.setItem("lang", lang); } catch { /* Private browsing may disable storage. */ }
  }, [lang]);
  const value = useMemo<I18n>(() => ({ lang, setLang, locale: lang === "ar" ? "ar-TN" : "fr-TN", t: text => translate(text, lang) }), [lang]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}

export function useI18n(): I18n {
  const context = useContext(Context);
  if (!context) throw new Error("useI18n requires I18nProvider");
  return context;
}

export function LanguageSwitch() {
  const { lang, setLang } = useI18n();
  return <div className="language-switch" role="group" aria-label={lang === "ar" ? "لغة الواجهة" : "Interface language"}>
    <button type="button" lang="fr" dir="ltr" aria-pressed={lang === "fr"} onClick={() => setLang("fr")}>FR</button>
    <button type="button" lang="ar" dir="rtl" aria-pressed={lang === "ar"} onClick={() => setLang("ar")}>العربية</button>
  </div>;
}
