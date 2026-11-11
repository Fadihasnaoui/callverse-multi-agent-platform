import { useI18n } from "@/i18n";
import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { api, ApiError } from "@/api";
import styles from "./Settings.module.css";

type FacebookPageOption = { page_id: string; page_name?: string; instagram?: { ig_user_id: string; username?: string } };

/* ---- Style Rules ---- */
interface StyleRules {
  banned_words: string[];
  max_emojis: number;
  end_with_question: boolean;
  max_caption_chars: number;
  max_hashtags: number;
}

const DEFAULT_RULES: StyleRules = {
  banned_words: [],
  max_emojis: 2,
  end_with_question: false,
  max_caption_chars: 500,
  max_hashtags: 10,
};

/* ---- Glossary ---- */
interface GlossaryPair {
  id: string;
  wrong: string;
  correct: string;
  times_seen: number;
}

/* ---- Memory ---- */
interface MemoryEntry {
  id: string;
  scope: string;
  task: string | null;
  content: string;
  source: string;
  pinned: boolean;
  created_at: string;
}

const SCOPES = ["all", "global", "content", "scheduling", "voice", "leads", "learning"];
const SCOPE_HELP: Record<string, string> = {
  all: "Everything the agent currently remembers.", global: "Audience, brand and non-negotiable business facts.",
  content: "Caption preferences and lessons from approvals or edits.", scheduling: "Preferred timing and cadence.",
  voice: "Tunisian pronunciation and transcript corrections.", leads: "Examples of real versus false lead intent.",
  learning: "Performance conclusions and experiment hypotheses.",
};

type Tab = "business" | "platforms" | "assets" | "rules" | "glossary" | "memory";
const TAB_LABELS: Record<Tab, string> = {
  business: "Business",
  platforms: "Platforms",
  assets: "Photos",
  rules: "Style Rules",
  glossary: "Derja Glossary",
  memory: "Memory",
};

export function Settings() {
  const { t } = useI18n();
  const { businessId } = useParams();
  const [tab, setTab] = useState<Tab>("business");

  return (
    <div>
      <div className={styles.tabs}>
        {(Object.keys(TAB_LABELS) as Tab[]).map((tabKey) => (
          <button
            key={tabKey}
            className={`btn-ghost ${tab === tabKey ? styles.activeTab : ""}`}
            onClick={() => setTab(tabKey)}
          >
            {t(TAB_LABELS[tabKey])}
          </button>
        ))}
      </div>

      {tab === "business" && <BusinessTab businessId={businessId!} />}
      {tab === "platforms" && <PlatformsTab businessId={businessId!} />}
      {tab === "assets" && <AssetsTab />}
      {tab === "rules" && <RulesTab businessId={businessId!} />}
      {tab === "glossary" && <GlossaryTab businessId={businessId!} />}
      {tab === "memory" && <MemoryTab businessId={businessId!} />}
    </div>
  );
}

/* ========== Business Info Tab ========== */
function BusinessTab({ businessId }: { businessId: string }) {
  const { t } = useI18n();
  const [name, setName] = useState("");
  const [industry, setIndustry] = useState("");
  const [brandVoice, setBrandVoice] = useState("");
  const [status, setStatus] = useState("onboarding");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    api.get<{ name: string; industry: string; brand_voice: string; status: string }>(`/businesses/${businessId}`)
      .then((b) => { setName(b.name); setIndustry(b.industry); setBrandVoice(b.brand_voice); setStatus(b.status); });
  }, [businessId]);

  async function save() {
    setSaving(true);
    setMsg("");
    try {
      await api.put(`/businesses/${businessId}`, { name, industry, brand_voice: brandVoice, status });
      setMsg("Saved ✓");
      setTimeout(() => setMsg(""), 2000);
    } catch (err) {
      setMsg(err instanceof ApiError ? err.body : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={styles.section}>
      <h3>{t("Business Details")}</h3>
      <div className={styles.formGrid}>
        <label>
          {t("Name")}<input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <label>
          {t("Industry")}<input value={industry} onChange={(e) => setIndustry(e.target.value)} placeholder={t("e.g. travel, hotel, restaurant")} />
        </label>
        <label>
          {t("Status")}<select value={status} onChange={(e) => setStatus(e.target.value)}>
            <option value="onboarding">{t("Onboarding")}</option>
            <option value="active">{t("Active")}</option>
            <option value="paused">{t("Paused")}</option>
          </select>
        </label>
      </div>
      <label className={styles.fullField}>
        {t("Brand voice")}<textarea
          value={brandVoice}
          onChange={(e) => setBrandVoice(e.target.value)}
          rows={3}
          placeholder={t("Describe how this brand speaks (e.g. chaleureux, professionnel, jeune…)")}
        />
      </label>
      <div className={styles.saveRow}>
        <button className="btn-primary" onClick={save} disabled={saving}>{saving ? t("Saving…") : t("Save")}</button>
        {msg && <span className={styles.msg}>{t(msg)}</span>}
      </div>
    </div>
  );
}

/* ========== Platforms Tab ========== */
function PlatformsTab({ businessId }: { businessId: string }) {
  const { t } = useI18n();
  const [platform, setPlatform] = useState("facebook");
  const [token, setToken] = useState("");
  const [pageId, setPageId] = useState("");
  const [igId, setIgId] = useState("");
  const [pages, setPages] = useState<FacebookPageOption[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [connected, setConnected] = useState<string[]>([]);

  useEffect(() => {
    api.get<{ connected: string[] }>(`/businesses/${businessId}/overview`)
      .then((o) => setConnected(o.connected));
  }, [businessId]);

  async function connect() {
    if (!token) return;
    if (platform === "facebook" && !pageId.trim()) { setMsg("Facebook Page ID is required for live metrics"); return; }
    if (platform === "instagram" && !igId.trim()) { setMsg("Instagram User ID is required for live metrics"); return; }
    setSaving(true);
    setMsg("");
    const meta: Record<string, string> = {};
    if (platform === "facebook" && pageId) meta.page_id = pageId;
    if (platform === "instagram" && igId) meta.ig_user_id = igId;
    try {
      await api.post(`/businesses/${businessId}/credentials`, {
        platform,
        access_token: token,
        meta,
      });
      setToken("");
      setPageId("");
      setIgId("");
      setMsg(`${platform} connected ✓`);
      setConnected((c) => c.includes(platform) ? c : [...c, platform]);
      setTimeout(() => setMsg(""), 3000);
    } catch (err) {
      setMsg(err instanceof ApiError ? err.body : "Connect failed");
    } finally {
      setSaving(false);
    }
  }

  async function discoverPages() {
    if (!token) return;
    setDiscovering(true); setMsg("");
    try {
      const result = await api.post<{ pages: FacebookPageOption[] }>(`/businesses/${businessId}/credentials/discover`, {
        platform: "facebook", access_token: token, meta: {},
      });
      setPages(result.pages || []);
      if (!(result.pages || []).length) setMsg("No Facebook Pages were found for this token");
    } catch (err) { setMsg(err instanceof ApiError ? err.body : "Could not discover Facebook Pages"); }
    finally { setDiscovering(false); }
  }

  return (
    <div className={styles.section}>
      <h3>{t("Connect Platforms")}</h3>
      <p className={styles.hint}>{t("Currently connected: ")}{connected.length > 0 ? connected.map(t).join("، ") : t("none")}</p>

      <div className={styles.formGrid}>
        <label>
          {t("Platform")}<select value={platform} onChange={(e) => setPlatform(e.target.value)}>
            <option value="facebook">{t("Facebook")}</option>
            <option value="instagram">{t("Instagram")}</option>
          </select>
        </label>
        <label>
          {t("Access token")}<input type="password" value={token} onChange={(e) => setToken(e.target.value)} placeholder={t("Paste token")} />
        </label>
        {platform === "facebook" && (
          <label>
            {t("Page ID")}<input value={pageId} onChange={(e) => setPageId(e.target.value)} placeholder={t("e.g. 123456789")} required />
          </label>
        )}
        {platform === "instagram" && (
          <label>
            {t("IG User ID")}<input value={igId} onChange={(e) => setIgId(e.target.value)} placeholder={t("e.g. 17841400...")} required />
          </label>
        )}
      </div>
      {platform === "facebook" && <div className={styles.discoverRow}>
        <button className="btn-ghost" onClick={discoverPages} disabled={discovering || !token}>{discovering ? "Finding Pages…" : "Find my Facebook Pages"}</button>
        {pages.map((page) => <button key={page.page_id} className={styles.pageChoice} onClick={() => { setPageId(page.page_id); if (page.instagram?.ig_user_id) setIgId(page.instagram.ig_user_id); }}>
          <strong>{page.page_name || page.page_id}</strong><small>Use this Page ID{page.instagram?.username ? ` · Instagram @${page.instagram.username} found` : ""}</small>
        </button>)}
      </div>}
      <div className={styles.saveRow}>
        <button className="btn-primary" onClick={connect} disabled={saving || !token || (platform === "facebook" && !pageId.trim()) || (platform === "instagram" && !igId.trim())}>
          {saving ? t("Connecting…") : t("Connect")}
        </button>
        {msg && <span className={styles.msg}>{t(msg)}</span>}
      </div>
    </div>
  );
}

/* ========== Assets Tab (global, not per-business) ========== */
function AssetsTab() {
  const { t } = useI18n();
  const [hotelName, setHotelName] = useState("");
  const [destination, setDestination] = useState("");
  const [results, setResults] = useState<{ photo_urls: string[]; source: string } | null>(null);
  const [searching, setSearching] = useState(false);
  // Manual register
  const [regUrls, setRegUrls] = useState("");
  const [regMsg, setRegMsg] = useState("");

  async function search() {
    if (!hotelName || !destination) return;
    setSearching(true);
    setResults(null);
    try {
      const r = await api.get<{ photo_urls: string[]; source: string }>(
        `/assets/search?hotel_name=${encodeURIComponent(hotelName)}&destination=${encodeURIComponent(destination)}`
      );
      setResults(r);
    } catch (err) {
      setResults({ photo_urls: [], source: err instanceof ApiError ? err.body : "search failed" });
    } finally {
      setSearching(false);
    }
  }

  async function registerManual() {
    const urls = regUrls.split("\n").map((u) => u.trim()).filter(Boolean);
    if (!urls.length || !hotelName || !destination) return;
    try {
      await api.post("/assets/manual", { hotel_name: hotelName, destination, photo_urls: urls });
      setRegMsg("Registered ✓");
      setRegUrls("");
      setTimeout(() => setRegMsg(""), 3000);
    } catch (err) {
      setRegMsg(err instanceof ApiError ? err.body : "Register failed");
    }
  }

  return (
    <div className={styles.section}>
      <h3>{t("Photo Assets")}</h3>
      <p className={styles.hint}>{t("Search the cache (Google Places or manually registered) and add photos for hotels.")}</p>

      <div className={styles.formGrid}>
        <label>
          {t("Hotel name")}<input value={hotelName} onChange={(e) => setHotelName(e.target.value)} placeholder={t("e.g. El Mouradi Hammamet")} />
        </label>
        <label>
          {t("Destination")}<input value={destination} onChange={(e) => setDestination(e.target.value)} placeholder={t("e.g. Hammamet")} />
        </label>
      </div>
      <div className={styles.saveRow} style={{ marginBottom: "1rem" }}>
        <button className="btn-primary" onClick={search} disabled={searching || !hotelName || !destination}>
          {searching ? t("Searching…") : t("Search")}
        </button>
      </div>

      {results && (
        <div style={{ marginBottom: "1rem" }}>
          <p className={styles.hint}>{results.photo_urls.length} {t(" photos from ")}{results.source}</p>
          <div className={styles.photoGrid}>
            {results.photo_urls.map((url, i) => (
              <img key={i} src={url} alt="" className={styles.photoThumb} />
            ))}
          </div>
        </div>
      )}

      <h4 style={{ marginBottom: "0.5rem" }}>{t("Register manual photos")}</h4>
      <p className={styles.hint}>{t("Paste photo URLs (one per line) to add to the cache for the hotel above.")}</p>
      <textarea
        className={styles.urlsArea}
        value={regUrls}
        onChange={(e) => setRegUrls(e.target.value)}
        rows={4}
        placeholder={"https://hotel-site.com/photo1.jpg\nhttps://hotel-site.com/photo2.jpg"}
      />
      <div className={styles.saveRow}>
        <button className="btn-primary" onClick={registerManual} disabled={!regUrls.trim() || !hotelName || !destination}>
          {t("Register")}</button>
        {regMsg && <span className={styles.msg}>{t(regMsg)}</span>}
      </div>
    </div>
  );
}

/* ========== Style Rules Tab ========== */
function RulesTab({ businessId }: { businessId: string }) {
  const { t } = useI18n();
  const [rules, setRules] = useState<StyleRules>(DEFAULT_RULES);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [bannedInput, setBannedInput] = useState("");

  useEffect(() => {
    api.get<StyleRules>(`/businesses/${businessId}/rules`).then(setRules).catch(() => {});
  }, [businessId]);

  async function save() {
    setSaving(true);
    setMsg("");
    try {
      await api.put(`/businesses/${businessId}/rules`, rules);
      setMsg("Saved ✓");
      setTimeout(() => setMsg(""), 2000);
    } catch (err) {
      setMsg(err instanceof ApiError ? err.body : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  function addBanned() {
    const w = bannedInput.trim().toLowerCase();
    if (w && !rules.banned_words.includes(w)) {
      setRules({ ...rules, banned_words: [...rules.banned_words, w] });
      setBannedInput("");
    }
  }

  function removeBanned(word: string) {
    setRules({ ...rules, banned_words: rules.banned_words.filter((w) => w !== word) });
  }

  return (
    <div className={styles.section}>
      <h3>{t("Style Rules")}</h3>
      <p className={styles.hint}>{t("These rules are enforced by code — the AI cannot break them.")}</p>

      <div className={styles.field}>
        <label>{t("Banned words")}</label>
        <div className={styles.tags}>
          {rules.banned_words.map((w) => (
            <span key={w} className={styles.tag}>
              {w} <button onClick={() => removeBanned(w)}>×</button>
            </span>
          ))}
        </div>
        <div className={styles.inlineInput}>
          <input
            value={bannedInput}
            onChange={(e) => setBannedInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addBanned())}
            placeholder={t("Add word…")}
          />
          <button className="btn-ghost" onClick={addBanned}>{t("Add")}</button>
        </div>
      </div>

      <div className={styles.row}>
        <label>
          {t("Max emojis")}<input
            type="number" min={0} max={20} value={rules.max_emojis}
            onChange={(e) => setRules({ ...rules, max_emojis: Number(e.target.value) })}
          />
        </label>
        <label>
          {t("Max caption chars")}<input
            type="number" min={50} max={2200} value={rules.max_caption_chars}
            onChange={(e) => setRules({ ...rules, max_caption_chars: Number(e.target.value) })}
          />
        </label>
        <label>
          {t("Max hashtags")}<input
            type="number" min={0} max={30} value={rules.max_hashtags}
            onChange={(e) => setRules({ ...rules, max_hashtags: Number(e.target.value) })}
          />
        </label>
      </div>

      <label className={styles.check}>
        <input
          type="checkbox" checked={rules.end_with_question}
          onChange={(e) => setRules({ ...rules, end_with_question: e.target.checked })}
        />
        {t("Always end caption with a question")}</label>

      <div className={styles.saveRow}>
        <button className="btn-primary" onClick={save} disabled={saving}>
          {saving ? t("Saving…") : t("Save rules")}
        </button>
        {msg && <span className={styles.msg}>{t(msg)}</span>}
      </div>
    </div>
  );
}

/* ========== Glossary Tab ========== */
function GlossaryTab({ businessId }: { businessId: string }) {
  const { t } = useI18n();
  const [pairs, setPairs] = useState<GlossaryPair[]>([]);
  const [wrong, setWrong] = useState("");
  const [correct, setCorrect] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(() => {
    api.get<GlossaryPair[]>(`/businesses/${businessId}/glossary`).then(setPairs).catch(() => {});
  }, [businessId]);

  useEffect(load, [load]);

  async function add() {
    if (!wrong.trim() || !correct.trim()) return;
    try {
      await api.post(`/businesses/${businessId}/glossary`, { wrong: wrong.trim(), correct: correct.trim() });
      setWrong("");
      setCorrect("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Add failed");
      setTimeout(() => setError(""), 3000);
    }
  }

  async function remove(id: string) {
    await api.del(`/businesses/${businessId}/glossary/${id}`);
    load();
  }

  return (
    <div className={styles.section}>
      <h3>{t("Derja Glossary")}</h3>
      <p className={styles.hint}>{t("Word corrections applied to speech-to-text output. Pairs seen ≥ 2 times are auto-applied.")}</p>

      {error && <p className={styles.error}>{t(error)}</p>}

      <div className={styles.inlineInput}>
        <input value={wrong} onChange={(e) => setWrong(e.target.value)} placeholder={t("Wrong form")} />
        <span>→</span>
        <input value={correct} onChange={(e) => setCorrect(e.target.value)} placeholder={t("Correct form")} />
        <button className="btn-primary" onClick={add}>{t("Add")}</button>
      </div>

      {pairs.length === 0 ? (
        <p className={styles.muted}>{t("No glossary pairs yet.")}</p>
      ) : (
        <table className={styles.table}>
          <thead>
            <tr><th>{t("Wrong")}</th><th>{t("Correct")}</th><th>{t("Seen")}</th><th></th></tr>
          </thead>
          <tbody>
            {pairs.map((p) => (
              <tr key={p.id}>
                <td>{p.wrong}</td>
                <td>{p.correct}</td>
                <td>{p.times_seen}</td>
                <td><button className="btn-ghost" onClick={() => remove(p.id)}>×</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/* ========== Memory Tab ========== */
function MemoryTab({ businessId }: { businessId: string }) {
  const { t, locale } = useI18n();
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [scope, setScope] = useState("all");
  const [newContent, setNewContent] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const load = useCallback(() => {
    const query = scope === "all" ? "" : `?scope=${scope}`;
    api.get<MemoryEntry[]>(`/businesses/${businessId}/memory${query}`).then(setEntries).catch(() => {});
  }, [businessId, scope]);

  useEffect(load, [load]);

  async function add() {
    if (!newContent.trim() || scope === "all") return;
    try {
      await api.post(`/businesses/${businessId}/memory`, { scope, content: newContent.trim(), source: "staff" });
      setNewContent("");
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.body : "Add failed");
      setTimeout(() => setError(""), 3000);
    }
  }

  async function seedOzone() {
    try { const result = await api.post<{ seeded: number; total: number }>(`/businesses/${businessId}/memory/seed-ozone`); setNotice(`${result.seeded} Ozone starter memories loaded`); load(); }
    catch (err) { setError(err instanceof ApiError ? err.body : "Could not load starter memory"); }
  }

  async function remove(id: string) {
    await api.del(`/businesses/${businessId}/memory/${id}`);
    load();
  }

  return (
    <div className={styles.section}>
      <div className={styles.memoryHeader}><div><h3>{t("Agent Memory")}</h3><p className={styles.hint}>{t("Visible, editable knowledge used by the agent. Scope separation prevents unrelated lessons from leaking into prompts.")}</p></div><button className="btn-primary" onClick={seedOzone}>{t("Load Ozone baseline")}</button></div>

      <div className={styles.scopeFilter}>
        {SCOPES.map((s) => (
          <button
            key={s}
            className={`btn-ghost ${scope === s ? styles.activeTab : ""}`}
            onClick={() => setScope(s)}
          >
            {t(s)}
          </button>
        ))}
      </div>
      <p className={styles.scopeHelp}>{t(SCOPE_HELP[scope])}</p>

      {error && <p className={styles.error}>{t(error)}</p>}
      {notice && <p className={styles.memoryNotice}>{t(notice)}</p>}

      {scope !== "all" && <div className={styles.inlineInput}>
        <input
          value={newContent}
          onChange={(e) => setNewContent(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), add())}
          placeholder={t(`Add to "${t(scope)}" memory…`)}
          style={{ flex: 1 }}
        />
        <button className="btn-primary" onClick={add}>{t("Add")}</button>
      </div>}

      {entries.length === 0 ? (
        <p className={styles.muted}>{t("No memories in \"")}{t(scope)}{t("\" scope.")}</p>
      ) : (
        <div className={styles.memList}>
          {entries.map((m) => (
            <div key={m.id} className={`card ${styles.memCard}`}>
              <div className={styles.memTop}>
                {scope === "all" && <span className={styles.scopeBadge}>{t(m.scope)}</span>}
                <span className={styles.memSource}>{t(m.source)}</span>
                {m.pinned && <span className={`badge badge-approved`}>{t("pinned")}</span>}
                <span className={styles.memTime}>
                  {new Date(m.created_at).toLocaleDateString(locale, { day: "numeric", month: "short" })}
                </span>
                <button className="btn-ghost" onClick={() => remove(m.id)} title={t("Delete")}>×</button>
              </div>
              <p className={styles.memContent}>{m.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
