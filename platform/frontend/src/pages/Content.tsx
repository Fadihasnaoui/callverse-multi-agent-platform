import { useI18n } from "@/i18n";
import { useEffect, useState, useCallback, useRef } from "react";
import { useParams } from "react-router-dom";
import { api, ApiError } from "@/api";
import type { ContentItem } from "@/types";
import styles from "./Content.module.css";

type Filter = "all" | "pending" | "approved" | "published" | "failed" | "rejected" | "publishing";

/* ---- Generate form state ---- */
interface GenForm { platform: string; topic: string; content_type: string }
const EMPTY_GEN: GenForm = { platform: "facebook", topic: "", content_type: "static_post" };

interface Recommendation { type: string; topic: string; reason: string; weight: number }
interface ContentRecs {
  recommendations: Recommendation[];
  directives_summary: string | null;
  benchmark_insight: string[] | null;
  sentiment_summary: { overall_sentiment: number; buying_intent: number; top_topics: string[] } | null;
}

function ImageLightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div className={styles.lightboxOverlay} onClick={onClose}>
      <button className={styles.lightboxClose} onClick={onClose}>×</button>
      <img src={src} alt="" className={styles.lightboxImg} onClick={(e) => e.stopPropagation()} />
    </div>
  );
}

export function Content() {
  const { t } = useI18n();
  const { businessId } = useParams();
  const [items, setItems] = useState<ContentItem[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Generate form
  const [genOpen, setGenOpen] = useState(false);
  const [gen, setGen] = useState<GenForm>(EMPTY_GEN);
  const [generating, setGenerating] = useState(false);

  // Expanded detail panel
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // AI recommendations
  const [recs, setRecs] = useState<ContentRecs | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    api.get<ContentItem[]>(`/businesses/${businessId}/content`)
      .then(setItems)
      .finally(() => setLoading(false));
  }, [businessId]);

  useEffect(load, [load]);

  useEffect(() => {
    api.get<ContentRecs>(`/businesses/${businessId}/content-recommendations`)
      .then(setRecs)
      .catch(() => {});
  }, [businessId]);

  const filtered = filter === "all" ? items : items.filter((i) => i.status === filter);
  const counts = (s: string) => items.filter((i) => i.status === s).length;

  function flash(msg: string) {
    setError(msg);
    setTimeout(() => setError(""), 5000);
  }

  /* ---- Generate content ---- */
  async function generate() {
    if (!gen.topic.trim()) return;
    setGenerating(true);
    try {
      await api.post(`/businesses/${businessId}/content`, gen);
      setGenOpen(false);
      setGen(EMPTY_GEN);
      load();
    } catch (err) {
      flash(err instanceof ApiError ? err.body : "Generation failed");
    } finally {
      setGenerating(false);
    }
  }

  /* ---- Resolve (approve / reject / edit) ---- */
  async function resolve(id: string, decision: string, editedCaption?: string, scheduledFor?: string, notes?: string) {
    try {
      await api.post(`/businesses/${businessId}/content/${id}/resolve`, {
        decision,
        edited_caption: editedCaption ?? null,
        scheduled_for: scheduledFor ?? null,
        reviewer_notes: notes ?? null,
        expected_version: items.find(item => item.id === id)?.content_version,
      });
      load();
    } catch (err) {
      flash(err instanceof ApiError ? err.body : "Action failed");
    }
  }

  /* ---- Other actions ---- */
  async function action(id: string, endpoint: string, body?: unknown) {
    try {
      await api.post(`/businesses/${businessId}/content/${id}/${endpoint}`, body);
      load();
    } catch (err) {
      flash(err instanceof ApiError ? err.body : `${endpoint} failed`);
    }
  }

  return (
    <div>
      <JobsPanel businessId={businessId!} onReload={load} />
      <div className={styles.header}>
        <h2>{t("Content")}</h2>
        <button className="btn-primary" onClick={() => setGenOpen(!genOpen)}>
          {genOpen ? t("Cancel") : t("+ Generate")}
        </button>
      </div>

      {/* AI Recommendations panel */}
      {genOpen && recs && (recs.recommendations.length > 0 || recs.sentiment_summary) && (
        <div className={`card ${styles.recsPanel}`}>
          <div className={styles.recsPanelHeader}>
            <span className={styles.recsIcon}>✦</span>
            <strong>{t("AI recommends")}</strong>
            {recs.sentiment_summary && (
              <span className={styles.recsBadge}>
                {t("Sentiment")}{recs.sentiment_summary.overall_sentiment > 0 ? "+" : ""}{recs.sentiment_summary.overall_sentiment}{t("% · Intent")}{recs.sentiment_summary.buying_intent}/100
              </span>
            )}
          </div>
          {recs.recommendations.length > 0 && (
            <div className={styles.recsList}>
              {recs.recommendations.slice(0, 5).map((r) => (
                <button
                  key={r.topic}
                  className={styles.recItem}
                  onClick={() => setGen({ ...gen, topic: r.topic })}
                  title={r.reason}
                >
                  <span className={styles.recType}>{t(r.type)}</span>
                  <span className={styles.recTopic}>{r.topic}</span>
                  <span className={styles.recReason}>{r.reason}</span>
                </button>
              ))}
            </div>
          )}
          {recs.benchmark_insight && recs.benchmark_insight.length > 0 && (
            <div className={styles.recsBenchmark}>
              {recs.benchmark_insight.map((ins) => (
                <p key={ins}>{ins}</p>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Generate form */}
      {genOpen && (
        <div className={`card ${styles.genForm}`}>
          <label>
            {t("Topic")}<input
              value={gen.topic} onChange={(e) => setGen({ ...gen, topic: e.target.value })}
              placeholder={t("e.g. Hôtel El Mouradi Hammamet — piscine chauffée")}
              style={{ width: "100%" }}
            />
          </label>
          <div className={styles.genRow}>
            <label>
              {t("Platform")}<select value={gen.platform} onChange={(e) => setGen({ ...gen, platform: e.target.value })}>
                <option value="facebook">{t("Facebook")}</option>
                <option value="instagram">{t("Instagram")}</option>
              </select>
            </label>
            <label>
              {t("Type")}<select value={gen.content_type} onChange={(e) => setGen({ ...gen, content_type: e.target.value })}>
                <option value="static_post">{t("Static Post")}</option>
                <option value="reel">{t("Reel")}</option>
                <option value="avatar_clip">{t("Avatar Clip")}</option>
              </select>
            </label>
            <button className="btn-primary" onClick={generate} disabled={generating || !gen.topic.trim()}>
              {generating ? t("Generating…") : t("Generate")}
            </button>
          </div>
        </div>
      )}

      {error && <p className={styles.error}>{t(error)}</p>}

      {/* Filters */}
      <div className={styles.filters}>
        {(["all", "pending", "approved", "publishing", "published", "failed", "rejected"] as Filter[]).map((f) => (
          <button
            key={f}
            className={`btn-ghost ${filter === f ? styles.activeFilter : ""}`}
            onClick={() => setFilter(f)}
          >
            {t(f)} {f !== "all" ? `(${counts(f)})` : `(${items.length})`}
          </button>
        ))}
      </div>

      {/* List */}
      {loading ? (
        <p>{t("Loading…")}</p>
      ) : filtered.length === 0 ? (
        <p className={styles.empty}>{t("No content")}{filter !== "all" ? t(` with status "${t(filter)}"`) : ""}.</p>
      ) : (
        <div className={styles.list}>
          {filtered.map((item) => (
            <ContentCard
              key={item.id}
              item={item}
              businessId={businessId!}
              expanded={expandedId === item.id}
              onToggle={() => setExpandedId(expandedId === item.id ? null : item.id)}
              onResolve={resolve}
              onAction={action}
              onError={flash}
              onReload={load}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function JobsPanel({businessId,onReload}:{businessId:string;onReload:()=>void}) {
  const [jobs,setJobs]=useState<{id:string;kind:string;status:string;error:string|null;progress:Record<string,string>}[]>([]);
  const [error,setError]=useState('');
  useEffect(()=>{
    let active=true;
    const refresh=()=>api.get<typeof jobs>(`/businesses/${businessId}/content/jobs`).then(rows=>{if(active)setJobs(rows);}).catch(()=>{});
    void refresh(); const timer=setInterval(refresh,4000);
    return ()=>{active=false;clearInterval(timer);};
  },[businessId]);
  if (!jobs.length) return null;
  return <details className="card"><summary>Production jobs ({jobs.length})</summary>{error&&<p role="alert">{error}</p>}{jobs.map(job=><div key={job.id} style={{padding:'.5rem'}}>
    <strong>{job.kind}</strong> · {job.status} · {Object.keys(job.progress||{}).length} saved steps
    {job.error&&<p>{job.error}</p>}
    {(job.status==='unknown'||job.status==='failed')&&<button onClick={async()=>{
      if(!window.confirm('Review the existing drafts/media and provider charges first. Restarting can repeat an operation whose external result is unknown. Have you checked and chosen to restart?'))return;
      try {await api.post(`/businesses/${businessId}/content/jobs/${job.id}/restart`,{reviewed_outcome:true});onReload();}catch(e){setError(e instanceof Error?e.message:'Job failed');}
    }}>Restart after review</button>}
  </div>)}</details>;
}

/* ======== Single content card with all actions ======== */

interface CardProps {
  item: ContentItem;
  businessId: string;
  expanded: boolean;
  onToggle: () => void;
  onResolve: (id: string, decision: string, caption?: string, schedule?: string, notes?: string) => Promise<void>;
  onAction: (id: string, endpoint: string, body?: unknown) => Promise<void>;
  onError: (msg: string) => void;
  onReload: () => void;
}

function ContentCard({ item, businessId, expanded, onToggle, onResolve, onAction, onError, onReload }: CardProps) {
  const { t, locale } = useI18n();
  const [editCaption, setEditCaption] = useState(item.caption || "");
  const [editNotes, setEditNotes] = useState("");
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState("");
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null);
  // Attach photos
  const [photoForm, setPhotoForm] = useState(false);
  const [hotelName, setHotelName] = useState("");
  const [destination, setDestination] = useState("");
  // Avatar
  const [avatarTranscript, setAvatarTranscript] = useState("");
  const audioRef = useRef<MediaRecorder | null>(null);
  const [recording, setRecording] = useState(false);
  // Suggest schedule
  const [suggestion, setSuggestion] = useState<string | null>(null);

  async function withBusy(label: string, fn: () => Promise<void>) {
    setBusy(label);
    try { await fn(); } finally { setBusy(""); }
  }

  const isPending = item.status === "pending";
  const isFailed = item.status === "failed";
  const isApproved = item.status === "approved";
  const isVideo = item.type === "reel" || item.type === "avatar_clip";
  const isAvatar = item.type === "avatar_clip";
  const locked = item.status === "publishing" || item.status === "published" || item.status === 'publish_unknown';

  return (
    <div className={`card ${styles.item}`}>
      {/* Header row */}
      <div className={styles.itemHeader} onClick={onToggle} style={{ cursor: "pointer" }}>
        <span className={`badge badge-${item.status}`}>{t(item.status)}</span>
        <span className={styles.meta}>{t(item.platform)} · {t(item.type)}</span>
        {item.scheduled_for && (
          <span className={styles.time}>
            {new Date(item.scheduled_for).toLocaleDateString(locale, {
              weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
            })}
          </span>
        )}
        <span className={styles.expand}>{expanded ? "▾" : "▸"}</span>
      </div>

      {/* Caption preview */}
      {!expanded && <p className={styles.captionPreview}>{(item.caption || "—").slice(0, 120)}</p>}

      {/* Expanded detail */}
      {expanded && (
        <div className={styles.detail}>
          {/* Caption */}
          {item.status === 'publish_unknown' && <div role="alert" className={styles.error}>
            <p>{t('Publishing outcome is unknown. Check the connected Page before retrying.')}</p>
            <button disabled={!!busy} onClick={() => {
              const id = window.prompt('Paste the published post ID after checking the Page, or leave blank if the post is definitely absent.');
              if (id === null) return;
              if (!id.trim() && !window.confirm('I checked the Page and confirm this post is absent. Return it to pending review?')) return;
              void onAction(item.id,'reconcile',{remote_post_id:id.trim() || null,confirmed_absent:!id.trim()});
            }}>{t('Reconcile publishing result')}</button>
          </div>}
          {editing ? (
            <div className={styles.editArea}>
              <textarea
                value={editCaption}
                onChange={(e) => setEditCaption(e.target.value)}
                rows={4}
              />
              <input
                placeholder={t("Reviewer notes (optional)")}
                value={editNotes}
                onChange={(e) => setEditNotes(e.target.value)}
              />
              <div className={styles.editActions}>
                <button className="btn-primary" disabled={!!busy} onClick={() =>
                  withBusy("saving", () => onResolve(item.id, "edit", editCaption, undefined, editNotes).then(() => setEditing(false)))
                }>{t("Save & approve")}</button>
                <button className="btn-ghost" onClick={() => setEditing(false)}>{t("Cancel")}</button>
              </div>
            </div>
          ) : (
            <>
              <p className={styles.caption}>{item.caption || "—"}</p>
              {item.hashtags && item.hashtags.length > 0 && (
                <p className={styles.hashtags}>{item.hashtags.join(" ")}</p>
              )}
            </>
          )}

          {/* Photos */}
          {item.photo_urls && item.photo_urls.length > 0 && (
            <div className={styles.photos}>
              {item.photo_urls.map((url, i) => (
                <img key={i} src={url} alt="" className={styles.photo} onClick={() => setLightboxSrc(url)} style={{ cursor: "pointer" }} />
              ))}
            </div>
          )}

          {item.visual_brief && (
            <div className={styles.visualBrief}>
              <strong>{t("Creative direction")}</strong>
              <p>{item.visual_brief}</p>
            </div>
          )}

          {/* Video */}
          {item.video_url && (
            <video src={item.video_url} controls className={styles.video} />
          )}
          {item.music_url && <audio src={item.music_url} controls />}

          {/* Publish error */}
          {item.publish_error && (
            <p className={styles.publishError}>{t("Error: ")}{item.publish_error}</p>
          )}

          {/* Avatar info */}
          {isAvatar && (
            <div className={styles.avatarSection}>
              <p className={styles.avatarLabel}>
                {t("Avatar:")}{item.avatar_language === "tn" ? t("Tunisian") : t("French")}
                {item.avatar_transcript_status && ` · transcript ${item.avatar_transcript_status}`}
                {item.avatar_job_id && ` · job ${item.avatar_job_id.slice(0, 8)}…`}
              </p>
              {item.avatar_text && <p className={styles.avatarText}>"{item.avatar_text}"</p>}
            </div>
          )}

          {/* ---- Action buttons ---- */}
          {busy && <p className={styles.busyLabel}>{t(busy)}…</p>}

          <div className={styles.actions}>
            {/* Approve / Reject / Edit */}
            {isPending && !editing && (
              <>
                <button className="btn-primary" disabled={!!busy}
                  onClick={() => withBusy("approving", () => onResolve(item.id, "approve"))}>
                  {t("✓ Approve")}</button>
                <button className="btn-ghost" disabled={!!busy}
                  onClick={() => withBusy("rejecting", () => onResolve(item.id, "reject"))}>
                  {t("✗ Reject")}</button>
                <button className="btn-ghost" disabled={!!busy}
                  onClick={() => { setEditCaption(item.caption || ""); setEditing(true); }}>
                  {t("✎ Edit")}</button>
              </>
            )}

            {/* Suggest schedule */}
            {(isPending || isApproved) && !locked && (
              <button className="btn-ghost" disabled={!!busy} onClick={() =>
                withBusy("suggesting", async () => {
                  const r = await api.post<{ proposed_datetime: string }>(`/businesses/${businessId}/content/${item.id}/suggest-schedule`);
                  setSuggestion(r.proposed_datetime);
                  onReload();
                })
              }>{t("🕐 Suggest time")}</button>
            )}

            {/* Attach photos */}
            {!locked && (
              <button className="btn-ghost" disabled={!!busy}
                onClick={() => setPhotoForm(!photoForm)}>
                {t("📷 Attach photos")}</button>
            )}

            {!locked && (
              <button className="btn-ghost" disabled={!!busy}
                onClick={() => withBusy("generating AI visual", () => onAction(item.id, "generate-image"))}>
                {t("✦ Generate AI visual")}</button>
            )}

            {item.type === "reel" && !locked && (
              <button className="btn-ghost" disabled={!!busy || !!item.music_url}
                onClick={() => withBusy("generating music", () => onAction(item.id, "music"))}>
                {item.music_url ? t("♫ Music ready") : t("♫ Add music")}
              </button>
            )}

            {/* Render reel */}
            {isVideo && !locked && (
              <button className="btn-ghost" disabled={!!busy}
                onClick={() => withBusy("rendering", () => onAction(item.id, "render"))}>
                {t("🎬 Render video")}</button>
            )}

            {/* Avatar actions */}
            {isAvatar && !locked && (
              <>
                {/* Record dictation (Tunisian) */}
                <button className="btn-ghost" disabled={!!busy || recording}
                  onClick={async () => {
                    if (recording && audioRef.current) {
                      audioRef.current.stop();
                      setRecording(false);
                      return;
                    }
                    try {
                      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
                      const chunks: BlobPart[] = [];
                      recorder.ondataavailable = (e) => chunks.push(e.data);
                      recorder.onstop = async () => {
                        stream.getTracks().forEach((t) => t.stop());
                        const blob = new Blob(chunks, { type: "audio/webm" });
                        setBusy("transcribing");
                        try {
                          const res = await fetch(`/businesses/${businessId}/content/${item.id}/avatar/dictation`, {
                            method: "POST",
                            credentials: "include",
                            headers: { "Content-Type": "audio/webm" },
                            body: blob,
                          });
                          if (!res.ok) throw new Error(await res.text());
                          const data = await res.json();
                          setAvatarTranscript(data.transcript || "");
                          onReload();
                        } catch (err) {
                          onError(err instanceof Error ? err.message : "Dictation failed");
                        } finally {
                          setBusy("");
                        }
                      };
                      recorder.start();
                      audioRef.current = recorder;
                      setRecording(true);
                    } catch {
                      onError("Microphone access denied");
                    }
                  }}>
                  {recording ? t("⏹ Stop recording") : t("🎙 Record Tunisian")}
                </button>

                {/* Transcript review */}
                {(item.avatar_transcript_status === "pending" || item.avatar_transcript_status === "pending_review" || avatarTranscript) && (
                  <>
                    <input
                      className={styles.transcriptInput}
                      value={avatarTranscript || item.avatar_text || ""}
                      onChange={(e) => setAvatarTranscript(e.target.value)}
                      placeholder={t("Review transcript…")}
                    />
                    <button className="btn-primary" disabled={!!busy}
                      onClick={() => withBusy("approving transcript", async () => {
                        await api.post(`/businesses/${businessId}/content/${item.id}/avatar/transcript`, {
                          transcript: avatarTranscript || item.avatar_text, approve: true,
                        });
                        setAvatarTranscript("");
                        onReload();
                      })}>{t("✓ Approve transcript")}</button>
                  </>
                )}

                {/* Generate avatar (French or after transcript approved) */}
                <button className="btn-ghost" disabled={!!busy}
                  onClick={() => withBusy("generating avatar", () =>
                    onAction(item.id, "avatar?language=fr"))}>
                  {t("🤖 Generate FR avatar")}</button>
                {item.avatar_transcript_status === "approved" && (
                  <button className="btn-ghost" disabled={!!busy}
                    onClick={() => withBusy("generating avatar", () =>
                      onAction(item.id, "avatar?language=tn"))}>
                    {t("🤖 Generate TN avatar")}</button>
                )}

                {/* Refresh avatar status */}
                {item.avatar_job_id && (
                  <button className="btn-ghost" disabled={!!busy}
                    onClick={() => withBusy("refreshing", () => onAction(item.id, "avatar/refresh"))}>
                    {t("🔄 Refresh avatar")}</button>
                )}
              </>
            )}

            {/* Publish now */}
            {isApproved && (
              <button className="btn-primary" disabled={!!busy}
                onClick={() => withBusy("publishing", () => onAction(item.id, "publish-now"))}>
                {t("🚀 Publish now")}</button>
            )}

            {/* Retry failed */}
            {isFailed && (
              <button className="btn-primary" disabled={!!busy}
                onClick={() => withBusy("retrying", () => onAction(item.id, "retry"))}>
                {t("🔁 Retry")}</button>
            )}

            {/* Delete */}
            {!locked && (
              <button className={styles.deleteBtn} disabled={!!busy}
                onClick={() => {
                  if (confirm("Delete this post?"))
                    withBusy("deleting", async () => {
                      await api.del(`/businesses/${businessId}/content/${item.id}`);
                      onReload();
                    });
                }}>
                {t("🗑 Delete")}</button>
            )}
          </div>

          {/* Suggestion display */}
          {(suggestion || item.proposed_schedule) && !locked && (
            <div className={styles.suggestion}>
              <p>{t("AI suggests:")}{new Date(suggestion || item.proposed_schedule!).toLocaleString(locale)}</p>
              <button disabled={!!busy} onClick={() => withBusy('applying reviewed schedule',()=>onResolve(item.id,'approve',undefined,suggestion || item.proposed_schedule!))}>{t('Apply time and approve reviewed content')}</button>
            </div>
          )}

          {/* Attach photos form */}
          {photoForm && (
            <div className={styles.photoForm}>
              <input placeholder={t("Hotel name")} value={hotelName} onChange={(e) => setHotelName(e.target.value)} />
              <input placeholder={t("Destination")} value={destination} onChange={(e) => setDestination(e.target.value)} />
              <button className="btn-primary" disabled={!!busy || !hotelName || !destination}
                onClick={() => withBusy("attaching photos", async () => {
                  await onAction(item.id, "attach-photos", { hotel_name: hotelName, destination });
                  setPhotoForm(false);
                  setHotelName("");
                  setDestination("");
                })}>{t("Attach")}</button>
              <button className="btn-ghost" onClick={() => setPhotoForm(false)}>{t("Cancel")}</button>
            </div>
          )}
        </div>
      )}

      {lightboxSrc && <ImageLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />}
    </div>
  );
}
