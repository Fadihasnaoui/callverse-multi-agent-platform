import { useI18n } from "@/i18n";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/api";
import styles from "./Catalogue.module.css";

interface CatalogueEntry {
  hotel_name: string;
  destination: string;
  photo_urls: string[];
  source: string;
  updated_at: string | null;
}

export function Catalogue() {
  const { t, locale } = useI18n();
  const [entries, setEntries] = useState<CatalogueEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [lightbox, setLightbox] = useState<string | null>(null);

  // Upload form
  const [formOpen, setFormOpen] = useState(false);
  const [hotel, setHotel] = useState("");
  const [dest, setDest] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [draggingOver, setDraggingOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(() => {
    setLoading(true);
    api.get<CatalogueEntry[]>("/assets/catalogue")
      .then(setEntries)
      .catch(() => flash("Failed to load catalogue"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  function flash(msg: string) {
    setError(msg);
    setTimeout(() => setError(""), 5000);
  }

  function addFiles(incoming: FileList | File[]) {
    const arr = Array.from(incoming).filter(
      (f) => f.type === "image/jpeg" || f.type === "image/png" || f.type === "image/webp"
    );
    if (arr.length === 0) {
      flash("Only JPEG, PNG or WebP images are accepted");
      return;
    }
    const merged = [...files, ...arr].slice(0, 20);
    setFiles(merged);
    setPreviews(merged.map((f) => URL.createObjectURL(f)));
  }

  function removeFile(idx: number) {
    const next = files.filter((_, i) => i !== idx);
    setFiles(next);
    setPreviews(next.map((f) => URL.createObjectURL(f)));
  }

  async function upload() {
    if (!hotel.trim() || !dest.trim() || files.length === 0) return;
    setUploading(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("hotel_name", hotel.trim());
      fd.append("destination", dest.trim());
      files.forEach((f) => fd.append("files", f));

      const res = await fetch("/assets/upload", {
        method: "POST",
        credentials: "include",
        body: fd,
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text);
      }
      const data = await res.json();
      setSuccess(`${data.uploaded} photos uploaded for ${hotel} (${dest})`);
      setTimeout(() => setSuccess(""), 5000);
      setHotel("");
      setDest("");
      setFiles([]);
      setPreviews([]);
      setFormOpen(false);
      load();
    } catch (err) {
      flash(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function deleteEntry(hotelName: string, destination: string) {
    if (!confirm(t(`Delete all photos for "${hotelName}" in ${destination}?`))) return;
    try {
      await fetch(
        `/assets/catalogue?hotel_name=${encodeURIComponent(hotelName)}&destination=${encodeURIComponent(destination)}`,
        { method: "DELETE", credentials: "include" },
      );
      load();
    } catch {
      flash("Delete failed");
    }
  }

  return (
    <div>
      <div className={styles.header}>
        <h2>{t("Photo Catalogue")}</h2>
        <button className="btn-primary" onClick={() => setFormOpen(!formOpen)}>
          {formOpen ? t("Cancel") : t("+ Upload Photos")}
        </button>
      </div>

      {error && <p className={styles.error}>{t(error)}</p>}
      {success && <p className={styles.success}>{t(success)}</p>}

      {formOpen && (
        <div className={`card ${styles.uploadForm}`}>
          <div className={styles.uploadRow}>
            <label>
              {t("Hotel / Place name")}<input value={hotel} onChange={(e) => setHotel(e.target.value)} placeholder={t("e.g. El Mouradi Hammamet")} />
            </label>
            <label>
              {t("Destination")}<input value={dest} onChange={(e) => setDest(e.target.value)} placeholder={t("e.g. Hammamet")} />
            </label>
          </div>

          <div
            className={`${styles.dropZone} ${draggingOver ? styles.dropZoneDragging : ""}`}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDraggingOver(true); }}
            onDragLeave={() => setDraggingOver(false)}
            onDrop={(e) => { e.preventDefault(); setDraggingOver(false); addFiles(e.dataTransfer.files); }}
          >
            {files.length === 0
              ? t("Drop images here or click to browse (JPEG, PNG, WebP — max 20)")
              : t(`${files.length} file${files.length > 1 ? "s" : ""} selected`)}
            <input
              ref={fileRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              multiple
              hidden
              onChange={(e) => { if (e.target.files) addFiles(e.target.files); }}
            />
          </div>

          {previews.length > 0 && (
            <div className={styles.previewGrid}>
              {previews.map((src, i) => (
                <div key={i} className={styles.thumbWrap}>
                  <img src={src} alt="" className={styles.previewThumb} />
                  <button className={styles.removeThumb} onClick={() => removeFile(i)}>x</button>
                </div>
              ))}
            </div>
          )}

          <button
            className="btn-primary"
            disabled={uploading || !hotel.trim() || !dest.trim() || files.length === 0}
            onClick={upload}
          >
            {uploading ? t("Uploading...") : t(`Upload ${files.length} photo${files.length !== 1 ? "s" : ""}`)}
          </button>
        </div>
      )}

      {loading ? (
        <p>{t("Loading...")}</p>
      ) : entries.length === 0 ? (
        <p className={styles.empty}>
          {t("No photos in the catalogue yet. Upload hotel and destination photos so the AI agent can use them in posts.")}</p>
      ) : (
        <div className={styles.list}>
          {entries.map((e) => (
            <div key={`${e.hotel_name}|${e.destination}`} className={`card ${styles.entry}`}>
              <div className={styles.entryHeader}>
                <div>
                  <div className={styles.entryTitle}>{e.hotel_name}</div>
                  <div className={styles.entryDest}>{e.destination}</div>
                </div>
                <button className={styles.deleteEntry} onClick={() => deleteEntry(e.hotel_name, e.destination)}>
                  {t("Delete")}</button>
              </div>
              <div className={styles.entryMeta}>
                <span>{e.photo_urls.length} {t(" photo")}{e.photo_urls.length !== 1 ? "s" : ""}</span>
                <span>{e.source}</span>
                {e.updated_at && <span>{new Date(e.updated_at).toLocaleDateString(locale)}</span>}
              </div>
              <div className={styles.entryPhotos}>
                {e.photo_urls.map((url, i) => (
                  <img
                    key={i}
                    src={url}
                    alt=""
                    className={styles.entryPhoto}
                    onClick={() => setLightbox(url)}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {lightbox && (
        <div className={styles.lightboxOverlay} onClick={() => setLightbox(null)}>
          <button className={styles.lightboxClose} onClick={() => setLightbox(null)}>x</button>
          <img src={lightbox} alt="" className={styles.lightboxImg} onClick={(e) => e.stopPropagation()} />
        </div>
      )}
    </div>
  );
}
