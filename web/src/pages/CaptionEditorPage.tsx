/**
 * Caption Editor — visual editor for phonetic caption jobs.
 *
 * Loaded at /captions (job list) and /captions/:id (editor for a specific job).
 * Calls the /api/caption/jobs/* endpoints — no LLM, pure FFmpeg re-burn.
 */

import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Download, Play, RefreshCw, ChevronLeft } from "lucide-react";
import { Button, Spinner, Typography } from "@nous-research/ui";
import { fetchJSON } from "@/lib/api";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CaptionSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  lang: "en" | "vi";
  phonetic: string;
}

interface CaptionStyle {
  font: string;
  font_size: number;
  primary_color: string;
  outline_color: string;
  outline_width: number;
  alignment: number;
  margin_bottom: number;
  max_line_length: number;
}

interface CaptionJob {
  id: string;
  created_at: string;
  video_path: string;
  output_path: string;
  style: CaptionStyle;
  segments: CaptionSegment[];
}

interface JobSummary {
  id: string;
  created_at: string;
  video_filename: string;
  segment_count: number;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiPut(path: string, body: unknown): Promise<void> {
  await fetchJSON(path, {
    method: "PUT",
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  });
}

async function apiPost(path: string): Promise<unknown> {
  return fetchJSON(path, { method: "POST" });
}

// ---------------------------------------------------------------------------
// Job list page
// ---------------------------------------------------------------------------

export function CaptionJobsPage() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchJSON<JobSummary[]>("/api/caption/jobs")
      .then(setJobs)
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="flex justify-center p-12"><Spinner /></div>;
  if (error) return <div className="p-6 text-red-500">{error}</div>;

  return (
    <div className="p-6 max-w-3xl">
      <Typography variant="h2" className="mb-4">Caption Jobs</Typography>
      {jobs.length === 0 ? (
        <p className="text-muted-foreground">
          No caption jobs yet. Send a video via Telegram and Hermes will create one.
        </p>
      ) : (
        <div className="space-y-2">
          {jobs.map((job) => (
            <Link
              key={job.id}
              to={`/captions/${job.id}`}
              className="flex items-center justify-between rounded-lg border p-4 hover:bg-muted/50 transition-colors"
            >
              <div>
                <div className="font-medium">{job.video_filename || job.id}</div>
                <div className="text-sm text-muted-foreground">
                  {job.segment_count} segments · {job.created_at ? new Date(job.created_at).toLocaleString() : ""}
                </div>
              </div>
              <Play className="w-4 h-4 text-muted-foreground" />
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Editor page
// ---------------------------------------------------------------------------

export default function CaptionEditorPage() {
  const { id: jobId } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [job, setJob] = useState<CaptionJob | null>(null);
  const [segments, setSegments] = useState<CaptionSegment[]>([]);
  const [style, setStyle] = useState<CaptionStyle | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [burning, setBurning] = useState(false);
  const [burnError, setBurnError] = useState<string | null>(null);
  const [burnTimestamp, setBurnTimestamp] = useState(Date.now());
  const videoRef = useRef<HTMLVideoElement>(null);

  // Load job on mount
  useEffect(() => {
    if (!jobId) return;
    setLoading(true);
    fetchJSON<CaptionJob>(`/api/caption/jobs/${jobId}`)
      .then((data) => {
        setJob(data);
        setSegments(data.segments);
        setStyle(data.style);
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [jobId]);

  const updateSegment = (idx: number, field: keyof CaptionSegment, value: string) => {
    setSegments((prev) =>
      prev.map((s, i) => (i === idx ? { ...s, [field]: value } : s))
    );
  };

  const toggleLang = (idx: number) => {
    setSegments((prev) =>
      prev.map((s, i) =>
        i === idx ? { ...s, lang: s.lang === "en" ? "vi" : "en", phonetic: s.lang === "en" ? s.phonetic : "" } : s
      )
    );
  };

  const handleReburn = async () => {
    if (!jobId || !style) return;
    setBurning(true);
    setBurnError(null);

    try {
      // Persist edits first
      await apiPut(`/api/caption/jobs/${jobId}/segments`, { segments });
      await apiPut(`/api/caption/jobs/${jobId}/style`, { style });
      // Trigger re-burn
      await apiPost(`/api/caption/jobs/${jobId}/burn`);
      // Reload video player with cache-busting timestamp
      setBurnTimestamp(Date.now());
      if (videoRef.current) videoRef.current.load();
    } catch (e: unknown) {
      setBurnError(String(e));
    } finally {
      setBurning(false);
    }
  };

  const handleDownload = async () => {
    const token = window.__HERMES_SESSION_TOKEN__ ?? "";
    const res = await fetch(`/api/caption/jobs/${jobId}/download`, {
      headers: { "X-Hermes-Session-Token": token },
    });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `captioned_${jobId}.mp4`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loading) return <div className="flex justify-center p-12"><Spinner /></div>;
  if (error) return <div className="p-6 text-red-500">Error: {error}</div>;
  if (!job || !style) return null;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 py-3 border-b">
        <button onClick={() => navigate("/captions")} className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
          <ChevronLeft className="w-4 h-4" />
          All jobs
        </button>
        <span className="text-muted-foreground">/</span>
        <span className="text-sm font-medium truncate">{job.video_path.split("/").pop()}</span>
      </div>

      {/* Main two-column layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: video player */}
        <div className="w-2/5 min-w-0 flex flex-col gap-3 p-4 border-r">
          <video
            ref={videoRef}
            key={burnTimestamp}
            controls
            className="w-full rounded-lg bg-black"
            src={`/api/caption/jobs/${jobId}/video?t=${burnTimestamp}`}
          />
          <div className="flex gap-2">
            <Button
              onClick={handleReburn}
              disabled={burning}
              className="flex-1 flex items-center justify-center gap-2"
            >
              {burning ? <Spinner className="w-4 h-4" /> : <RefreshCw className="w-4 h-4" />}
              {burning ? "Re-burning…" : "Re-burn"}
            </Button>
            <Button variant="outline" onClick={handleDownload} className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Download
            </Button>
          </div>
          {burnError && <p className="text-sm text-red-500">{burnError}</p>}

          {/* Style panel */}
          <details className="group">
            <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground py-1">
              Style settings
            </summary>
            <div className="mt-2 space-y-2 text-sm">
              <StyleField label="Font" value={style.font} onChange={(v) => setStyle((s) => s && ({ ...s, font: v }))} />
              <StyleNumberField label="Font size" value={style.font_size} onChange={(v) => setStyle((s) => s && ({ ...s, font_size: v }))} />
              <StyleColorField label="Text color" value={style.primary_color} onChange={(v) => setStyle((s) => s && ({ ...s, primary_color: v }))} />
              <StyleColorField label="Outline color" value={style.outline_color} onChange={(v) => setStyle((s) => s && ({ ...s, outline_color: v }))} />
              <StyleNumberField label="Outline width" value={style.outline_width} onChange={(v) => setStyle((s) => s && ({ ...s, outline_width: v }))} />
              <StyleNumberField label="Margin bottom" value={style.margin_bottom} onChange={(v) => setStyle((s) => s && ({ ...s, margin_bottom: v }))} />
            </div>
          </details>
        </div>

        {/* Right: segment editor */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <Typography variant="h3" className="mb-3">
            Segments ({segments.length})
          </Typography>
          {segments.map((seg, idx) => (
            <div key={seg.id} className="rounded-lg border p-3 space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-16 shrink-0">
                  {formatTime(seg.start)}–{formatTime(seg.end)}
                </span>
                <button
                  onClick={() => toggleLang(idx)}
                  className={`px-2 py-0.5 rounded text-xs font-medium shrink-0 transition-colors ${
                    seg.lang === "vi"
                      ? "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300"
                      : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300"
                  }`}
                >
                  {seg.lang.toUpperCase()}
                </button>
                <input
                  className="flex-1 min-w-0 bg-transparent border-b border-transparent hover:border-input focus:border-primary outline-none px-1"
                  value={seg.text}
                  onChange={(e) => updateSegment(idx, "text", e.target.value)}
                  placeholder="Caption text"
                />
              </div>
              {seg.lang === "vi" && (
                <div className="flex items-center gap-2 pl-[4.5rem]">
                  <span className="text-xs text-muted-foreground shrink-0">phonetic</span>
                  <input
                    className="flex-1 min-w-0 bg-transparent border-b border-transparent hover:border-input focus:border-primary outline-none px-1 italic text-muted-foreground"
                    value={seg.phonetic}
                    onChange={(e) => updateSegment(idx, "phonetic", e.target.value)}
                    placeholder="[pronunciation guide]"
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small style field helpers
// ---------------------------------------------------------------------------

function StyleField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 shrink-0 text-muted-foreground">{label}</span>
      <input
        className="flex-1 bg-transparent border rounded px-2 py-0.5 text-sm outline-none focus:border-primary"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

function StyleNumberField({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-28 shrink-0 text-muted-foreground">{label}</span>
      <input
        type="number"
        className="w-24 bg-transparent border rounded px-2 py-0.5 text-sm outline-none focus:border-primary"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

function StyleColorField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  // value is ASS &HAABBGGRR format — display as hex for editing convenience
  const toHex = (ass: string): string => {
    const m = ass.match(/^&H[0-9A-Fa-f]{2}([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})([0-9A-Fa-f]{2})$/);
    if (!m) return "#ffffff";
    const [, bb, gg, rr] = m;
    return `#${rr}${gg}${bb}`.toLowerCase();
  };
  const toAss = (hex: string): string => {
    const m = hex.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    if (!m) return value;
    const [, r, g, b] = m;
    return `&H00${b}${g}${r}`.toUpperCase();
  };

  return (
    <div className="flex items-center gap-2">
      <span className="w-28 shrink-0 text-muted-foreground">{label}</span>
      <input
        type="color"
        className="w-8 h-6 rounded cursor-pointer border"
        value={toHex(value)}
        onChange={(e) => onChange(toAss(e.target.value))}
      />
      <span className="text-xs text-muted-foreground">{toHex(value)}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}
