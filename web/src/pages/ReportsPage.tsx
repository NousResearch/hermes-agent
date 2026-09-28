import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router";
import { FileText, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { authedFetch, fetchJSON, HERMES_BASE_PATH } from "@/lib/api";
import { reportArtifactPath, rewriteReportLinks } from "@/lib/report-download";

type ReportArtifact = { filename: string; sha256: string };
type Report = {
  report_id: string;
  run_id: string;
  prompt_version: string;
  title: string;
  report_url: string;
  artifacts: ReportArtifact[];
};

export default function ReportsPage() {
  const { reportId } = useParams<{ reportId?: string }>();
  const [reports, setReports] = useState<Report[]>([]);
  const [html, setHtml] = useState("");
  const [error, setError] = useState<string | null>(null);
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const loadAbortRef = useRef<AbortController | null>(null);

  const load = useCallback(async () => {
    setError(null);
    loadAbortRef.current?.abort();
    const controller = new AbortController();
    loadAbortRef.current = controller;
    setHtml("");
    try {
      const result = await fetchJSON<{ reports: Report[] }>("/api/reports", { signal: controller.signal });
      setReports(result.reports);
      const selected = reportId || result.reports[0]?.report_id;
      if (selected) {
        const response = await authedFetch(`/api/reports/${encodeURIComponent(selected)}/report.html`, { signal: controller.signal });
        if (!response.ok) throw new Error(`Report konnte nicht geladen werden (${response.status})`);
        setHtml(rewriteReportLinks(await response.text(), selected, HERMES_BASE_PATH));
      } else {
        setHtml("");
      }
    } catch (cause) {
      if (controller.signal.aborted) return;
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [reportId]);

  useEffect(() => {
    void load();
    return () => loadAbortRef.current?.abort();
  }, [load]);

  const downloadArtifact = useCallback(async (filename: string) => {
    const activeReportId = reportId || reports[0]?.report_id;
    const activeReport = reports.find((item) => item.report_id === activeReportId);
    const requestPath = activeReport && reportArtifactPath(activeReport.report_id, filename, activeReport.artifacts);
    if (!requestPath) { setError("Artefakt gehört nicht zum ausgewählten Report"); return; }
    try {
      setError(null);
      const response = await authedFetch(requestPath);
      if (!response.ok) throw new Error(`Download fehlgeschlagen (${response.status})`);
      const expectedHash = activeReport.artifacts.find((item) => item.filename === filename)?.sha256;
      if (response.headers.get("X-Report-SHA256") !== expectedHash) throw new Error("Download-Hash stimmt nicht mit dem Reportkatalog überein");
      const href = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(href), 60_000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [reportId, reports]);

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      const payload = event.data as { type?: string; url?: string };
      if (event.source !== iframeRef.current?.contentWindow) return;
      if (!(["hermes-report-download", "hermes-report-detail"].includes(payload.type || "")) || !payload.url) return;
      const target = new URL(payload.url, window.location.origin);
      const activeReportId = reportId || reports[0]?.report_id;
      const activePrefix = activeReportId
        ? `${HERMES_BASE_PATH}/api/reports/${encodeURIComponent(activeReportId)}/`
        : "";
      if (!activePrefix || target.origin !== window.location.origin || !target.pathname.startsWith(activePrefix)) return;
      if (payload.type === "hermes-report-detail") {
        if (target.pathname !== `${activePrefix}f001-details.html`) return;
        const activeReport = reports.find((item) => item.report_id === activeReportId);
        const expectedHash = activeReport?.artifacts.find((item) => item.filename === "f001-details.html")?.sha256;
        if (!expectedHash) return;
        void (async () => {
          try {
            const response = await authedFetch(`/api/reports/${encodeURIComponent(activeReportId!)}/f001-details.html`);
            if (!response.ok || response.headers.get("X-Report-SHA256") !== expectedHash) throw new Error("Report-Detail ist nicht verifiziert");
            setHtml(rewriteReportLinks(await response.text(), activeReportId!, HERMES_BASE_PATH));
          } catch (cause) {
            setError(cause instanceof Error ? cause.message : String(cause));
          }
        })();
        return;
      }
      void downloadArtifact(decodeURIComponent(target.pathname.slice(activePrefix.length)));
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [downloadArtifact, reportId, reports]);

  const selected = useMemo(() => reportId ? reports.find((item) => item.report_id === reportId) : reports[0], [reports, reportId]);
  return <div className="flex min-h-0 flex-1 flex-col gap-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div><h1 className="text-xl font-semibold">SEO-Reports</h1><p className="text-sm text-muted-foreground">Authentifizierte, hashgebundene Hermes-Reports</p></div>
      <Button onClick={() => void load()}><RefreshCw className="mr-2 h-4 w-4" />Aktualisieren</Button>
    </div>
    {error && <div role="alert" className="rounded border border-destructive/40 p-3 text-sm">{error}</div>}
    <div className="flex flex-wrap gap-2">{reports.map((item) => <Link key={item.report_id} to={`/reports/${encodeURIComponent(item.report_id)}`} className="rounded border px-3 py-2 text-sm hover:bg-muted"><FileText className="mr-2 inline h-4 w-4" />{item.title}</Link>)}</div>
    {selected && <div aria-label="Report downloads" className="flex flex-wrap gap-2">{selected.artifacts.map((artifact) => <Button key={artifact.filename} type="button" onClick={() => void downloadArtifact(artifact.filename)}>Download {artifact.filename}</Button>)}</div>}
    {selected && html ? <iframe ref={iframeRef} title={selected.title} sandbox="allow-scripts" srcDoc={html} className="min-h-[70vh] flex-1 rounded border bg-background" /> : <div className="rounded border p-6 text-sm">Kein Report verfügbar.</div>}
  </div>;
}
