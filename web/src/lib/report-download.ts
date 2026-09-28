export type ReportDownloadArtifact = { filename: string; sha256: string };

export function reportArtifactPath(
  reportId: string,
  filename: string,
  artifacts: ReportDownloadArtifact[],
): string | null {
  if (!artifacts.some((artifact) => artifact.filename === filename)) return null;
  return `/api/reports/${encodeURIComponent(reportId)}/${encodeURIComponent(filename)}`;
}

export function rewriteReportLinks(source: string, reportId: string, basePath: string): string {
  const prefix = `${basePath}/api/reports/${encodeURIComponent(reportId)}/`;
  const rewritten = source.replace(/(href\s*=\s*["'])([^"'#][^"']*)(["'])/gi, (_m, start, href, end) => {
    if (/^(?:https?:|data:|mailto:|javascript:)/i.test(href)) return `${start}${href}${end}`;
    if (href === `/reports/${reportId}/f001-details.html`) return `${start}${prefix}f001-details.html${end}`;
    if (href.startsWith("/api/reports/")) return `${start}${basePath}${href}${end}`;
    if (href.startsWith(`${basePath}/api/reports/`) || href.startsWith("/")) return `${start}${href}${end}`;
    return `${start}${prefix}${encodeURIComponent(href)}${end}`;
  });
  const bridgePrefix = `${basePath}/api/reports/`;
  const bridge = `<script>(()=>{const p=${JSON.stringify(bridgePrefix)};window.addEventListener('click',e=>{const a=e.target.closest&&e.target.closest('a[href]');if(!a||!a.getAttribute('href').startsWith(p))return;e.preventDefault();window.parent.postMessage({type:a.pathname.endsWith('/f001-details.html')?'hermes-report-detail':'hermes-report-download',url:a.href},'*')})})()</script>`;
  const policy = `<meta http-equiv="Content-Security-Policy" content="sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'">`;
  const narrowLayout = `<style>body{overflow-wrap:anywhere}</style>`;
  const withPolicy = /<head\b[^>]*>/i.test(rewritten)
    ? rewritten.replace(/<head\b[^>]*>/i, (tag) => `${tag}${policy}${narrowLayout}`)
    : `${policy}${narrowLayout}${rewritten}`;
  return /<\/head>/i.test(withPolicy)
    ? withPolicy.replace(/<\/head>/i, `${bridge}</head>`)
    : `${withPolicy}${bridge}`;
}
