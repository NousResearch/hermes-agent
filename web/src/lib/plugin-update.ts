import { api, type AgentPluginScanFinding } from "@/lib/api";

/** The page copy this flow needs; optional keys fall back to English at the call site. */
export interface PluginUpdateCopy {
  consentBody: (name: string, sha: string) => string;
  scanBody: (name: string, count: number) => string;
  scanBlocked: (name: string) => string;
}

const SEVERITY_ORDER = ["critical", "high", "medium", "low"];
const FINDINGS_SHOWN = 8;

const severityRank = (severity: string) => {
  const rank = SEVERITY_ORDER.indexOf(severity);
  return rank === -1 ? SEVERITY_ORDER.length : rank;
};

/** One line per finding, worst first, capped so a native confirm stays readable. */
export function scanFindingLines(findings: AgentPluginScanFinding[]): string[] {
  const sorted = [...findings].sort((a, b) => severityRank(a.severity) - severityRank(b.severity));
  const lines = sorted
    .slice(0, FINDINGS_SHOWN)
    .map((f) => `${f.severity.toUpperCase()}  ${f.file}:${f.line}  ${f.description || f.category}`);
  return sorted.length > FINDINGS_SHOWN ? [...lines, `+${sorted.length - FINDINGS_SHOWN}`] : lines;
}

/** Update one plugin, asking before each override the backend asks for: a catalog pin that widens the
 *  plugin, or new code the security scan rated caution. Each question is asked at most once. Resolves
 *  true when the update applied; throws on a refusal (a dangerous verdict included). */
export async function runDashboardPluginUpdate(
  name: string,
  copy: PluginUpdateCopy,
  ask: (message: string) => boolean = (message) => window.confirm(message),
): Promise<boolean> {
  const options = { acceptCapabilities: false, force: false };
  for (;;) {
    const res = await api.updateAgentPlugin(name, options);
    if (res.consent_required && !options.acceptCapabilities) {
      // The new pin widens the plugin; the backend changed nothing until confirmed.
      const body = [copy.consentBody(name, (res.sha ?? "").slice(0, 8)), ...(res.delta_lines ?? [])];
      if (!ask(body.join("\n"))) return false;
      options.acceptCapabilities = true;
      continue;
    }
    if (res.scan_blocked && res.scan_verdict === "caution" && !options.force) {
      const findings = res.scan_findings ?? [];
      if (!ask([copy.scanBody(name, findings.length), ...scanFindingLines(findings)].join("\n"))) return false;
      options.force = true;
      continue;
    }
    if (res.scan_blocked) throw new Error(copy.scanBlocked(name));
    return true;
  }
}
