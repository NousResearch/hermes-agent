import { useEffect, useState } from "react";
import { Globe } from "lucide-react";
import { fetchJSON, getManagementProfile } from "@/lib/api";
import { errorMessage } from "@/lib/api-error";
import { useI18n } from "@/i18n";

interface EgressRequest {
  ts: number;
  method: string;
  host: string;
  path: string;
  status: number;
  ms: number;
  action: string;
}

interface EgressRequestsResponse {
  enabled: boolean;
  requests: EgressRequest[];
}

function getEgressRequests(sessionId: string, profile = getManagementProfile()) {
  const query = new URLSearchParams({ session: sessionId });
  if (profile) query.set("profile", profile);
  return fetchJSON<EgressRequestsResponse>(`/api/egress/requests?${query}`);
}

interface SessionEgressLogProps {
  sessionId: string;
  profile?: string;
}

// Outbound requests the session's sandbox made through the egress proxy. Renders nothing
// unless egress is enabled: the route answers `enabled: false` and this panel stays absent.
export function SessionEgressLog({ sessionId, profile }: SessionEgressLogProps) {
  const { t } = useI18n();
  const [data, setData] = useState<EgressRequestsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getEgressRequests(sessionId, profile)
      .then((resp) => {
        if (!cancelled) setData(resp);
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, profile]);

  if (error) return <p className="mt-4 text-xs text-destructive">{error}</p>;
  if (!data?.enabled) return null;

  return (
    <section className="mt-4 border-t border-border pt-3" aria-label={t.sessions.egressTitle}>
      <h4 className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-muted-foreground">
        <Globe className="h-3.5 w-3.5" />
        {t.sessions.egressTitle}
      </h4>
      {data.requests.length === 0 ? (
        <p className="text-xs text-muted-foreground">{t.sessions.egressEmpty}</p>
      ) : (
        <table className="w-full table-fixed font-mono text-xs">
          <tbody>
            {data.requests.map((r) => (
              <tr key={`${r.ts}-${r.method}-${r.host}-${r.path}`} className="border-b border-border/40">
                <td className="w-20 py-1 text-muted-foreground">
                  {new Date(r.ts * 1000).toLocaleTimeString()}
                </td>
                <td className="w-20 py-1">{r.method}</td>
                <td className="truncate py-1" title={`${r.host}${r.path}`}>
                  {r.host}
                  <span className="text-muted-foreground">{r.path}</span>
                </td>
                <td className={`w-12 py-1 text-right ${r.status >= 400 ? "text-destructive" : ""}`}>
                  {r.status}
                </td>
                <td className="w-16 py-1 text-right text-muted-foreground">{Math.round(r.ms)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
