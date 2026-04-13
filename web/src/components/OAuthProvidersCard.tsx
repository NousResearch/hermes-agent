import { useEffect, useState, useCallback, useRef } from "react";
import { ShieldCheck, ShieldOff, Copy, ExternalLink, RefreshCw, LogOut, Terminal, LogIn } from "lucide-react";
import { api, type OAuthProvider } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { OAuthLoginModal } from "@/components/OAuthLoginModal";

/**
 * OAuthProvidersCard — surfaces every OAuth-capable LLM provider with its
 * current connection status, a truncated token preview when connected, and
 * action buttons (Copy CLI command for setup, Disconnect for cleanup).
 *
 * Phase 1 scope: read-only status + disconnect + copy-to-clipboard CLI
 * command. Phase 2 will add in-browser PKCE / device-code flows so users
 * never need to drop to a terminal.
 */

interface Props {
  onError?: (msg: string) => void;
  onSuccess?: (msg: string) => void;
}

const FLOW_LABELS: Record<OAuthProvider["flow"], string> = {
  pkce: "Browser login (PKCE)",
  device_code: "Device code",
  external: "External CLI",
};

function formatExpiresAt(expiresAt: string | null | undefined): string | null {
  if (!expiresAt) return null;
  try {
    const dt = new Date(expiresAt);
    if (Number.isNaN(dt.getTime())) return null;
    const now = Date.now();
    const diff = dt.getTime() - now;
    if (diff < 0) return "expired";
    const mins = Math.floor(diff / 60_000);
    if (mins < 60) return `expires in ${mins}m`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `expires in ${hours}h`;
    const days = Math.floor(hours / 24);
    return `expires in ${days}d`;
  } catch {
    return null;
  }
}

export function OAuthProvidersCard({ onError, onSuccess }: Props) {
  const [providers, setProviders] = useState<OAuthProvider[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  // Provider that the login modal is currently open for. null = modal closed.
  const [loginFor, setLoginFor] = useState<OAuthProvider | null>(null);

  // Use refs for callbacks to avoid re-creating refresh() when parent re-renders
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const refresh = useCallback(() => {
    setLoading(true);
    api
      .getOAuthProviders()
      .then((resp) => setProviders(resp.providers))
      .catch((e) => onErrorRef.current?.(`Failed to load providers: ${e}`))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleCopy = async (provider: OAuthProvider) => {
    try {
      await navigator.clipboard.writeText(provider.cli_command);
      setCopiedId(provider.id);
      onSuccess?.(`Copied: ${provider.cli_command}`);
      setTimeout(() => setCopiedId((v) => (v === provider.id ? null : v)), 1500);
    } catch {
      onError?.("Clipboard write failed — copy the command manually");
    }
  };

  const handleDisconnect = async (provider: OAuthProvider) => {
    if (!confirm(`Disconnect ${provider.name}? You'll need to log in again to use this provider.`)) {
      return;
    }
    setBusyId(provider.id);
    try {
      await api.disconnectOAuthProvider(provider.id);
      onSuccess?.(`${provider.name} disconnected`);
      refresh();
    } catch (e) {
      onError?.(`Disconnect failed: ${e}`);
    } finally {
      setBusyId(null);
    }
  };

  const connectedCount = providers?.filter((p) => p.status.logged_in).length ?? 0;
  const totalCount = providers?.length ?? 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-base">Provider Logins (OAuth)</CardTitle>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={refresh}
            disabled={loading}
            className="text-xs"
          >
            <RefreshCw className={`h-3 w-3 mr-1 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
        <CardDescription>
          {connectedCount} of {totalCount} OAuth providers connected. Login flows currently
          run via the CLI; click <em>Copy command</em> and paste into a terminal to set up.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading && providers === null && (
          <div className="flex items-center justify-center py-8">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          </div>
        )}
        {providers && providers.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-8">
            No OAuth-capable providers detected.
          </p>
        )}
        <div className="flex flex-col divide-y divide-border">
          {providers?.map((p) => {
            const expiresLabel = formatExpiresAt(p.status.expires_at);
            const isBusy = busyId === p.id;
            return (
              <div
                key={p.id}
                className="flex items-center justify-between gap-4 py-3"
              >
                {/* Left: status icon + name + source */}
                <div className="flex items-start gap-3 min-w-0 flex-1">
                  {p.status.logged_in ? (
                    <ShieldCheck className="h-5 w-5 text-success shrink-0 mt-0.5" />
                  ) : (
                    <ShieldOff className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                  )}
                  <div className="flex flex-col min-w-0 gap-0.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-medium text-sm">{p.name}</span>
                      <Badge variant="outline" className="text-[11px] uppercase tracking-wide">
                        {FLOW_LABELS[p.flow]}
                      </Badge>
                      {p.status.logged_in && (
                        <Badge variant="success" className="text-[11px]">
                          Connected
                        </Badge>
                      )}
                      {expiresLabel === "expired" && (
                        <Badge variant="destructive" className="text-[11px]">
                          Expired
                        </Badge>
                      )}
                      {expiresLabel && expiresLabel !== "expired" && (
                        <Badge variant="outline" className="text-[11px]">
                          {expiresLabel}
                        </Badge>
                      )}
                    </div>
                    {p.status.logged_in && p.status.token_preview && (
                      <code className="text-xs text-muted-foreground font-mono-ui truncate">
                        token{" "}
                        <span className="text-foreground">{p.status.token_preview}</span>
                        {p.status.source_label && (
                          <span className="text-muted-foreground/70">
                            {" "}· {p.status.source_label}
                          </span>
                        )}
                      </code>
                    )}
                    {!p.status.logged_in && (
                      <span className="text-xs text-muted-foreground/80">
                        Not connected. Run{" "}
                        <code className="text-foreground bg-secondary/40 px-1 rounded">
                          {p.cli_command}
                        </code>{" "}
                        in a terminal.
                      </span>
                    )}
                    {p.status.error && (
                      <span className="text-xs text-destructive">
                        {p.status.error}
                      </span>
                    )}
                  </div>
                </div>
                {/* Right: action buttons */}
                <div className="flex items-center gap-1.5 shrink-0">
                  {p.docs_url && (
                    <a
                      href={p.docs_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex"
                      title={`Open ${p.name} docs`}
                    >
                      <Button variant="ghost" size="sm" className="h-7 w-7 p-0">
                        <ExternalLink className="h-3.5 w-3.5" />
                      </Button>
                    </a>
                  )}
                  {!p.status.logged_in && p.flow !== "external" && (
                    <Button
                      variant="default"
                      size="sm"
                      onClick={() => setLoginFor(p)}
                      className="text-xs h-7"
                      title={`Start ${p.flow === "pkce" ? "browser" : "device code"} login`}
                    >
                      <LogIn className="h-3 w-3 mr-1" />
                      Login
                    </Button>
                  )}
                  {!p.status.logged_in && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleCopy(p)}
                      className="text-xs h-7"
                      title="Copy CLI command (for external / fallback)"
                    >
                      {copiedId === p.id ? (
                        <>Copied ✓</>
                      ) : (
                        <>
                          <Copy className="h-3 w-3 mr-1" />
                          CLI
                        </>
                      )}
                    </Button>
                  )}
                  {p.status.logged_in && p.flow !== "external" && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleDisconnect(p)}
                      disabled={isBusy}
                      className="text-xs h-7"
                    >
                      {isBusy ? (
                        <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                      ) : (
                        <LogOut className="h-3 w-3 mr-1" />
                      )}
                      Disconnect
                    </Button>
                  )}
                  {p.status.logged_in && p.flow === "external" && (
                    <span className="text-[11px] text-muted-foreground italic px-2">
                      <Terminal className="h-3 w-3 inline mr-0.5" />
                      Managed externally
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </CardContent>
      {loginFor && (
        <OAuthLoginModal
          provider={loginFor}
          onClose={() => {
            setLoginFor(null);
            refresh();  // always refresh on close so token preview updates after login
          }}
          onSuccess={(msg) => onSuccess?.(msg)}
          onError={(msg) => onError?.(msg)}
        />
      )}
    </Card>
  );
}
