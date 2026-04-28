import { useEffect, useState, useCallback, useRef } from "react";
import { ShieldCheck, ShieldOff, ExternalLink, RefreshCw, LogOut, Terminal, LogIn } from "lucide-react";
import { api, type OAuthProvider } from "@/lib/api";
import { Button, CopyButton } from "@nous-research/ui";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { OAuthLoginModal } from "@/components/OAuthLoginModal";
import { useI18n } from "@/i18n";

interface Props {
  onError?: (msg: string) => void;
  onSuccess?: (msg: string) => void;
}

function formatExpiresAt(expiresAt: string | null | undefined, expiresInTemplate: string): string | null {
  if (!expiresAt) return null;
  try {
    const dt = new Date(expiresAt);
    if (Number.isNaN(dt.getTime())) return null;
    const now = Date.now();
    const diff = dt.getTime() - now;
    if (diff < 0) return "expired";
    const mins = Math.floor(diff / 60_000);
    if (mins < 60) return expiresInTemplate.replace("{time}", `${mins}m`);
    const hours = Math.floor(mins / 60);
    if (hours < 24) return expiresInTemplate.replace("{time}", `${hours}h`);
    const days = Math.floor(hours / 24);
    return expiresInTemplate.replace("{time}", `${days}d`);
  } catch {
    return null;
  }
}

export function OAuthProvidersCard({ onError, onSuccess }: Props) {
  const [providers, setProviders] = useState<OAuthProvider[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [loginFor, setLoginFor] = useState<OAuthProvider | null>(null);
  const { t } = useI18n();

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

  const handleDisconnect = async (provider: OAuthProvider) => {
    if (!confirm(`${t.oauth.disconnect} ${provider.name}?`)) {
      return;
    }
    setBusyId(provider.id);
    try {
      await api.disconnectOAuthProvider(provider.id);
      onSuccess?.(`${provider.name} ${t.oauth.disconnect.toLowerCase()}ed`);
      refresh();
    } catch (e) {
      onError?.(`${t.oauth.disconnect} failed: ${e}`);
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
            <CardTitle className="text-base">{t.oauth.providerLogins}</CardTitle>
          </div>
          <Button
            outlined
            onClick={refresh}
            disabled={loading}
            prefix={<RefreshCw className={loading ? "animate-spin" : undefined} />}
          >
            {t.common.refresh}
          </Button>
        </div>
        <CardDescription>
          {t.oauth.description.replace("{connected}", String(connectedCount)).replace("{total}", String(totalCount))}
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
            {t.oauth.noProviders}
          </p>
        )}
        <div className="flex flex-col divide-y divide-border">
          {providers?.map((p) => {
            const expiresLabel = formatExpiresAt(p.status.expires_at, t.oauth.expiresIn);
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
                        {t.oauth.flowLabels[p.flow]}
                      </Badge>
                      {p.status.logged_in && (
                        <Badge variant="success" className="text-[11px]">
                          {t.oauth.connected}
                        </Badge>
                      )}
                      {expiresLabel === "expired" && (
                        <Badge variant="destructive" className="text-[11px]">
                          {t.oauth.expired}
                        </Badge>
                      )}
                      {expiresLabel && expiresLabel !== "expired" && (
                        <Badge variant="outline" className="text-[11px]">
                          {expiresLabel}
                        </Badge>
                      )}
                    </div>
                    {p.status.logged_in && p.status.token_preview && (
                      <code className="text-xs font-mono-ui truncate">
                        <span className="opacity-50">token{" "}</span>
                        {p.status.token_preview}
                        {p.status.source_label && (
                          <span className="opacity-40">
                            {" "}· {p.status.source_label}
                          </span>
                        )}
                      </code>
                    )}
                    {!p.status.logged_in && (
                      <span className="text-xs text-muted-foreground/80">
                        {t.oauth.notConnected.split("{command}")[0]}
                        <code className="text-foreground bg-secondary/40 px-1">
                          {p.cli_command}
                        </code>
                        {t.oauth.notConnected.split("{command}")[1]}
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
                      <Button outlined className="!p-1.5 aspect-square">
                        <ExternalLink className="h-3.5 w-3.5" />
                      </Button>
                    </a>
                  )}
                  {!p.status.logged_in && p.flow !== "external" && (
                    <Button
                      onClick={() => setLoginFor(p)}
                      prefix={<LogIn />}
                    >
                      {t.oauth.login}
                    </Button>
                  )}
                  {!p.status.logged_in && (
                    <CopyButton
                      text={p.cli_command}
                      label={t.oauth.cli}
                      copiedLabel={t.oauth.copied}
                    />
                  )}
                  {p.status.logged_in && p.flow !== "external" && (
                    <Button
                      outlined
                      onClick={() => handleDisconnect(p)}
                      disabled={isBusy}
                      prefix={
                        isBusy ? (
                          <RefreshCw className="animate-spin" />
                        ) : (
                          <LogOut />
                        )
                      }
                    >
                      {t.oauth.disconnect}
                    </Button>
                  )}
                  {p.status.logged_in && p.flow === "external" && (
                    <span className="text-[11px] text-muted-foreground italic px-2">
                      <Terminal className="h-3 w-3 inline mr-0.5" />
                      {t.oauth.managedExternally}
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
            refresh();
          }}
          onSuccess={(msg) => onSuccess?.(msg)}
          onError={(msg) => onError?.(msg)}
        />
      )}
    </Card>
  );
}
