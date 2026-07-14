import { useEffect, useState, useCallback, useRef } from "react";
import {
  ShieldCheck,
  ShieldOff,
  ExternalLink,
  RefreshCw,
  Terminal,
} from "lucide-react";
import { api, type OAuthProvider } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { CopyButton } from "@nous-research/ui/ui/components/command-block";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { ConfirmDialog } from "@nous-research/ui/ui/components/confirm-dialog";
import { OAuthLoginModal } from "@/components/OAuthLoginModal";
import { useI18n } from "@/i18n";

interface Props {
  onError?: (msg: string) => void;
  onSuccess?: (msg: string) => void;
}

function formatExpiresAt(
  expiresAt: string | null | undefined,
  expiresInTemplate: string,
): string | null {
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
  const [disconnectTarget, setDisconnectTarget] =
    useState<OAuthProvider | null>(null);
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
    setBusyId(provider.id);
    setDisconnectTarget(null);
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

  const connectedCount =
    providers?.filter((p) => p.status.logged_in).length ?? 0;
  const totalCount = providers?.length ?? 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-5 w-5 text-muted-foreground" />
            <CardTitle className="text-base">
              {t.oauth.providerLogins}
            </CardTitle>
          </div>
          <Button
            ghost
            size="icon"
            className="text-muted-foreground hover:text-foreground"
            onClick={refresh}
            disabled={loading}
            aria-label={t.common.refresh}
          >
            {loading ? <Spinner /> : <RefreshCw />}
          </Button>
        </div>
        <CardDescription>
          {t.oauth.description
            .replace("{connected}", String(connectedCount))
            .replace("{total}", String(totalCount))}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {loading && providers === null && (
          <div className="flex items-center justify-center py-8">
            <Spinner className="text-xl text-primary" />
          </div>
        )}
        {providers && providers.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-8">
            {t.oauth.noProviders}
          </p>
        )}
        <div className="flex flex-col divide-y divide-border">
          {providers?.map((p) => {
            const expiresLabel = formatExpiresAt(
              p.status.expires_at,
              t.oauth.expiresIn,
            );
            const isBusy = busyId === p.id;
            return (
              <div
                key={p.id}
                className="flex items-center justify-between gap-4 py-3"
              >
                <div className="flex items-start gap-3 min-w-0 flex-1">
                  {p.status.logged_in ? (
                    <ShieldCheck className="h-5 w-5 text-success shrink-0 mt-0.5" />
                  ) : (
                    <ShieldOff className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                  )}
                  <div className="flex flex-col min-w-0 gap-0.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-medium text-sm">{p.name}</span>
                      <Badge
                        tone="outline"
                        className="text-xs tracking-wide"
                      >
                        {t.oauth.flowLabels[p.flow]}
                      </Badge>
                      {p.status.logged_in && (
                        <Badge tone="success" className="text-xs">
                          {t.oauth.connected}
                        </Badge>
                      )}
                      {expiresLabel === "expired" && (
                        <Badge tone="destructive" className="text-xs">
                          {t.oauth.expired}
                        </Badge>
                      )}
                      {expiresLabel && expiresLabel !== "expired" && (
                        <Badge tone="outline" className="text-xs">
                          {expiresLabel}
                        </Badge>
                      )}
                    </div>
                    {p.status.logged_in && p.status.token_preview && (
                      <span className="truncate text-xs font-mono-ui text-text-secondary">
                        <span className="text-text-tertiary">token </span>
                        {p.status.token_preview}
                        {p.status.source_label && (
                          <span className="text-text-tertiary">
                            {" "}
                            · {p.status.source_label}
                          </span>
                        )}
                      </span>
                    )}
                    {!p.status.logged_in && (
                      <>
                        <span className="text-xs text-text-secondary">
                          {t.oauth.notConnected.split("{command}")[0].trimEnd()}
                          {t.oauth.notConnected.split("{command}")[1] ?? ""}
                        </span>

                        <div className="flex min-w-0 flex-wrap items-center gap-2">
                          <code className="font-courier truncate text-xs opacity-60">
                            {p.cli_command}
                          </code>

                          <CopyButton
                            text={p.cli_command}
                            label={t.oauth.cli}
                            copiedLabel={t.oauth.copied}
                          />
                        </div>
                      </>
                    )}
                    {p.status.error && (
                      <span className="text-xs text-destructive">
                        {p.status.error}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  {p.docs_url && (
                    <a
                      href={p.docs_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex"
                      title={`Open ${p.name} docs`}
                    >
                      <Button ghost size="icon">
                        <ExternalLink />
                      </Button>
                    </a>
                  )}
                  {!p.status.logged_in && p.flow !== "external" && (
                    <Button
                      size="sm"
                      className="uppercase"
                      onClick={() => setLoginFor(p)}
                    >
                      {t.oauth.login}
                    </Button>
                  )}
                  {p.status.logged_in && p.flow !== "external" && (
                    <Button
                      size="sm"
                      outlined
                      className="uppercase"
                      onClick={() => setDisconnectTarget(p)}
                      disabled={isBusy}
                      prefix={isBusy ? <Spinner /> : undefined}
                    >
                      {t.oauth.disconnect}
                    </Button>
                  )}
                  {p.status.logged_in && p.flow === "external" && (
                    <span className="text-xs text-text-tertiary italic px-2">
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
      <ConfirmDialog
        open={disconnectTarget !== null}
        onCancel={() => setDisconnectTarget(null)}
        onConfirm={() => {
          if (disconnectTarget) void handleDisconnect(disconnectTarget);
        }}
        title={`${t.oauth.disconnect} ${disconnectTarget?.name ?? ""}?`}
        description={`This will remove the stored OAuth tokens for ${disconnectTarget?.name ?? "this provider"}. You will need to re-authenticate to use it again.`}
        destructive
        confirmLabel={t.oauth.disconnect}
      />
    </Card>
  );
}
