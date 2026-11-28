import { useState } from "react";
import { Download } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { CatalogMessagingPlatform } from "@/lib/api";
import { errorMessage } from "@/lib/api-error";
import { useI18n } from "@/i18n";
import type { Translations } from "@/i18n";

type Copy = Translations["channelsCatalog"];

const BADGES: Record<
  CatalogMessagingPlatform["badge"],
  { tone: "success" | "secondary"; label: (c: Copy) => string }
> = {
  official: { tone: "success", label: (c) => c.badgeOfficial },
  verified: { tone: "secondary", label: (c) => c.badgeVerified },
};

// Reason codes the backend sends in `blocked`; anything newer shows the backend's English sentence.
const BLOCKED_COPY: Record<string, (c: Copy) => string> = {
  lazy_installs_off: (c) => c.lazyInstallsOff,
};

interface RowFailure {
  error: string;
  command: string;
}

function RunCommand({ command, copy }: { command: string; copy: Copy }) {
  if (!command) return null;
  return (
    <span className="text-xs text-muted-foreground">
      {copy.run} <code className="font-courier">{command}</code>
    </span>
  );
}

/**
 * Catalog messaging platforms this profile does not have yet (GET /api/messaging/platforms
 * `available`). Install adds and enables the plugin; `onInstalled` gets the platform ids it
 * registered so the page can open the new card's credential form.
 */
export function AvailablePlatforms({
  rows,
  onInstalled,
  showToast,
}: {
  rows: CatalogMessagingPlatform[];
  onInstalled: (platformIds: string[]) => Promise<void>;
  showToast: (message: string, type: "success" | "error") => void;
}) {
  const { t } = useI18n();
  const copy = t.channelsCatalog;
  const [busy, setBusy] = useState<string | null>(null);
  const [failures, setFailures] = useState<Record<string, RowFailure>>({});

  if (rows.length === 0) return null;

  const install = async (row: CatalogMessagingPlatform) => {
    setBusy(row.name);
    setFailures((prev) => {
      const next = { ...prev };
      delete next[row.name];
      return next;
    });
    const fail = (error: string, command: string) => {
      setFailures((prev) => ({ ...prev, [row.name]: { error, command } }));
      showToast(`${copy.installFailed(row.title)}: ${error}`, "error");
    };
    try {
      const result = await api.installMessagingCatalogPlatform(row.name);
      if (!result.ok) {
        fail(result.error, result.install_command || row.install_command);
        return;
      }
      showToast(copy.installed(row.title), "success");
      if (result.warnings.length > 0) showToast(result.warnings.join(" "), "error");
      await onInstalled(result.platforms);
    } catch (e) {
      fail(errorMessage(e), row.install_command);
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="grid gap-3" aria-labelledby="channels-catalog-title">
      <div className="grid gap-1">
        <h2 id="channels-catalog-title" className="font-mondwest text-sm tracking-wider">
          {copy.title}
        </h2>
        <p className="text-xs text-muted-foreground">{copy.hint}</p>
      </div>

      {rows.map((row) => {
        const badge = BADGES[row.badge];
        const failure = failures[row.name];
        const installing = busy === row.name;
        const blockedText = row.blocked
          ? (BLOCKED_COPY[row.blocked]?.(copy) ?? row.blocked_reason)
          : "";
        return (
          <Card key={row.name} className="border-border" data-catalog-platform={row.name}>
            <CardContent className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex min-w-0 flex-col gap-0.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-mondwest normal-case text-sm font-medium">{row.title}</span>
                  {badge && <Badge tone={badge.tone}>{badge.label(copy)}</Badge>}
                </div>
                {row.summary && (
                  <span className="text-xs text-muted-foreground">{row.summary}</span>
                )}
                {row.disclosure && (
                  <span className="text-xs text-muted-foreground">
                    {copy.disclosure} {row.disclosure}
                  </span>
                )}
                {row.blocked && (
                  <>
                    <span className="text-xs text-warning">{blockedText}</span>
                    <RunCommand command={row.install_command} copy={copy} />
                  </>
                )}
                {failure && (
                  <>
                    <span className="text-xs text-destructive">{failure.error}</span>
                    <RunCommand command={failure.command} copy={copy} />
                  </>
                )}
              </div>

              {!row.blocked && (
                <Button
                  size="sm"
                  className="shrink-0 self-start uppercase sm:self-center"
                  disabled={busy !== null}
                  onClick={() => void install(row)}
                  prefix={installing ? <Spinner /> : <Download className="h-4 w-4" />}
                >
                  {installing ? copy.installing : copy.install}
                </Button>
              )}
            </CardContent>
          </Card>
        );
      })}
    </section>
  );
}
