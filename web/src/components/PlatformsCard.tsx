import { AlertTriangle, PowerOff, Radio, Wifi, WifiOff } from "lucide-react";
import type { PlatformStatus } from "@/lib/api";
import { isoTimeAgo } from "@/lib/utils";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { useI18n } from "@/i18n";

export function PlatformsCard({ platforms }: PlatformsCardProps) {
  const { t } = useI18n();
  const platformStateBadge: Record<
    string,
    { tone: "success" | "warning" | "destructive" | "outline"; label: string }
  > = {
    connected: { tone: "success", label: t.status.connected },
    disconnected: { tone: "warning", label: t.status.disconnected },
    disabled: { tone: "outline", label: t.status.disabled ?? "Disabled" },
    fatal: { tone: "destructive", label: t.status.error },
  };

  return (
    <Card className="min-w-0 max-w-full overflow-hidden">
      <CardHeader className="min-w-0 p-3 sm:p-4">
        <div className="flex min-w-0 items-center gap-2">
          <Radio className="h-5 w-5 shrink-0 text-muted-foreground" />
          <CardTitle className="min-w-0 break-words text-base [overflow-wrap:anywhere]">
            {t.status.connectedPlatforms}
          </CardTitle>
        </div>
      </CardHeader>

      <CardContent className="grid min-w-0 gap-2 p-3 sm:gap-3 sm:p-4">
        {platforms.map(([name, info]) => {
          const display = platformStateBadge[info.state] ?? {
            tone: "outline" as const,
            label: info.state,
          };
          const IconComponent =
            info.state === "connected"
              ? Wifi
              : info.state === "fatal"
                ? AlertTriangle
                : info.state === "disabled"
                  ? PowerOff
                  : WifiOff;

          return (
            <div
              key={name}
              className="flex min-w-0 w-full flex-wrap items-start justify-between gap-2 border border-border p-2.5 sm:flex-nowrap sm:items-center sm:p-3"
            >
              <div className="flex min-w-[min(100%,12rem)] flex-1 items-start gap-2.5 sm:items-center sm:gap-3">
                <IconComponent
                  className={`h-4 w-4 shrink-0 ${
                    info.state === "connected"
                      ? "text-success"
                      : info.state === "fatal"
                        ? "text-destructive"
                        : info.state === "disabled"
                          ? "text-muted-foreground"
                          : "text-warning"
                  }`}
                />

                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="font-mondwest normal-case min-w-0 break-words text-sm font-medium capitalize [overflow-wrap:anywhere] sm:truncate">
                    {name}
                  </span>

                  {info.error_message && (
                    <span
                      className={`font-mondwest normal-case min-w-0 break-words text-xs [overflow-wrap:anywhere] ${
                        info.state === "disabled"
                          ? "text-muted-foreground"
                          : "text-destructive"
                      }`}
                    >
                      {info.error_message}
                    </span>
                  )}

                  {info.updated_at && (
                    <span className="font-mondwest normal-case min-w-0 break-words text-xs text-muted-foreground [overflow-wrap:anywhere]">
                      {t.status.lastUpdate}: {isoTimeAgo(info.updated_at)}
                    </span>
                  )}
                </div>
              </div>

              <Badge
                tone={display.tone}
                className="max-w-full shrink-0 self-start whitespace-normal sm:self-center"
              >
                {display.tone === "success" && (
                  <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                )}
                {display.label}
              </Badge>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

interface PlatformsCardProps {
  platforms: [string, PlatformStatus][];
}
