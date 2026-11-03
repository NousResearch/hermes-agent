import { ExternalLink, Network } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import type { TeamGatewayInfo } from "@/lib/api";

const PLATFORM_NAMES: Record<string, string> = {
  discord: "Discord",
  slack: "Slack",
  telegram: "Telegram",
  whatsapp: "WhatsApp",
};

function platformList(ids: string[]): string {
  return ids.map((id) => PLATFORM_NAMES[id] ?? id).join(", ");
}

/**
 * The whole Channels page when Team Gateway owns this agent's messaging. The gateway switches
 * every direct bot off at each start there, so per-platform setup here would only lead nowhere:
 * channels are configured on the agent's page in Nous Portal.
 */
export function TeamGatewayChannels({ teamGateway }: { teamGateway: TeamGatewayInfo }) {
  return (
    <Card className="border-border">
      <CardContent className="flex flex-col gap-4 p-6">
        <div className="flex items-start gap-3">
          <Network className="h-5 w-5 shrink-0 mt-0.5 text-primary" />
          <div className="flex flex-col gap-1">
            <span className="font-mondwest normal-case text-sm font-medium">
              Messaging runs through Team Gateway
            </span>
            <span className="text-sm text-muted-foreground">
              This agent reaches chat platforms through Team Gateway's shared bots, so it needs no
              bot tokens of its own. Choose which platforms it uses and who can message it on the
              agent's page in Nous Portal.
            </span>
            {teamGateway.platforms.length > 0 && (
              <span className="text-xs text-muted-foreground">
                Carried for this agent: {platformList(teamGateway.platforms)}
              </span>
            )}
          </div>
        </div>
        <Button
          size="sm"
          className="w-fit uppercase"
          onClick={() => window.open(teamGateway.portal_url, "_blank", "noopener,noreferrer")}
          prefix={<ExternalLink className="h-4 w-4" />}
        >
          Manage agents in Nous Portal
        </Button>
      </CardContent>
    </Card>
  );
}
