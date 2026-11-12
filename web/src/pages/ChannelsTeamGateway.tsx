import { ExternalLink, Network } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import type { MessagingPlatform, TeamGatewayInfo } from "@/lib/api";

/**
 * Shown in place of the messaging platforms when Team Gateway owns this agent's messaging. The
 * gateway switches every direct messaging bot off at each start there, so their setup here would
 * configure nothing: the agent's channels are configured in Nous Portal.
 */
export function TeamGatewayChannels({
  platforms,
  teamGateway,
}: {
  platforms: MessagingPlatform[];
  teamGateway: TeamGatewayInfo;
}) {
  const names = new Map(platforms.map((p) => [p.id, p.name]));
  const carried = teamGateway.platforms.map((id) => names.get(id) ?? id).join(", ");
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
              bot tokens of its own. Choose which platforms it uses and who can message it under
              Manage Agents in Nous Portal.
            </span>
            {carried && (
              <span className="text-xs text-muted-foreground">
                Available for this agent: {carried}
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
