import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { cn } from "@/lib/utils";
import type { CrewNode } from "@/types/crew";
import { CrewHealthBadge } from "./CrewHealthBadge";

export interface CrewProfileCardProps {
  node: CrewNode;
  compact?: boolean;
  onClick?: (node: CrewNode) => void;
  className?: string;
}

export function CrewProfileCard({ node, compact = false, onClick, className }: CrewProfileCardProps) {
  const profile = node.profile;
  return (
    <Card
      className={cn(
        "border-border/70 bg-card/80 transition hover:border-primary/40",
        onClick && "cursor-pointer hover:bg-muted/30",
        className,
      )}
      onClick={() => onClick?.(node)}
    >
      <CardContent className={cn("space-y-3", compact ? "p-3" : "p-4")}>
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-foreground">{node.display_name}</div>
            <div className="truncate text-xs text-muted-foreground">{profile.name}</div>
          </div>
          <CrewHealthBadge health={node.health} reasons={node.health_reasons} />
        </div>
        <div className="space-y-1 text-xs text-muted-foreground">
          <div className="truncate"><span className="text-foreground">Role:</span> {node.role}</div>
          <div className="truncate"><span className="text-foreground">Department:</span> {node.department}</div>
          <div className="truncate"><span className="text-foreground">Manager:</span> {node.manager ?? "—"}</div>
        </div>
        <div className="flex flex-wrap gap-1.5 text-xs">
          <Badge tone="secondary">{profile.gateway_status}</Badge>
          <Badge tone={profile.has_env ? "secondary" : "outline"}>.env {profile.has_env ? "exists" : "missing"}</Badge>
          <Badge tone={profile.has_soul ? "secondary" : "outline"}>SOUL {profile.has_soul ? "exists" : "missing"}</Badge>
          <Badge tone="outline">{profile.skill_count} skills</Badge>
          {node.metadata_status === "missing" && <Badge tone="outline">Needs classification</Badge>}
        </div>
        {!compact && (
          <div className="text-xs text-muted-foreground">
            Model/provider: {profile.model || "—"} / {profile.provider || "—"}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
