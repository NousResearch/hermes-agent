import { X } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import type { CrewNode } from "@/types/crew";
import { CrewHealthBadge } from "./CrewHealthBadge";

export interface CrewProfileDrawerProps {
  node: CrewNode | null;
  onClose: () => void;
}

function Field({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="break-words text-sm text-foreground">{value == null || value === "" ? "—" : String(value)}</div>
    </div>
  );
}

export function CrewProfileDrawer({ node, onClose }: CrewProfileDrawerProps) {
  if (!node) return null;
  const profile = node.profile;
  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-background/60 backdrop-blur-sm" role="dialog" aria-modal="true">
      <div className="h-full w-full max-w-xl overflow-y-auto border-l border-border bg-background p-5 shadow-xl">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-xl font-semibold text-foreground">{node.display_name}</h2>
              <CrewHealthBadge health={node.health} reasons={node.health_reasons} />
            </div>
            <p className="text-sm text-muted-foreground">{profile.name} · read-only crew profile detail</p>
          </div>
          <Button ghost size="sm" onClick={onClose} aria-label="Close crew profile drawer">
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="space-y-4">
          <Card><CardContent className="grid gap-3 p-4 sm:grid-cols-2">
            <Field label="Role" value={node.role} />
            <Field label="Level" value={node.level} />
            <Field label="Department" value={node.department} />
            <Field label="Manager" value={node.manager} />
            <Field label="Board" value={node.board} />
            <Field label="Metadata" value={node.metadata_status} />
          </CardContent></Card>

          <Card><CardContent className="grid gap-3 p-4 sm:grid-cols-2">
            <Field label="Runtime" value={profile.gateway_status} />
            <Field label="Last seen" value={profile.last_seen_at} />
            <Field label="Current task" value={profile.current_task?.title ?? profile.current_task?.id} />
            <Field label="Recent errors" value={profile.recent_error_count ?? 0} />
          </CardContent></Card>

          <Card><CardContent className="grid gap-3 p-4 sm:grid-cols-2">
            <Field label="Telegram bot" value={node.telegram_bot} />
            <Field label="Telegram topic" value={node.telegram_topic} />
            <Field label="Model" value={profile.model} />
            <Field label="Provider" value={profile.provider} />
            <Field label="Skills" value={profile.skill_count} />
            <Field label="Toolsets" value={(profile.toolsets ?? []).join(", ")} />
          </CardContent></Card>

          <Card><CardContent className="space-y-3 p-4">
            <h3 className="text-sm font-semibold text-foreground">Config Health</h3>
            <div className="flex flex-wrap gap-2">
              <Badge tone={profile.has_env ? "secondary" : "outline"}>.env {profile.has_env ? "exists" : "missing"}</Badge>
              <Badge tone={profile.has_soul ? "secondary" : "outline"}>SOUL.md {profile.has_soul ? "exists" : "missing"}</Badge>
              <Badge tone="outline">Secrets are not read or displayed</Badge>
            </div>
            <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
              {node.health_reasons.map((reason) => <li key={reason}>{reason}</li>)}
            </ul>
          </CardContent></Card>

          <Card><CardContent className="space-y-2 p-4 text-sm text-muted-foreground">
            <h3 className="text-sm font-semibold text-foreground">Guardrails</h3>
            <p>Read-only MVP: no start, stop, restart, edit, delete, or deployment controls are available here.</p>
          </CardContent></Card>
        </div>
      </div>
    </div>
  );
}
