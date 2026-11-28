import { Label } from "@nous-research/ui/ui/components/label";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useI18n } from "@/i18n";

interface WebhookDeliveryTargetProps {
  value: string;
  onChange: (target: string) => void;
}

export function WebhookDeliveryTarget({ value, onChange }: WebhookDeliveryTargetProps) {
  const { profiles } = useProfileScope();
  const { t } = useI18n();
  const explicit = value.startsWith("bot-chat:");
  const mode = explicit ? "bot-chat" : value;
  const targetProfile = explicit ? value.slice("bot-chat:".length) : "__executing__";
  return (
    <div className="grid gap-2">
      <Label htmlFor="webhook-deliver">Deliver to</Label>
      <select id="webhook-deliver" value={mode} onChange={event => onChange(event.target.value)}
        className="h-9 w-full border border-midground/15 bg-background px-3 py-1 font-courier text-sm text-midground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/30">
        <option value="log">Log</option>
        <option value="bot-chat">Bot Chat</option>
        <option value="telegram">Telegram</option>
        <option value="discord">Discord</option>
        <option value="slack">Slack</option>
        <option value="email">Email</option>
        <option value="github_comment">GitHub comment</option>
      </select>
      {mode === "bot-chat" && (
        <div className="grid gap-2">
          <Label htmlFor="webhook-target-profile">{t.common.webhookTargetProfile ?? "Target profile"}</Label>
          <select id="webhook-target-profile" value={targetProfile}
            onChange={event => onChange(event.target.value === "__executing__" ? "bot-chat" : `bot-chat:${event.target.value}`)}
            className="h-9 w-full border border-midground/15 bg-background px-3 py-1 font-courier text-sm text-midground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/30">
            <option value="__executing__">{t.common.webhookExecutingProfile ?? "Executing profile"}</option>
            {profiles.map(profile => <option key={profile} value={profile}>{profile}</option>)}
          </select>
          <p className="text-xs text-muted-foreground">{t.common.webhookBotChatHint ?? "I deliver to the canonical local Bot Chat. The receiving Bot Chat may run its model and incur cost, even with deliver only."}</p>
        </div>
      )}
    </div>
  );
}

export function WebhookExecutionProfile({ profile }: { profile?: string }) {
  const { t } = useI18n();
  return <span className="text-xs text-muted-foreground">{t.common.webhookExecutionProfile ?? "Execution profile"}: {profile ?? "default"}</span>;
}
