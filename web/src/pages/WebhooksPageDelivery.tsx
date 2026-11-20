import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
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
      <Select id="webhook-deliver" value={mode} onValueChange={onChange}>
        <SelectOption value="log">Log</SelectOption>
        <SelectOption value="bot-chat">Bot Chat</SelectOption>
        <SelectOption value="telegram">Telegram</SelectOption>
        <SelectOption value="discord">Discord</SelectOption>
        <SelectOption value="slack">Slack</SelectOption>
        <SelectOption value="email">Email</SelectOption>
        <SelectOption value="github_comment">GitHub comment</SelectOption>
      </Select>
      {mode === "bot-chat" && (
        <div className="grid gap-2">
          <Label htmlFor="webhook-target-profile">{t.common.webhookTargetProfile ?? "Target profile"}</Label>
          <Select id="webhook-target-profile" value={targetProfile}
            onValueChange={profile => onChange(profile === "__executing__" ? "bot-chat" : `bot-chat:${profile}`)}>
            <SelectOption value="__executing__">{t.common.webhookExecutingProfile ?? "Executing profile"}</SelectOption>
            {profiles.map(profile => <SelectOption key={profile} value={profile}>{profile}</SelectOption>)}
          </Select>
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
