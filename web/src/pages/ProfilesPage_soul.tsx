import { Button } from "@nous-research/ui/ui/components/button";
import { Label } from "@nous-research/ui/ui/components/label";
import { useI18n } from "@/i18n";
import type { useSoulEditor } from "./ProfilesPage_soulState";

export function SoulEditorBody({
  name,
  soul,
}: {
  name: string;
  soul: ReturnType<typeof useSoulEditor>;
}) {
  const { t } = useI18n();
  return (
    <>
      <Label
        htmlFor="profile-soul-editor"
        className="font-mondwest text-display text-xs tracking-wider text-muted-foreground"
      >
        {t.profiles.soulSection}
      </Label>

      {soul.readError && (
        <div className="flex items-start justify-between gap-3 border border-destructive/50 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          <span>{soul.readError}</span>
          <Button size="sm" onClick={() => void soul.load(name)}>
            {t.common.retry}
          </Button>
        </div>
      )}

      <textarea
        id="profile-soul-editor"
        className="flex min-h-[280px] w-full border border-input bg-transparent px-3 py-2 text-sm font-mono shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-40"
        placeholder={t.profiles.soulPlaceholder}
        value={soul.text}
        disabled={soul.unread}
        onChange={(e) => soul.setText(e.target.value)}
      />

      <div className="flex justify-end">
        <Button
          size="sm"
          className="uppercase"
          onClick={() => void soul.save(name)}
          disabled={soul.saving || soul.unread}
        >
          {soul.saving ? t.common.saving : t.common.save}
        </Button>
      </div>
    </>
  );
}
