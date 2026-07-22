import { Typography } from "@nous-research/ui/ui/components/typography/index";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";
import type { SidebarStatus } from "@/hooks/useSidebarStatus";

export function SidebarFooter({ status }: SidebarFooterProps) {
  const { t } = useI18n();

  // SidebarFooter only reads `version`. Pull it from whichever SidebarStatus
  // branch has the last-known-good data; while `loading` we show the
  // em-dash placeholder, and during `unreachable` we keep the last-seen
  // version so the user knows what they were looking at.
  const data =
    status.kind === "live"
      ? status.data
      : status.kind === "unreachable"
        ? status.lastData
        : null;

  return (
    <div
      className={cn(
        "flex shrink-0 items-center justify-between gap-2",
        "px-5 py-2.5",
        "border-t border-current/10",
      )}
    >
      <Typography
        className="font-mono-ui text-xs tabular-nums tracking-[0.08em] text-text-tertiary lowercase"
      >
        {data?.version != null ? `v${data.version}` : "—"}
      </Typography>

      <a
        href="https://nousresearch.com"
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          "font-sans text-display text-xs tracking-[0.12em] text-midground",
          "transition-opacity hover:opacity-90",
          "focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/40",
        )}
      >
        {t.app.footer.org}
      </a>
    </div>
  );
}

interface SidebarFooterProps {
  status: SidebarStatus;
}