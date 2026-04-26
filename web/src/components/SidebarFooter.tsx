import { useSidebarStatus } from "@/hooks/useSidebarStatus";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";

export function SidebarFooter() {
  const status = useSidebarStatus();
  const { t } = useI18n();

  return (
    <div
      className={cn(
        "flex shrink-0 items-center justify-between gap-2",
        "border-t border-sidebar-border px-5 py-2.5",
      )}
    >
      <span className="font-mono text-xs tabular-nums text-muted-foreground">
        {status?.version != null ? `v${status.version}` : "—"}
      </span>

      <a
        href="https://nousresearch.com"
        target="_blank"
        rel="noopener noreferrer"
        className="text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {t.app.footer.org}
      </a>
    </div>
  );
}
