import { useLayoutEffect } from "react";
import { ExternalLink } from "lucide-react";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { PluginSlot } from "@/plugins";

export const HERMES_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/";

export default function DocsPage() {
  const { t } = useI18n();
  const { setEnd } = usePageHeader();

  useLayoutEffect(() => {
    setEnd(
      <a
        href={HERMES_DOCS_URL}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          buttonVariants({ variant: "outline", size: "sm" }),
          "h-7 text-xs",
        )}
      >
        <ExternalLink className="mr-1.5 h-3 w-3" />
        {t.app.openDocumentation}
      </a>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, t]);

  return (
    <div
      className={cn(
        "flex min-h-0 w-full min-w-0 flex-1 flex-col",
        "pt-1 sm:pt-2",
      )}
    >
      <PluginSlot name="docs:top" />
      <iframe
        title={t.app.nav.documentation}
        src={HERMES_DOCS_URL}
        className={cn(
          "min-h-0 w-full min-w-0 flex-1",
          "rounded-sm border border-current/20",
          "bg-background",
        )}
        sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
        referrerPolicy="no-referrer-when-downgrade"
      />
      <PluginSlot name="docs:bottom" />
    </div>
  );
}
