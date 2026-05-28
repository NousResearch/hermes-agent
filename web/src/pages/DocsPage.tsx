import { useLayoutEffect, useState } from "react";
import { BookOpen, ExternalLink, FileText, Globe } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@/components/ui/card";
import GuidePage from "@/pages/GuidePage";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";
import { PluginSlot } from "@/plugins";

export const HERMES_DOCS_URL = "https://hermes-agent.nousresearch.com/docs/";

const DS_BUTTON_OUTLINED_LINK_CN = cn(
  "group relative inline-grid grid-cols-[auto_1fr_auto] items-center",
  "px-[.9em_.75em] py-[1.25em] gap-2",
  "leading-0 font-bold tracking-[0.2em] uppercase",
  "text-midground bg-transparent shadow-midground",
  "shadow-[inset_-1px_-1px_0_0_#00000080,inset_1px_1px_0_0_#ffffff80]",
);

export default function DocsPage() {
  const { t } = useI18n();
  const { setEnd } = usePageHeader();
  const [section, setSection] = useState<"guide" | "official">("guide");

  useLayoutEffect(() => {
    setEnd(
      <a
        href={HERMES_DOCS_URL}
        target="_blank"
        rel="noopener noreferrer"
        className={DS_BUTTON_OUTLINED_LINK_CN}
      >
        <ExternalLink className="size-3.5" />
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
        "flex min-h-0 w-full min-w-0 flex-1 flex-col gap-4",
        "pt-1 sm:pt-2",
      )}
    >
      <PluginSlot name="docs:top" />
      <Card>
        <CardContent className="space-y-4 py-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-sm">
                <BookOpen className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">Documentation Center</span>
              </div>
              <p className="text-xs text-muted-foreground">
                这里是文档入口，不把本地 9119 操作手册和 Hermes 官方文档混成一篇。
              </p>
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                outlined={section !== "guide"}
                onClick={() => setSection("guide")}
              >
                9119 操作手册
              </Button>
              <Button
                size="sm"
                outlined={section !== "official"}
                onClick={() => setSection("official")}
              >
                Hermes 官方文档
              </Button>
            </div>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <button
              type="button"
              onClick={() => setSection("guide")}
              className={cn(
                "border p-4 text-left transition-colors",
                section === "guide"
                  ? "border-primary bg-primary/10"
                  : "border-border bg-secondary/10 hover:bg-secondary/30",
              )}
            >
              <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                <FileText className="h-4 w-4 text-muted-foreground" />
                9119 操作手册
              </div>
              <p className="text-xs text-muted-foreground">
                只讲你的本地后台怎么用：Agents、模型策略、Delegations、Logs、Skills、Plugins。
              </p>
            </button>
            <button
              type="button"
              onClick={() => setSection("official")}
              className={cn(
                "border p-4 text-left transition-colors",
                section === "official"
                  ? "border-primary bg-primary/10"
                  : "border-border bg-secondary/10 hover:bg-secondary/30",
              )}
            >
              <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                <Globe className="h-4 w-4 text-muted-foreground" />
                Hermes 官方文档
              </div>
              <p className="text-xs text-muted-foreground">
                上游 Hermes Agent 的通用产品文档，通过独立 iframe 查看。
              </p>
            </button>
          </div>
        </CardContent>
      </Card>
      {section === "guide" ? (
        <GuidePage embedded />
      ) : (
        <iframe
          title={t.app.nav.documentation}
          src={HERMES_DOCS_URL}
          className={cn(
            "min-h-[720px] w-full min-w-0 flex-1",
            "rounded-sm border border-current/20",
            // Docusaurus paints over a transparent <html> / <body> and
            // relies on the browser's canvas color (light by default) to
            // fill the viewport. Inheriting the dashboard's dark color
            // scheme makes that canvas dark, so the docs body text — which
            // is tuned for a light canvas — becomes near-invisible. Force a
            // light color scheme + white background on the iframe element so
            // the docs render cleanly regardless of the active dashboard
            // theme or the user's prefers-color-scheme.
            "[color-scheme:light] bg-white",
          )}
          sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
          referrerPolicy="no-referrer-when-downgrade"
        />
      )}
      <PluginSlot name="docs:bottom" />
    </div>
  );
}
