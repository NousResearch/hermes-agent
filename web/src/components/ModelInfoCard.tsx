import { useEffect, useRef, useState } from "react";
import { Brain, Eye, Gauge, Lightbulb, Wrench } from "lucide-react";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { ModelInfoResponse } from "@/lib/api";
import { formatTokenCount } from "@/lib/format";
import { useI18n } from "@/i18n";

function tr(locale: string, en: string, ja: string): string {
  return locale === "ja" ? ja : en;
}

interface ModelInfoCardProps {
  /** Current model string from config state — used to detect changes */
  currentModel: string;
  /** Bumped after config saves to trigger re-fetch */
  refreshKey?: number;
}

export function ModelInfoCard({
  currentModel,
  refreshKey = 0,
}: ModelInfoCardProps) {
  const { locale } = useI18n();
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const lastFetchKeyRef = useRef("");

  useEffect(() => {
    if (!currentModel) return;
    // Re-fetch when model changes OR when refreshKey bumps (after save)
    const fetchKey = `${currentModel}:${refreshKey}`;
    if (fetchKey === lastFetchKeyRef.current) return;
    lastFetchKeyRef.current = fetchKey;
    setLoading(true);
    api
      .getModelInfo()
      .then(setInfo)
      .catch(() => setInfo(null))
      .finally(() => setLoading(false));
  }, [currentModel, refreshKey]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-2 text-xs text-muted-foreground">
        <Spinner className="text-xs" />
        {tr(locale, "Loading model info…", "モデル情報を読み込み中…")}
      </div>
    );
  }

  if (!info || !info.model || info.effective_context_length <= 0) return null;

  const caps = info.capabilities;
  const hasCaps = caps && Object.keys(caps).length > 0;

  return (
    <div className="border border-border/60 bg-muted/30 px-3 py-2.5 space-y-2">
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Gauge className="h-3.5 w-3.5" />
          <span className="font-medium">{tr(locale, "Context Window", "コンテキスト長")}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono font-semibold text-foreground">
            {formatTokenCount(info.effective_context_length)}
          </span>
          {info.config_context_length > 0 ? (
            <span className="text-amber-500/80 text-[10px]">
              ({tr(locale, "override", "上書き")} — auto: {formatTokenCount(info.auto_context_length)})
            </span>
          ) : (
            <span className="text-muted-foreground/60 text-[10px]">
              {tr(locale, "auto-detected", "自動検出")}
            </span>
          )}
        </div>
      </div>

      {hasCaps && caps.max_output_tokens && caps.max_output_tokens > 0 && (
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5 text-muted-foreground">
            <Lightbulb className="h-3.5 w-3.5" />
            <span className="font-medium">{tr(locale, "Max Output", "最大出力")}</span>
          </div>
          <span className="font-mono font-semibold text-foreground">
            {formatTokenCount(caps.max_output_tokens)}
          </span>
        </div>
      )}

      {hasCaps && (
        <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
          {caps.supports_tools && (
            <span className="inline-flex items-center gap-1 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-600 dark:text-emerald-400">
              <Wrench className="h-2.5 w-2.5" /> {tr(locale, "Tools", "ツール")}
            </span>
          )}
          {caps.supports_vision && (
            <span className="inline-flex items-center gap-1 bg-blue-500/10 px-2 py-0.5 text-[10px] font-medium text-blue-600 dark:text-blue-400">
              <Eye className="h-2.5 w-2.5" /> Vision
            </span>
          )}
          {caps.supports_reasoning && (
            <span className="inline-flex items-center gap-1 bg-purple-500/10 px-2 py-0.5 text-[10px] font-medium text-purple-600 dark:text-purple-400">
              <Brain className="h-2.5 w-2.5" /> {tr(locale, "Reasoning", "推論")}
            </span>
          )}
          {caps.model_family && (
            <span className="inline-flex items-center gap-1 bg-muted px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
              {caps.model_family}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
