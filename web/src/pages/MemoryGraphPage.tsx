import { useLayoutEffect, useMemo } from "react";
import { ExternalLink } from "lucide-react";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";
import { PluginSlot } from "@/plugins";

function memoryGraphSrc(): string {
  const base =
    (typeof window !== "undefined" &&
      (window as Window & { __HERMES_BASE_PATH__?: string }).__HERMES_BASE_PATH__) ||
    "";
  return `${base}/memory-graph/obsidian-memory-graph.html`;
}

export default function MemoryGraphPage() {
  const src = useMemo(() => memoryGraphSrc(), []);
  const { setEnd } = usePageHeader();

  useLayoutEffect(() => {
    setEnd(
      <a
        href={src}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          "group relative inline-grid grid-cols-[auto_1fr_auto] items-center",
          "px-[.9em_.75em] py-[1.25em] gap-2",
          "leading-0 font-bold tracking-[0.2em] uppercase",
          "text-midground bg-transparent shadow-midground",
          "shadow-[inset_-1px_-1px_0_0_#00000080,inset_1px_1px_0_0_#ffffff80]",
        )}
      >
        <ExternalLink className="size-3.5" />
        Open VR tab
      </a>,
    );
    return () => {
      setEnd(null);
    };
  }, [setEnd, src]);

  return (
    <div
      className={cn(
        "flex min-h-0 w-full min-w-0 flex-1 flex-col",
        "pt-1 sm:pt-2",
      )}
    >
      <PluginSlot name="memory-graph:top" />
      <iframe
        title="Obsidian Memory Graph 3D"
        src={src}
        className={cn(
          "min-h-0 w-full min-w-0 flex-1",
          "rounded-sm border border-current/20 bg-[#030308]",
        )}
        allow="xr-spatial-tracking"
        sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
        referrerPolicy="no-referrer-when-downgrade"
      />
      <PluginSlot name="memory-graph:bottom" />
    </div>
  );
}
