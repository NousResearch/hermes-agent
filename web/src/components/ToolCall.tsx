import { useState } from "react";
import { ChevronDown, ChevronRight, Wrench } from "lucide-react";

export type ToolStatus = "running" | "done";

export interface ToolCallState {
  id: string;
  name: string;
  preview: string;
  status: ToolStatus;
}

export function ToolCall({ tool }: { tool: ToolCallState }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="my-1 border border-warning/20 bg-warning/5">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-xs text-warning cursor-pointer hover:bg-warning/10 transition-colors"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? (
          <ChevronDown className="h-3 w-3 shrink-0" />
        ) : (
          <ChevronRight className="h-3 w-3 shrink-0" />
        )}
        <Wrench className="h-3 w-3 shrink-0" />
        <span className="font-mono-ui font-medium truncate">{tool.name}</span>
        {tool.status === "running" && (
          <span className="ml-auto h-1.5 w-1.5 rounded-full bg-warning animate-pulse shrink-0" />
        )}
        {tool.status === "done" && (
          <span className="ml-auto text-warning/50 shrink-0">done</span>
        )}
      </button>
      {open && tool.preview && (
        <pre className="border-t border-warning/20 px-3 py-2 text-xs text-warning/80 overflow-x-auto whitespace-pre-wrap font-mono">
          {tool.preview}
        </pre>
      )}
    </div>
  );
}
