import { useState } from "react";
import { Send, Square, GitBranch, Eye, PlusCircle, MinusCircle, FileDiff, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useCodeWorkspaceStore } from "@/stores/codeStore";

type ExecutionMode = "read" | "safe_edit" | "approval" | "full";

const MODES: { id: ExecutionMode; label: string; description: string; color: string }[] = [
  { id: "read", label: "Read", description: "Read-only, no changes", color: "text-muted-foreground" },
  { id: "safe_edit", label: "Safe Edit", description: "Edits with review", color: "text-blue-400" },
  { id: "approval", label: "Approval", description: "Executes with approval", color: "text-warning" },
  { id: "full", label: "Full", description: "Unrestricted execution", color: "text-destructive" },
];

interface CockpitBottomBarProps {
  workspaceId: string | null;
  sessionActive: boolean;
  onReviewChanges?: () => void;
  onPrepareCommit?: () => void;
}

export function CockpitBottomBar({
  workspaceId,
  sessionActive,
  onReviewChanges,
  onPrepareCommit,
}: CockpitBottomBarProps) {
  const { gitDiff } = useCodeWorkspaceStore();
  const [mode, setMode] = useState<ExecutionMode>("approval");
  const [requestInput, setRequestInput] = useState("");
  const [sending, setSending] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  const diff = workspaceId ? gitDiff[workspaceId] : undefined;
  const changedFiles = diff?.diffs.length ?? 0;
  const additions = diff?.total_additions ?? 0;
  const deletions = diff?.total_deletions ?? 0;

  const handleSendRequest = async () => {
    if (!requestInput.trim()) return;
    setSending(true);
    try {
      // Dispatch via chat panel input — the actual sending goes through ChatPanel
      // This component emits an event that ChatPanel picks up
      window.dispatchEvent(
        new CustomEvent("cockpit:send-request", { detail: { message: requestInput.trim(), mode } })
      );
      setRequestInput("");
    } finally {
      setSending(false);
    }
  };

  const currentMode = MODES.find((m) => m.id === mode)!;

  if (collapsed) {
    return (
      <div className="flex items-center justify-between px-4 py-1.5 border-t border-border bg-card/50">
        <div className="flex items-center gap-3">
          <button onClick={() => setCollapsed(false)} className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors">
            <ChevronUp className="h-3 w-3 rotate-180" />
            <span className="text-[10px] font-compressed tracking-widest uppercase">Expand</span>
          </button>
          {changedFiles > 0 && (
            <Badge variant="outline" className="text-[10px] gap-1">
              <FileDiff className="h-3 w-3" />
              {changedFiles} file{changedFiles !== 1 ? "s" : ""} changed
            </Badge>
          )}
          {sessionActive && (
            <Badge variant="success" className="text-[10px]">
              <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse mr-1" />
              Agent running
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-[10px] font-compressed tracking-widest uppercase ${currentMode.color}`}>
            {currentMode.label}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="border-t border-border bg-card/50">
      {/* Top row — file changes summary */}
      <div className="flex items-center justify-between px-4 py-1.5">
        <div className="flex items-center gap-3">
          {changedFiles > 0 ? (
            <>
              <button onClick={onReviewChanges} className="flex items-center gap-1.5 hover:opacity-80 transition-opacity">
                <FileDiff className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-xs">
                  <span className="font-bold">{changedFiles}</span>
                  <span className="text-muted-foreground ml-1">file{changedFiles !== 1 ? "s" : ""}</span>
                </span>
              </button>
              <div className="flex items-center gap-1.5">
                <PlusCircle className="h-3 w-3 text-success" />
                <span className="text-xs text-success font-mono">+{additions}</span>
                <MinusCircle className="h-3 w-3 text-destructive ml-1" />
                <span className="text-xs text-destructive font-mono">-{deletions}</span>
              </div>
            </>
          ) : (
            <span className="text-xs text-muted-foreground">No changes detected</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {onReviewChanges && changedFiles > 0 && (
            <Button size="sm" variant="ghost" className="h-7 text-[10px]" onClick={onReviewChanges}>
              <Eye className="h-3 w-3 mr-1" />
              Review
            </Button>
          )}
          {onPrepareCommit && changedFiles > 0 && (
            <Button size="sm" variant="ghost" className="h-7 text-[10px]" onClick={onPrepareCommit}>
              <GitBranch className="h-3 w-3 mr-1" />
              Prepare commit
            </Button>
          )}
          <button onClick={() => setCollapsed(true)} className="text-muted-foreground hover:text-foreground transition-colors">
            <ChevronUp className="h-3 w-3" />
          </button>
        </div>
      </div>

      {/* Mode selector */}
      <div className="flex items-center gap-1 px-4 pb-1.5">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`px-2 py-0.5 text-[10px] font-compressed tracking-widest uppercase border transition-colors ${
              mode === m.id
                ? `${m.color} border-current bg-foreground/5`
                : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/20"
            }`}
            title={m.description}
          >
            {m.label}
          </button>
        ))}
        <span className="text-[10px] text-muted-foreground ml-2">{currentMode.description}</span>
      </div>

      {/* Input row */}
      <div className="flex items-center gap-2 px-4 pb-3">
        <input
          type="text"
          value={requestInput}
          onChange={(e) => setRequestInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSendRequest()}
          placeholder={
            mode === "read"
              ? "Request a read operation..."
              : mode === "safe_edit"
                ? "Describe the change you need..."
                : mode === "approval"
                  ? "What should the agent do? (requires approval)"
                  : "Full access — what should the agent do?"
          }
          className="flex-1 px-3 py-1.5 text-xs border border-border rounded bg-background placeholder:text-muted-foreground focus:outline-none focus:border-foreground/30"
          disabled={sending}
        />
        <Button
          size="sm"
          className="h-8 px-4"
          onClick={handleSendRequest}
          disabled={sending || !requestInput.trim()}
        >
          {sending ? (
            <span className="h-3 w-3 border-2 border-background/30 border-t-background rounded-full animate-spin" />
          ) : sessionActive ? (
            <Square className="h-3 w-3" />
          ) : (
            <Send className="h-3 w-3" />
          )}
        </Button>
      </div>
    </div>
  );
}
