import { useState } from "react";
import { Terminal, X, Copy, Check, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { CodeCommand } from "@/types/code";

interface CommandOutputPanelProps {
  commands: CodeCommand[];
  onCancel?: (commandId: string) => void;
}

export function CommandOutputPanel({ commands, onCancel }: CommandOutputPanelProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const copyOutput = async (cmd: CodeCommand) => {
    const text = `stdout:\n${cmd.stdout}\nstderr:\n${cmd.stderr}`;
    await navigator.clipboard.writeText(text);
    setCopiedId(cmd.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getStatusBadge = (cmd: CodeCommand) => {
    switch (cmd.status) {
      case "running":
        return (
          <Badge variant="success" className="text-[10px]">
            <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
            Running
          </Badge>
        );
      case "completed":
        return (
          <Badge variant="outline" className="text-[10px]">
            Exit: {cmd.exit_code ?? 0}
          </Badge>
        );
      case "failed":
        return (
          <Badge variant="destructive" className="text-[10px]">
            Failed
          </Badge>
        );
      case "timeout":
        return (
          <Badge variant="destructive" className="text-[10px]">
            Timeout
          </Badge>
        );
      case "cancelled":
        return (
          <Badge variant="outline" className="text-[10px]">
            Cancelled
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="text-[10px]">
            {cmd.status}
          </Badge>
        );
    }
  };

  if (commands.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Terminal className="h-4 w-4" />
            Commands
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-muted-foreground">
            <Terminal className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No commands executed yet</p>
            <p className="text-xs">Commands will appear here when run during the session</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Terminal className="h-4 w-4" />
          Commands ({commands.length})
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {commands.map((cmd) => {
          const isExpanded = expandedIds.has(cmd.id);
          const hasStderr = cmd.stderr && cmd.stderr.trim().length > 0;
          const outputLines = (cmd.stdout || "").split("\n");
          const displayLines = isExpanded ? outputLines : outputLines.slice(0, 10);
          const isTruncated = outputLines.length > 10 && !isExpanded;

          return (
            <div key={cmd.id} className="border rounded-lg p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <code className="text-xs font-mono truncate">
                    {cmd.command.substring(0, 60)}
                    {cmd.command.length > 60 && "..."}
                  </code>
                  {getStatusBadge(cmd)}
                </div>
                <div className="flex gap-1 shrink-0">
                  {cmd.duration_ms && (
                    <span className="text-[10px] text-muted-foreground">
                      {(cmd.duration_ms / 1000).toFixed(1)}s
                    </span>
                  )}
                  <Button
                    onClick={() => copyOutput(cmd)}
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                  >
                    {copiedId === cmd.id ? (
                      <Check className="h-3 w-3 text-success" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                  {cmd.status === "running" && onCancel && (
                    <Button
                      onClick={() => onCancel(cmd.id)}
                      variant="ghost"
                      size="sm"
                      className="h-7 w-7 p-0"
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              </div>

              {hasStderr && (
                <div className="flex items-start gap-1 text-xs">
                  <AlertCircle className="h-3 w-3 text-destructive shrink-0 mt-0.5" />
                  <pre className="text-destructive/80 font-mono text-[10px] whitespace-pre-wrap break-all">
                    {cmd.stderr.substring(0, 500)}
                    {cmd.stderr.length > 500 && "..."}
                  </pre>
                </div>
              )}

              {cmd.stdout && (
                <pre className="text-xs font-mono text-muted-foreground whitespace-pre-wrap break-all bg-muted/50 p-2 rounded max-h-40 overflow-auto">
                  {displayLines.join("\n")}
                  {isTruncated && (
                    <button
                      onClick={() => toggleExpand(cmd.id)}
                      className="text-xs text-primary hover:underline mt-1 block"
                    >
                      Show {outputLines.length - 10} more lines
                    </button>
                  )}
                </pre>
              )}

              {!isExpanded && outputLines.length > 10 && (
                <button
                  onClick={() => toggleExpand(cmd.id)}
                  className="text-xs text-primary hover:underline"
                >
                  Show less
                </button>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
