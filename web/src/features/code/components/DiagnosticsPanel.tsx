import { useState } from "react";
import { AlertTriangle, RefreshCw, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { DiagnosticsResult, Diagnostic } from "@/types/code";

interface DiagnosticsPanelProps {
  diagnostics?: DiagnosticsResult;
  loading?: boolean;
  onRunDiagnostics?: () => void;
  onRestartLSP?: () => void;
}

export function DiagnosticsPanel({
  diagnostics,
  loading,
  onRunDiagnostics,
  onRestartLSP,
}: DiagnosticsPanelProps) {
  const [filter, setFilter] = useState<"all" | "error" | "warning" | "info">("all");

  const getSeverityIcon = (severity: Diagnostic["severity"]) => {
    switch (severity) {
      case "error":
        return <AlertTriangle className="h-3 w-3 text-destructive" />;
      case "warning":
        return <AlertTriangle className="h-3 w-3 text-yellow-500" />;
      case "info":
        return <AlertTriangle className="h-3 w-3 text-blue-500" />;
      case "hint":
        return <AlertTriangle className="h-3 w-3 text-muted-foreground" />;
    }
  };

  const filteredDiagnostics = diagnostics?.diagnostics.filter((d) => {
    if (filter === "all") return true;
    return d.severity === filter;
  }) || [];

  const groupedByFile = filteredDiagnostics.reduce(
    (acc, diag) => {
      if (!acc[diag.file]) {
        acc[diag.file] = [];
      }
      acc[diag.file].push(diag);
      return acc;
    },
    {} as Record<string, Diagnostic[]>
  );

  if (!diagnostics && !loading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Diagnostics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-muted-foreground">
            <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No diagnostics available</p>
            <p className="text-xs">Run diagnostics to check for errors in your code</p>
            {onRunDiagnostics && (
              <Button onClick={onRunDiagnostics} variant="outline" size="sm" className="mt-3">
                <Play className="h-4 w-4 mr-1" />
                Run Diagnostics
              </Button>
            )}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Diagnostics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-6">
            <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!diagnostics) return null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Diagnostics
          </CardTitle>
          <div className="flex gap-2">
            {onRestartLSP && (
              <Button onClick={onRestartLSP} variant="ghost" size="sm" className="h-7 text-xs">
                Restart LSP
              </Button>
            )}
            {onRunDiagnostics && (
              <Button onClick={onRunDiagnostics} variant="outline" size="sm" className="h-7 text-xs">
                <RefreshCw className="h-3 w-3 mr-1" />
                Refresh
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Summary */}
        <div className="flex gap-2">
          {diagnostics.summary.errors > 0 && (
            <Badge variant="destructive" className="text-[10px]">
              {diagnostics.summary.errors} errors
            </Badge>
          )}
          {diagnostics.summary.warnings > 0 && (
            <Badge variant="warning" className="text-[10px]">
              {diagnostics.summary.warnings} warnings
            </Badge>
          )}
          {diagnostics.summary.info > 0 && (
            <Badge variant="outline" className="text-[10px]">
              {diagnostics.summary.info} info
            </Badge>
          )}
          <Badge variant="outline" className="text-[10px]">
            Total: {diagnostics.summary.total}
          </Badge>
        </div>

        {/* Filter */}
        <Tabs
          value={filter}
          onValueChange={(v) => setFilter(v as typeof filter)}
        >
          <TabsList className="w-full">
            <TabsTrigger value="all" className="flex-1 text-xs">
              All ({diagnostics.summary.total})
            </TabsTrigger>
            <TabsTrigger value="error" className="flex-1 text-xs">
              Errors ({diagnostics.summary.errors})
            </TabsTrigger>
            <TabsTrigger value="warning" className="flex-1 text-xs">
              Warnings ({diagnostics.summary.warnings})
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {/* Diagnostics List */}
        {filteredDiagnostics.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            No {filter === "all" ? "" : filter} diagnostics found
          </p>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {Object.entries(groupedByFile).map(([file, diags]) => (
              <div key={file} className="border rounded p-2 space-y-1">
                <span className="text-xs font-mono font-medium truncate block">{file}</span>
                {diags.map((diag, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-xs">
                    {getSeverityIcon(diag.severity)}
                    <span className="text-muted-foreground shrink-0">
                      {diag.line !== null ? `${diag.line}` : ""}
                      {diag.column !== null ? `:${diag.column}` : ""}
                    </span>
                    <span className="truncate">{diag.message}</span>
                    {diag.source && (
                      <Badge variant="outline" className="text-[10px] shrink-0">
                        {diag.source}
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

        {diagnostics.commands_run.length > 0 && (
          <div className="text-[10px] text-muted-foreground">
            Commands run: {diagnostics.commands_run.join(", ")}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
