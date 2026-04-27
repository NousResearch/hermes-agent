import { useState } from "react";
import { FileCode, Copy, Check, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { CodeArtifact, GitDiff as GitDiffType } from "@/types/code";

interface DiffPreviewPanelProps {
  artifacts?: CodeArtifact[];
  gitDiff?: GitDiffType;
  onRefresh?: () => void;
  loading?: boolean;
}

export function DiffPreviewPanel({ artifacts = [], gitDiff, onRefresh, loading }: DiffPreviewPanelProps) {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [copiedPath, setCopiedPath] = useState<string | null>(null);

  const handleCopy = async (text: string, path: string) => {
    await navigator.clipboard.writeText(text);
    setCopiedPath(path);
    setTimeout(() => setCopiedPath(null), 2000);
  };

  const allDiffs = gitDiff?.diffs || [];
  const allFiles = [
    ...artifacts.map((a) => ({
      path: a.path,
      status: a.status,
      additions: a.additions,
      deletions: a.deletions,
      diff: a.diff,
      isArtifact: true,
    })),
    ...allDiffs.map((d) => ({
      path: d.path,
      status: d.status,
      additions: d.additions,
      deletions: d.deletions,
      diff: d.diff,
      isArtifact: false,
    })),
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "added":
        return "text-green-500 bg-green-500/10";
      case "modified":
        return "text-yellow-500 bg-yellow-500/10";
      case "deleted":
        return "text-red-500 bg-red-500/10";
      default:
        return "text-muted-foreground bg-muted";
    }
  };

  if (allFiles.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FileCode className="h-4 w-4" />
              Diff Preview
            </CardTitle>
            {onRefresh && (
              <Button onClick={onRefresh} variant="ghost" size="sm" className="h-7 w-7 p-0">
                <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-muted-foreground">
            <FileCode className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No artifacts or diffs yet</p>
            <p className="text-xs">When Hermes modifies files, they will appear here</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const selectedDiff = selectedFile ? allFiles.find((f) => f.path === selectedFile) : null;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <FileCode className="h-4 w-4" />
            Diff Preview ({allFiles.length} files)
          </CardTitle>
          {onRefresh && (
            <Button onClick={onRefresh} variant="ghost" size="sm" className="h-7 w-7 p-0">
              <RefreshCw className={`h-3 w-3 ${loading ? "animate-spin" : ""}`} />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="list" className="w-full">
          <TabsList className="w-full">
            <TabsTrigger value="list" className="flex-1 text-xs">
              Files ({allFiles.length})
            </TabsTrigger>
            <TabsTrigger value="diff" className="flex-1 text-xs" disabled={!selectedFile}>
              Diff View
            </TabsTrigger>
          </TabsList>
          <TabsContent value="list" className="mt-2">
            <div className="space-y-1 max-h-60 overflow-y-auto">
              {allFiles.map((file) => (
                <div
                  key={file.path}
                  onClick={() => setSelectedFile(file.path)}
                  className={`flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-muted ${
                    selectedFile === file.path ? "bg-muted" : ""
                  }`}
                >
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${getStatusColor(file.status)}`}>
                    {file.status.charAt(0).toUpperCase()}
                  </span>
                  <span className="text-xs font-mono truncate flex-1">{file.path}</span>
                  <div className="flex gap-2 text-[10px] shrink-0">
                    {file.additions > 0 && (
                      <span className="text-green-500">+{file.additions}</span>
                    )}
                    {file.deletions > 0 && (
                      <span className="text-red-500">-{file.deletions}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
          <TabsContent value="diff" className="mt-2">
            {selectedDiff ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[10px]">
                      {selectedDiff.status}
                    </Badge>
                    <span className="text-xs font-mono truncate">{selectedDiff.path}</span>
                  </div>
                  <Button
                    onClick={() => handleCopy(selectedDiff.diff, selectedDiff.path)}
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                  >
                    {copiedPath === selectedDiff.path ? (
                      <Check className="h-3 w-3 text-success" />
                    ) : (
                      <Copy className="h-3 w-3" />
                    )}
                  </Button>
                </div>
                <pre className="text-xs font-mono bg-muted/50 p-3 rounded overflow-auto max-h-80 whitespace-pre">
                  {selectedDiff.diff || "No diff available"}
                </pre>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground text-center py-4">
                Select a file to view its diff
              </p>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
