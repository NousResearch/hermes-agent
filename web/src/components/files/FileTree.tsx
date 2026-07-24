import { ChevronDown, ChevronRight, FileIcon, FileText, Folder, FolderOpen } from "lucide-react";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import type { ManagedFileEntry } from "@/lib/api";
import { previewKind } from "@/lib/file-browser";

interface FileTreeProps {
  path: string;
  label: string;
  depth?: number;
  currentPath: string;
  selectedPath: string | null;
  entriesByDirectory: Record<string, ManagedFileEntry[]>;
  expandedDirectories: Set<string>;
  loadingDirectories: Set<string>;
  onToggle: (path: string) => void;
  onOpenDirectory: (path: string) => void;
  onOpenFile: (entry: ManagedFileEntry) => void;
}

export function FileTree(props: FileTreeProps) {
  const {
    path,
    label,
    depth = 0,
    currentPath,
    selectedPath,
    entriesByDirectory,
    expandedDirectories,
    loadingDirectories,
    onToggle,
    onOpenDirectory,
    onOpenFile,
  } = props;
  const expanded = expandedDirectories.has(path);
  const entries = entriesByDirectory[path];

  return (
    <>
      <div
        className={`group flex min-w-0 items-center pr-2 text-sm ${
          currentPath === path ? "bg-primary/10 text-foreground" : "text-text-secondary"
        }`}
        style={{ paddingLeft: `${Math.max(0, depth) * 12 + 4}px` }}
      >
        <button
          type="button"
          onClick={() => onToggle(path)}
          className="flex h-7 w-6 shrink-0 items-center justify-center text-text-tertiary"
          aria-label={`${expanded ? "Collapse" : "Expand"} ${label}`}
          aria-expanded={expanded}
        >
          {loadingDirectories.has(path) ? (
            <Spinner />
          ) : expanded ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </button>
        <button
          type="button"
          onClick={() => onOpenDirectory(path)}
          className="flex min-w-0 flex-1 items-center gap-2 py-1 text-left hover:text-foreground"
          title={path}
        >
          {expanded ? (
            <FolderOpen className="h-4 w-4 shrink-0 text-warning" />
          ) : (
            <Folder className="h-4 w-4 shrink-0 text-warning" />
          )}
          <span className="truncate font-mono text-xs">{label}</span>
        </button>
      </div>

      {expanded && entries?.map((entry) =>
        entry.is_directory ? (
          <FileTree
            key={entry.path}
            {...props}
            path={entry.path}
            label={entry.name}
            depth={depth + 1}
          />
        ) : (
          <button
            key={entry.path}
            type="button"
            onClick={() => onOpenFile(entry)}
            className={`flex w-full min-w-0 items-center gap-2 py-1.5 pr-2 text-left font-mono text-xs transition hover:bg-background/45 hover:text-foreground ${
              selectedPath === entry.path
                ? "bg-primary/10 text-foreground"
                : "text-text-secondary"
            }`}
            style={{ paddingLeft: `${(depth + 1) * 12 + 30}px` }}
            title={entry.path}
          >
            {previewKind(entry.name, entry.mime_type, entry.size ?? 0) === "markdown" ? (
              <FileText className="h-3.5 w-3.5 shrink-0 text-primary" />
            ) : (
              <FileIcon className="h-3.5 w-3.5 shrink-0 text-text-tertiary" />
            )}
            <span className="truncate">{entry.name}</span>
          </button>
        ),
      )}
    </>
  );
}
