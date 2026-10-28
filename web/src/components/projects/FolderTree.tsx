import type { FolderEntry } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";

interface FolderTreeProps {
  folders: FolderEntry[];
  missing: string[];
  onApplyTemplate?: () => void;
}

export function FolderTree({ folders, missing, onApplyTemplate }: FolderTreeProps) {
  return (
    <div>
      <div className="font-mono text-[11.5px] leading-relaxed text-midground/70 space-y-0.5">
        {folders.map((f) => (
          <div key={f.path}>
            <span className="text-midground font-medium">{f.path}</span>
            {f.isPrimary && (
              <span className="ml-1.5 text-midground/40">primary</span>
            )}
            {!f.isPrimary && (
              <span className="ml-1.5 text-midground/40">
                {f.sessionCount} sessions
              </span>
            )}
          </div>
        ))}
        {missing.length > 0 && (
          <div className="mt-1">
            <span className="text-[11px] font-mono text-yellow-500/80 uppercase tracking-wider">
              Missing:
            </span>
            {missing.map((m) => (
              <div key={m} className="text-yellow-500/70 ml-2">
                {m}
              </div>
            ))}
          </div>
        )}
      </div>
      {missing.length > 0 && onApplyTemplate && (
        <Button
          size="sm"
          outlined
          className="mt-2 text-xs"
          onClick={onApplyTemplate}
        >
          Apply template
        </Button>
      )}
    </div>
  );
}
