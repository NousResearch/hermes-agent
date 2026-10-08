import { Label } from "@nous-research/ui/ui/components/label";

interface ContextFileListEditorProps {
  value: unknown;
  onChange: (value: string[]) => void;
}

export function ContextFileListEditor({ value, onChange }: ContextFileListEditorProps) {
  return (
    <div className="grid gap-1.5">
      <Label className="text-sm" htmlFor="external-context-files">External Files</Label>
      <textarea
        id="external-context-files"
        className="flex min-h-[112px] w-full border border-input bg-transparent px-3 py-2 font-mono text-xs shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        value={Array.isArray(value) ? value.join("\n") : String(value ?? "")}
        onChange={(e) => onChange(e.target.value.split(/\r?\n/))}
        onBlur={(e) => onChange(e.currentTarget.value.split(/\r?\n/).map((path) => path.trim()).filter(Boolean))}
        placeholder="one path per line"
        rows={4}
      />
    </div>
  );
}
