interface VariantPickerProps {
  value: "workbench" | "overview";
  onChange: (v: "workbench" | "overview") => void;
}

export function VariantPicker({ value, onChange }: VariantPickerProps) {
  return (
    <div
      role="tablist"
      aria-label="View variant"
      className="inline-flex border border-midground/20 rounded overflow-hidden text-[10.5px] font-mono uppercase tracking-widest"
    >
      {(["workbench", "overview"] as const).map((v) => (
        <button
          key={v}
          role="tab"
          aria-selected={value === v}
          onClick={() => onChange(v)}
          className={`px-3 py-1.5 cursor-pointer transition-colors ${
            value === v
              ? "bg-midground/15 text-midground"
              : "text-midground/40 hover:text-midground/70"
          }`}
        >
          {v}
        </button>
      ))}
    </div>
  );
}

const STORAGE_KEY = "hermes.projects.variant";

export function readVariant(): "workbench" | "overview" {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "overview") return "overview";
  } catch {
    // localStorage unavailable
  }
  return "workbench";
}

export function writeVariant(v: "workbench" | "overview"): void {
  try {
    localStorage.setItem(STORAGE_KEY, v);
  } catch {
    // ignore
  }
}
