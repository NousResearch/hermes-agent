import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

export interface ReasoningEffortSelectProps {
  scope: "main" | "delegation";
  refreshKey: number;
  profile: string;
  onSaved(): void;
}

export function ReasoningEffortSelect({ scope, refreshKey, profile, onSaved }: ReasoningEffortSelectProps) {
  const [selection, setSelection] = useState({ profile, scope, value: "" });
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef(0);
  const isCurrent = selection.profile === profile && selection.scope === scope;
  const value = isCurrent ? selection.value : "";

  useEffect(() => {
    let active = true;
    const id = ++requestId.current;
    setLoading(true);
    setBusy(false);
    setError("");
    api.getReasoningEffort(profile)
      .then((data) => {
        if (active && requestId.current === id) {
          setSelection({ profile, scope, value: scope === "main" ? data.main_raw : data.delegation_raw });
        }
      })
      .catch(() => { if (active && requestId.current === id) setError("Failed to load reasoning effort"); })
      .finally(() => { if (active && requestId.current === id) setLoading(false); });
    return () => { active = false; };
  }, [scope, refreshKey, profile]);

  const save = async (next: string) => {
    const previous = value;
    const saveId = requestId.current;
    setSelection({ profile, scope, value: next });
    setBusy(true);
    setError("");
    try {
      const result = await api.setReasoningEffort(scope, next, profile);
      if (!result.ok || result.scope !== scope || result.raw !== next) throw new Error("Saved value could not be verified");
      if (requestId.current !== saveId) return;
      onSaved();
    } catch {
      if (requestId.current === saveId) {
        setSelection({ profile, scope, value: previous });
        setError("Failed to save reasoning effort");
      }
    } finally {
      if (requestId.current === saveId) setBusy(false);
    }
  };

  return <div className="flex items-center gap-1">
    <select aria-label={`${scope} reasoning effort`} value={value} disabled={loading || busy || !isCurrent} onChange={(event) => void save(event.target.value)} className="border border-border bg-background px-1.5 py-1 text-xs">
      <option value="">{scope === "delegation" ? "Inherit parent" : "Provider default"}</option>
      {["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"].map((effort) => <option key={effort} value={effort}>{effort}</option>)}
    </select>
    {loading && <span role="status" className="text-xs text-text-tertiary">Loading…</span>}
    {error && <span role="alert" className="text-xs text-red-500">{error}</span>}
  </div>;
}
