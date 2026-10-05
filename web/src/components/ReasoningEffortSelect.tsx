import { useEffect, useRef, useState } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { api } from "@/lib/api";

export interface ReasoningEffortSelectProps {
  scope: "main" | "delegation";
  refreshKey: number;
  profile: string;
  onSaved(): void;
}

type EffortData = {
  main_raw: string; delegation_raw: string; main_effective: string;
  main_source: "model_override" | "global" | "provider_default"; main_model: string;
  main_custom?: string; delegation_custom?: string;
};
const choices = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"];

export function ReasoningEffortSelect({ scope, refreshKey, profile, onSaved }: ReasoningEffortSelectProps) {
  const [selection, setSelection] = useState({ profile, scope, value: "" });
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState<EffortData | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const requestId = useRef(0);
  const isCurrent = selection.profile === profile && selection.scope === scope;
  const value = isCurrent ? selection.value : "";

  useEffect(() => {
    let active = true;
    const id = ++requestId.current;
    setLoading(true); setBusy(false); setError(""); setEditing(false); setDraft("");
    api.getReasoningEffort(profile).then((result) => {
      if (active && requestId.current === id) {
        const next = result as EffortData;
        setData(next);
        setSelection({ profile, scope, value: scope === "main" ? next.main_raw : next.delegation_raw });
      }
    }).catch(() => { if (active && requestId.current === id) setError("Failed to load reasoning effort"); })
      .finally(() => { if (active && requestId.current === id) setLoading(false); });
    return () => { active = false; };
  }, [scope, refreshKey, profile]);

  const save = async (next: string, target: "global" | "model" = "global") => {
    const previous = value; const saveId = requestId.current;
    setBusy(true); setError("");
    if (target === "global") setSelection({ profile, scope, value: next });
    try {
      const result = await api.setReasoningEffort(scope, next, profile, target, target === "model" ? data?.main_model : undefined) as EffortData & { ok: boolean; scope: string; raw: string };
      if (!result.ok || result.scope !== scope || result.raw !== next) throw new Error("Saved value could not be verified");
      if (requestId.current !== saveId) return;
      setData(result);
      setSelection({ profile, scope, value: scope === "main" ? result.main_raw : result.delegation_raw });
      setEditing(false); setDraft("");
      onSaved();
    } catch {
      if (requestId.current === saveId) { if (target === "global") setSelection({ profile, scope, value: previous }); setError("Failed to save reasoning effort"); }
    } finally { if (requestId.current === saveId) setBusy(false); }
  };

  const custom = scope === "main" ? data?.main_custom : data?.delegation_custom;
  const mainSource = data?.main_source;
  const displayedValue = value === "none" ? "Disabled" : value === "__custom__" ? custom || "Custom value" : value || "Not set";
  const beginEdit = () => {
    const draftValue = data?.main_effective === "none" ? "none" : (data?.main_effective || "medium");
    setDraft(choices.includes(draftValue) ? draftValue : "none");
    setEditing(true); setError("");
  };
  return <div className="flex flex-wrap items-center gap-2">
    <select aria-label={`${scope} reasoning effort`} value={value} disabled={loading || busy || !isCurrent} onChange={(event) => void save(event.target.value)} className="border border-border bg-background px-1.5 py-1 text-xs">
      <option value="">{scope === "delegation" ? "Inherit parent" : "Provider default"}</option>
      <option value="none">Disabled</option>
      {choices.filter((effort) => effort !== "none").map((effort) => <option key={effort} value={effort}>{effort}</option>)}
      {value === "__custom__" && <option value="__custom__">Custom: {custom || "configured"}</option>}
    </select>
    {value === "__custom__" && <span role="status" className="text-xs text-text-tertiary">Configured custom value: {custom || "(not provided)"}</span>}
    {scope === "main" && <div className="flex flex-wrap items-center gap-2 text-xs text-text-tertiary">
      <span role="status">Effective for {data?.main_model || "selected model"}: {data?.main_effective || "provider default"} ({mainSource === "model_override" ? "model override" : mainSource === "global" ? "global setting" : "provider default"}).</span>
      {editing ? <>
        <label className="flex items-center gap-1">Model override
          <select aria-label="Model override effort" value={draft} disabled={busy || !isCurrent} onChange={(event) => setDraft(event.target.value)} className="border border-border bg-background px-1.5 py-1 text-xs">
            <option value="none">Disabled</option>
            {choices.filter((effort) => effort !== "none").map((effort) => <option key={effort} value={effort}>{effort}</option>)}
          </select>
        </label>
        <Button size="sm" outlined disabled={busy || !isCurrent} onClick={() => void save(draft, "model")}>Save</Button>
        <Button size="sm" outlined disabled={busy} onClick={() => { setEditing(false); setDraft(""); }}>Cancel</Button>
      </> : <>
        {mainSource === "model_override" && <Button size="sm" outlined disabled={loading || busy || !isCurrent} onClick={beginEdit}>Edit model override</Button>}
        {mainSource === "model_override" && <Button size="sm" outlined disabled={loading || busy || !isCurrent} onClick={() => void save("", "model")}>Use global setting</Button>}
        {mainSource !== "model_override" && <Button size="sm" outlined disabled={loading || busy || !isCurrent} onClick={beginEdit}>Edit model override</Button>}
      </>}
      <span>Global reasoning default: {displayedValue}</span>
    </div>}
    {loading && <span role="status" className="text-xs text-text-tertiary">Loading…</span>}
    {error && <span role="alert" className="text-xs text-red-500">{error}</span>}
  </div>;
}