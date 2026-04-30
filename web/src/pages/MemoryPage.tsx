import { useEffect, useMemo, useState } from "react";
import { Brain, Plus, Save, Trash2, UserRound } from "lucide-react";
import { api } from "@/lib/api";
import type { BuiltinMemoryResponse, MemoryProvidersResponse } from "@/lib/api";
import { Toast } from "@/components/Toast";
import { useToast } from "@/hooks/useToast";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

type Target = "memory" | "user";

function usageLabel(used: number, limit: number) {
  const pct = limit > 0 ? Math.round((used / limit) * 100) : 0;
  return `${used}/${limit} chars (${pct}%)`;
}

export default function MemoryPage() {
  const [memory, setMemory] = useState<BuiltinMemoryResponse | null>(null);
  const [providers, setProviders] = useState<MemoryProvidersResponse | null>(null);
  const [drafts, setDrafts] = useState<Record<Target, string[]>>({ memory: [], user: [] });
  const [saving, setSaving] = useState<Target | null>(null);
  const { toast, showToast } = useToast();

  const load = () => {
    Promise.all([api.getBuiltinMemory(), api.getMemoryProviders()])
      .then(([mem, providerInfo]) => {
        setMemory(mem);
        setProviders(providerInfo);
        setDrafts({ memory: mem.memory.entries, user: mem.user.entries });
      })
      .catch(() => showToast("Failed to load memory", "error"));
  };

  useEffect(() => {
    load();
  }, []);

  const dirty = useMemo(() => {
    if (!memory) return { memory: false, user: false };
    return {
      memory: JSON.stringify(drafts.memory) !== JSON.stringify(memory.memory.entries),
      user: JSON.stringify(drafts.user) !== JSON.stringify(memory.user.entries),
    };
  }, [drafts, memory]);

  const saveTarget = async (target: Target) => {
    setSaving(target);
    try {
      const updated = await api.saveBuiltinMemory(target, drafts[target]);
      setMemory(updated);
      setDrafts({ memory: updated.memory.entries, user: updated.user.entries });
      showToast("Memory saved", "success");
    } catch (err) {
      showToast(`Save failed: ${err}`, "error");
    } finally {
      setSaving(null);
    }
  };

  const renderTarget = (target: Target) => {
    if (!memory) return null;
    const info = memory[target];
    const Icon = target === "memory" ? Brain : UserRound;
    const title = target === "memory" ? "Agent Memory" : "User Profile";

    return (
      <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
        <Card>
          <CardHeader className="flex-row items-center justify-between gap-3">
            <CardTitle className="flex items-center gap-2">
              <Icon className="h-4 w-4" />
              {title}
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge variant={dirty[target] ? "warning" : "outline"}>
                {usageLabel(info.usage, info.limit)}
              </Badge>
              <Button
                size="sm"
                onClick={() => saveTarget(target)}
                disabled={!dirty[target] || saving === target}
              >
                <Save className="h-3.5 w-3.5" />
                Save
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {drafts[target].length === 0 && (
              <div className="border border-border p-4 text-sm text-muted-foreground">
                No entries yet.
              </div>
            )}
            {drafts[target].map((entry, index) => (
              <div key={index} className="flex gap-2">
                <textarea
                  value={entry}
                  onChange={(e) => {
                    const next = [...drafts[target]];
                    next[index] = e.target.value;
                    setDrafts((prev) => ({ ...prev, [target]: next }));
                  }}
                  className="min-h-24 flex-1 resize-y border border-border bg-background/40 p-3 font-courier text-sm outline-none focus:border-foreground/30 focus:ring-1 focus:ring-foreground/20"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  className="text-muted-foreground hover:text-destructive"
                  onClick={() => {
                    const next = drafts[target].filter((_, i) => i !== index);
                    setDrafts((prev) => ({ ...prev, [target]: next }));
                  }}
                  aria-label="Delete memory entry"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
            <Button
              variant="outline"
              onClick={() => setDrafts((prev) => ({ ...prev, [target]: [...prev[target], ""] }))}
            >
              <Plus className="h-4 w-4" />
              Add Entry
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Provider</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3 text-sm">
            <div className="flex items-center justify-between border border-border p-3">
              <span className="text-muted-foreground">Active</span>
              <Badge variant={providers?.external_configured ? "success" : "outline"}>
                {providers?.active_provider || "builtin"}
              </Badge>
            </div>
            <div className="border border-border p-3 text-xs leading-relaxed text-muted-foreground">
              Built-in curated memory is injected at session start. Changes here affect new sessions and do not mutate active prompt caches.
            </div>
            {providers?.external_configured && (
              <div className="border border-success/30 bg-success/5 p-3 text-xs text-success">
                External provider is configured. Deep provider search/editing is outside this MVP.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  };

  if (!memory) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />
      <div className="flex items-center gap-3">
        <Brain className="h-5 w-5 text-muted-foreground" />
        <h1 className="text-base font-semibold">Memory</h1>
        <span className="text-xs text-muted-foreground">Curated long-term context</span>
      </div>

      <Tabs defaultValue="memory">
        {(active, setActive) => (
          <>
            <TabsList>
              <TabsTrigger active={active === "memory"} value="memory" onClick={() => setActive("memory")}>
                Agent Memory
              </TabsTrigger>
              <TabsTrigger active={active === "user"} value="user" onClick={() => setActive("user")}>
                User Profile
              </TabsTrigger>
            </TabsList>
            {renderTarget(active as Target)}
          </>
        )}
      </Tabs>
    </div>
  );
}
