/**
 * MemoryInspector — slide-over panel showing the agent's built-in memory
 * files (MEMORY.md and USER.md) with view/edit capability.
 */

import { Brain, Save, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { Markdown } from "@/components/Markdown";
import { Button } from "@nous-research/ui/ui/components/button";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type TabKey = "user" | "memory";

interface MemoryContent {
  memory: string;
  user: string;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export function MemoryInspector({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [content, setContent] = useState<MemoryContent>({
    memory: "",
    user: "",
  });
  const [activeTab, setActiveTab] = useState<TabKey>("user");
  const [editMode, setEditMode] = useState(false);
  const [editValue, setEditValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      void api
        .getMemoryContent()
        .then((data) => {
          setContent(data);
          setError(null);
        })
        .catch((e) => {
          setError(e.message);
        });
    }
  }, [open]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaved(false);
    try {
      const key = activeTab === "user" ? "user" : "memory";
      await api.putMemoryContent({ [key]: editValue });
      setContent((prev) => ({ ...prev, [key]: editValue }));
      setEditMode(false);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save");
    } finally {
      setSaving(false);
    }
  }, [activeTab, editValue]);

  const handleEdit = useCallback(() => {
    const key = activeTab === "user" ? "user" : "memory";
    setEditValue(content[key]);
    setEditMode(true);
  }, [activeTab, content]);

  const tabLabel = (key: TabKey) =>
    key === "user" ? "User Profile" : "Memory";
  const fileName = (key: TabKey) => (key === "user" ? "USER.md" : "MEMORY.md");
  const fileSize = (key: TabKey) => {
    const text = content[key];
    return `${text.length} bytes`;
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div className="relative z-10 flex h-[80vh] w-full max-w-2xl flex-col overflow-hidden rounded-xl border border-border bg-background shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            <span className="font-semibold text-foreground">
              Memory Inspector
            </span>
          </div>
          <Button
            ghost
            size="icon"
            onClick={onClose}
            className="h-6 w-6 text-text-secondary hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border">
          {(["user", "memory"] as TabKey[]).map((tab) => (
            <button
              key={tab}
              onClick={() => {
                setActiveTab(tab);
                setEditMode(false);
              }}
              className={cn(
                "flex-1 px-4 py-2.5 text-xs font-medium transition-colors",
                "border-b-2",
                activeTab === tab
                  ? "border-primary text-foreground"
                  : "border-transparent text-text-secondary hover:text-foreground",
              )}
            >
              {tabLabel(tab)}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {error && (
            <div className="border border-destructive/50 bg-destructive/10 text-destructive px-3 py-2 text-xs">
              {error}
            </div>
          )}

          {editMode ? (
            <div className="flex h-full flex-col">
              <div className="flex items-center justify-between border-b border-border px-3 py-2 text-xs text-text-secondary">
                <span>
                  Editing {fileName(activeTab)} — {fileSize(activeTab)}
                </span>
                <div className="flex gap-2">
                  <Button
                    ghost
                    size="sm"
                    onClick={() => setEditMode(false)}
                    className="h-6 px-2 text-xs"
                  >
                    Cancel
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleSave}
                    disabled={saving}
                    className="h-6 gap-1 px-2 text-xs"
                  >
                    <Save className="h-3 w-3" />
                    {saving ? "Saving…" : saved ? "Saved" : "Save"}
                  </Button>
                </div>
              </div>
              <textarea
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                className="flex-1 resize-none bg-background p-3 font-mono text-xs text-foreground outline-none"
                spellCheck={false}
              />
            </div>
          ) : (
            <div className="h-full overflow-y-auto p-4">
              <div className="mb-3 flex items-center justify-between text-xs text-text-secondary">
                <span>{fileName(activeTab)}</span>
                <span>{fileSize(activeTab)}</span>
              </div>
              <div className="prose prose-invert prose-sm max-w-none">
                <Markdown content={content[activeTab]} />
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-border px-4 py-2">
          {!editMode && (
            <Button
              size="sm"
              onClick={handleEdit}
              className="h-7 gap-1.5 px-3 text-xs"
            >
              <Save className="h-3 w-3" />
              Edit
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
