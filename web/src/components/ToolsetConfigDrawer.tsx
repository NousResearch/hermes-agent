import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Check, ExternalLink, Loader2, Terminal, X } from "lucide-react";
import { api } from "@/lib/api";
import type {
  ToolsetConfig,
  ToolsetInfo,
  ToolsetProvider,
} from "@/lib/api";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { cn, themedBody } from "@/lib/utils";
import { useModalBehavior } from "@/hooks/useModalBehavior";

interface Props {
  /** The toolset whose backends are being configured. */
  toolset: ToolsetInfo;
  /** Optional profile to scope config reads/writes to (Skills page profile
   *  selector). Omitted = the dashboard process's own profile. */
  profile?: string;
  onClose: () => void;
  /** Called after a toggle/provider/key change so the parent grid refreshes. */
  onChanged: () => void;
}

/**
 * Full configuration surface for a single toolset's backends — the dashboard
 * equivalent of selecting a toolset in the `hermes tools` curses UI: toggle
 * the toolset on/off, pick a provider, enter API keys, and run a provider's
 * post-setup install hook (npm/pip/binary) with a live log tail.
 */
export function ToolsetConfigDrawer({ toolset, profile, onClose, onChanged }: Props) {
  const { toast, showToast } = useToast();
  const [config, setConfig] = useState<ToolsetConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [enabled, setEnabled] = useState(toolset.enabled);
  const [toggling, setToggling] = useState(false);
  const [selecting, setSelecting] = useState<string | null>(null);
  const [activeProvider, setActiveProvider] = useState<string | null>(null);
  // Per-env-var draft input values, keyed by env var name.
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [savingProvider, setSavingProvider] = useState<string | null>(null);
  const [isSet, setIsSet] = useState<Record<string, boolean>>({});

  // Post-setup install log tail state.
  const [postSetupRunning, setPostSetupRunning] = useState(false);
  const [postSetupLog, setPostSetupLog] = useState<string[]>([]);
  const [postSetupKey, setPostSetupKey] = useState<string | null>(null);
  // Bumped each time a post-setup is kicked off, to (re)trigger the poll
  // effect below. Mirrors the SkillsPage HubBrowser action-poll pattern so
  // the recursive timer lives inside the effect (lint-clean — no ref
  // mutation, no self-referencing memo).
  const [postSetupTrigger, setPostSetupTrigger] = useState(0);

  const loadConfig = useCallback(() => {
    // Promise-chain shape (not async/await with a leading synchronous
    // setLoading) so callers in a useEffect don't trip
    // react-hooks/set-state-in-effect — setState only fires inside the
    // async .then/.catch/.finally callbacks.
    return api
      .getToolsetConfig(toolset.name, profile)
      .then((cfg) => {
        setConfig(cfg);
        setActiveProvider(cfg.active_provider);
        const seed: Record<string, boolean> = {};
        for (const p of cfg.providers) {
          for (const e of p.env_vars) seed[e.key] = e.is_set;
        }
        setIsSet(seed);
      })
      .catch(() => showToast("Failed to load toolset config", "error"))
      .finally(() => setLoading(false));
  }, [toolset.name, profile, showToast]);

  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  // Poll the post-setup action's log until it exits. Driven by
  // postSetupTrigger; the recursive timer + cleanup live entirely inside the
  // effect (matches the SkillsPage HubBrowser pattern — lint-clean).
  useEffect(() => {
    if (postSetupTrigger === 0) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const st = await api.getActionStatus("tools-post-setup", 300);
        if (cancelled) return;
        setPostSetupLog(st.lines);
        if (st.running) {
          timer = setTimeout(() => void poll(), 1200);
        } else {
          setPostSetupRunning(false);
          const ok = st.exit_code === 0;
          showToast(
            ok ? "Post-setup complete" : "Post-setup finished with errors",
            ok ? "success" : "error",
          );
          // Refresh — a backend may now report itself configured/available.
          void loadConfig();
          onChanged();
        }
      } catch {
        if (!cancelled) {
          setPostSetupRunning(false);
          showToast("Lost track of the post-setup process", "error");
        }
      }
    };
    // Small delay so the spawned action has a log file to read.
    timer = setTimeout(() => void poll(), 800);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [postSetupTrigger, showToast, loadConfig, onChanged]);

  const handleToggle = async (next: boolean) => {
    setToggling(true);
    try {
      await api.toggleToolset(toolset.name, next, profile);
      setEnabled(next);
      showToast(
        `${toolset.label || toolset.name} ${next ? "enabled" : "disabled"}`,
        "success",
      );
      onChanged();
    } catch {
      showToast("Failed to toggle toolset", "error");
    } finally {
      setToggling(false);
    }
  };

  const handleSelectProvider = async (provider: ToolsetProvider) => {
    setSelecting(provider.name);
    try {
      await api.selectToolsetProvider(toolset.name, provider.name, profile);
      setActiveProvider(provider.name);
      showToast(`Provider set to ${provider.name}`, "success");
      onChanged();
    } catch (e) {
      showToast(
        e instanceof Error ? e.message : "Failed to select provider",
        "error",
      );
    } finally {
      setSelecting(null);
    }
  };

  const handleSaveKeys = async (provider: ToolsetProvider) => {
    const env: Record<string, string> = {};
    for (const e of provider.env_vars) {
      const v = drafts[e.key];
      if (v && v.trim()) env[e.key] = v.trim();
    }
    if (Object.keys(env).length === 0) {
      showToast("Enter at least one value to save", "error");
      return;
    }
    setSavingProvider(provider.name);
    try {
      const res = await api.saveToolsetEnv(toolset.name, env, profile);
      setIsSet((prev) => ({ ...prev, ...res.is_set }));
      // Clear saved drafts so the inputs reset to the "saved" placeholder.
      setDrafts((prev) => {
        const next = { ...prev };
        for (const k of res.saved) delete next[k];
        return next;
      });
      showToast(
        res.saved.length
          ? `Saved ${res.saved.length} key${res.saved.length > 1 ? "s" : ""}`
          : "Nothing to save",
        "success",
      );
      onChanged();
    } catch (e) {
      showToast(
        e instanceof Error ? e.message : "Failed to save keys",
        "error",
      );
    } finally {
      setSavingProvider(null);
    }
  };

  const handleRunPostSetup = async (provider: ToolsetProvider) => {
    if (!provider.post_setup) return;
    setPostSetupRunning(true);
    setPostSetupLog([]);
    setPostSetupKey(provider.post_setup);
    try {
      await api.runToolsetPostSetup(toolset.name, provider.post_setup, profile);
      // Bump the trigger so the poll effect (re)starts tailing the log.
      setPostSetupTrigger((n) => n + 1);
    } catch (e) {
      setPostSetupRunning(false);
      showToast(
        e instanceof Error ? e.message : "Failed to start post-setup",
        "error",
      );
    }
  };

  const labelText = toolset.label?.trim() || toolset.name;
  const dialogRef = useModalBehavior({ open: true, onClose });

  return createPortal(
    <div
      ref={dialogRef}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 p-4"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="toolset-config-title"
      tabIndex={-1}
    >
      <div
        className={cn(
          themedBody,
          "relative w-full max-w-2xl max-h-[85vh] border border-border bg-card shadow-2xl flex flex-col",
        )}
      >
        <Button
          ghost
          size="xs"
          className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
          onClick={onClose}
          aria-label="Close"
        >
          <X />
        </Button>

        {/* Header — toolset identity + enable toggle */}
        <header className="p-5 pb-3 border-b border-border">
          <div className="flex items-center gap-3 pr-8">
            <span
              id="toolset-config-title"
              className="font-mondwest text-display text-base tracking-wider"
            >
              {labelText}
            </span>
            <Badge tone={enabled ? "success" : "outline"} className="text-xs">
              {enabled ? "Active" : "Inactive"}
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {toolset.description}
          </p>
          <div className="mt-3 flex items-center gap-2">
            <Switch
              checked={enabled}
              onCheckedChange={(v) => void handleToggle(v)}
              disabled={toggling}
              aria-label="Enable toolset"
            />
            <span className="text-xs text-muted-foreground">
              {enabled ? "Enabled for the agent" : "Disabled"}
            </span>
          </div>
        </header>

        {/* Body — provider matrix */}
        <div className="flex-1 min-h-0 overflow-y-auto p-5 pt-4 space-y-4">
          {loading ? (
            <div className="flex items-center justify-center py-10">
              <Spinner />
            </div>
          ) : !config?.has_category ? (
            <p className="text-sm text-muted-foreground py-6 text-center">
              This toolset has no configurable backends — toggle it on or off
              above. It works with no provider selection or API keys.
            </p>
          ) : config.providers.length === 0 ? (
            <p className="text-sm text-muted-foreground py-6 text-center">
              No providers are available for this toolset in this install.
            </p>
          ) : (
            config.providers.map((provider) => {
              const isActive = provider.name === activeProvider;
              return (
                <div
                  key={provider.name}
                  className={cn(
                    "border border-border p-3",
                    isActive && "border-emerald-500/60 bg-emerald-500/5",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-medium text-sm">
                        {provider.name}
                      </span>
                      {provider.badge && (
                        <Badge tone="secondary" className="text-xs">
                          {provider.badge}
                        </Badge>
                      )}
                      {provider.requires_nous_auth && (
                        <Badge tone="outline" className="text-xs">
                          Nous Portal
                        </Badge>
                      )}
                    </div>
                    {isActive ? (
                      <Badge tone="success" className="text-xs shrink-0">
                        <Check className="h-3 w-3 mr-0.5" /> Selected
                      </Badge>
                    ) : (
                      <Button
                        size="sm"
                        outlined
                        onClick={() => void handleSelectProvider(provider)}
                        disabled={selecting !== null}
                      >
                        {selecting === provider.name ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          "Select"
                        )}
                      </Button>
                    )}
                  </div>
                  {provider.tag && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {provider.tag}
                    </p>
                  )}

                  {/* API key inputs */}
                  {provider.env_vars.length > 0 && (
                    <div className="mt-3 space-y-2.5">
                      {provider.env_vars.map((ev) => (
                        <div key={ev.key} className="space-y-1">
                          <div className="flex items-center justify-between gap-2">
                            <Label
                              htmlFor={`env-${ev.key}`}
                              className="text-xs font-mono"
                            >
                              {ev.key}
                            </Label>
                            {isSet[ev.key] && (
                              <Badge tone="success" className="text-xs">
                                Saved
                              </Badge>
                            )}
                          </div>
                          <Input
                            id={`env-${ev.key}`}
                            type="password"
                            autoComplete="new-password"
                            autoCapitalize="off"
                            autoCorrect="off"
                            spellCheck={false}
                            className="h-8 rounded-none text-xs font-mono"
                            placeholder={
                              isSet[ev.key]
                                ? "•••••••• (saved — leave blank to keep)"
                                : ev.prompt || ev.key
                            }
                            value={drafts[ev.key] ?? ""}
                            onChange={(e) =>
                              setDrafts((prev) => ({
                                ...prev,
                                [ev.key]: e.target.value,
                              }))
                            }
                          />
                          {ev.url && (
                            <a
                              href={ev.url}
                              target="_blank"
                              rel="noreferrer"
                              className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                            >
                              <ExternalLink className="h-3 w-3" /> Get a key
                            </a>
                          )}
                        </div>
                      ))}
                      <Button
                        size="sm"
                        onClick={() => void handleSaveKeys(provider)}
                        disabled={savingProvider !== null}
                      >
                        {savingProvider === provider.name ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          "Save keys"
                        )}
                      </Button>
                    </div>
                  )}

                  {/* Post-setup install hook */}
                  {provider.post_setup && (
                    <div className="mt-3 border-t border-border pt-3">
                      <p className="text-xs text-muted-foreground mb-1.5">
                        This backend needs a one-time install
                        {" "}
                        <span className="font-mono">
                          ({provider.post_setup})
                        </span>
                        . Runs on this host — may take a few minutes.
                      </p>
                      <Button
                        size="sm"
                        outlined
                        className={cn(
                          postSetupRunning &&
                            postSetupKey === provider.post_setup &&
                            "[&_svg]:animate-spin",
                        )}
                        onClick={() => void handleRunPostSetup(provider)}
                        disabled={postSetupRunning}
                        prefix={
                          postSetupRunning &&
                          postSetupKey === provider.post_setup ? (
                            <Loader2 />
                          ) : (
                            <Terminal />
                          )
                        }
                      >
                        {postSetupRunning &&
                        postSetupKey === provider.post_setup
                          ? "Installing…"
                          : "Run setup"}
                      </Button>
                    </div>
                  )}
                </div>
              );
            })
          )}

          {/* Post-setup live log */}
          {(postSetupRunning || postSetupLog.length > 0) && (
            <div className="border border-border">
              <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border bg-muted/30">
                <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
                <span className="text-xs font-mono text-muted-foreground">
                  post-setup: {postSetupKey}
                </span>
                {postSetupRunning && (
                  <Loader2 className="h-3 w-3 animate-spin ml-auto text-muted-foreground" />
                )}
              </div>
              <pre className="max-h-48 overflow-y-auto p-3 text-xs font-mono whitespace-pre-wrap text-text-secondary">
                {postSetupLog.length ? postSetupLog.join("\n") : "Starting…"}
              </pre>
            </div>
          )}
        </div>
      </div>
      <Toast toast={toast} />
    </div>,
    document.body,
  );
}
