import { useEffect, useState } from "react";
import { Cpu, Check, X, RefreshCw, Zap, DollarSign, Code2, Eye, Brain, Server, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { codeApi } from "@/lib/codeApi";
import type { ProviderInfo, ProviderSelection } from "@/types/code";

interface ProviderModelSwitcherProps {
  compact?: boolean;
  onSelectionChange?: (provider: string, model: string) => void;
}

type Preset = {
  id: string;
  label: string;
  icon: React.ReactNode;
  description: string;
};

const PRESETS: Preset[] = [
  { id: "fast", label: "Fast", icon: <Zap className="h-3 w-3" />, description: "Low latency" },
  { id: "cheap", label: "Cheap", icon: <DollarSign className="h-3 w-3" />, description: "Low cost" },
  { id: "code", label: "Code", icon: <Code2 className="h-3 w-3" />, description: "Best for code" },
  { id: "review", label: "Review", icon: <Eye className="h-3 w-3" />, description: "Code review" },
  { id: "reasoning", label: "Reasoning", icon: <Brain className="h-3 w-3" />, description: "Strong reasoning" },
  { id: "local", label: "Local", icon: <Server className="h-3 w-3" />, description: "Offline capable" },
];

export function ProviderModelSwitcher({ compact = false, onSelectionChange }: ProviderModelSwitcherProps) {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [currentSelection, setCurrentSelection] = useState<ProviderSelection | null>(null);
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<"success" | "error" | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    loadProviders();
  }, []);

  const loadProviders = async () => {
    setLoading(true);
    try {
      const data = await codeApi.getProviders();
      setProviders(data.providers);
      setCurrentSelection(data.current);
      setSelectedProvider(data.current.provider);
      setSelectedModel(data.current.model);
    } catch {
      // Provider may not be available
    } finally {
      setLoading(false);
    }
  };

  const handleProviderChange = (providerId: string) => {
    setSelectedProvider(providerId);
    setTestResult(null);
    const provider = providers.find((p) => p.id === providerId);
    if (provider && provider.models.length > 0) {
      setSelectedModel(provider.models[0].id);
    }
  };

  const handleSave = async () => {
    if (!selectedProvider) return;
    setSaving(true);
    setTestResult(null);
    try {
      await codeApi.selectProvider(selectedProvider, selectedModel);
      setCurrentSelection({ provider: selectedProvider, model: selectedModel });
      onSelectionChange?.(selectedProvider, selectedModel);
      setTestResult(null);
    } catch {
      setTestResult("error");
    } finally {
      setSaving(false);
    }
  };

  const handleTest = async () => {
    if (!selectedProvider) return;
    setTesting(true);
    setTestResult(null);
    try {
      await codeApi.selectProvider(selectedProvider, selectedModel);
      setTestResult("success");
      setTimeout(() => setTestResult(null), 3000);
    } catch {
      setTestResult("error");
    } finally {
      setTesting(false);
    }
  };

  const handlePreset = (preset: Preset) => {
    // Apply preset heuristics — pick first provider that matches, then best model
    const provider =
      preset.id === "local"
        ? providers.find((p) => p.id.includes("local") || p.id.includes("ollama") || p.id.includes("lmstudio"))
        : providers.find((p => p.status === "configured")) ?? providers[0];

    if (provider) {
      setSelectedProvider(provider.id);
      const model =
        preset.id === "cheap"
          ? provider.models.find((m) => m.id.includes("mini") || m.id.includes("lite")) ?? provider.models[0]
          : preset.id === "code"
            ? provider.models.find((m) => m.id.includes("coder") || m.id.includes("code")) ?? provider.models[0]
            : preset.id === "reasoning"
              ? provider.models.find((m) => m.id.includes("reason") || m.id.includes("think")) ?? provider.models[0]
              : preset.id === "fast"
                ? provider.models[0]
                : provider.models[0];
      setSelectedModel(model.id);
      setTestResult(null);
    }
  };

  const currentProvider = providers.find((p) => p.id === selectedProvider);
  const hasChanges =
    selectedProvider !== currentSelection?.provider || selectedModel !== currentSelection?.model;

  const statusColor = (status: ProviderInfo["status"]) => {
    switch (status) {
      case "configured": return "text-success";
      case "missing_token": return "text-warning";
      case "invalid_token": return "text-destructive";
      default: return "text-muted-foreground";
    }
  };

  const statusLabel = (status: ProviderInfo["status"]) => {
    switch (status) {
      case "configured": return "Configured";
      case "missing_token": return "No token";
      case "invalid_token": return "Invalid token";
      default: return "Unknown";
    }
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <Cpu className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        {loading ? (
          <RefreshCw className="h-3 w-3 animate-spin text-muted-foreground" />
        ) : currentSelection?.provider ? (
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 hover:opacity-80 transition-opacity"
          >
            <span className="text-[10px] font-compressed tracking-widest uppercase text-muted-foreground">
              {currentSelection.provider}
            </span>
            <span className="text-[10px] font-mono text-foreground/70">
              /{currentSelection.model.split("/").pop()}
            </span>
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          </button>
        ) : (
          <span className="text-[10px] text-muted-foreground font-compressed tracking-widest uppercase">
            No provider
          </span>
        )}

        {expanded && (
          <div className="absolute top-full right-0 mt-2 z-50 w-64">
            <Card className="shadow-xl border-border/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-xs">Provider / Model</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <QuickProviderSelect
                  providers={providers}
                  selectedProvider={selectedProvider}
                  selectedModel={selectedModel}
                  onProviderChange={handleProviderChange}
                  onModelChange={setSelectedModel}
                />
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" className="flex-1 h-7 text-[10px]"
                    onClick={handleTest} disabled={testing || !selectedProvider}>
                    {testing ? <RefreshCw className="h-3 w-3 animate-spin" /> : "Test"}
                  </Button>
                  <Button size="sm" className="flex-1 h-7 text-[10px]"
                    onClick={handleSave} disabled={saving || !hasChanges || !selectedProvider}>
                    {saving ? <RefreshCw className="h-3 w-3 animate-spin" /> : "Apply"}
                  </Button>
                </div>
                {testResult && (
                  <div className={`flex items-center gap-1 text-[10px] ${testResult === "success" ? "text-success" : "text-destructive"}`}>
                    {testResult === "success" ? <Check className="h-3 w-3" /> : <X className="h-3 w-3" />}
                    {testResult === "success" ? "Connection OK" : "Connection failed"}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Cpu className="h-4 w-4" />
          Provider
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading ? (
          <div className="flex items-center gap-2 py-4 justify-center">
            <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Loading...</span>
          </div>
        ) : providers.length === 0 ? (
          <div className="text-center py-4">
            <p className="text-xs text-muted-foreground mb-2">No providers configured</p>
            <p className="text-[10px] text-muted-foreground/70">
              Add providers via config or env vars
            </p>
          </div>
        ) : (
          <>
            {/* Current selection display */}
            {currentSelection?.provider && (
              <div className="flex items-center gap-2 p-2 border border-border rounded bg-muted/30">
                <div className="flex flex-col min-w-0">
                  <span className="text-xs font-medium truncate">
                    {currentSelection.provider}
                  </span>
                  <span className="text-[10px] text-muted-foreground font-mono truncate">
                    {currentSelection.model}
                  </span>
                </div>
                <Badge variant="success" className="shrink-0 text-[9px]">
                  Active
                </Badge>
              </div>
            )}

            {/* Provider + Model selects */}
            <QuickProviderSelect
              providers={providers}
              selectedProvider={selectedProvider}
              selectedModel={selectedModel}
              onProviderChange={handleProviderChange}
              onModelChange={setSelectedModel}
            />

            {/* Presets */}
            <div className="space-y-1.5">
              <span className="text-[10px] text-muted-foreground font-compressed tracking-widest uppercase">
                Presets
              </span>
              <div className="grid grid-cols-3 gap-1">
                {PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => handlePreset(preset)}
                    className="flex flex-col items-center gap-0.5 p-1.5 border border-border rounded text-center hover:bg-foreground/5 transition-colors"
                    title={preset.description}
                  >
                    <span className="text-muted-foreground">{preset.icon}</span>
                    <span className="text-[9px] font-compressed tracking-widest uppercase">
                      {preset.label}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Status per provider */}
            <div className="space-y-1">
              <span className="text-[10px] text-muted-foreground font-compressed tracking-widest uppercase">
                Available Providers
              </span>
              {providers.map((p) => (
                <div key={p.id} className="flex items-center justify-between py-0.5">
                  <span className="text-xs truncate mr-2">{p.name}</span>
                  <div className="flex items-center gap-1.5 shrink-0">
                    <span className={`text-[10px] ${statusColor(p.status)}`}>
                      {statusLabel(p.status)}
                    </span>
                    {p.id === selectedProvider && (
                      <Check className="h-3 w-3 text-success" />
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-1">
              <Button
                size="sm"
                variant="outline"
                className="flex-1 h-8"
                onClick={handleTest}
                disabled={testing || !selectedProvider}
              >
                {testing ? (
                  <RefreshCw className="h-3 w-3 animate-spin" />
                ) : testResult === "success" ? (
                  <Check className="h-3 w-3 text-success" />
                ) : testResult === "error" ? (
                  <X className="h-3 w-3 text-destructive" />
                ) : (
                  "Test"
                )}
                {testResult && (
                  <span className="ml-1 text-[10px]">
                    {testResult === "success" ? "OK" : "Failed"}
                  </span>
                )}
              </Button>
              <Button
                size="sm"
                className="flex-1 h-8"
                onClick={handleSave}
                disabled={saving || !hasChanges || !selectedProvider}
              >
                {saving ? <RefreshCw className="h-3 w-3 animate-spin" /> : "Apply"}
              </Button>
            </div>

            {/* Model info */}
            {currentProvider && selectedModel && (
              <div className="text-[10px] text-muted-foreground space-y-0.5">
                {(() => {
                  const model = currentProvider.models.find((m) => m.id === selectedModel);
                  if (!model) return null;
                  return (
                    <>
                      <div className="flex gap-3">
                        {model.contextWindow && (
                          <span>Context: {Math.round(model.contextWindow / 1000)}K</span>
                        )}
                        {model.supportsTools && (
                          <Badge variant="outline" className="text-[9px] h-4">Tools</Badge>
                        )}
                        {model.supportsVision && (
                          <Badge variant="outline" className="text-[9px] h-4">Vision</Badge>
                        )}
                      </div>
                    </>
                  );
                })()}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function QuickProviderSelect({
  providers,
  selectedProvider,
  selectedModel,
  onProviderChange,
  onModelChange,
}: {
  providers: ProviderInfo[];
  selectedProvider: string;
  selectedModel: string;
  onProviderChange: (id: string) => void;
  onModelChange: (id: string) => void;
}) {
  const currentProvider = providers.find((p) => p.id === selectedProvider);

  return (
    <div className="space-y-2">
      <div className="space-y-1">
        <label className="text-[10px] text-muted-foreground font-compressed tracking-widest uppercase">
          Provider
        </label>
        <select
          value={selectedProvider}
          onChange={(e) => onProviderChange(e.target.value)}
          className="w-full px-2 py-1.5 text-xs border border-border rounded bg-background text-foreground"
        >
          <option value="">Select provider</option>
          {providers.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.status === "configured" ? "✓" : p.status === "missing_token" ? "⚠" : "✗"})
            </option>
          ))}
        </select>
      </div>

      {currentProvider && currentProvider.models.length > 0 && (
        <div className="space-y-1">
          <label className="text-[10px] text-muted-foreground font-compressed tracking-widest uppercase">
            Model
          </label>
          <select
            value={selectedModel}
            onChange={(e) => onModelChange(e.target.value)}
            className="w-full px-2 py-1.5 text-xs border border-border rounded bg-background text-foreground"
          >
            {currentProvider.models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
                {m.contextWindow ? ` (${Math.round(m.contextWindow / 1000)}K)` : ""}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}
