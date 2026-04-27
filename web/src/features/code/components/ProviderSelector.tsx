import { useEffect, useState } from "react";
import { Cpu, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectOption,
} from "@/components/ui/select";
import { codeApi } from "@/lib/codeApi";
import type { ProviderInfo, ProviderSelection } from "@/types/code";

interface ProviderSelectorProps {
  onProviderChange?: (provider: string, model: string) => void;
}

export function ProviderSelector({ onProviderChange }: ProviderSelectorProps) {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [currentSelection, setCurrentSelection] = useState<ProviderSelection>({
    provider: "",
    model: "",
  });
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

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
      // Error handling - providers may not be available
    } finally {
      setLoading(false);
    }
  };

  const handleProviderChange = async (providerId: string) => {
    setSelectedProvider(providerId);
    const provider = providers.find((p) => p.id === providerId);
    if (provider && provider.models.length > 0) {
      setSelectedModel(provider.models[0].id);
    }
  };

  const handleSave = async () => {
    if (!selectedProvider) return;
    setSaving(true);
    setSaved(false);
    try {
      await codeApi.selectProvider(selectedProvider, selectedModel);
      setCurrentSelection({ provider: selectedProvider, model: selectedModel });
      onProviderChange?.(selectedProvider, selectedModel);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      // Error handling
    } finally {
      setSaving(false);
    }
  };

  const currentProvider = providers.find((p) => p.id === selectedProvider);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          <Cpu className="h-4 w-4" />
          Provider
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {loading ? (
          <div className="text-xs text-muted-foreground">Loading providers...</div>
        ) : (
          <>
            <Select value={selectedProvider || ""} onValueChange={handleProviderChange} className="w-full">
              {!selectedProvider && <SelectOption value="">Select provider</SelectOption>}
              {providers.map((provider) => {
                let label = provider.name;
                if (provider.status === "configured") label += " (Configured)";
                if (provider.status === "missing_token") label += " (No Token)";
                return (
                  <SelectOption key={provider.id} value={provider.id}>
                    {label}
                  </SelectOption>
                );
              })}
            </Select>

            {currentProvider && currentProvider.models.length > 0 && (
              <Select value={selectedModel || ""} onValueChange={setSelectedModel} className="w-full">
                {!selectedModel && <SelectOption value="">Select model</SelectOption>}
                {currentProvider.models.map((model) => {
                  let label = model.name;
                  if (model.contextWindow) label += ` (${Math.round(model.contextWindow / 1000)}K)`;
                  if (model.supportsTools) label += " [Tools]";
                  return (
                    <SelectOption key={model.id} value={model.id}>
                      {label}
                    </SelectOption>
                  );
                })}
              </Select>
            )}

            <Button
              onClick={handleSave}
              disabled={!selectedProvider || saving}
              size="sm"
              className="w-full"
            >
              {saving ? (
                "Saving..."
              ) : saved ? (
                <>
                  <Check className="h-4 w-4 mr-1" />
                  Saved
                </>
              ) : (
                "Apply"
              )}
            </Button>

            {currentSelection.provider && (
              <div className="text-[10px] text-muted-foreground">
                Current: {currentSelection.provider}/{currentSelection.model}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
