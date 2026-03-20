import { useEffect, useRef, useState } from "react";
import { Download, RotateCcw, Save, Upload } from "lucide-react";
import { api } from "@/lib/api";
import { getNestedValue, setNestedValue } from "@/lib/nested";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";
import { AutoField } from "@/components/AutoField";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function ConfigPage() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null);
  const [schema, setSchema] = useState<Record<string, Record<string, unknown>> | null>(null);
  const [defaults, setDefaults] = useState<Record<string, unknown> | null>(null);
  const [saving, setSaving] = useState(false);
  const { toast, showToast } = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.getConfig().then(setConfig).catch(() => {});
    api.getSchema().then((s) => setSchema(s as Record<string, Record<string, unknown>>)).catch(() => {});
    api.getDefaults().then(setDefaults).catch(() => {});
  }, []);

  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    try {
      await api.saveConfig(config);
      showToast("Configuration saved", "success");
    } catch (e) {
      showToast(`Failed to save: ${e}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    if (defaults) setConfig(structuredClone(defaults));
  };

  const handleExport = () => {
    if (!config) return;
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "hermes-config.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const imported = JSON.parse(reader.result as string);
        setConfig(imported);
        showToast("Config imported — review and save", "success");
      } catch {
        showToast("Invalid JSON file", "error");
      }
    };
    reader.readAsText(file);
  };

  if (!config || !schema) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  const categories = [...new Set(Object.values(schema).map((s) => String(s.category ?? "general")))];

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <div className="flex items-center justify-between flex-wrap gap-2">
        <p className="text-sm text-muted-foreground">
          Edit <code>~/.hermes/config.yaml</code>
        </p>

        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="h-3 w-3" />
            Export
          </Button>

          <Button variant="outline" size="sm" onClick={() => fileInputRef.current?.click()}>
            <Upload className="h-3 w-3" />
            Import
          </Button>

          <input ref={fileInputRef} type="file" accept=".json,.yaml,.yml" className="hidden" onChange={handleImport} />

          <Button variant="outline" size="sm" onClick={handleReset}>
            <RotateCcw className="h-3 w-3" />
            Reset
          </Button>

          <Button size="sm" onClick={handleSave} disabled={saving}>
            <Save className="h-3 w-3" />
            {saving ? "Saving..." : "Save"}
          </Button>
        </div>
      </div>

      <Tabs defaultValue={categories[0]}>
        {(active, setActive) => (
          <>
            <TabsList className="flex-wrap">
              {categories.map((cat) => (
                <TabsTrigger key={cat} value={cat} active={active === cat} onClick={() => setActive(cat)}>
                  {cat.charAt(0).toUpperCase() + cat.slice(1)}
                </TabsTrigger>
              ))}
            </TabsList>

            <Card>
              <CardHeader>
                <CardTitle className="text-base capitalize">{active}</CardTitle>
              </CardHeader>

              <CardContent className="grid gap-6">
                {Object.entries(schema)
                  .filter(([, s]) => String(s.category ?? "general") === active)
                  .map(([key, s]) => (
                    <AutoField
                      key={key}
                      schemaKey={key}
                      schema={s}
                      value={getNestedValue(config, key)}
                      onChange={(v) => setConfig(setNestedValue(config, key, v))}
                    />
                  ))}
              </CardContent>
            </Card>
          </>
        )}
      </Tabs>
    </div>
  );
}
