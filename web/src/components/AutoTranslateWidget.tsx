import { useCallback, useState } from "react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { api } from "@/lib/api";
import { useI18n, LOCALE_META } from "@/i18n";
import type { Locale } from "@/i18n";
import { Languages, Sparkles } from "lucide-react";

interface TranslateResult {
  translated: number;
  missing: number;
  translations: Record<string, string>;
  message?: string;
}

/**
 * Auto-translate widget — uses the current Hermes model to translate
 * missing dashboard UI strings to a selected locale.
 */
export function AutoTranslateWidget() {
  const { t, locale: currentLocale } = useI18n();
  const { toast, showToast } = useToast();
  const [targetLocale, setTargetLocale] = useState<Locale>(currentLocale === "en" ? "zh" : currentLocale);
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TranslateResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleTranslate = useCallback(async () => {
    if (targetLocale === "en") {
      showToast("Select a non-English locale first", "error");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.i18nTranslate({
        locale: targetLocale,
        model: model.trim() || undefined,
        dry_run: false,
      });
      setResult(res);
      if (res.translated > 0) {
        showToast(`Translated ${res.translated} strings to ${LOCALE_META[targetLocale as Locale]?.name || targetLocale}`, "success");
      } else {
        showToast(res.message || "No missing strings", "success");
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      showToast(msg, "error");
    } finally {
      setLoading(false);
    }
  }, [targetLocale, model, showToast]);

  const locales = Object.entries(LOCALE_META).filter(([code]) => code !== "en");

  return (
    <div className="space-y-3 p-3">
      <Toast toast={toast} />
      <div className="flex items-center gap-2">
        <Sparkles className="h-4 w-4 text-primary" />
        <span className="text-xs font-semibold uppercase tracking-wider">
          Auto-translate UI
        </span>
      </div>

      <p className="text-xs text-muted-foreground">
        Use the current Hermes model to translate missing dashboard strings.
      </p>

      {/* Locale selector */}
      <div className="space-y-1">
        <label className="text-xs font-medium">Target language</label>
        <div className="flex flex-wrap gap-1">
          {locales.map(([code, meta]) => (
            <button
              key={code}
              onClick={() => setTargetLocale(code as Locale)}
              className={`rounded px-2 py-1 text-xs transition-colors ${
                targetLocale === code
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted hover:bg-muted/70"
              }`}
            >
              {meta.name}
            </button>
          ))}
        </div>
      </div>

      {/* Model override */}
      <div className="space-y-1">
        <label className="text-xs font-medium">
          Model <span className="text-muted-foreground">(default: current)</span>
        </label>
        <Input
          className="h-7 text-xs"
          placeholder="e.g. deepseek-v4-pro"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        />
      </div>

      {/* Translate button */}
      <Button
        size="sm"
        className="w-full"
        onClick={handleTranslate}
        disabled={loading || targetLocale === "en"}
        prefix={loading ? <Spinner /> : <Languages className="h-3.5 w-3.5" />}
      >
        {loading ? "Translating…" : `Translate to ${LOCALE_META[targetLocale as Locale]?.name || targetLocale}`}
      </Button>

      {/* Results */}
      {error && (
        <div className="rounded border border-destructive/30 bg-destructive/5 p-2">
          <p className="text-xs text-destructive">{error}</p>
        </div>
      )}

      {result && !error && (
        <div className="rounded border border-border bg-muted/30 p-2 space-y-1">
          <div className="flex items-center gap-2">
            <Badge tone="success">{result.translated} translated</Badge>
            {result.missing > 0 && (
              <Badge tone="warning">{result.missing} total missing</Badge>
            )}
          </div>
          {result.translated > 0 && (
            <p className="text-xs text-muted-foreground">
              ✓ Translations saved. Refresh the page to see changes, or switch language in the picker above.
            </p>
          )}
          {result.translated === 0 && !error && (
            <p className="text-xs text-muted-foreground">
              ✓ All strings are already translated for this locale.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
