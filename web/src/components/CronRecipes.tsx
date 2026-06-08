import { useCallback, useEffect, useState } from "react";
import { Clock, Wand2 } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { api } from "@/lib/api";
import type { CronRecipe, CronRecipeField } from "@/lib/api";
import { cn, themedBody } from "@/lib/utils";

interface CronRecipesProps {
  profile: string;
  /** Called after a recipe is instantiated so the parent can refresh its job list. */
  onCreated?: () => void;
}

/** Initial form values for a recipe = each field's default (or ""). */
function initialValues(recipe: CronRecipe): Record<string, string> {
  const out: Record<string, string> = {};
  for (const f of recipe.fields) out[f.name] = f.default ?? "";
  return out;
}

function FieldInput({
  field,
  value,
  onChange,
}: {
  field: CronRecipeField;
  value: string;
  onChange: (v: string) => void;
}) {
  if (field.type === "enum" || field.type === "weekdays") {
    return (
      <Select value={value} onValueChange={(v) => onChange(v)}>
        {field.options.map((opt) => (
          <SelectOption key={opt} value={opt}>
            {opt}
          </SelectOption>
        ))}
      </Select>
    );
  }
  if (field.type === "time") {
    return (
      <Input
        type="time"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }
  // text
  return (
    <Input
      type="text"
      value={value}
      placeholder={field.help || field.label}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function RecipeCard({
  recipe,
  profile,
  showToast,
  onCreated,
}: {
  recipe: CronRecipe;
  profile: string;
  showToast: (message: string, type: "error" | "success") => void;
  onCreated?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [values, setValues] = useState<Record<string, string>>(() => initialValues(recipe));
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = useCallback(async () => {
    setSubmitting(true);
    setError(null);
    try {
      const job = await api.instantiateCronRecipe({ recipe: recipe.key, values }, profile);
      const when = job.schedule_display ? ` — ${job.schedule_display}` : "";
      showToast(`${recipe.title} scheduled${when}`, "success");
      setOpen(false);
      setValues(initialValues(recipe));
      onCreated?.();
    } catch (e) {
      // 422 from the API carries the slot-level validation message.
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg.replace(/^\d+:\s*/, ""));
    } finally {
      setSubmitting(false);
    }
  }, [recipe, values, profile, showToast, onCreated]);

  return (
    <Card className={cn("overflow-hidden", themedBody)}>
      <CardContent className="space-y-3 p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Wand2 className="h-4 w-4 shrink-0 opacity-70" />
              <span className="font-medium">{recipe.title}</span>
            </div>
            <p className="mt-1 text-sm opacity-70">{recipe.description}</p>
            <div className="mt-2 flex flex-wrap gap-1">
              {recipe.tags.map((t) => (
                <Badge key={t} tone="secondary">
                  {t}
                </Badge>
              ))}
            </div>
          </div>
          <Button
            ghost={open}
            size="sm"
            onClick={() => setOpen((o) => !o)}
          >
            {open ? "Cancel" : "Set up"}
          </Button>
        </div>

        {open && (
          <div className="space-y-3 border-t pt-3">
            {recipe.fields.map((f) => (
              <div key={f.name} className="space-y-1">
                <Label htmlFor={`${recipe.key}-${f.name}`}>{f.label}</Label>
                <FieldInput
                  field={f}
                  value={values[f.name] ?? ""}
                  onChange={(v) => setValues((prev) => ({ ...prev, [f.name]: v }))}
                />
                {f.help && f.type !== "text" ? (
                  <p className="text-xs opacity-60">{f.help}</p>
                ) : null}
              </div>
            ))}
            {error ? (
              <p className="text-sm text-red-500" role="alert">
                {error}
              </p>
            ) : null}
            <div className="flex items-center gap-2">
              <Button onClick={() => void submit()} disabled={submitting}>
                {submitting ? <Spinner className="h-4 w-4" /> : <Clock className="h-4 w-4" />}
                Schedule it
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Cron Recipes gallery — the form-where-there's-a-screen surface. Each recipe
 * card expands into an inline form (one field per typed slot); submitting POSTs
 * to /api/cron/recipes/instantiate which fills the recipe and creates the job
 * via the same create_job path as everything else.
 */
export function CronRecipes({ profile, onCreated }: CronRecipesProps) {
  const { toast, showToast } = useToast();
  const [recipes, setRecipes] = useState<CronRecipe[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getCronRecipes()
      .then((r) => {
        if (!cancelled) setRecipes(r.recipes);
      })
      .catch((e) => {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loadError) {
    return <p className="text-sm text-red-500">Couldn't load recipes: {loadError}</p>;
  }
  if (recipes === null) {
    return (
      <div className="flex items-center gap-2 opacity-70">
        <Spinner className="h-4 w-4" /> Loading recipes…
      </div>
    );
  }
  if (recipes.length === 0) {
    return <p className="opacity-70">No cron recipes available.</p>;
  }

  return (
    <>
      <Toast toast={toast} />
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {recipes.map((r) => (
          <RecipeCard
            key={r.key}
            recipe={r}
            profile={profile}
            showToast={showToast}
            onCreated={onCreated}
          />
        ))}
      </div>
    </>
  );
}

export default CronRecipes;
