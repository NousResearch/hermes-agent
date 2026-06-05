import { useEffect, useState } from "react";
import { Star, X } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import { useI18n } from "@/i18n";
import { ModelPickerDialog } from "@/components/ModelPickerDialog";

interface Props {
  profileName: string;
  onClose(): void;
  onError(message: string): void;
}

export function ProfileModelDialog({ profileName, onClose, onError }: Props) {
  const { t } = useI18n();
  const [current, setCurrent] = useState<{
    provider: string;
    model: string;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [picker, setPicker] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .getProfileModel(profileName)
      .then((res) => {
        if (cancelled) return;
        setCurrent({ provider: res.provider, model: res.model });
      })
      .catch((e) => {
        if (cancelled) return;
        onError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [profileName, onError]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const provider = current?.provider ?? "";
  const model = current?.model ?? "";

  return (
    <>
      <div
        className="fixed inset-0 z-[90] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
        onClick={(e) => e.target === e.currentTarget && onClose()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="profile-model-dialog-title"
      >
        <div className="relative w-full max-w-lg border border-border bg-card shadow-2xl">
          <Button
            ghost
            size="icon"
            onClick={onClose}
            className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
            aria-label={t.common.close}
          >
            <X />
          </Button>

          <header className="p-5 pb-3 border-b border-border">
            <h2
              id="profile-model-dialog-title"
              className="font-display text-base tracking-wider uppercase"
            >
              {t.profiles.configureModelTitle.replace("{name}", profileName)}
            </h2>
            <p className="text-xs text-muted-foreground mt-1">
              {t.profiles.configureModelSubtitle}
            </p>
          </header>

          <div className="p-5 space-y-3">
            <div className="flex items-center justify-between gap-3 bg-muted/20 border border-border/50 px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 mb-0.5">
                  <Star className="h-3 w-3 text-primary" />
                  <span className="text-xs font-medium uppercase tracking-wider">
                    {t.profiles.profileModel}
                  </span>
                </div>
                <div className="text-xs font-mono text-muted-foreground truncate">
                  {loading ? (
                    <span className="inline-flex items-center gap-2">
                      <Spinner className="text-xs" /> {t.common.loading}
                    </span>
                  ) : (
                    <>
                      {provider || t.profiles.modelUnset}
                      {provider && model && " · "}
                      {model || (provider ? t.profiles.modelUnset : "")}
                    </>
                  )}
                </div>
              </div>
              <Button
                size="sm"
                onClick={() => setPicker(true)}
                disabled={loading}
                className="text-xs"
              >
                {t.profiles.change}
              </Button>
            </div>
          </div>

          <footer className="border-t border-border p-3 flex items-center justify-end">
            <Button outlined onClick={onClose}>
              {t.common.close}
            </Button>
          </footer>
        </div>
      </div>

      {picker && (
        <ModelPickerDialog
          loader={() => api.getProfileModelOptions(profileName)}
          alwaysGlobal
          title={t.profiles.setProfileModelTitle.replace("{name}", profileName)}
          onApply={async ({ provider, model }) => {
            await api.setProfileModel(profileName, { provider, model });
            setCurrent({ provider, model });
          }}
          onClose={() => setPicker(false)}
        />
      )}
    </>
  );
}
