import { useEffect, useState } from "react";
import { Check, ChevronDown, Users } from "lucide-react";
import { api, setActiveProfile, getActiveProfile } from "@/lib/api";
import type { ProfileInfo } from "@/lib/api";
import { cn } from "@/lib/utils";

export function ProfileSwitcher() {
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<string>(() => getActiveProfile() ?? "default");

  useEffect(() => {
    api
      .getProfiles()
      .then((res) => {
        setProfiles(res.profiles);
        if (res.profiles.length > 0) {
          setActive((prev) => {
            const exists = res.profiles.some((p) => p.name === prev);
            return exists
              ? prev
              : (res.profiles.find((p) => p.is_default)?.name ?? res.profiles[0].name);
          });
        }
      })
      .catch((err) => {
        console.error("Failed to load profiles", err);
      });
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("hermes.activeProfile", active);
    } catch {
      // ignore
    }
    setActiveProfile(active === "default" ? null : active);
  }, [active]);

  useEffect(() => {
    try {
      const stored = localStorage.getItem("hermes.activeProfile");
      if (stored && stored !== active) {
        setActive(stored);
      }
    } catch {
      // ignore
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const activeLabel = (() => {
    const p = profiles.find((x) => x.name === active);
    if (!p) return active;
    return p.is_default ? "default" : p.name;
  })();

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={cn(
          "flex h-9 items-center gap-2 rounded-lg border border-border bg-card px-2.5 py-1.5",
          "text-xs font-medium text-foreground shadow-sm transition-colors hover:bg-secondary",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        )}
        aria-label="Switch Hermes profile"
      >
        <Users className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="hidden sm:inline">{activeLabel}</span>
        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
      </button>

      {open && (
        <div className="absolute right-0 z-50 mt-2 w-64 overflow-hidden rounded-xl border border-border bg-background shadow-xl backdrop-blur-xl">
          <div className="max-h-72 overflow-auto">
            {profiles.length === 0 ? (
              <div className="px-3 py-2 text-xs text-muted-foreground">No profiles</div>
            ) : (
              profiles.map((p) => {
                const isActive = p.name === active;
                return (
                  <button
                    key={p.name}
                    type="button"
                    className={cn(
                      "w-full px-3 py-2.5 text-left text-sm transition-colors",
                      "hover:bg-secondary",
                      isActive ? "bg-secondary text-foreground" : "text-foreground",
                    )}
                    onClick={() => {
                      setActive(p.name);
                      setOpen(false);
                      window.location.reload();
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate font-medium">{p.is_default ? "default" : p.name}</span>
                      <div className="flex items-center gap-2">
                        {p.gateway_running && (
                          <span className="rounded-full bg-emerald-500/12 px-2 py-0.5 text-[10px] font-medium text-emerald-400">gateway</span>
                        )}
                        {isActive && <Check className="h-3.5 w-3.5 text-primary" />}
                      </div>
                    </div>
                    <div className="truncate text-xs text-muted-foreground">{p.path}</div>
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
