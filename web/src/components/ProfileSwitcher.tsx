import { useEffect, useState } from "react";
import { ChevronDown, Users } from "lucide-react";
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
          "flex items-center gap-2",
          "px-2 py-1",
          "border border-current/20",
          "hover:bg-midground/5",
          "text-[0.65rem] sm:text-[0.7rem] tracking-[0.15em]",
        )}
        aria-label="Switch Hermes profile"
      >
        <Users className="h-3.5 w-3.5 opacity-60" />
        <span className="hidden sm:inline">{activeLabel}</span>
        <ChevronDown className="h-3.5 w-3.5 opacity-60" />
      </button>

      {open && (
        <div className="absolute right-0 mt-2 w-56 border border-current/20 bg-background-base/95 backdrop-blur-sm z-50">
          <div className="max-h-72 overflow-auto">
            {profiles.length === 0 ? (
              <div className="px-3 py-2 text-xs opacity-60">No profiles</div>
            ) : (
              profiles.map((p) => {
                const isActive = p.name === active;
                return (
                  <button
                    key={p.name}
                    type="button"
                    className={cn(
                      "w-full text-left px-3 py-2 text-xs",
                      "hover:bg-midground/5",
                      isActive ? "bg-midground/10" : "",
                    )}
                    onClick={() => {
                      setActive(p.name);
                      setOpen(false);
                      window.location.reload();
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-mono-ui">{p.is_default ? "default" : p.name}</span>
                      {p.gateway_running && (
                        <span className="text-[10px] opacity-60">gateway</span>
                      )}
                    </div>
                    <div className="text-[10px] opacity-50 truncate">{p.path}</div>
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
