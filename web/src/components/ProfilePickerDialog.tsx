/**
 * ProfilePickerDialog — inline dropdown for switching the active profile.
 *
 * Embedded in ChatSidebar.tsx alongside the Model picker.  Shows all
 * available profiles with the active one marked.  Selecting a different
 * profile calls the /api/profiles/{name}/activate endpoint, then
 * signals to the parent that the chat should reconnect the PTY so the
 * new profile's hermes --tui starts under the updated HERMES_HOME.
 */

import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Check, RefreshCw, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

interface ProfileInfo {
  name: string;
  path: string;
  is_default: boolean;
  model: string | null;
  provider: string | null;
  has_env: boolean;
  skill_count: number;
}

interface ProfilePickerDialogProps {
  /** Current active profile name (from metrics) */
  activeProfile: string;
  onClose(): void;
  /** Called after a profile is successfully activated — parent reloads PTY. */
  onProfileActivated(newProfile: string): void;
}

export function ProfilePickerDialog({
  activeProfile,
  onClose,
  onProfileActivated,
}: ProfilePickerDialogProps) {
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activating, setActivating] = useState<string | null>(null);
  const closedRef = useRef(false);

  useEffect(() => {
    closedRef.current = false;
    api
      .getProfiles()
      .then((r) => {
        if (closedRef.current) return;
        // Sort: default first, then alphabetical
        setProfiles(
          [...r.profiles].sort((a, b) => {
            if (a.is_default) return -1;
            if (b.is_default) return 1;
            return a.name.localeCompare(b.name);
          }),
        );
        setLoading(false);
      })
      .catch((e) => {
        if (closedRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });

    return () => {
      closedRef.current = true;
    };
  }, []);

  // Esc closes.
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

  const activate = async (name: string) => {
    if (activating) return;
    setActivating(name);
    try {
      await api.activateProfile(name);
      onProfileActivated(name);
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setActivating(null);
    }
  };

  const portalRoot =
    typeof document !== "undefined" ? document.body : null;
  if (!portalRoot) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-label="Switch Agent Profile"
    >
      <div className="relative w-full max-w-sm max-h-[70vh] border border-border bg-card shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border shrink-0">
          <div>
            <h2 className="font-display text-sm tracking-wider uppercase">
              Agent Profile
            </h2>
            <p className="text-[0.65rem] text-muted-foreground mt-0.5">
              Switch profile — new chat sessions use the selected agent
            </p>
          </div>
          <Button
            ghost
            size="icon"
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground shrink-0"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Profile list */}
        <div className="flex-1 min-h-0 overflow-y-auto py-1">
          {loading && (
            <div className="flex items-center gap-2 p-4 text-xs text-muted-foreground">
              <Spinner className="text-xs" />
              loading profiles…
            </div>
          )}

          {error && !loading && (
            <div className="p-4 text-xs text-destructive">{error}</div>
          )}

          {!loading && !error && profiles.length === 0 && (
            <div className="p-4 text-xs text-muted-foreground italic">
              no profiles found
            </div>
          )}

          {!loading &&
            profiles.map((p) => {
              const isActive = p.name === activeProfile;
              const isBusy = activating === p.name;
              return (
                <ListItem
                  key={p.name}
                  onClick={() => !isActive && activate(p.name)}
                  className={cn(
                    "px-4 py-2.5 text-sm cursor-pointer transition-colors",
                    isActive
                      ? "bg-primary/10 cursor-default"
                      : "hover:bg-accent hover:text-accent-foreground",
                  )}
                  aria-current={isActive ? "true" : undefined}
                >
                  <div className="flex items-center gap-3 min-w-0 w-full">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium truncate">
                          {p.name}
                        </span>
                        {p.is_default && (
                          <span className="text-[0.6rem] uppercase tracking-wider text-muted-foreground shrink-0">
                            default
                          </span>
                        )}
                        {p.has_env && (
                          <span className="text-[0.6rem] uppercase tracking-wider text-muted-foreground shrink-0">
                            env
                          </span>
                        )}
                      </div>
                      {p.model && (
                        <div className="text-[0.65rem] text-muted-foreground/80 font-mono truncate mt-0.5">
                          {p.model}
                        </div>
                      )}
                    </div>

                    {isBusy ? (
                      <Spinner className="text-xs shrink-0" />
                    ) : isActive ? (
                      <Check className="h-4 w-4 text-primary shrink-0" />
                    ) : (
                      <RefreshCw className="h-3.5 w-3.5 text-muted-foreground/50 shrink-0 opacity-0 group-hover:opacity-100" />
                    )}
                  </div>
                </ListItem>
              );
            })}
        </div>

        {/* Footer hint */}
        <div className="px-4 py-2 border-t border-border shrink-0">
          <p className="text-[0.6rem] text-muted-foreground">
            Profile changes apply to new chat sessions. Active chats are not
            restarted automatically.
          </p>
        </div>
      </div>
    </div>,
    portalRoot,
  );
}
