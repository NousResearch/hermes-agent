/**
 * SkillPicker — compact launcher in the dashboard Chat sidebar.
 *
 * Lists skills from GET /api/skills, filters by name/description, and
 * emits a slash-skill command (`/name`) for the parent to type into the
 * PTY. Fail-open: a failed fetch leaves an empty list; a missing
 * launch callback is a no-op.
 */

import { Card } from "@nous-research/ui/ui/components/card";
import { useEffect, useMemo, useState } from "react";

import { api, type SkillInfo } from "@/lib/api";
import { filterSkills, launchSkillCommand } from "@/lib/skill-picker";

interface SkillPickerProps {
  profile?: string;
  onLaunch?: (cmd: string) => void;
}

export function SkillPicker({ profile, onLaunch }: SkillPickerProps) {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void api
      .getSkills(profile || undefined)
      .then((rows) => {
        if (cancelled) return;
        setSkills(Array.isArray(rows) ? rows : []);
        setError(null);
      })
      .catch(() => {
        if (cancelled) return;
        setSkills([]);
        setError("could not load skills");
      });
    return () => {
      cancelled = true;
    };
  }, [profile]);

  const visible = useMemo(() => filterSkills(skills, query), [skills, query]);

  return (
    <Card className="flex flex-col gap-2 px-3 py-2">
      <div className="text-display text-xs tracking-wider text-text-tertiary">
        skills
      </div>

      <input
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="search"
        aria-label="search skills"
        className="w-full min-w-0 border-b border-current/10 bg-transparent py-0.5 text-xs outline-none placeholder:text-text-tertiary"
      />

      {error && (
        <div className="text-xs text-text-secondary">{error}</div>
      )}

      <div className="max-h-40 overflow-y-auto">
        {visible.length === 0 ? (
          <div className="py-1 text-xs text-text-tertiary">no skills</div>
        ) : (
          visible.map((skill) => {
            const usage =
              typeof skill.usage === "number" && Number.isFinite(skill.usage)
                ? skill.usage
                : null;
            return (
              <button
                key={skill.name}
                type="button"
                title={skill.description || skill.name}
                onClick={() => {
                  const cmd = launchSkillCommand(skill.name);
                  if (cmd) onLaunch?.(cmd);
                }}
                className="flex w-full min-w-0 items-center justify-between gap-2 py-1 text-left text-xs hover:underline"
              >
                <span className="truncate">{skill.name}</span>
                {usage !== null && (
                  <span className="shrink-0 text-text-tertiary">{usage}</span>
                )}
              </button>
            );
          })
        )}
      </div>
    </Card>
  );
}
