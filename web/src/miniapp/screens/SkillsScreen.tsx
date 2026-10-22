import { useEffect, useState, type MouseEvent } from "react";
import type { SkillInfo } from "@/lib/api";
import { get, put } from "../api";
import { useMiniApp } from "../context";

export function SkillsScreen({ onOpen }: { onOpen: (name: string) => void }) {
  const { isAdmin, showToast } = useMiniApp();
  const [skills, setSkills] = useState<SkillInfo[] | null>(null);
  const [pending, setPending] = useState<Record<string, boolean>>({});

  const load = () => get<SkillInfo[]>("/api/skills").then(setSkills).catch(() => setSkills([]));
  useEffect(() => {
    load();
  }, []);

  if (!skills) return null;

  const toggle = async (skill: SkillInfo, e: MouseEvent) => {
    e.stopPropagation();
    const next = !(pending[skill.name] ?? skill.enabled);
    setPending((p) => ({ ...p, [skill.name]: next }));
    try {
      await put("/api/skills/toggle", { name: skill.name, enabled: next });
      showToast(next ? `${skill.name} enabled` : `${skill.name} disabled`);
      load();
    } catch {
      setPending((p) => ({ ...p, [skill.name]: skill.enabled }));
      showToast("Couldn't update skill");
    }
  };

  const enabledCount = skills.filter((s) => pending[s.name] ?? s.enabled).length;

  return (
    <div style={{ padding: "16px 14px 24px", display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", padding: "0 4px" }}>
        <div style={{ fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--t3)", fontFamily: "var(--mono)" }}>
          Skills
        </div>
        <div style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--t3)", whiteSpace: "nowrap" }}>
          {skills.length} · {enabledCount} enabled
        </div>
      </div>
      {skills.map((s) => {
        const enabled = pending[s.name] ?? s.enabled;
        return (
          <div
            key={s.name}
            onClick={isAdmin ? () => onOpen(s.name) : undefined}
            style={{
              background: "var(--card)",
              border: "1px solid var(--line)",
              borderRadius: 14,
              padding: "12px 14px",
              cursor: isAdmin ? "pointer" : "default",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 13.5,
                  fontWeight: 600,
                  color: "var(--mid)",
                  flex: 1,
                  minWidth: 0,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {s.name}
              </span>
              <span
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 9.5,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  color: "var(--t3)",
                  border: "1px solid var(--line)",
                  borderRadius: 6,
                  padding: "2px 6px",
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                }}
              >
                {s.category}
              </span>
              {!isAdmin && (
                <span
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: 99,
                    flexShrink: 0,
                    background: enabled ? "var(--success)" : "var(--line2)",
                  }}
                />
              )}
              {isAdmin && (
                <div
                  onClick={(e) => toggle(s, e)}
                  role="switch"
                  aria-checked={enabled}
                  style={{
                    width: 38,
                    height: 23,
                    borderRadius: 99,
                    position: "relative",
                    cursor: "pointer",
                    flexShrink: 0,
                    transition: "background 180ms ease",
                    background: enabled ? "var(--success)" : "var(--line2)",
                  }}
                >
                  <span
                    style={{
                      position: "absolute",
                      top: 2.5,
                      width: 18,
                      height: 18,
                      borderRadius: 99,
                      background: "var(--bg)",
                      transition: "left 180ms ease",
                      boxShadow: "0 1px 3px rgba(0,0,0,0.3)",
                      left: enabled ? 17.5 : 2.5,
                    }}
                  />
                </div>
              )}
              {isAdmin && (
                <svg width="7" height="12" viewBox="0 0 7 12" style={{ flexShrink: 0 }}>
                  <path
                    d="M1 1l5 5-5 5"
                    stroke="currentColor"
                    strokeOpacity={0.4}
                    strokeWidth="1.8"
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </div>
            <div
              style={{
                fontSize: 12,
                color: "var(--t2)",
                marginTop: 5,
                lineHeight: 1.45,
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
              }}
            >
              {s.description}
            </div>
          </div>
        );
      })}
    </div>
  );
}
