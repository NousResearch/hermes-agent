import { useEffect, useState } from "react";
import type { SkillContent } from "@/lib/api";
import { get } from "../api";

export function SkillDetailPane({ name }: { name: string }) {
  const [content, setContent] = useState<SkillContent | null>(null);

  useEffect(() => {
    get<SkillContent>(`/api/skills/content?name=${encodeURIComponent(name)}`)
      .then(setContent)
      .catch(() => setContent(null));
  }, [name]);

  return (
    <div style={{ padding: "16px 14px 24px", display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 15,
            fontWeight: 650,
            color: "var(--mid)",
            flex: 1,
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {name}
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
            padding: "2.5px 7px",
            whiteSpace: "nowrap",
            flexShrink: 0,
          }}
        >
          read-only
        </span>
      </div>
      {content && (
        <div style={{ display: "flex", gap: 14, fontFamily: "var(--mono)", fontSize: 11, color: "var(--t3)" }}>
          <span style={{ marginLeft: "auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {content.path}
          </span>
        </div>
      )}
      <div
        style={{
          background: "var(--card)",
          border: "1px solid var(--line)",
          borderRadius: 14,
          padding: "14px 15px",
          fontFamily: "var(--mono)",
          fontSize: 11.5,
          lineHeight: 1.65,
          color: "var(--t2)",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {content ? content.content : "Loading…"}
      </div>
    </div>
  );
}
