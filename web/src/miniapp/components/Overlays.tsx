import { useEffect } from "react";
import type { ConfirmSpec } from "../context";
import { haptic, hideMainButton, isInsideTelegram, showMainButton } from "../telegram";

const PALETTES: Array<{ key: string; label: string; bg: string; mid: string; accent: string }> = [
  { key: "solarpunk", label: "Solarpunk", bg: "#f0edda", mid: "#2f5238", accent: "#a97f14" },
  { key: "dieselpunk", label: "Dieselpunk", bg: "#1a1714", mid: "#d8c49c", accent: "#e2a13c" },
  { key: "biopunk", label: "Biopunk", bg: "#0a1811", mid: "#c4ef6a", accent: "#a06bf5" },
  { key: "cyberpunk", label: "Cyberpunk", bg: "#05070f", mid: "#62c9ff", accent: "#ff3fa5" },
  { key: "steampunk", label: "Steampunk", bg: "#251611", mid: "#d2a75d", accent: "#c2703d" },
];

export function DimBackdrop({ onClick }: { onClick: () => void }) {
  return (
    <div
      onClick={onClick}
      style={{
        position: "absolute",
        inset: 0,
        background: "rgba(0,0,0,0.45)",
        zIndex: 50,
        animation: "miniapp-fade-in 140ms ease-out",
      }}
    />
  );
}

export function Toast({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      style={{
        position: "absolute",
        top: 118,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 40,
        background: "var(--mid)",
        color: "var(--bg)",
        fontSize: 12,
        fontWeight: 600,
        padding: "8px 14px",
        borderRadius: 99,
        whiteSpace: "nowrap",
        animation: "miniapp-toast-in 160ms ease-out",
        boxShadow: "0 6px 18px rgba(0,0,0,0.25)",
      }}
    >
      {message}
    </div>
  );
}

export function PaletteSheet({
  palette,
  onPick,
}: {
  palette: string;
  onPick: (key: string) => void;
}) {
  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 60,
        background: "var(--headbg)",
        borderTop: "1px solid var(--line)",
        borderRadius: "22px 22px 0 0",
        padding: "14px 16px calc(40px + var(--sab))",
        animation: "miniapp-sheet-in 220ms cubic-bezier(0.23,1,0.32,1)",
      }}
    >
      <div
        style={{
          width: 36,
          height: 4,
          borderRadius: 99,
          background: "var(--line2)",
          margin: "0 auto 12px",
        }}
      />
      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.14em",
          textTransform: "uppercase",
          color: "var(--t3)",
          fontFamily: "var(--mono)",
          padding: "0 4px 10px",
        }}
      >
        Palette · saved to device
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        {PALETTES.map((p) => {
          const active = p.key === palette;
          return (
            <button
              key={p.key}
              onClick={() => onPick(p.key)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 11,
                padding: "11px 12px",
                borderRadius: 13,
                cursor: "pointer",
                textAlign: "left",
                background: active ? "var(--card2)" : "transparent",
                border: `1px solid ${active ? "var(--line2)" : "transparent"}`,
              }}
            >
              <span style={{ display: "flex", gap: 4, flexShrink: 0 }}>
                <span
                  style={{
                    width: 15,
                    height: 15,
                    borderRadius: 99,
                    background: p.bg,
                    border: "1px solid rgba(128,128,128,0.35)",
                  }}
                />
                <span style={{ width: 15, height: 15, borderRadius: 99, background: p.mid }} />
                <span style={{ width: 15, height: 15, borderRadius: 99, background: p.accent }} />
              </span>
              <span style={{ flex: 1, fontSize: 13.5, fontWeight: 600, color: "var(--mid)" }}>
                {p.label}
              </span>
              {active && (
                <svg width="14" height="11" viewBox="0 0 14 11" fill="none">
                  <path
                    d="M1.5 5.5l4 4L12.5 1.5"
                    stroke="currentColor"
                    strokeWidth="2.2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function ConfirmSheet({ confirm, onClose }: { confirm: ConfirmSpec; onClose: () => void }) {
  useEffect(() => {
    const run = () => {
      onClose();
      haptic(confirm.destructive ? "warning" : "success");
      void confirm.run();
    };
    showMainButton({ text: confirm.label, destructive: confirm.destructive, onClick: run });
    return () => hideMainButton(run);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [confirm]);

  const doConfirm = () => {
    onClose();
    haptic(confirm.destructive ? "warning" : "success");
    void confirm.run();
  };

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 60,
        background: "var(--headbg)",
        borderTop: "1px solid var(--line)",
        borderRadius: "22px 22px 0 0",
        padding: "18px 16px 12px",
        animation: "miniapp-sheet-in 220ms cubic-bezier(0.23,1,0.32,1)",
      }}
    >
      <div style={{ fontSize: 15, fontWeight: 650, color: "var(--mid)" }}>{confirm.title}</div>
      <div style={{ fontSize: 12.5, color: "var(--t2)", lineHeight: 1.55, marginTop: 6 }}>
        {confirm.body}
      </div>
      <button
        onClick={onClose}
        style={{
          display: "block",
          width: "100%",
          marginTop: 14,
          padding: 11,
          borderRadius: 11,
          border: "1px solid var(--line)",
          background: "transparent",
          fontSize: 13.5,
          fontWeight: 600,
          color: "var(--t2)",
          cursor: "pointer",
        }}
      >
        Cancel
      </button>
      {/* Fallback for outside-Telegram preview only: Telegram's real
          MainButton (wired above via showMainButton) is the actual control
          on-device, docked below this sheet. Rendering this unconditionally
          used to duplicate it -- on-device the user saw both the native
          MainButton AND this button stacked underneath the sheet. */}
      {!isInsideTelegram() && (
        <button
          onClick={doConfirm}
          style={{
            display: "block",
            width: "calc(100% + 32px)",
            margin: "12px -16px 0",
            padding: "15px 0 calc(27px + var(--sab))",
            border: "none",
            background: confirm.destructive ? "var(--destr)" : "var(--accent)",
            color: confirm.destructive ? "#fff" : "var(--on-accent)",
            fontSize: 15,
            fontWeight: 650,
            cursor: "pointer",
            letterSpacing: "0.01em",
          }}
        >
          {confirm.label}
        </button>
      )}
    </div>
  );
}

export function LogPopup({
  title,
  text,
  onClose,
  onCopy,
}: {
  title: string;
  text: string;
  onClose: () => void;
  onCopy: () => void;
}) {
  return (
    <div
      style={{
        position: "absolute",
        left: 16,
        right: 16,
        top: "50%",
        transform: "translateY(-50%)",
        zIndex: 60,
        background: "var(--headbg)",
        border: "1px solid var(--line2)",
        borderRadius: 16,
        padding: "14px 15px",
        animation: "miniapp-sheet-in 200ms cubic-bezier(0.23,1,0.32,1)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span
          style={{
            fontFamily: "var(--mono)",
            fontSize: 13,
            fontWeight: 650,
            color: "var(--mid)",
            flex: 1,
            minWidth: 0,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {title}
        </span>
        <button
          onClick={onClose}
          aria-label="Close"
          style={{
            width: 28,
            height: 28,
            borderRadius: 8,
            border: "none",
            background: "transparent",
            cursor: "pointer",
            color: "var(--t3)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
            <path
              d="M1.5 1.5l9 9M10.5 1.5l-9 9"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </div>
      <div
        style={{
          background: "var(--bg)",
          border: "1px solid var(--line)",
          borderRadius: 11,
          padding: "11px 12px",
          fontFamily: "var(--mono)",
          fontSize: 10.5,
          lineHeight: 1.6,
          color: "var(--t2)",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          maxHeight: 320,
          overflowY: "auto",
          userSelect: "text",
        }}
      >
        {text}
      </div>
      <button
        onClick={onCopy}
        style={{
          display: "block",
          width: "100%",
          marginTop: 11,
          padding: 10,
          borderRadius: 11,
          border: "1px solid var(--line2)",
          background: "transparent",
          fontSize: 13,
          fontWeight: 600,
          color: "var(--mid)",
          cursor: "pointer",
        }}
      >
        Copy log
      </button>
    </div>
  );
}
