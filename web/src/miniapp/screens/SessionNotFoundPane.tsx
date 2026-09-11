export function SessionNotFoundPane({ onBack }: { onBack: () => void }) {
  return (
    <div
      style={{
        padding: "90px 32px 24px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 14,
        textAlign: "center",
      }}
    >
      <div
        style={{
          width: 58,
          height: 58,
          borderRadius: 99,
          border: "1.5px dashed var(--line2)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "var(--mono)",
          fontSize: 20,
          color: "var(--t3)",
        }}
      >
        ?
      </div>
      <div style={{ fontSize: 15, fontWeight: 650, color: "var(--mid)" }}>Session not found</div>
      <div style={{ fontSize: 12.5, color: "var(--t2)", lineHeight: 1.55, maxWidth: 250 }}>
        It may have been deleted or expired. Nothing is broken.
      </div>
      <button
        onClick={onBack}
        style={{
          marginTop: 6,
          padding: "9px 18px",
          borderRadius: 11,
          border: "1px solid var(--line2)",
          background: "transparent",
          fontSize: 13,
          fontWeight: 600,
          color: "var(--mid)",
          cursor: "pointer",
        }}
      >
        Back to sessions
      </button>
    </div>
  );
}
