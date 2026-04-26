import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

export function Toast({ toast }: { toast: { message: string; type: "success" | "error" } | null }) {
  const [visible, setVisible] = useState(false);
  const [current, setCurrent] = useState(toast);

  useEffect(() => {
    if (toast) {
      setCurrent(toast);
      setVisible(true);
    } else {
      setVisible(false);
      const timer = setTimeout(() => setCurrent(null), 200);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  if (!current) return null;

  return createPortal(
    <div
      role="status"
      aria-live="polite"
      className={`fixed top-16 right-4 z-50 rounded-lg border px-4 py-2.5 text-sm font-medium shadow-lg backdrop-blur-sm ${
        current.type === "success"
          ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
          : "border-destructive/20 bg-destructive/10 text-destructive"
      }`}
      style={{
        animation: visible ? "toast-in 200ms ease-out forwards" : "toast-out 200ms ease-in forwards",
      }}
    >
      {current.message}
    </div>,
    document.body,
  );
}
