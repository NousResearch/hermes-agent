import { useEffect, useRef } from "react";

/**
 * Hook that adds standard modal behaviors when `open` is true:
 * - Escape key calls `onClose`
 * - Body scroll is locked
 * - Focus is restored to the previously focused element on close
 *
 * Returns a ref to attach to the modal container (for optional future focus trapping).
 */
export function useModalBehavior({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;

    const prevActive = document.activeElement as HTMLElement | null;

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };

    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      prevActive?.focus?.();
    };
  }, [open, onClose]);

  return containerRef;
}
