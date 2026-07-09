import { useEffect, useRef } from "react";

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "textarea:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

type InertState = {
  element: Element;
  ariaHidden: string | null;
  inert: boolean;
};

function getFocusable(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR)).filter(
    (element) =>
      !element.hasAttribute("disabled") &&
      element.getAttribute("aria-hidden") !== "true" &&
      element.offsetParent !== null,
  );
}

function getBackgroundSiblings(container: HTMLElement): Element[] {
  const candidates = new Set<Element>();
  for (let current: HTMLElement | null = container; current && current !== document.body;) {
    const ancestor: HTMLElement | null = current.parentElement;
    if (ancestor === null) break;
    const activeChild: HTMLElement = current;
    (Array.from(ancestor.children) as Element[]).forEach((child) => {
      if (child !== activeChild && !container.contains(child)) {
        candidates.add(child);
      }
    });
    current = ancestor;
  }

  return Array.from(candidates);
}

/**
 * Hook that adds standard modal behaviors when `open` is true:
 * - Escape key calls `onClose`
 * - Body scroll is locked
 * - Focus is moved into the modal, trapped with Tab/Shift+Tab, and restored
 * - Background siblings are marked inert/aria-hidden while the modal is open
 *
 * Returns a ref to attach to the modal container that owns `role="dialog"`.
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

    const container = containerRef.current;
    const prevActive = document.activeElement as HTMLElement | null;
    const prevOverflow = document.body.style.overflow;
    const inertElements = container ? getBackgroundSiblings(container) : [];
    const inertState: InertState[] = inertElements.map((element) => ({
      element,
      ariaHidden: element.getAttribute("aria-hidden"),
      inert: element.hasAttribute("inert"),
    }));

    inertElements.forEach((element) => {
      element.setAttribute("aria-hidden", "true");
      element.setAttribute("inert", "");
    });

    const focusInitial = window.requestAnimationFrame(() => {
      const current = containerRef.current;
      if (!current) return;
      const explicit = current.querySelector<HTMLElement>("[data-autofocus]");
      const first = explicit ?? getFocusable(current)[0] ?? current;
      first.focus?.();
    });

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab") return;
      const current = containerRef.current;
      if (!current) return;
      const focusable = getFocusable(current);
      if (focusable.length === 0) {
        e.preventDefault();
        current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";

    return () => {
      window.cancelAnimationFrame(focusInitial);
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      inertState.forEach(({ element, ariaHidden, inert }) => {
        if (ariaHidden === null) element.removeAttribute("aria-hidden");
        else element.setAttribute("aria-hidden", ariaHidden);
        if (!inert) element.removeAttribute("inert");
      });
      if (prevActive && document.contains(prevActive)) {
        prevActive.focus?.();
      }
    };
  }, [open, onClose]);

  return containerRef;
}
