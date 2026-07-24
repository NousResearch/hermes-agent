import { useEffect, useRef, type RefObject } from "react";

const FOCUSABLE_SELECTOR = [
  "[data-modal-initial-focus]",
  "[autofocus]",
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  "[tabindex]:not([tabindex='-1'])",
].join(",");

interface IsolatedElementState {
  count: number;
  inert: boolean;
  inertAttribute: boolean;
  ariaHidden: string | null;
}

const isolatedElements = new Map<HTMLElement, IsolatedElementState>();
const overlayStack: symbol[] = [];
let bodyLockCount = 0;
let bodyOverflowBeforeLock = "";

function isUsableFocusTarget(element: HTMLElement): boolean {
  if (element.closest("[inert], [hidden], [aria-hidden='true']")) return false;
  const style = window.getComputedStyle(element);
  return style.display !== "none" && style.visibility !== "hidden";
}

function focusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(
    container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
  ).filter(isUsableFocusTarget);
}

function focusFirstElement(container: HTMLElement): void {
  const preferred = container.querySelector<HTMLElement>(
    "[data-modal-initial-focus], [autofocus]",
  );
  const target =
    (preferred && isUsableFocusTarget(preferred) ? preferred : null) ??
    focusableElements(container)[0];

  if (target) {
    target.focus();
    return;
  }

  container.focus();
}

function acquireIsolation(element: HTMLElement): void {
  const existing = isolatedElements.get(element);
  if (existing) {
    existing.count += 1;
    return;
  }

  isolatedElements.set(element, {
    count: 1,
    inert: element.inert,
    inertAttribute: element.hasAttribute("inert"),
    ariaHidden: element.getAttribute("aria-hidden"),
  });
  element.inert = true;
  element.setAttribute("inert", "");
  element.setAttribute("aria-hidden", "true");
}

function releaseIsolation(element: HTMLElement): void {
  const state = isolatedElements.get(element);
  if (!state) return;
  state.count -= 1;
  if (state.count > 0) return;

  element.inert = state.inert;
  if (!state.inertAttribute) element.removeAttribute("inert");
  if (state.ariaHidden === null) {
    element.removeAttribute("aria-hidden");
  } else {
    element.setAttribute("aria-hidden", state.ariaHidden);
  }
  isolatedElements.delete(element);
}

/**
 * Isolate every DOM branch outside the active overlay. Walking to `body`
 * makes this work for both in-tree drawers and modals portaled to `body`.
 */
function isolateBackground(container: HTMLElement): HTMLElement[] {
  const isolated: HTMLElement[] = [];
  let branch: HTMLElement = container;

  while (branch.parentElement) {
    const parent = branch.parentElement;
    for (const sibling of Array.from(parent.children)) {
      if (sibling === branch || !(sibling instanceof HTMLElement)) continue;
      acquireIsolation(sibling);
      isolated.push(sibling);
    }
    if (parent === document.body) break;
    branch = parent;
  }

  return isolated;
}

function lockBodyScroll(): void {
  if (bodyLockCount === 0) {
    bodyOverflowBeforeLock = document.body.style.overflow;
    document.body.style.overflow = "hidden";
  }
  bodyLockCount += 1;
}

function unlockBodyScroll(): void {
  bodyLockCount = Math.max(0, bodyLockCount - 1);
  if (bodyLockCount === 0) {
    document.body.style.overflow = bodyOverflowBeforeLock;
  }
}

interface ModalBehaviorOptions {
  open: boolean;
  onClose: () => void;
  /** Drawers provide their own geometry; ordinary modal shells use the shared viewport classes. */
  modalLayout?: boolean;
  /** Optional stable opener for overlays whose trigger is rendered through a portal/context slot. */
  restoreFocusRef?: RefObject<HTMLElement | null>;
}

/**
 * Shared accessible overlay behavior for dashboard dialogs and drawers:
 * focus containment/restoration, Escape dismissal, background isolation,
 * body-scroll locking, and mobile-safe modal viewport classes.
 */
export function useModalBehavior({
  open,
  onClose,
  modalLayout = true,
  restoreFocusRef,
}: ModalBehaviorOptions) {
  const containerRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    if (!open) return;

    const container = containerRef.current;
    if (!container) return;

    const overlayId = Symbol("dashboard-overlay");
    const previousActive =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    const restoreFocusTarget = restoreFocusRef?.current ?? previousActive;
    const hadTabIndex = container.hasAttribute("tabindex");
    const previousTabIndex = container.getAttribute("tabindex");
    const addedBackdropClass =
      modalLayout && !container.classList.contains("hermes-modal-backdrop");
    const panel = modalLayout
      ? container.firstElementChild instanceof HTMLElement
        ? container.firstElementChild
        : null
      : null;
    const addedPanelClass =
      panel !== null &&
      !panel.classList.contains("hermes-modal-panel-viewport");

    if (!hadTabIndex) container.setAttribute("tabindex", "-1");
    if (addedBackdropClass) container.classList.add("hermes-modal-backdrop");
    if (addedPanelClass) panel?.classList.add("hermes-modal-panel-viewport");

    overlayStack.push(overlayId);
    const isolated = isolateBackground(container);
    lockBodyScroll();

    const isTopOverlay = () => overlayStack.at(-1) === overlayId;
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isTopOverlay()) return;

      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        onCloseRef.current();
        return;
      }

      if (event.key !== "Tab") return;
      const focusable = focusableElements(container);
      if (focusable.length === 0) {
        event.preventDefault();
        container.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (!container.contains(active)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };
    const onFocusIn = (event: FocusEvent) => {
      if (
        isTopOverlay() &&
        event.target instanceof Node &&
        !container.contains(event.target)
      ) {
        focusFirstElement(container);
      }
    };

    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("focusin", onFocusIn);
    if (!container.contains(document.activeElement)) {
      focusFirstElement(container);
    }

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("focusin", onFocusIn);
      const stackIndex = overlayStack.lastIndexOf(overlayId);
      if (stackIndex >= 0) overlayStack.splice(stackIndex, 1);
      isolated.reverse().forEach(releaseIsolation);
      unlockBodyScroll();

      if (addedBackdropClass) container.classList.remove("hermes-modal-backdrop");
      if (addedPanelClass) panel?.classList.remove("hermes-modal-panel-viewport");
      if (!hadTabIndex) {
        container.removeAttribute("tabindex");
      } else if (previousTabIndex !== null) {
        container.setAttribute("tabindex", previousTabIndex);
      }

      if (restoreFocusTarget?.isConnected) restoreFocusTarget.focus();
    };
  }, [modalLayout, open, restoreFocusRef]);

  return containerRef;
}
