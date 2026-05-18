import {
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import { Typography } from "@/components/NouiTypography";
import { cn } from "@/lib/utils";

const CLOSE_DRAG_MIN_PX = 72;
const CLOSE_DRAG_RATIO = 0.18;
const SHEET_TRANSITION_MS = 280;

/**
 * Mobile-first picker shell: fixed backdrop + bottom sheet, portaled to `body`
 * so nested overflow/transform in the sidebar cannot clip menus (theme /
 * language switchers). Open/close uses slide + fade; teardown is delayed until
 * the exit animation finishes so animations can complete.
 *
 * Drag the header/handle downward to dismiss (skipped when reduced motion is on).
 */
export function BottomPickSheet({
  backdropDismissLabel = "Dismiss",
  children,
  onClose,
  open,
  title,
}: BottomPickSheetProps) {
  const [renderPortal, setRenderPortal] = useState(open);
  const [entered, setEntered] = useState(false);
  const [dragOffsetPx, setDragOffsetPx] = useState(0);
  const [dragActive, setDragActive] = useState(false);

  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sheetRef = useRef<HTMLDivElement>(null);
  const dragTrackingRef = useRef(false);
  const dragStartYRef = useRef(0);
  const dragOffsetRef = useRef(0);

  const reducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const syncDragPx = (next: number) => {
    dragOffsetRef.current = next;
    setDragOffsetPx(next);
  };

  useEffect(() => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }

    const ms = reducedMotion ? 0 : SHEET_TRANSITION_MS;

    let openRafId = 0;
    let exitRafId = 0;

    if (open) {
      openRafId = requestAnimationFrame(() => {
        dragTrackingRef.current = false;
        dragOffsetRef.current = 0;
        setDragActive(false);
        setDragOffsetPx(0);
        setRenderPortal(true);
        requestAnimationFrame(() => {
          requestAnimationFrame(() => setEntered(true));
        });
      });
    } else {
      exitRafId = requestAnimationFrame(() => {
        dragTrackingRef.current = false;
        setDragActive(false);
        setEntered(false);
        closeTimerRef.current = window.setTimeout(() => {
          dragOffsetRef.current = 0;
          setDragOffsetPx(0);
          setRenderPortal(false);
          closeTimerRef.current = null;
        }, ms);
      });
    }

    return () => {
      cancelAnimationFrame(openRafId);
      cancelAnimationFrame(exitRafId);
      if (closeTimerRef.current) {
        clearTimeout(closeTimerRef.current);
        closeTimerRef.current = null;
      }
    };
  }, [open, reducedMotion]);

  useEffect(() => {
    if (!renderPortal) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [renderPortal]);

  if (!renderPortal || typeof document === "undefined") return null;

  const durationClass = reducedMotion ? "duration-0" : "duration-[280ms]";

  const draggingVisual = dragActive || dragOffsetPx > 0;

  const onDragPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (reducedMotion || !entered) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;

    dragTrackingRef.current = true;
    setDragActive(true);
    dragStartYRef.current = e.clientY;
    syncDragPx(0);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const onDragPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragTrackingRef.current) return;
    const dy = e.clientY - dragStartYRef.current;
    const next = Math.max(0, dy);
    const sheetH = sheetRef.current?.offsetHeight ?? 560;
    syncDragPx(Math.min(next, sheetH));
  };

  const endDrag = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!dragTrackingRef.current) return;
    dragTrackingRef.current = false;
    setDragActive(false);
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }

    const sheetH = sheetRef.current?.offsetHeight ?? 560;
    const threshold = Math.max(CLOSE_DRAG_MIN_PX, sheetH * CLOSE_DRAG_RATIO);
    const d = dragOffsetRef.current;

    if (d >= threshold) {
      onClose();
      return;
    }
    syncDragPx(0);
  };

  return createPortal(
    <div className="fixed inset-0 z-[200] flex flex-col justify-end">
      <button
        type="button"
        aria-label={backdropDismissLabel}
        className={cn(
          "absolute inset-0 bg-black/55 backdrop-blur-[2px]",
          "transition-opacity ease-out motion-reduce:transition-none",
          durationClass,
          entered ? "opacity-100" : "opacity-0",
        )}
        onClick={onClose}
      />

      <div
        aria-label={title}
        aria-modal="true"
        ref={sheetRef}
        className={cn(
          "relative flex max-h-[85dvh] min-h-0 flex-col rounded-t-xl border border-current/20",
          "bg-background-base/98 pb-[max(1rem,env(safe-area-inset-bottom))]",
          "shadow-[0_-12px_40px_-8px_rgba(0,0,0,0.55)] backdrop-blur-md",
          "ease-out motion-reduce:transition-none transform-gpu",
          draggingVisual ? "transition-none" : cn("transition-transform", durationClass),
          entered ? "translate-y-0" : "translate-y-full",
        )}
        role="dialog"
        style={
          entered && dragOffsetPx > 0
            ? { transform: `translateY(${dragOffsetPx}px)` }
            : undefined
        }
      >
        <div
          className={cn(
            "flex shrink-0 flex-col gap-2 border-b border-current/15 px-4 pb-3 pt-2",
            "touch-none select-none",
            reducedMotion ? "cursor-default" : "cursor-grab active:cursor-grabbing",
          )}
          onPointerCancel={endDrag}
          onPointerDown={onDragPointerDown}
          onPointerMove={onDragPointerMove}
          onPointerUp={endDrag}
        >
          <div
            aria-hidden
            className="mx-auto h-1 w-10 shrink-0 rounded-full bg-current/20"
          />

          <Typography
            mondwest
            className="text-[0.65rem] tracking-[0.15em] uppercase text-midground/70"
          >
            {title}
          </Typography>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
          {children}
        </div>
      </div>
    </div>,
    document.body,
  );
}

interface BottomPickSheetProps {
  backdropDismissLabel?: string;
  children: ReactNode;
  onClose: () => void;
  open: boolean;
  title: string;
}
