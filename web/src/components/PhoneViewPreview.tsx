import { useEffect, useMemo, useRef } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";

import { useI18n } from "@/i18n";
import { buildPhonePreviewUrl } from "@/lib/phone-view";

interface PhoneViewPreviewProps {
  onClose: () => void;
}

const PHONE_VIEW_CLOSE_ID = "phone-view-close";

/**
 * The phone-adaptive view: a fixed 390px-wide same-origin iframe showing the
 * current route. A real viewport, so the app's own responsive rules and
 * `matchMedia` calls produce the phone layout — the preview is the phone
 * version, not a scaled screenshot of the desktop one.
 *
 * The preview runs with `?phone-preview=1`; that flag hides the view controls
 * inside it and keeps ChatPage from spawning a second live PTY (which would
 * supersede the tab that opened the preview).
 */
export function PhoneViewPreview({ onClose }: PhoneViewPreviewProps) {
  const { t } = useI18n();
  const labels = t.app.viewControls;
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  const previewUrl = useMemo(() => buildPhonePreviewUrl(), []);
  const dialogLabel = labels?.phonePreview ?? "Phone preview";
  const closeLabel = labels?.closePhoneView ?? "Close phone view";

  useEffect(() => {
    restoreFocusRef.current = document.activeElement as HTMLElement | null;
    // The UI Button does not forward refs; focus its rendered element by id.
    document.getElementById(PHONE_VIEW_CLOSE_ID)?.focus();

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previousOverflow;
      restoreFocusRef.current?.focus?.();
    };
  }, [onClose]);

  return createPortal(
    <div
      className="fixed inset-0 z-[300] flex flex-col items-center justify-center gap-3 bg-black/70 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={dialogLabel}
    >
      <div className="flex items-center gap-3">
        <Button
          id={PHONE_VIEW_CLOSE_ID}
          outlined
          onClick={onClose}
          aria-label={closeLabel}
        >
          <span className="inline-flex items-center gap-1.5">
            <X className="h-3.5 w-3.5" />
            {closeLabel}
          </span>
        </Button>
        <span aria-hidden className="text-xs tracking-wide text-white/60">
          390 × 844
        </span>
      </div>

      <div className="phone-view-frame">
        <iframe
          src={previewUrl}
          title={dialogLabel}
          className="block h-full w-full border-0 bg-background-base"
        />
      </div>
    </div>,
    document.body,
  );
}
