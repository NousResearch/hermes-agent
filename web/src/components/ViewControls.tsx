import { Maximize2, Minimize2, Smartphone } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";

import { useFullscreen } from "@/hooks/useFullscreen";
import { useI18n } from "@/i18n";
import { cn } from "@/lib/utils";

interface ViewControlsProps {
  phoneView: boolean;
  onTogglePhoneView: () => void;
}

/**
 * The header's view toolbar: real fullscreen for the dashboard shell, and a
 * phone-adaptive preview that renders the same app at a phone viewport.
 *
 * Icon-only by design — the accessible name comes from `aria-label`/`title`
 * (localised), never from an emoji or a visible caption that would fight the
 * header's single-row layout.
 */
export function ViewControls({ phoneView, onTogglePhoneView }: ViewControlsProps) {
  const { t } = useI18n();
  const { isFullscreen, supported, toggleFullscreen } = useFullscreen();
  const labels = t.app.viewControls;

  const fullscreenLabel =
    (isFullscreen ? labels?.exitFullscreen : labels?.fullscreen) ??
    (isFullscreen ? "Exit fullscreen" : "Fullscreen");
  const phoneLabel = labels?.phoneView ?? "Phone adaptive version";

  return (
    <div
      className="flex items-center gap-0.5"
      role="group"
      aria-label={labels?.controls ?? "View controls"}
    >
      {supported && (
        <Button
          ghost
          size="icon"
          onClick={toggleFullscreen}
          aria-label={fullscreenLabel}
          title={fullscreenLabel}
          className="text-text-secondary hover:text-midground"
        >
          {isFullscreen ? (
            <Minimize2 className="h-4 w-4" />
          ) : (
            <Maximize2 className="h-4 w-4" />
          )}
        </Button>
      )}

      <Button
        ghost
        size="icon"
        onClick={onTogglePhoneView}
        aria-label={phoneLabel}
        aria-pressed={phoneView}
        title={phoneLabel}
        className={cn(
          "text-text-secondary hover:text-midground",
          phoneView && "bg-midground/10 text-midground",
        )}
      >
        <Smartphone className="h-4 w-4" />
      </Button>
    </div>
  );
}
