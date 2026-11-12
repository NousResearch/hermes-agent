import { useCallback, useEffect, useState } from "react";

interface FullscreenState {
  /** True while the document occupies the screen. */
  isFullscreen: boolean;
  /** False when the browser has no Fullscreen API (the control hides itself). */
  supported: boolean;
  toggleFullscreen: () => void;
}

/**
 * Real fullscreen for the dashboard shell, driven by the Fullscreen API.
 *
 * State tracks `fullscreenchange` rather than the click, so the control also
 * reflects Esc / the browser's own fullscreen exit. Failures (denied
 * permission, unsupported element) are swallowed: the dashboard stays usable
 * and the button simply does nothing.
 */
export function useFullscreen(): FullscreenState {
  const supported =
    typeof document !== "undefined" &&
    typeof document.documentElement.requestFullscreen === "function";

  const [isFullscreen, setIsFullscreen] = useState(
    () =>
      typeof document !== "undefined" && Boolean(document.fullscreenElement),
  );

  useEffect(() => {
    if (!supported) return;
    const sync = () => setIsFullscreen(Boolean(document.fullscreenElement));
    sync();
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, [supported]);

  const toggleFullscreen = useCallback(() => {
    if (!supported) return;
    if (document.fullscreenElement) {
      void document.exitFullscreen().catch(() => {});
      return;
    }
    void document.documentElement.requestFullscreen().catch(() => {});
  }, [supported]);

  return { isFullscreen, supported, toggleFullscreen };
}
