import { useCallback, useState } from "react";

import {
  isPhonePreviewMode,
  readStoredPhoneView,
  writeStoredPhoneView,
} from "@/lib/phone-view";

/**
 * The header's phone-view toggle. Persisted per browser so the choice survives
 * a reload, but never active inside the preview iframe itself (that would nest
 * previews and, worse, let the iframe open a preview of its own).
 */
export function usePhoneView() {
  const [phoneView, setPhoneView] = useState(
    () => !isPhonePreviewMode() && readStoredPhoneView(),
  );

  const togglePhoneView = useCallback(() => {
    setPhoneView((prev) => {
      const next = !prev;
      writeStoredPhoneView(next);
      return next;
    });
  }, []);

  const closePhoneView = useCallback(() => {
    writeStoredPhoneView(false);
    setPhoneView(false);
  }, []);

  return { phoneView, togglePhoneView, closePhoneView };
}
