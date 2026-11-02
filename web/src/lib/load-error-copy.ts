import { en } from "@/i18n/en";

/** Fill the `{what}` / `{detail}` placeholders of the "could not load" card copy.
 *  Kept out of the component so the copy shape is testable without rendering. */
export function loadErrorCopy(
  t: { loadFailed?: string; loadFailedDetails?: string },
  what: string,
  detail?: string | null,
): { title: string; details: string | null } {
  const loadFailed = t.loadFailed ?? en.common.loadFailed!;
  const loadFailedDetails = t.loadFailedDetails ?? en.common.loadFailedDetails!;
  return {
    title: loadFailed.replace("{what}", what),
    details: detail ? loadFailedDetails.replace("{detail}", detail) : null,
  };
}
