import { useI18n } from "@/i18n";
import { resolveApiHint } from "@/lib/api-hints";
import { useProfileScope } from "@/contexts/useProfileScope";

/** Muted footer naming the REST route and `hermes` command behind the current page, so the
 *  dashboard teaches its own scripting equivalent. Renders nothing for pages without one. */
export function ApiHint({ pathname }: { pathname: string }) {
  const { t } = useI18n();
  const { profile, currentProfile } = useProfileScope();
  const hint = resolveApiHint(pathname, profile || currentProfile);
  // Chat, docs and plugin pages have no entry and render nothing.
  if (!hint) return null;
  return (
    <footer
      aria-label={t.app.apiHint.label}
      className="mt-6 flex flex-wrap items-center gap-x-2 gap-y-1 font-mono-ui text-xs text-text-tertiary"
    >
      <code>{hint.api}</code>
      <span aria-hidden>·</span>
      <code>{hint.cli}</code>
      <span aria-hidden>·</span>
      <a
        href={hint.docsUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="underline-offset-2 hover:underline"
      >
        {t.app.apiHint.docs}
      </a>
    </footer>
  );
}
