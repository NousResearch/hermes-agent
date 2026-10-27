import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useI18n } from "@/i18n";

/**
 * Confirm + full-page reload after a model change.
 *
 * Changing the main model persists to config.yaml, but the RUNNING chat keeps
 * its model until its session is rebuilt. A full reload (fresh PTY session that
 * boots its agent from the just-saved config) is the reliable way to apply it —
 * the in-place hot-swap and partial remount both proved unreliable. We confirm
 * first because the reload starts a fresh chat (the current one stays resumable
 * in Sessions and the agent's memory is kept).
 *
 * Shared by the chat sidebar picker and the Models page so both behave
 * identically. `model` is the short model name awaiting confirmation, or null
 * when the dialog is closed. `notice`: the backend's switch warning (Nous window
 * plans, Large model), shown under the body; absent for everyone else.
 */
export function ModelReloadConfirm({
  model,
  description,
  notice,
  onCancel,
}: {
  model: string | null;
  /** Override the default body copy (e.g. the Models-page phrasing). */
  description?: string;
  notice?: string | null;
  onCancel: () => void;
}) {
  const { t } = useI18n();
  const P = t.modelPicker;
  const body =
    description ??
    (P?.reloadDescription ??
      "Switching to {model} starts a fresh chat. Your current chat stays in your Sessions list and the agent's memory is kept. Reload now to apply it?"
    ).replace("{model}", model ?? "");
  return (
    <ConfirmDialog
      open={model !== null}
      title={P?.reloadTitle ?? "Switch model?"}
      description={notice ? `${body}\n\n⚠ ${notice}` : body}
      confirmLabel={P?.reload ?? "Reload"}
      onConfirm={() => window.location.reload()}
      onCancel={onCancel}
    />
  );
}
