import { ConfirmDialog } from "@/components/ConfirmDialog";

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
 * Important: by the time this dialog renders, the model is **already saved**.
 * The copy says so explicitly — the dialog is asking about reloading, not
 * about saving. There is no "don't save" affordance here; the save is a
 * precondition for the dialog appearing. Callers should NOT refresh the
 * sidebar badge (or otherwise signal "the model changed") until the user
 * has resolved this dialog — otherwise the user sees a UI change *before*
 * they've been asked to confirm it, and Cancel looks like a no-op.
 *
 * Shared by the chat sidebar picker and the Models page so both behave
 * identically. `model` is the short model name awaiting confirmation, or null
 * when the dialog is closed.
 */
export function ModelReloadConfirm({
  model,
  description,
  onCancel,
}: {
  model: string | null;
  /** Override the default body copy (e.g. the Models-page phrasing). */
  description?: string;
  onCancel: () => void;
}) {
  return (
    <ConfirmDialog
      open={model !== null}
      title="Switch model?"
      description={
        description ??
        `Model saved. Reload to start a fresh chat with ${model ?? "the new model"}? Your current chat stays in Sessions and the agent's memory is kept.`
      }
      confirmLabel="Reload"
      onConfirm={() => window.location.reload()}
      onCancel={onCancel}
    />
  );
}
