import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { useI18n } from "@/i18n";
import { DeleteConfirmDialog } from "@/components/DeleteConfirmDialog";

interface UseSkillDeleteOptions {
  profile?: string | null;
  showToast: (message: string, kind: "success" | "error") => void;
  onDeleted: (name: string) => void;
}

/** Skills page delete: a confirm dialog, then DELETE /api/skills, which archives the skill
 *  (restorable) through the same backend as Desktop's Capabilities "Archive". */
export function useSkillDelete({ profile, showToast, onDeleted }: UseSkillDeleteOptions) {
  const { t } = useI18n();
  const [target, setTarget] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  const fill = (text: string) => text.replace("{name}", target ?? "");

  const confirm = useCallback(async () => {
    if (!target) return;
    setDeleting(true);
    try {
      await api.deleteSkill(target, profile || undefined);
      onDeleted(target);
      showToast((t.skills.skillDeleted ?? "{name} archived").replace("{name}", target), "success");
      setTarget(null);
    } catch (e) {
      showToast(String(e), "error");
    } finally {
      setDeleting(false);
    }
  }, [target, profile, onDeleted, showToast, t]);

  const dialog = (
    <DeleteConfirmDialog
      open={target !== null}
      loading={deleting}
      title={fill(t.skills.deleteSkillConfirmTitle ?? "Delete skill \u201c{name}\u201d?")}
      description={fill(
        t.skills.deleteSkillConfirmMessage ??
          "The skill moves to the profile's skills archive. Restore it with `hermes curator restore {name}`.",
      )}
      onCancel={() => setTarget(null)}
      onConfirm={confirm}
    />
  );

  return { request: setTarget, dialog, label: t.skills.deleteSkill ?? "Delete skill" };
}
