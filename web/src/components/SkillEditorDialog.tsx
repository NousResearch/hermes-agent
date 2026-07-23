import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@nous-research/ui/ui/components/dialog";

/* ------------------------------------------------------------------ */
/*  SkillEditorDialog — create or edit a SKILL.md from the dashboard   */
/*                                                                      */
/*  Headless/VPS users have no editor besides this: the only other way */
/*  to author a custom skill is SSH + a terminal editor. Create mode   */
/*  posts a brand-new skill (name + optional category + SKILL.md);     */
/*  edit mode loads the existing SKILL.md raw text and rewrites it.    */
/*  Validation (frontmatter, name, size) happens server-side via the   */
/*  same path the agent's skill_manage tool uses, so errors come back  */
/*  as actionable messages rendered inline.                            */
/* ------------------------------------------------------------------ */

const CREATE_TEMPLATE = `---
name: my-skill
description: One-line description of when to use this skill.
---

# My Skill

Numbered steps, exact commands, and pitfalls go here.
`;

export interface SkillEditorDialogProps {
  open: boolean;
  /** Skill name to edit, or null for create mode. */
  editName: string | null;
  /** Profile to scope reads/writes to ("" = the dashboard's own profile). */
  profile?: string;
  onClose: () => void;
  /** Called after a successful save so the page can refresh its list. */
  onSaved: (name: string) => void;
}

export function SkillEditorDialog({
  open,
  editName,
  profile,
  onClose,
  onSaved,
}: SkillEditorDialogProps) {
  // The body is remounted via `key` every time the dialog opens or the
  // target skill changes, so all form state initializes through useState
  // initializers — no reset-on-open effect (react-hooks/set-state-in-effect).
  return (
    <Dialog open={open} onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="hermes-modal-panel-viewport flex h-[calc(100dvh-1.5rem)] max-w-3xl flex-col overflow-hidden sm:h-auto max-sm:[&_[data-slot=dialog-close]]:h-11 max-sm:[&_[data-slot=dialog-close]]:w-11">
        {open && (
          <EditorBody
            key={editName ?? "__create__"}
            editName={editName}
            profile={profile}
            onClose={onClose}
            onSaved={onSaved}
          />
        )}
      </DialogContent>
    </Dialog>
  );
}

function EditorBody({
  editName,
  profile,
  onClose,
  onSaved,
}: Omit<SkillEditorDialogProps, "open">) {
  const isEdit = editName !== null;
  const [name, setName] = useState("");
  const [category, setCategory] = useState("");
  const [content, setContent] = useState(isEdit ? "" : CREATE_TEMPLATE);
  const [loading, setLoading] = useState(isEdit);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!editName) return;
    let cancelled = false;
    api
      .getSkillContent(editName, profile || undefined)
      .then((res) => !cancelled && setContent(res.content))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [editName, profile]);

  const handleSave = async () => {
    setError(null);
    if (!isEdit && !name.trim()) {
      setError("Skill name is required.");
      return;
    }
    if (!content.trim()) {
      setError("SKILL.md content is required.");
      return;
    }
    setSaving(true);
    try {
      if (isEdit) {
        await api.updateSkillContent(editName, content, profile || undefined);
        onSaved(editName);
      } else {
        const trimmed = name.trim();
        await api.createSkill(
          {
            name: trimmed,
            content,
            category: category.trim() || undefined,
          },
          profile || undefined,
        );
        onSaved(trimmed);
      }
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <DialogHeader>
        <DialogTitle>
          {isEdit ? `Edit skill: ${editName}` : "New skill"}
        </DialogTitle>
        <DialogDescription>
          {isEdit
            ? "Rewrite this skill's SKILL.md. Frontmatter (name, description) is validated on save."
            : "Author a custom skill — YAML frontmatter plus markdown instructions. It becomes available to the agent and attachable to cron jobs."}
        </DialogDescription>
      </DialogHeader>

      <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto overscroll-contain p-3 sm:overflow-visible sm:p-0">
        {!isEdit && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="grid gap-1.5">
              <Label htmlFor="skill-editor-name">Name</Label>
              <Input
                id="skill-editor-name"
                autoFocus
                placeholder="my-skill"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="skill-editor-category">Category (optional)</Label>
              <Input
                id="skill-editor-category"
                placeholder="devops"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
              />
            </div>
          </div>
        )}

        <div className="grid min-h-0 flex-1 gap-1.5">
          <Label htmlFor="skill-editor-content">SKILL.md</Label>
          {loading ? (
            <div className="flex items-center justify-center py-16">
              <Spinner className="text-xl text-primary" />
            </div>
          ) : (
            <textarea
              id="skill-editor-content"
              spellCheck={false}
              className="h-full min-h-36 max-h-[45dvh] w-full resize-y border border-border bg-background/40 px-3 py-2 font-mono text-xs leading-relaxed shadow-sm placeholder:text-muted-foreground focus-visible:border-foreground/25 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground/30 sm:min-h-[320px] sm:max-h-[55vh]"
              value={content}
              onChange={(e) => setContent(e.target.value)}
            />
          )}
        </div>

        {error && (
          <p className="whitespace-pre-wrap text-xs text-destructive">
            {error}
          </p>
        )}

        <div className="sticky bottom-0 grid shrink-0 grid-cols-2 gap-2 bg-background-base py-1 sm:static sm:flex sm:items-center sm:justify-end sm:py-0">
          <Button
            className="min-h-11 sm:min-h-0"
            ghost
            size="sm"
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            className="min-h-11 uppercase sm:min-h-0"
            onClick={handleSave}
            disabled={saving || loading}
            prefix={saving ? <Spinner /> : undefined}
          >
            {saving ? "Saving…" : isEdit ? "Save changes" : "Create skill"}
          </Button>
        </div>
      </div>
    </>
  );
}
