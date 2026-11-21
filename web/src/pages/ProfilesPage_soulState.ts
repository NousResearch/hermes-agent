import { useCallback, useRef, useState } from "react";
import { api } from "@/lib/api";
import { errorMessage } from "@/lib/api-error";
import { useI18n } from "@/i18n";

type ShowToast = (message: string, kind: "success" | "error") => void;

/** SOUL.md editor state for the Profiles page.
 *
 *  `unread` is set from the moment a read starts until it succeeds: while it
 *  is set the editor and Save are disabled, because saving the empty
 *  placeholder would overwrite a SOUL.md we never read (a 500 on GET — e.g.
 *  an unreadable file — used to leave an editable empty box whose Save wiped
 *  the persona). Inspired by Claude Cowork's fix for "Global instructions
 *  opening empty and overwriting the saved instructions when the file could
 *  not be read" (desktop changelog v6.28899.2). */
export function useSoulEditor(showToast: ShowToast, onSaved: () => void) {
  const { t } = useI18n();
  const [text, setText] = useState("");
  const [saving, setSaving] = useState(false);
  const [unread, setUnread] = useState(false);
  const [readError, setReadError] = useState<string | null>(null);
  // Tracks the latest SOUL request so out-of-order responses don't overwrite
  // newer state when the user switches profiles or closes the editor.
  const activeRequest = useRef<string | null>(null);

  const load = useCallback(
    async (name: string) => {
      setText("");
      setUnread(true);
      setReadError(null);
      activeRequest.current = name;
      try {
        const soul = await api.getProfileSoul(name);
        if (activeRequest.current === name) {
          setText(soul.content);
          setUnread(false);
        }
      } catch (e) {
        if (activeRequest.current === name) {
          setReadError(errorMessage(e));
          showToast(`${t.status.error}: ${errorMessage(e)}`, "error");
        }
      }
    },
    [showToast, t.status.error],
  );

  const save = async (name: string) => {
    if (unread) return;
    setSaving(true);
    try {
      await api.updateProfileSoul(name, text);
      showToast(`${t.profiles.soulSaved}: ${name}`, "success");
      activeRequest.current = null;
      onSaved();
    } catch (e) {
      showToast(`${t.status.error}: ${errorMessage(e)}`, "error");
    } finally {
      setSaving(false);
    }
  };

  const forget = useCallback(() => {
    activeRequest.current = null;
  }, []);

  return { text, setText, saving, unread, readError, load, save, forget };
}
