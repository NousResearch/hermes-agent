import { useCallback, useEffect, useMemo, useState } from "react";
import { Pencil, Plus, Save, Trash2, X } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { useI18n } from "@/i18n";
import { api, getManagementProfile } from "@/lib/api";
import { errorMessage } from "@/lib/api-error";
import { cn, themedBody } from "@/lib/utils";
import {
  buildWhatsAppRouteRow,
  extractProfileRoutes,
  persistWhatsAppProfileRoutes,
  resolveHostProfileName,
  validateWhatsAppRouteChatId,
  validateWhatsAppRouteName,
  validateWhatsAppRouteProfile,
  whatsappProfileRoutesFromList,
  type WhatsAppProfileRouteRow,
} from "@/lib/whatsapp-profile-routes";

type RouteModalState =
  | { mode: "add" }
  | { mode: "edit"; index: number; preserve: Record<string, unknown> };

export function WhatsAppProfileRoutesPanel({
  showToast,
}: {
  showToast: (message: string, type: "success" | "error") => void;
}) {
  const { t } = useI18n();
  const copy = t.channelsPage?.whatsappRoutes;

  const [hostProfile, setHostProfile] = useState("default");
  const [profileNames, setProfileNames] = useState<string[]>([]);
  const [routes, setRoutes] = useState<WhatsAppProfileRouteRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [modal, setModal] = useState<RouteModalState | null>(null);
  const [draftName, setDraftName] = useState("");
  const [draftChatId, setDraftChatId] = useState("");
  const [draftProfile, setDraftProfile] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const installedSet = useMemo(() => new Set(profileNames), [profileNames]);
  const managementProfile = getManagementProfile();
  const viewingSecondary =
    Boolean(managementProfile) && managementProfile !== hostProfile;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const { profiles } = await api.getProfiles();
      const host = resolveHostProfileName(profiles);
      setHostProfile(host);
      setProfileNames(profiles.map((p) => p.name));
      const config = await api.getConfig(host);
      setRoutes(whatsappProfileRoutesFromList(extractProfileRoutes(config)));
    } catch (e) {
      showToast(
        copy?.loadFailed ?? `Could not load WhatsApp routes: ${errorMessage(e)}`,
        "error",
      );
    } finally {
      setLoading(false);
    }
  }, [copy?.loadFailed, showToast]);

  useEffect(() => {
    void load();
  }, [load]);

  const closeModal = useCallback(() => {
    setModal(null);
    setFieldErrors({});
  }, []);
  const modalRef = useModalBehavior({ open: modal !== null, onClose: closeModal });

  const openAdd = () => {
    setDraftName("");
    setDraftChatId("");
    setDraftProfile(profileNames[0] ?? "");
    setFieldErrors({});
    setModal({ mode: "add" });
  };

  const openEdit = (index: number) => {
    const row = routes[index];
    setDraftName(row.name);
    setDraftChatId(row.chat_id);
    setDraftProfile(row.profile);
    setFieldErrors({});
    setModal({ mode: "edit", index, preserve: { ...row } });
  };

  const validateDraft = (): WhatsAppProfileRouteRow | null => {
    const errors: Record<string, string> = {};
    const nameErr = validateWhatsAppRouteName(draftName);
    if (nameErr) errors.name = nameErr;
    const chatErr = validateWhatsAppRouteChatId(draftChatId);
    if (chatErr) errors.chat_id = chatErr;
    const profileErr = validateWhatsAppRouteProfile(draftProfile, installedSet);
    if (profileErr) errors.profile = profileErr;
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      return null;
    }
    const preserve =
      modal?.mode === "edit" ? modal.preserve : null;
    return buildWhatsAppRouteRow(
      { name: draftName, chat_id: draftChatId, profile: draftProfile },
      preserve,
    );
  };

  const applyModal = () => {
    const row = validateDraft();
    if (!row || !modal) return;
    if (modal.mode === "add") {
      setRoutes((prev) => [...prev, row]);
    } else {
      setRoutes((prev) => prev.map((r, i) => (i === modal.index ? row : r)));
    }
    closeModal();
  };

  const removeRow = (index: number) => {
    setRoutes((prev) => prev.filter((_, i) => i !== index));
  };

  const saveRoutes = async () => {
    setSaving(true);
    try {
      await persistWhatsAppProfileRoutes(
        {
          getConfig: (profile) => api.getConfig(profile),
          saveConfig: (partial, profile) => api.saveConfig(partial, profile),
        },
        hostProfile,
        routes,
      );
      showToast(
        copy?.saved ??
          "Routes saved. They take effect the next time the gateway process starts.",
        "success",
      );
      await load();
    } catch (e) {
      showToast(`Failed to save routes: ${errorMessage(e)}`, "error");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="rounded-sm border border-border bg-background/35 p-4">
      <div className="grid gap-2">
        <span className="font-mondwest text-sm text-foreground">
          {copy?.title ?? "One number, many profiles"}
        </span>
        <p className="text-xs text-muted-foreground">
          {copy?.hostIntro ??
            "WhatsApp bridge and multiplex routing live on the host (default) profile config, not the profile picker selection."}
        </p>
        <p className="text-xs text-muted-foreground">
          {(copy?.hostProfileLabel ?? "Host profile")}:{" "}
          <code className="font-courier">{hostProfile}</code>
          {" · "}
          {copy?.multiplexOn ?? "Multiplex routing is on (gateway.multiplex_profiles)."}
        </p>
        {viewingSecondary && (
          <p className="text-xs text-muted-foreground">
            {copy?.ingressOwnedByHost ??
              "WhatsApp ingress is owned by the host profile; enabling WhatsApp on a secondary profile does not start a second bridge."}
          </p>
        )}
      </div>

      {loading ? (
        <div className="flex justify-center py-8">
          <Spinner className="text-xl text-primary" />
        </div>
      ) : (
        <>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[420px] border-collapse text-xs">
              <thead>
                <tr className="border-b border-border text-left text-muted-foreground">
                  <th className="py-2 pr-3 font-medium">{copy?.colName ?? "Name"}</th>
                  <th className="py-2 pr-3 font-medium">{copy?.colChatId ?? "Chat id"}</th>
                  <th className="py-2 pr-3 font-medium">{copy?.colProfile ?? "Profile"}</th>
                  <th className="py-2 w-24" />
                </tr>
              </thead>
              <tbody>
                {routes.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-4 text-muted-foreground">
                      {copy?.empty ?? "No WhatsApp routes yet. Add a group JID or DM target."}
                    </td>
                  </tr>
                ) : (
                  routes.map((row, index) => (
                    <tr key={`${row.name}-${index}`} className="border-b border-border/60">
                      <td className="py-2 pr-3 font-medium">{row.name}</td>
                      <td className="py-2 pr-3 font-courier break-all">{row.chat_id}</td>
                      <td className="py-2 pr-3 font-courier">{row.profile}</td>
                      <td className="py-2">
                        <div className="flex gap-1">
                          <Button
                            ghost
                            size="icon"
                            aria-label={copy?.edit ?? "Edit route"}
                            onClick={() => openEdit(index)}
                          >
                            <Pencil className="h-3.5 w-3.5" />
                          </Button>
                          <Button
                            ghost
                            size="icon"
                            className="text-destructive"
                            aria-label={copy?.remove ?? "Remove route"}
                            onClick={() => removeRow(index)}
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <Button
              size="sm"
              outlined
              prefix={<Plus className="h-4 w-4" />}
              onClick={openAdd}
              disabled={profileNames.length === 0}
            >
              {copy?.addRoute ?? "Add route"}
            </Button>
            <Button
              size="sm"
              className="uppercase"
              onClick={() => void saveRoutes()}
              disabled={saving}
              prefix={saving ? <Spinner /> : <Save className="h-4 w-4" />}
            >
              {saving ? (copy?.saving ?? "Saving…") : (copy?.saveRoutes ?? "Save routes")}
            </Button>
          </div>
        </>
      )}

      {modal && (
        <div
          ref={modalRef}
          className={cn(
            "fixed inset-0 z-[100] flex min-h-dvh items-start justify-center overflow-y-auto bg-background/85 px-4",
            "pb-[calc(1rem+env(safe-area-inset-bottom))] pt-[calc(1rem+env(safe-area-inset-top))]",
            "sm:items-center sm:p-4",
          )}
          onClick={(e) => e.target === e.currentTarget && closeModal()}
          role="dialog"
          aria-modal="true"
          aria-labelledby="wa-route-modal-title"
        >
          <div
            className={cn(
              themedBody,
              "relative flex max-h-[calc(100dvh-2rem)] w-full max-w-md flex-col border border-border bg-card shadow-2xl sm:max-h-[90dvh]",
            )}
          >
            <Button
              ghost
              size="icon"
              onClick={closeModal}
              className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
              aria-label={t.common.close}
            >
              <X />
            </Button>
            <header className="border-b border-border p-5 pb-3">
              <h2
                id="wa-route-modal-title"
                className="font-mondwest text-display text-base tracking-wider"
              >
                {modal.mode === "add"
                  ? (copy?.modalAddTitle ?? "Add WhatsApp route")
                  : (copy?.modalEditTitle ?? "Edit WhatsApp route")}
              </h2>
            </header>
            <div className="grid gap-4 overflow-y-auto p-5">
              <div className="grid gap-1.5">
                <Label htmlFor="wa-route-name">{copy?.fieldName ?? "Name"} *</Label>
                <Input
                  id="wa-route-name"
                  value={draftName}
                  aria-invalid={Boolean(fieldErrors.name)}
                  onChange={(e) => {
                    setDraftName(e.target.value);
                    setFieldErrors((prev) => {
                      if (!prev.name) return prev;
                      const next = { ...prev };
                      delete next.name;
                      return next;
                    });
                  }}
                />
                {fieldErrors.name && (
                  <span className="text-xs text-destructive">{fieldErrors.name}</span>
                )}
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="wa-route-chat">{copy?.fieldChatId ?? "Chat id"} *</Label>
                <Input
                  id="wa-route-chat"
                  className="font-courier"
                  placeholder="120363…@g.us or 15551234567"
                  value={draftChatId}
                  aria-invalid={Boolean(fieldErrors.chat_id)}
                  onChange={(e) => {
                    setDraftChatId(e.target.value);
                    setFieldErrors((prev) => {
                      if (!prev.chat_id) return prev;
                      const next = { ...prev };
                      delete next.chat_id;
                      return next;
                    });
                  }}
                />
                <span className="text-xs text-muted-foreground">
                  {copy?.fieldChatIdHint ??
                    "Groups must end with @g.us. DMs may use a phone number or user JID."}
                </span>
                {fieldErrors.chat_id && (
                  <span className="text-xs text-destructive">{fieldErrors.chat_id}</span>
                )}
              </div>
              <div className="grid gap-1.5">
                <Label htmlFor="wa-route-profile">{copy?.fieldProfile ?? "Target profile"} *</Label>
                <Select
                  id="wa-route-profile"
                  value={draftProfile}
                  onValueChange={(value) => {
                    setDraftProfile(value);
                    setFieldErrors((prev) => {
                      if (!prev.profile) return prev;
                      const next = { ...prev };
                      delete next.profile;
                      return next;
                    });
                  }}
                >
                  {profileNames.map((name) => (
                    <SelectOption key={name} value={name}>
                      {name}
                    </SelectOption>
                  ))}
                </Select>
                {fieldErrors.profile && (
                  <span className="text-xs text-destructive">{fieldErrors.profile}</span>
                )}
              </div>
              <div className="flex flex-col-reverse gap-2 pt-1 sm:flex-row sm:justify-end">
                <Button ghost size="sm" onClick={closeModal}>
                  {t.common.cancel}
                </Button>
                <Button size="sm" className="uppercase" onClick={applyModal}>
                  {copy?.modalApply ?? "Apply"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
