import { useCallback, useEffect, useMemo, useState } from "react";
import { Pencil, Plus, RefreshCw, Save, Trash2, X } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { useModalBehavior } from "@/hooks/useModalBehavior";
import { useI18n } from "@/i18n";
import { useProfileScope } from "@/contexts/useProfileScope";
import { api } from "@/lib/api";
import { errorMessage } from "@/lib/api-error";
import { cn, themedBody } from "@/lib/utils";
import {
  buildWhatsAppRouteRow,
  extractProfileRoutes,
  formatProfileRouteOption,
  formatWhatsAppLastSeen,
  multiplexProfilesFlag,
  persistWhatsAppProfileRoutes,
  resolveHostProfileName,
  whatsappMultiplexerEngaged,
  whatsappRouteManager,
  unassignedWhatsAppChats,
  validateWhatsAppRouteChatId,
  validateWhatsAppRouteName,
  validateWhatsAppRouteProfile,
  whatsappChatKind,
  whatsappChatLabel,
  whatsappProfileRoutesFromList,
  whatsappRouteCountsByProfile,
  whatsappRouteNameForChat,
  type WhatsAppProfileRouteRow,
  type WhatsAppSeenChatView,
} from "@/lib/whatsapp-profile-routes";

type RouteModalState =
  | { mode: "add" }
  | { mode: "edit"; index: number; preserve: Record<string, unknown> };

export function WhatsAppProfileRoutesPanel({
  showToast,
  onLinkChange,
}: {
  showToast: (message: string, type: "success" | "error") => void;
  onLinkChange?: (link: { linked: boolean; manager: string }) => void;
}) {
  const { t } = useI18n();
  const copy = t.channelsPage?.whatsappRoutes;

  const { profile: selectedProfile, setProfile } = useProfileScope();
  const [hostProfile, setHostProfile] = useState("default");
  const [multiplexFlag, setMultiplexFlag] = useState<boolean | null>(null);
  const [sharedWith, setSharedWith] = useState<string[] | null>(null);
  const [standaloneReason, setStandaloneReason] = useState<string | null>(null);
  const [engaging, setEngaging] = useState(false);
  const [profileNames, setProfileNames] = useState<string[]>([]);
  const [routes, setRoutes] = useState<WhatsAppProfileRouteRow[]>([]);
  const [seenChats, setSeenChats] = useState<WhatsAppSeenChatView[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [refreshingChats, setRefreshingChats] = useState(false);
  const [modal, setModal] = useState<RouteModalState | null>(null);
  const [draftName, setDraftName] = useState("");
  const [draftChatId, setDraftChatId] = useState("");
  const [draftProfile, setDraftProfile] = useState("");
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const installedSet = useMemo(() => new Set(profileNames), [profileNames]);
  const routeCounts = useMemo(() => whatsappRouteCountsByProfile(routes), [routes]);
  const seenById = useMemo(() => {
    const map = new Map<string, WhatsAppSeenChatView>();
    for (const chat of seenChats) {
      map.set(chat.chat_id, chat);
      const local = chat.chat_id.split("@")[0];
      if (local && !map.has(local)) map.set(local, chat);
      if (!chat.chat_id.includes("@")) map.set(`${chat.chat_id}@g.us`, chat);
    }
    return map;
  }, [seenChats]);
  const unassignedChats = useMemo(
    () => unassignedWhatsAppChats(seenChats, routes),
    [seenChats, routes],
  );
  const link = useMemo(
    () => whatsappRouteManager({ selectedProfile, hostProfile, routes }),
    [selectedProfile, hostProfile, routes],
  );
  const engaged = whatsappMultiplexerEngaged(multiplexFlag, sharedWith, standaloneReason);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const { profiles } = await api.getProfiles();
      const host = resolveHostProfileName(profiles);
      setHostProfile(host);
      setProfileNames(profiles.map((p) => p.name));
      const [config, seen, status] = await Promise.all([
        api.getConfig(host, { includeDefaults: false }),
        api.getWhatsAppSeenChats().catch(() => ({ chats: [] as WhatsAppSeenChatView[] })),
        api.getStatus().catch(() => null),
      ]);
      setMultiplexFlag(multiplexProfilesFlag(config));
      setSharedWith(status?.gateway_shared_with ?? null);
      setStandaloneReason(status?.multiplex_standalone_reason ?? null);
      setRoutes(whatsappProfileRoutesFromList(extractProfileRoutes(config)));
      setSeenChats(seen.chats ?? []);
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

  useEffect(() => {
    onLinkChange?.(link);
  }, [link, onLinkChange]);

  const engageMultiplexer = async (next: boolean) => {
    if (!next) {
      showToast(
        copy?.multiplexStayOn ??
          "The host gateway stays on. One gateway per host cannot be switched off. Clear routes to stop sending chats to other profiles.",
        "error",
      );
      return;
    }
    setEngaging(true);
    try {
      await api.saveConfig({ gateway: { multiplex_profiles: true } }, hostProfile);
      setMultiplexFlag(true);
      showToast(
        copy?.multiplexEngaged ??
          `Multiplexer on. It is linked to ${hostProfile}. Restart the gateway if it is not already serving every profile.`,
        "success",
      );
    } catch (e) {
      showToast(`Could not turn the multiplexer on: ${errorMessage(e)}`, "error");
    } finally {
      setEngaging(false);
    }
  };

  const closeModal = useCallback(() => {
    setModal(null);
    setFieldErrors({});
  }, []);
  const modalRef = useModalBehavior({ open: modal !== null, onClose: closeModal });

  const refreshSeenChats = useCallback(async () => {
    setRefreshingChats(true);
    try {
      const seen = await api.getWhatsAppSeenChats();
      const next = seen.chats ?? [];
      setSeenChats(next);
      const open = unassignedWhatsAppChats(next, routes);
      setDraftChatId((current) => {
        if (next.length === 0) return current;
        return open.some((chat) => chat.chat_id === current) ? current : (open[0]?.chat_id ?? "");
      });
    } catch (e) {
      showToast(
        copy?.refreshChatsFailed ?? `Could not refresh WhatsApp chats: ${errorMessage(e)}`,
        "error",
      );
    } finally {
      setRefreshingChats(false);
    }
  }, [copy?.refreshChatsFailed, routes, showToast]);

  const openAdd = () => {
    setDraftName("");
    setDraftChatId(unassignedChats[0]?.chat_id ?? "");
    setDraftProfile(profileNames[0] ?? "");
    setFieldErrors({});
    setModal({ mode: "add" });
    void refreshSeenChats();
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
    const chatErr = validateWhatsAppRouteChatId(draftChatId);
    if (chatErr) errors.chat_id = chatErr;
    else if (
      modal?.mode === "add" &&
      seenChats.length > 0 &&
      !unassignedChats.some((chat) => chat.chat_id === draftChatId)
    ) {
      errors.chat_id =
        copy?.fieldChatEmpty ??
        "Choose a chat the bridge has already seen.";
    }
    const duplicate = routes.some(
      (row, index) =>
        row.chat_id.trim().toLowerCase() === draftChatId.trim().toLowerCase() &&
        !(modal?.mode === "edit" && index === modal.index),
    );
    if (!errors.chat_id && duplicate) {
      errors.chat_id = "That chat already has a route.";
    }
    const seen = seenById.get(draftChatId);
    const label = whatsappChatLabel(draftChatId, seen?.name, seen?.kind);
    const taken = new Set(
      routes
        .filter((_, index) => !(modal?.mode === "edit" && index === modal.index))
        .map((row) => row.name),
    );
    const name =
      modal?.mode === "edit" && draftName.trim()
        ? draftName.trim()
        : whatsappRouteNameForChat(label, draftChatId, taken);
    const nameErr = validateWhatsAppRouteName(name);
    if (nameErr) errors.name = nameErr;
    const profileErr = validateWhatsAppRouteProfile(draftProfile, installedSet);
    if (profileErr) errors.profile = profileErr;
    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      return null;
    }
    const preserve = modal?.mode === "edit" ? modal.preserve : null;
    return buildWhatsAppRouteRow(
      { name, chat_id: draftChatId, profile: draftProfile },
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
          getConfig: (profile) => api.getConfig(profile, { includeDefaults: false }),
          saveConfig: (partial, profile) => api.saveConfig(partial, profile),
        },
        hostProfile,
        routes,
      );
      showToast(
        copy?.saved ??
          "Routes saved. They apply on the next inbound message.",
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
        <div className="flex items-center justify-between gap-3 rounded-sm border border-border bg-background px-3 py-2">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-medium text-foreground">
                {copy?.multiplexSwitchLabel ?? "Multiplexer"}
              </span>
              <span
                className={cn(
                  "rounded-sm border px-1.5 py-0.5 text-[11px] font-medium uppercase tracking-wide",
                  engaged
                    ? "border-primary bg-primary/15 text-foreground"
                    : "border-foreground/40 bg-background text-foreground",
                )}
              >
                {engaged
                  ? (copy?.multiplexSwitchOn ?? "On")
                  : (copy?.multiplexSwitchAvailable ?? "Available")}
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {engaged
                ? (copy?.multiplexOn ?? "Multiplex routing is on.")
                : (copy?.multiplexOff ??
                  "Available, and off. Turn it on to route this number's chats to other profiles.")}{" "}
              {(copy?.linkedTo ?? "Linked to")}{" "}
              {link.linked ? (
                <button
                  type="button"
                  className="font-courier text-foreground underline"
                  onClick={() => setProfile(link.manager)}
                >
                  {link.manager}
                </button>
              ) : (
                <span className="font-courier text-foreground">{link.manager}</span>
              )}
              .
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {engaging ? <Spinner className="text-sm" /> : null}
            <Switch
              checked={engaged}
              disabled={link.linked || engaging}
              onCheckedChange={(next) => void engageMultiplexer(next)}
              aria-label={
                link.linked
                  ? `${copy?.multiplexSwitchLabel ?? "Multiplexer"}, ${copy?.linkedTo ?? "linked to"} ${link.manager}`
                  : (copy?.multiplexSwitchLabel ?? "Multiplexer")
              }
            />
          </div>
        </div>
        {link.linked ? (
          <p className="text-xs text-muted-foreground">
            {(copy?.linkedProfileWhatsAppOff ??
              "This profile receives chats from that number, so its own WhatsApp stays off.")}{" "}
            <button
              type="button"
              className="text-foreground underline"
              onClick={() => setProfile(link.manager)}
            >
              {copy?.openManagingProfile ?? `Open ${link.manager}`}
            </button>
          </p>
        ) : (
          <p className="text-xs text-muted-foreground">
            {copy?.ownNumberHint ??
              "Each profile can still pair its own WhatsApp number. That number is a separate bridge on the same gateway, not a second multiplexer."}
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
                  <th className="py-2 pr-3 font-medium">{copy?.colChat ?? "Chat"}</th>
                  <th className="py-2 pr-3 font-medium">{copy?.colKind ?? "Kind"}</th>
                  <th className="py-2 pr-3 font-medium">{copy?.colProfile ?? "Profile"}</th>
                  <th className="py-2 pr-3 font-medium">{copy?.colLastSeen ?? "Last seen"}</th>
                  <th className="py-2 w-24" />
                </tr>
              </thead>
              <tbody>
                {routes.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-4 text-muted-foreground">
                      {copy?.empty ??
                        "No WhatsApp routes yet. Add a chat, or enter its id if the bridge has not listed it."}
                    </td>
                  </tr>
                ) : (
                  routes.map((row, index) => {
                    const seen = seenById.get(row.chat_id);
                    const kind = whatsappChatKind(row.chat_id, seen?.kind);
                    const label = whatsappChatLabel(row.chat_id, seen?.name, seen?.kind);
                    const lastSeen = formatWhatsAppLastSeen(seen?.last_seen);
                    return (
                    <tr key={`${row.chat_id}-${index}`} className="border-b border-border/60">
                      <td className="py-2 pr-3 font-medium" title={row.chat_id}>
                        {label}
                      </td>
                      <td className="py-2 pr-3">
                        {kind === "group" ? (copy?.kindGroup ?? "Group") : (copy?.kindPerson ?? "Person")}
                      </td>
                      <td className="py-2 pr-3 font-courier">{row.profile}</td>
                      <td className="py-2 pr-3 text-muted-foreground">{lastSeen || "—"}</td>
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
                    );
                  })
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
              {unassignedChats.length > 0
                ? `${copy?.addRoute ?? "Add route"} (${unassignedChats.length})`
                : (copy?.addRoute ?? "Add route")}
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
            <div className="grid gap-4 overflow-visible p-5">
              <div className="grid gap-1.5">
                <Label htmlFor="wa-route-chat">{copy?.fieldChat ?? "Chat"} *</Label>
                {modal.mode === "edit" ? (
                  <p className="text-sm" title={draftChatId}>
                    {whatsappChatLabel(
                      draftChatId,
                      seenById.get(draftChatId)?.name,
                      seenById.get(draftChatId)?.kind,
                    )}
                  </p>
                ) : seenChats.length === 0 ? (
                  <Input
                    id="wa-route-chat"
                    value={draftChatId}
                    placeholder={
                      copy?.fieldChatManual ??
                      "Group JID (…@g.us), phone number, or user JID"
                    }
                    onChange={(event) => {
                      setDraftChatId(event.target.value);
                      setFieldErrors((prev) => {
                        if (!prev.chat_id) return prev;
                        const next = { ...prev };
                        delete next.chat_id;
                        return next;
                      });
                    }}
                  />
                ) : unassignedChats.length === 0 ? (
                  <p id="wa-route-chat" className="text-xs text-muted-foreground">
                    {copy?.fieldChatEmpty ??
                      "No unassigned chats yet. After a new message arrives, use Refresh chats."}
                  </p>
                ) : (
                  <Select
                    id="wa-route-chat"
                    value={draftChatId}
                    onValueChange={(value) => {
                      setDraftChatId(value);
                      setFieldErrors((prev) => {
                        if (!prev.chat_id) return prev;
                        const next = { ...prev };
                        delete next.chat_id;
                        return next;
                      });
                    }}
                  >
                    {unassignedChats.map((chat) => (
                      <SelectOption key={chat.chat_id} value={chat.chat_id}>
                        {whatsappChatLabel(chat.chat_id, chat.name, chat.kind)}
                      </SelectOption>
                    ))}
                  </Select>
                )}
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
                      {formatProfileRouteOption(name, routeCounts.get(name) ?? 0)}
                    </SelectOption>
                  ))}
                </Select>
                <span className="text-xs text-muted-foreground">
                  {copy?.fieldProfileCountHint ??
                    "The number is how many WhatsApp chats already route to that profile."}
                </span>
                {fieldErrors.profile && (
                  <span className="text-xs text-destructive">{fieldErrors.profile}</span>
                )}
              </div>
              <div className="flex flex-col gap-2 pt-1">
                {modal.mode === "add" && (
                  <Button
                    outlined
                    size="sm"
                    className="w-full"
                    onClick={() => void refreshSeenChats()}
                    disabled={refreshingChats}
                    prefix={refreshingChats ? <Spinner /> : <RefreshCw className="h-4 w-4" />}
                  >
                    {refreshingChats
                      ? (copy?.refreshingChats ?? "Refreshing…")
                      : (copy?.refreshChats ?? "Refresh chats")}
                  </Button>
                )}
                <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                  <Button ghost size="sm" onClick={closeModal}>
                    {t.common.cancel}
                  </Button>
                  <Button
                    size="sm"
                    className="uppercase"
                    onClick={applyModal}
                    disabled={
                      modal.mode === "add" &&
                      seenChats.length > 0 &&
                      unassignedChats.length === 0
                    }
                  >
                    {copy?.modalApply ?? "Apply"}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
