import { useEffect, useState, useMemo } from "react";
import {
  Terminal,
  Search,
  Plus,
  RefreshCw,
  Trash2,
  Edit2,
  AlertCircle,
  Eye,
} from "lucide-react";
import { api, type CommandInfo } from "@/lib/api";
import { useProfileScope } from "@/contexts/useProfileScope";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { Card, CardContent, CardHeader } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { Label } from "@nous-research/ui/ui/components/label";
import { Input } from "@nous-research/ui/ui/components/input";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@nous-research/ui/ui/components/dialog";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";

export default function CommandsPage() {
  const [commands, setCommands] = useState<CommandInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [search, setSearch] = useState("");
  const [tab, setTab] = useState<"all" | "builtin" | "custom">("all");
  const { profile: selectedProfile } = useProfileScope();
  const { showToast } = useToast();
  useI18n();
  const { setEnd } = usePageHeader();

  // Dialog states
  const [dialogOpen, setDialogOpen] = useState(false);
  const [editingCommand, setEditingCommand] = useState<CommandInfo | null>(null);
  
  // Form states
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [type, setType] = useState<"exec" | "alias">("exec");
  const [commandText, setCommandText] = useState("");
  const [targetText, setTargetText] = useState("");
  const [enabled, setEnabled] = useState(true);
  const [silentEmpty, setSilentEmpty] = useState(false);
  const [visTelegram, setVisTelegram] = useState(true);
  const [visDiscord, setVisDiscord] = useState(true);
  const [visCli, setVisCli] = useState(true);
  const [formError, setFormError] = useState("");
  const [saving, setSaving] = useState(false);

  const fetchCommands = async () => {
    setLoading(true);
    try {
      const data = await api.getCommands(selectedProfile || undefined);
      setCommands(data);
    } catch (err: any) {
      showToast("Failed to load commands", "error");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCommands();
  }, [selectedProfile]);

  // Sync action trigger
  const handleSync = async () => {
    setSyncing(true);
    try {
      const res = await api.syncCommands(selectedProfile || undefined);
      if (res.ok) {
        showToast("Platform commands synchronized successfully", "success");
      } else {
        showToast(res.detail || "Command synchronization failed", "error");
      }
    } catch (err: any) {
      showToast(`Sync failed: ${err.message || err}`, "error");
    } finally {
      setSyncing(false);
    }
  };

  // Header buttons setup
  useEffect(() => {
    setEnd(
      <div className="flex items-center gap-2">
        <Button
          outlined
          size="sm"
          onClick={handleSync}
          disabled={syncing}
          className="flex items-center gap-1.5"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${syncing ? "animate-spin" : ""}`} />
          Sync to Platforms
        </Button>
        <Button
          size="sm"
          onClick={() => handleOpenCreate()}
          className="flex items-center gap-1.5"
        >
          <Plus className="h-3.5 w-3.5" />
          Create Command
        </Button>
      </div>
    );
    return () => setEnd(null);
  }, [syncing, selectedProfile]);

  const handleOpenCreate = () => {
    setEditingCommand(null);
    setName("");
    setDescription("");
    setType("exec");
    setCommandText("");
    setTargetText("");
    setEnabled(true);
    setSilentEmpty(false);
    setVisTelegram(true);
    setVisDiscord(true);
    setVisCli(true);
    setFormError("");
    setDialogOpen(true);
  };

  const handleOpenEdit = (cmd: CommandInfo) => {
    setEditingCommand(cmd);
    setName(cmd.name);
    setDescription(cmd.description || "");
    setType(cmd.type === "builtin" ? "exec" : cmd.type);
    setCommandText(cmd.type === "exec" ? cmd.command : "");
    setTargetText(cmd.type === "alias" ? cmd.command : "");
    setEnabled(cmd.enabled);
    setSilentEmpty(cmd.silent_empty || false);
    setVisTelegram(cmd.visible?.telegram !== false);
    setVisDiscord(cmd.visible?.discord !== false);
    setVisCli(cmd.visible?.cli !== false);
    setFormError("");
    setDialogOpen(true);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setFormError("Command name is required.");
      return;
    }
    if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
      setFormError("Name must contain only letters, numbers, hyphens, and underscores.");
      return;
    }
    if (type === "exec" && !commandText.trim()) {
      setFormError("Shell command is required for Exec type.");
      return;
    }
    if (type === "alias" && !targetText.trim()) {
      setFormError("Target command is required for Alias type.");
      return;
    }

    setSaving(true);
    setFormError("");

    const payload = {
      name: name.trim(),
      description: description.trim(),
      type,
      command: type === "exec" ? commandText.trim() : targetText.trim(),
      enabled,
      visible: {
        telegram: visTelegram,
        discord: visDiscord,
        cli: visCli,
      },
      silent_empty: type === "exec" ? silentEmpty : false,
    };

    try {
      await api.upsertCustomCommand(payload, selectedProfile || undefined);
      showToast(`Command /${name} saved successfully`, "success");
      setDialogOpen(false);
      fetchCommands();
    } catch (err: any) {
      setFormError(err.message || "Failed to save command");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (cmd: CommandInfo) => {
    if (!confirm(`Are you sure you want to delete custom command /${cmd.name}?`)) {
      return;
    }
    try {
      await api.deleteCustomCommand(cmd.name, selectedProfile || undefined);
      showToast(`Command /${cmd.name} deleted`, "success");
      fetchCommands();
    } catch (err: any) {
      showToast(`Failed to delete command: ${err.message || err}`, "error");
    }
  };

  const handleToggleBuiltin = async (cmd: CommandInfo, newState: boolean) => {
    try {
      await api.updateBuiltinCommand(
        cmd.name,
        { enabled: newState },
        selectedProfile || undefined
      );
      setCommands((prev) =>
        prev.map((c) => (c.name === cmd.name ? { ...c, enabled: newState } : c))
      );
      showToast(`Command /${cmd.name} ${newState ? "enabled" : "disabled"}`, "success");
    } catch (err: any) {
      showToast(`Failed to update command: ${err.message || err}`, "error");
    }
  };

  const handleToggleBuiltinVisibility = async (cmd: CommandInfo, platform: string, newState: boolean) => {
    try {
      const currentVisible = cmd.visible || { telegram: true, discord: true, cli: true };
      const updatedVisible = { ...currentVisible, [platform]: newState };
      await api.updateBuiltinCommand(
        cmd.name,
        { visible: updatedVisible },
        selectedProfile || undefined
      );
      setCommands((prev) =>
        prev.map((c) => (c.name === cmd.name ? { ...c, visible: updatedVisible } : c))
      );
      showToast(`Visibility for platform '${platform}' updated`, "success");
    } catch (err: any) {
      showToast(`Failed to update visibility: ${err.message || err}`, "error");
    }
  };

  const filteredCommands = useMemo(() => {
    return commands.filter((cmd) => {
      const matchSearch =
        cmd.name.toLowerCase().includes(search.toLowerCase()) ||
        (cmd.description || "").toLowerCase().includes(search.toLowerCase());
      
      const matchTab =
        tab === "all" ||
        (tab === "builtin" && cmd.source === "builtin") ||
        (tab === "custom" && cmd.source === "custom");

      return matchSearch && matchTab;
    });
  }, [commands, search, tab]);

  return (
    <div className="space-y-6">
      {/* Search & Filter bar */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between bg-card/35 backdrop-blur-md border border-border p-4 rounded-xl shadow-sm">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search commands by name or description..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9 bg-background/50 border-border"
          />
        </div>
        
        <div className="flex items-center gap-1.5 bg-background/40 border border-border p-1 rounded-lg">
          <Button
            ghost={tab !== "all"}
            size="sm"
            onClick={() => setTab("all")}
            className="text-xs"
          >
            All
          </Button>
          <Button
            ghost={tab !== "builtin"}
            size="sm"
            onClick={() => setTab("builtin")}
            className="text-xs"
          >
            Built-in
          </Button>
          <Button
            ghost={tab !== "custom"}
            size="sm"
            onClick={() => setTab("custom")}
            className="text-xs"
          >
            Custom
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="flex h-[300px] flex-col items-center justify-center gap-4">
          <Spinner className="text-2xl text-primary" />
          <p className="text-sm text-muted-foreground">Loading commands configurations...</p>
        </div>
      ) : filteredCommands.length === 0 ? (
        <Card className="border border-dashed border-border flex flex-col items-center justify-center p-12 bg-card/20 backdrop-blur-sm">
          <Terminal className="h-10 w-10 text-muted-foreground/60 mb-3" />
          <h3 className="font-semibold text-lg">No commands found</h3>
          <p className="text-sm text-muted-foreground mt-1 max-w-xs text-center">
            {search
              ? "No commands match your search criteria. Try a different query."
              : "No commands defined. Click 'Create Command' to register a custom command."}
          </p>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filteredCommands.map((cmd) => (
            <Card
              key={cmd.name}
              className={`relative overflow-hidden transition-all duration-200 border bg-card/30 backdrop-blur-md hover:bg-card/45 hover:shadow-md ${
                !cmd.enabled ? "opacity-60 border-destructive/20" : "border-border"
              }`}
            >
              <CardHeader className="pb-3 flex flex-row items-start justify-between space-y-0">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-base font-bold text-foreground">
                      /{cmd.name}
                    </span>
                    <Badge
                      tone={cmd.source === "builtin" ? "secondary" : "default"}
                      className="text-[10px] px-1.5 py-0"
                    >
                      {cmd.source === "builtin" ? "Built-in" : "Custom"}
                    </Badge>
                  </div>
                  {cmd.source === "custom" && (
                    <div className="text-[11px] text-muted-foreground font-mono">
                      type: <span className="text-foreground">{cmd.type}</span>
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  {cmd.source === "custom" ? (
                    <>
                      <Button
                        ghost
                        size="xs"
                        onClick={() => handleOpenEdit(cmd)}
                        className="p-1 hover:bg-foreground/5"
                      >
                        <Edit2 className="h-3.5 w-3.5 text-muted-foreground hover:text-foreground" />
                      </Button>
                      <Button
                        ghost
                        size="xs"
                        onClick={() => handleDelete(cmd)}
                        className="p-1 hover:bg-destructive/10"
                      >
                        <Trash2 className="h-3.5 w-3.5 text-destructive/80 hover:text-destructive" />
                      </Button>
                    </>
                  ) : (
                    <Switch
                      checked={cmd.enabled}
                      onCheckedChange={(checked) => handleToggleBuiltin(cmd, checked)}
                      className="scale-90"
                    />
                  )}
                </div>
              </CardHeader>

              <CardContent className="space-y-4">
                <p className="text-xs text-muted-foreground leading-relaxed min-h-[32px]">
                  {cmd.description || "No description provided."}
                </p>

                {cmd.source === "custom" && (
                  <div className="bg-background/40 border border-border/60 p-2 rounded font-mono text-[10px] break-all max-h-[80px] overflow-y-auto">
                    <span className="text-muted-foreground">
                      {cmd.type === "exec" ? "$ " : "-> "}
                    </span>
                    <span className="text-foreground">{cmd.command}</span>
                  </div>
                )}

                {/* Platform Visibility Row */}
                <div className="pt-2 border-t border-border/50 flex flex-col gap-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-muted-foreground font-medium flex items-center gap-1">
                      <Eye className="h-3 w-3" /> Visibility
                    </span>
                    <div className="flex items-center gap-3">
                      {["telegram", "discord", "cli"].map((p) => {
                        const isVisible = cmd.visible?.[p] !== false;
                        return (
                          <div key={p} className="flex items-center gap-1">
                            {cmd.source === "custom" ? (
                              <Badge
                                tone={isVisible ? "success" : "secondary"}
                                className="text-[9px] px-1 py-0 capitalize"
                              >
                                {p}
                              </Badge>
                            ) : (
                              <button
                                onClick={() =>
                                  handleToggleBuiltinVisibility(cmd, p, !isVisible)
                                }
                                className={`text-[10px] px-1.5 py-0.5 rounded capitalize border transition-all ${
                                  isVisible
                                    ? "bg-success/10 border-success/35 text-success font-medium"
                                    : "bg-muted/15 border-muted-foreground/20 text-muted-foreground"
                                }`}
                              >
                                {p}
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  
                  {cmd.source === "custom" && (
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-muted-foreground">Status</span>
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={cmd.enabled}
                          onCheckedChange={(checked) => {
                            const payload = {
                              name: cmd.name,
                              description: cmd.description,
                              type: cmd.type,
                              command: cmd.command,
                              enabled: checked,
                              visible: cmd.visible,
                              silent_empty: cmd.silent_empty,
                            };
                            api.upsertCustomCommand(payload, selectedProfile || undefined)
                              .then(() => {
                                setCommands((prev) =>
                                  prev.map((c) => (c.name === cmd.name ? { ...c, enabled: checked } : c))
                                );
                                showToast(`Command /${cmd.name} ${checked ? "enabled" : "disabled"}`, "success");
                              })
                              .catch((err) => showToast(`Failed to toggle command: ${err}`, "error"));
                          }}
                          className="scale-75"
                        />
                        <span className={cmd.enabled ? "text-success font-medium" : "text-muted-foreground"}>
                          {cmd.enabled ? "Enabled" : "Disabled"}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Editor Modal */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="max-w-md bg-card border border-border backdrop-blur-xl">
          <DialogHeader>
            <DialogTitle className="text-lg font-bold flex items-center gap-2">
              <Terminal className="h-5 w-5 text-primary" />
              {editingCommand ? "Edit Custom Command" : "Create Custom Command"}
            </DialogTitle>
            <DialogDescription className="text-xs">
              Define a custom slash command running a local shell script or redirecting to another command.
            </DialogDescription>
          </DialogHeader>

          <form onSubmit={handleSave} className="space-y-4 pt-2">
            <div className="grid gap-1">
              <Label htmlFor="cmd-name" className="text-xs font-semibold">
                Name
              </Label>
              <div className="relative">
                <span className="absolute left-3 top-2.5 text-xs text-muted-foreground font-mono">
                  /
                </span>
                <Input
                  id="cmd-name"
                  placeholder="e.g. status-check"
                  value={name}
                  onChange={(e) => setName(e.target.value.toLowerCase().replace(/\s+/g, ""))}
                  disabled={!!editingCommand}
                  className="pl-6 bg-background/40 border-border text-xs font-mono"
                />
              </div>
            </div>

            <div className="grid gap-1">
              <Label htmlFor="cmd-desc" className="text-xs font-semibold">
                Description
              </Label>
              <Input
                id="cmd-desc"
                placeholder="A short description of what this command does"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="bg-background/40 border-border text-xs"
              />
            </div>

            <div className="grid gap-1">
              <Label htmlFor="cmd-type" className="text-xs font-semibold">
                Type
              </Label>
              <Select
                id="cmd-type"
                value={type}
                onValueChange={(val) => setType(val as "exec" | "alias")}
              >
                <SelectOption value="exec">Exec (Shell Subprocess)</SelectOption>
                <SelectOption value="alias">Alias (Redirect command)</SelectOption>
              </Select>
            </div>

            {type === "exec" ? (
              <div className="space-y-2">
                <div className="grid gap-1">
                  <Label htmlFor="cmd-shell" className="text-xs font-semibold">
                    Shell Command
                  </Label>
                  <textarea
                    id="cmd-shell"
                    placeholder="e.g. echo 'CPU Usage:' && top -b -n 1 | grep Cpu"
                    value={commandText}
                    onChange={(e) => setCommandText(e.target.value)}
                    className="min-h-[80px] w-full border border-border bg-background/30 rounded-lg p-2.5 font-mono text-[11px] leading-relaxed shadow-sm focus:outline-none focus:ring-1 focus:ring-primary"
                  />
                </div>
                <div className="flex items-center justify-between text-xs pt-1">
                  <div className="flex flex-col gap-0.5">
                    <span className="font-medium text-foreground">Silent on Empty Output</span>
                    <span className="text-[10px] text-muted-foreground">If the command returns no output, don't send any reply.</span>
                  </div>
                  <Switch
                    checked={silentEmpty}
                    onCheckedChange={setSilentEmpty}
                    className="scale-75"
                  />
                </div>
              </div>
            ) : (
              <div className="grid gap-1">
                <Label htmlFor="cmd-target" className="text-xs font-semibold">
                  Target Command
                </Label>
                <div className="relative">
                  <span className="absolute left-3 top-2.5 text-xs text-muted-foreground font-mono">
                    /
                  </span>
                  <Input
                    id="cmd-target"
                    placeholder="e.g. status"
                    value={targetText}
                    onChange={(e) => setTargetText(e.target.value)}
                    className="pl-6 bg-background/40 border-border text-xs font-mono"
                  />
                </div>
              </div>
            )}

            {/* Platform checkboxes */}
            <div className="pt-2 border-t border-border/40">
              <Label className="text-xs font-semibold mb-2 block">Platform Visibility</Label>
              <div className="grid grid-cols-3 gap-2">
                <label className="flex items-center gap-2 p-2 rounded-lg border border-border bg-background/20 cursor-pointer select-none hover:bg-background/40">
                  <input
                    type="checkbox"
                    checked={visTelegram}
                    onChange={(e) => setVisTelegram(e.target.checked)}
                    className="rounded border-border text-primary focus:ring-0 focus:ring-offset-0"
                  />
                  <span className="text-xs capitalize">Telegram</span>
                </label>
                <label className="flex items-center gap-2 p-2 rounded-lg border border-border bg-background/20 cursor-pointer select-none hover:bg-background/40">
                  <input
                    type="checkbox"
                    checked={visDiscord}
                    onChange={(e) => setVisDiscord(e.target.checked)}
                    className="rounded border-border text-primary focus:ring-0 focus:ring-offset-0"
                  />
                  <span className="text-xs capitalize">Discord</span>
                </label>
                <label className="flex items-center gap-2 p-2 rounded-lg border border-border bg-background/20 cursor-pointer select-none hover:bg-background/40">
                  <input
                    type="checkbox"
                    checked={visCli}
                    onChange={(e) => setVisCli(e.target.checked)}
                    className="rounded border-border text-primary focus:ring-0 focus:ring-offset-0"
                  />
                  <span className="text-xs capitalize">CLI</span>
                </label>
              </div>
            </div>

            <div className="flex items-center gap-2 pt-2">
              <Switch
                checked={enabled}
                onCheckedChange={setEnabled}
                className="scale-75"
              />
              <span className="text-xs text-muted-foreground">Enabled</span>
            </div>

            {formError && (
              <div className="flex items-center gap-1.5 p-2 rounded bg-destructive/10 border border-destructive/25 text-destructive text-xs">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <span>{formError}</span>
              </div>
            )}

            <div className="flex justify-end gap-2 pt-2">
              <Button
                outlined
                size="sm"
                onClick={() => setDialogOpen(false)}
                type="button"
              >
                Cancel
              </Button>
              <Button
                size="sm"
                type="submit"
                disabled={saving}
              >
                {saving ? "Saving..." : "Save Command"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}
