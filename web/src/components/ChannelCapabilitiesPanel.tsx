import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import { AlertTriangle, Network, ShieldCheck } from "lucide-react";

import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Switch } from "@nous-research/ui/ui/components/switch";

import { useI18n } from "@/i18n";
import type { Translations } from "@/i18n";
import { api, type ChannelCapability, type ChannelMcpPolicy } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  profile?: string;
  query: string;
  onError: (message: string) => void;
  onSaved: (message: string) => void;
}

interface SaveRequest {
  generation: number;
  profile: string;
}

const HIGH_IMPACT = new Set([
  "terminal",
  "file",
  "code_execution",
  "computer_use",
  "delegation",
  "cronjob",
]);

const CHANNEL_COPY = {
  abilitiesEnabled: "Enabled abilities",
  channelCapabilitiesDescription:
    "Choose the exact toolsets and MCP access available to messages from this channel.",
  channelCapabilitiesFailed: "Failed to save channel abilities.",
  channelCapabilitiesSaved: "Channel abilities saved.",
  channels: "Channel abilities",
  changesNewSessions: "Changes apply to new Agent sessions.",
  customBoundary: "Custom boundary",
  highImpact: "High impact",
  inheritedDefaults: "Inherited defaults",
  mcpAccess: "MCP access",
  mcpAll: "All enabled MCP servers",
  mcpNone: "No MCP servers",
  mcpSelected: "Selected MCP servers",
  noMcpServers: "No enabled MCP servers are available.",
  requiredAbilities: "Required channel abilities",
  requiredAbilitiesDescription:
    "These platform-native abilities are always available on this channel.",
  saveCapabilities: "Save abilities",
  savingCapabilities: "Saving…",
} as const;

type ChannelCopyKey = keyof typeof CHANNEL_COPY;

function channelText(skills: Translations["skills"], key: ChannelCopyKey): string {
  return skills[key] ?? CHANNEL_COPY[key];
}

export function ChannelCapabilitiesPanel({ profile, query, onError, onSaved }: Props) {
  const { t } = useI18n();
  const loadFailed = channelText(t.skills, "channelCapabilitiesFailed");
  const profileKey = profile ?? "";
  const [channels, setChannels] = useState<ChannelCapability[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [loadedProfile, setLoadedProfile] = useState<string | null>(null);
  const [savingRequest, setSavingRequest] = useState<SaveRequest | null>(null);
  const [toolsets, setToolsets] = useState<Set<string>>(new Set());
  const [mcpMode, setMcpMode] = useState<ChannelMcpPolicy["mode"]>("all");
  const [mcpServers, setMcpServers] = useState<Set<string>>(new Set());
  const saveGeneration = useRef(0);
  const activeProfileRef = useRef(profileKey);
  const saving = savingRequest?.profile === profileKey;

  useLayoutEffect(() => {
    // Invalidate save completions before the new profile can receive input.
    // The request token lets an obsolete completion clean up only itself.
    saveGeneration.current += 1;
    activeProfileRef.current = profileKey;
  }, [profileKey]);

  useEffect(() => {
    let cancelled = false;
    api
      .getChannelCapabilities(profile)
      .then((rows) => {
        if (cancelled) return;
        const first = rows[0];
        setChannels(rows);
        setSelected(first?.platform ?? null);
        setToolsets(
          new Set(first?.toolsets.filter((row) => row.enabled).map((row) => row.name)),
        );
        setMcpMode(first?.mcp.mode ?? "all");
        setMcpServers(new Set(first?.mcp.selected));
        setLoadedProfile(profileKey);
      })
      .catch(() => {
        if (cancelled) return;
        setChannels([]);
        setSelected(null);
        setToolsets(new Set());
        setMcpMode("all");
        setMcpServers(new Set());
        setLoadedProfile(profileKey);
        onError(loadFailed);
      });
    return () => {
      cancelled = true;
    };
  }, [loadFailed, onError, profile, profileKey]);

  const channel = useMemo(
    () =>
      loadedProfile === profileKey
        ? channels.find((row) => row.platform === selected) ?? null
        : null,
    [channels, loadedProfile, profileKey, selected],
  );

  const selectChannel = (nextChannel: ChannelCapability) => {
    setSelected(nextChannel.platform);
    setToolsets(
      new Set(nextChannel.toolsets.filter((row) => row.enabled).map((row) => row.name)),
    );
    setMcpMode(nextChannel.mcp.mode);
    setMcpServers(new Set(nextChannel.mcp.selected));
  };

  const visibleChannels = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return channels;
    return channels.filter(
      (row) =>
        row.label.toLowerCase().includes(needle) ||
        row.platform.toLowerCase().includes(needle) ||
        row.toolsets.some(
          (item) =>
            item.label.toLowerCase().includes(needle) ||
            item.name.toLowerCase().includes(needle) ||
            item.tools.some((tool) => tool.toLowerCase().includes(needle)),
        ),
    );
  }, [channels, query]);

  const updateSet = (
    setter: Dispatch<SetStateAction<Set<string>>>,
    name: string,
    enabled: boolean,
  ) =>
    setter((current) => {
      const next = new Set(current);
      if (enabled) next.add(name);
      else next.delete(name);
      return next;
    });

  const save = async () => {
    if (!channel || saving) return;
    const requestProfile = profileKey;
    const request = {
      generation: ++saveGeneration.current,
      profile: requestProfile,
    };
    setSavingRequest(request);
    const isCurrentRequest = () =>
      saveGeneration.current === request.generation &&
      activeProfileRef.current === requestProfile;
    try {
      const result = await api.updateChannelCapabilities(
        channel.platform,
        {
          toolsets: [...toolsets].sort(),
          mcp_mode: mcpMode,
          mcp_servers: mcpMode === "allowlist" ? [...mcpServers].sort() : [],
        },
        profile,
      );
      if (!isCurrentRequest()) return;
      setChannels((rows) =>
        rows.map((row) =>
          row.platform === result.channel.platform ? result.channel : row,
        ),
      );
      onSaved(channelText(t.skills, "channelCapabilitiesSaved"));
    } catch {
      if (isCurrentRequest()) onError(loadFailed);
    } finally {
      setSavingRequest((current) => (current === request ? null : current));
    }
  };

  if (loadedProfile !== profileKey) {
    return (
      <Card className="rounded-none">
        <CardContent className="py-12 text-center text-sm text-muted-foreground">
          {t.common.loading}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid min-h-[560px] gap-3 lg:grid-cols-[240px_minmax(0,1fr)]">
      <Card className="rounded-none">
        <CardHeader className="px-3 py-3">
          <CardTitle className="flex items-center gap-2 text-sm">
            <Network className="h-4 w-4" />
            {channelText(t.skills, "channels")}
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-1 px-2 pb-3">
          {visibleChannels.map((row) => (
            <button
              key={row.platform}
              type="button"
              className={cn(
                "flex w-full items-center gap-2 px-2 py-2 text-left text-sm",
                selected === row.platform
                  ? "bg-muted text-foreground"
                  : "text-muted-foreground hover:bg-muted/50 hover:text-foreground",
                "disabled:cursor-not-allowed disabled:opacity-60",
              )}
              disabled={saving}
              onClick={() => selectChannel(row)}
            >
              <span className="min-w-0 flex-1 truncate">{row.label}</span>
            </button>
          ))}
        </CardContent>
      </Card>

      {channel && (
        <Card className="rounded-none">
          <CardHeader className="px-4 py-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <CardTitle className="flex items-center gap-2 text-base">
                  <ShieldCheck className="h-4 w-4" />
                  {channel.label}
                </CardTitle>
                <p className="mt-1 text-xs text-muted-foreground">
                  {channelText(t.skills, "channelCapabilitiesDescription")}
                </p>
              </div>
              <Badge tone="secondary">
                {channel.explicit
                  ? channelText(t.skills, "customBoundary")
                  : channelText(t.skills, "inheritedDefaults")}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="grid gap-6 px-4 pb-5">
            <section>
              <div className="mb-2 flex items-center justify-between">
                <h3 className="text-sm font-medium">{channelText(t.skills, "abilitiesEnabled")}</h3>
                <span className="text-xs text-muted-foreground">
                  {toolsets.size}/{channel.toolsets.length}
                </span>
              </div>
              <div className="grid gap-x-5 gap-y-1 sm:grid-cols-2">
                {channel.toolsets.map((item) => (
                  <div key={item.name} className="flex items-start gap-3 py-2.5">
                    <Switch
                      aria-label={item.label}
                      checked={toolsets.has(item.name)}
                      disabled={saving || (toolsets.size === 1 && toolsets.has(item.name))}
                      onCheckedChange={(checked) =>
                        updateSet(setToolsets, item.name, checked)
                      }
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-sm font-medium">{item.label}</span>
                        {HIGH_IMPACT.has(item.name) && (
                          <Badge tone="warning" className="text-[10px]">
                            <AlertTriangle className="mr-1 h-3 w-3" />
                            {channelText(t.skills, "highImpact")}
                          </Badge>
                        )}
                      </div>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {item.description}
                      </p>
                      {item.tools.length > 0 && (
                        <p className="mt-1 truncate font-mono text-[10px] text-text-tertiary">
                          {item.tools.join(", ")}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            {channel.implicit_toolsets.length > 0 && (
              <section>
                <h3 className="mb-2 text-sm font-medium">
                  {channelText(t.skills, "requiredAbilities")}
                </h3>
                <p className="mb-2 text-xs text-muted-foreground">
                  {channelText(t.skills, "requiredAbilitiesDescription")}
                </p>
                <div className="flex flex-wrap gap-2">
                  {channel.implicit_toolsets.map((item) => (
                    <Badge key={item.name} tone="secondary">
                      {item.label}
                    </Badge>
                  ))}
                </div>
              </section>
            )}

            <section>
              <h3 className="mb-2 text-sm font-medium">{channelText(t.skills, "mcpAccess")}</h3>
              <div className="flex flex-wrap gap-2">
                {(["all", "none", "allowlist"] as const).map((mode) => (
                    <Button
                      disabled={saving}
                    key={mode}
                    size="sm"
                    outlined={mcpMode !== mode}
                    onClick={() => setMcpMode(mode)}
                  >
                    {mode === "all"
                      ? channelText(t.skills, "mcpAll")
                      : mode === "none"
                        ? channelText(t.skills, "mcpNone")
                        : channelText(t.skills, "mcpSelected")}
                  </Button>
                ))}
              </div>
              {mcpMode === "allowlist" && (
                <div className="mt-3 grid gap-2 sm:grid-cols-2">
                  {channel.mcp.available.length === 0 ? (
                    <p className="text-xs text-muted-foreground">
                      {channelText(t.skills, "noMcpServers")}
                    </p>
                  ) : (
                    channel.mcp.available.map((server) => (
                      <label key={server} className="flex items-center gap-2 text-sm">
                        <Switch
                          aria-label={server}
                          checked={mcpServers.has(server)}
                          disabled={saving}
                          onCheckedChange={(checked) =>
                            updateSet(setMcpServers, server, checked)
                          }
                        />
                        <span className="font-mono text-xs">{server}</span>
                      </label>
                    ))
                  )}
                </div>
              )}
            </section>

            <div className="flex items-center justify-between gap-3 border-t border-border pt-4">
              <p className="text-xs text-muted-foreground">
                {channelText(t.skills, "changesNewSessions")}
              </p>
              <Button onClick={() => void save()} disabled={saving}>
                {saving
                  ? channelText(t.skills, "savingCapabilities")
                  : channelText(t.skills, "saveCapabilities")}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
