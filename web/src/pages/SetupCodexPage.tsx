import { useEffect, useMemo, useState, type ComponentType, type ReactNode } from "react";
import {
  Bot,
  CheckCircle2,
  Clipboard,
  KeyRound,
  MessageCircle,
  Route,
  Settings2,
  ShieldCheck,
  Sparkles,
  Terminal,
  Users,
  Wrench,
} from "lucide-react";
import { api, type SetupCodexState } from "@/lib/api";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";

type TopicId =
  | "new-agent"
  | "telegram"
  | "feishu"
  | "model"
  | "toolsets"
  | "subagents"
  | "gateway"
  | "security";

type Topic = {
  id: TopicId;
  title: string;
  subtitle: string;
  icon: ComponentType<{ className?: string }>;
  tone: string;
  steps: string[];
  commands: string[];
};

const AGENT_TYPES = [
  { id: "personal", label: "个人助理", preset: "safe,skills,session_search" },
  { id: "coding", label: "代码助手", preset: "safe,terminal,file,web,skills" },
  { id: "group", label: "群聊 Bot", preset: "safe,skills,session_search" },
  { id: "research", label: "研究助手", preset: "safe,web,search,skills,session_search" },
];
const PLATFORM_CHOICES = ["cli", "telegram", "feishu", "api_server"];
const PROFILE_RE = /^[a-z][a-z0-9-]{1,31}$/;

const TOPICS: Topic[] = [
  {
    id: "new-agent",
    title: "新建 Agent",
    subtitle: "用 profile 创建一个独立身份、独立配置的 Hermes Agent。",
    icon: Bot,
    tone: "from-cyan-500/20 to-blue-500/10",
    steps: [
      "先决定用途：个人助理、代码助手、群聊 Bot、定时任务 Bot 或研究助手。",
      "给 profile 起一个小写英文名，例如 research-bot；不要把 token 或隐私放进名字。",
      "选择 clone 策略：小白建议先 --clone，继承当前可用模型与工具。",
      "进入新 profile 后检查模型、工具和入口平台，再做第一次 hello 测试。",
    ],
    commands: [
      "hermes profile create research-bot --clone",
      "hermes -p research-bot model",
      "hermes -p research-bot tools",
      "hermes -p research-bot chat -q \"hello\"",
      "hermes profile list",
    ],
  },
  {
    id: "telegram",
    title: "连接 Telegram",
    subtitle: "检查 Bot token、配对状态和 Gateway 运行状态。",
    icon: MessageCircle,
    tone: "from-sky-500/20 to-cyan-500/10",
    steps: [
      "在 BotFather 创建 bot 并拿到 token；不要在群聊里粘贴 token。",
      "把 token 存到 .env，然后重启 gateway。",
      "第一次私聊 bot 时使用 pairing flow 授权，而不是开放所有用户。",
    ],
    commands: [
      "hermes config env-path",
      "hermes gateway status",
      "hermes pairing list",
      "hermes pairing approve telegram <code>",
    ],
  },
  {
    id: "feishu",
    title: "连接 Feishu / Lark",
    subtitle: "检查 App 凭据、机器人事件订阅和 Gateway 状态。",
    icon: Route,
    tone: "from-indigo-500/20 to-violet-500/10",
    steps: [
      "确认飞书开放平台应用已创建，并启用机器人能力。",
      "凭据只存 .env；页面只显示是否配置，不显示 secret。",
      "确认事件订阅与权限范围后，再在飞书里 @bot 测试。",
    ],
    commands: [
      "hermes config env-path",
      "hermes gateway status",
      "hermes status --all",
    ],
  },
  {
    id: "model",
    title: "选择模型",
    subtitle: "主模型负责正常对话，auxiliary 模型负责视觉、压缩、搜索等辅助任务。",
    icon: Sparkles,
    tone: "from-amber-500/20 to-orange-500/10",
    steps: [
      "先确认主 provider/model 是否可用。",
      "需要长上下文或低成本时，再分别配置 auxiliary 任务。",
      "OAuth provider 出问题时优先用 hermes auth list / reset 排查。",
    ],
    commands: ["hermes model", "hermes auth list", "hermes status --all"],
  },
  {
    id: "toolsets",
    title: "配置工具权限",
    subtitle: "toolsets 是 Agent 的行动权限包；平台级 toolsets 可以给 Telegram/Feishu 不同权限。",
    icon: Wrench,
    tone: "from-emerald-500/20 to-green-500/10",
    steps: [
      "小白默认从 safe / skills / session_search 开始。",
      "terminal、messaging、cronjob、homeassistant 属于高风险工具，要按平台最小授权。",
      "修改工具权限后，新 session 才会加载新的 tool schema。",
    ],
    commands: ["hermes tools list", "hermes tools", "hermes config path"],
  },
  {
    id: "subagents",
    title: "开启 Subagents",
    subtitle: "delegate_task 适合短任务并行；长任务应该用 cron 或 background terminal。",
    icon: Users,
    tone: "from-fuchsia-500/20 to-pink-500/10",
    steps: [
      "默认 leaf subagent 不能继续委派，也不能 clarify/memory/send_message/execute_code。",
      "只有 max_spawn_depth > 1 且 orchestrator_enabled=true 时，orchestrator 才有意义。",
      "并发和深度会指数放大成本；先从 max_concurrent_children=2~3 开始。",
    ],
    commands: ["hermes config path", "hermes status --all"],
  },
  {
    id: "gateway",
    title: "排查 Gateway",
    subtitle: "确认 gateway 进程、平台连接状态、日志和重启方式。",
    icon: Terminal,
    tone: "from-slate-500/20 to-zinc-500/10",
    steps: [
      "先看 status，再看 gateway.log；不要盲目 --replace。",
      "systemd 管理时让 systemd 负责生命周期，ExecStart 不要带 --replace。",
      "本页 MVP 不会自动重启 gateway，只给复制命令。",
    ],
    commands: ["hermes gateway status", "hermes logs --level warning", "hermes doctor"],
  },
  {
    id: "security",
    title: "安全检查",
    subtitle: "确认 secret 不进聊天、不进页面、不进代码；高风险配置必须二次确认。",
    icon: ShieldCheck,
    tone: "from-red-500/20 to-rose-500/10",
    steps: [
      "API key、token、OAuth、private key 只看存在性，不显示原文。",
      "allowlist、platform_toolsets、terminal/messaging/cronjob 权限属于高风险变更。",
      "后续若做一键 apply，必须先展示 diff，再要求确认短语。",
    ],
    commands: ["hermes doctor", "hermes config check", "hermes tools list"],
  },
];

function yesNo(value?: boolean): string {
  return value ? "已配置" : "未配置";
}

function platformLabel(id: string): string {
  if (id === "api_server") return "API Server";
  if (id === "feishu") return "Feishu";
  return id.charAt(0).toUpperCase() + id.slice(1);
}

function CopyButton({ text, onCopied }: { text: string; onCopied: () => void }) {
  return (
    <Button
      ghost
      size="xs"
      className="shrink-0"
      onClick={async () => {
        await navigator.clipboard.writeText(text);
        onCopied();
      }}
    >
      <Clipboard className="h-3.5 w-3.5" />
      Copy
    </Button>
  );
}

function CommandBlock({ commands, onCopied }: { commands: string[]; onCopied: () => void }) {
  return (
    <div className="space-y-2">
      {commands.map((cmd) => (
        <div
          key={cmd}
          className="flex items-center justify-between gap-3 rounded-lg border bg-muted/40 px-3 py-2 font-mono text-xs"
        >
          <span className="overflow-x-auto whitespace-nowrap pr-2">{cmd}</span>
          <CopyButton text={cmd} onCopied={onCopied} />
        </div>
      ))}
    </div>
  );
}

function FieldLabel({ children }: { children: string }) {
  return <label className="text-xs font-medium text-muted-foreground">{children}</label>;
}

function NativeSelect({ value, onChange, children }: { value: string; onChange: (value: string) => void; children: ReactNode }) {
  return (
    <select
      className="h-9 rounded-md border bg-background px-3 text-sm"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {children}
    </select>
  );
}

export default function SetupCodexPage() {
  const [state, setState] = useState<SetupCodexState | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeId, setActiveId] = useState<TopicId>("new-agent");
  const [profileName, setProfileName] = useState("research-bot");
  const [agentType, setAgentType] = useState("research");
  const [entryPlatform, setEntryPlatform] = useState("telegram");
  const { toast, showToast } = useToast();

  useEffect(() => {
    api
      .getSetupCodexState()
      .then(setState)
      .catch(() => showToast("配置宝典状态加载失败", "error"))
      .finally(() => setLoading(false));
  }, []);

  const activeTopic = useMemo(
    () => TOPICS.find((topic) => topic.id === activeId) ?? TOPICS[0],
    [activeId],
  );
  const selectedAgentType = AGENT_TYPES.find((item) => item.id === agentType) ?? AGENT_TYPES[0];
  const profileValid = PROFILE_RE.test(profileName);
  const generatedAgentCommands = [
    `hermes profile create ${profileName || "<profile-name>"} --clone`,
    `hermes -p ${profileName || "<profile-name>"} model`,
    `hermes -p ${profileName || "<profile-name>"} config set agent.enabled_toolsets ${selectedAgentType.preset}`,
    entryPlatform === "cli"
      ? `hermes -p ${profileName || "<profile-name>"} chat -q "hello"`
      : `hermes -p ${profileName || "<profile-name>"} gateway status`,
    entryPlatform === "telegram"
      ? `hermes -p ${profileName || "<profile-name>"} pairing list`
      : `hermes -p ${profileName || "<profile-name>"} status --all`,
  ];
  const ActiveIcon = activeTopic.icon;

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 md:p-6">
      <div className="overflow-hidden rounded-3xl border bg-gradient-to-br from-background via-background to-muted/60 p-6 shadow-sm">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div className="space-y-4">
            <div className="inline-flex items-center gap-2 rounded-full border bg-background/70 px-3 py-1 text-xs text-muted-foreground">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              Setup Codex · read-only MVP
            </div>
            <div>
              <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">Hermes配置宝典</h1>
              <p className="mt-2 max-w-2xl text-sm text-muted-foreground md:text-base">
                官方新手村 + 配置驾驶舱。先读状态、解释风险、生成可复制命令；当前版本不会写 config、不会写 .env、不会执行命令、不会重启 Gateway。
              </p>
            </div>
          </div>
          <div className="grid gap-2 rounded-2xl border bg-background/70 p-4 text-sm sm:grid-cols-2 lg:min-w-[420px]">
            <div>
              <div className="text-xs text-muted-foreground">Model</div>
              <div className="font-medium">{state?.model.provider || "auto"} / {state?.model.model || "未设置"}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Gateway</div>
              <div className="font-medium">{state?.gateway.running ? "running" : state?.gateway.state || "stopped"}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Config</div>
              <div className="truncate font-mono text-xs">{state?.paths.config_path}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Secrets</div>
              <div className="font-medium">{state?.secrets.env_secret_keys_configured_count ?? 0} keys · values redacted</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {TOPICS.map((topic) => {
          const Icon = topic.icon;
          const active = topic.id === activeId;
          return (
            <button
              key={topic.id}
              className={`rounded-2xl border bg-gradient-to-br ${topic.tone} p-4 text-left transition hover:-translate-y-0.5 hover:shadow-md ${
                active ? "ring-2 ring-primary/50" : ""
              }`}
              onClick={() => setActiveId(topic.id)}
            >
              <Icon className="mb-3 h-5 w-5 text-primary" />
              <div className="font-medium">{topic.title}</div>
              <div className="mt-1 text-xs text-muted-foreground">{topic.subtitle}</div>
            </button>
          );
        })}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.35fr_0.85fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ActiveIcon className="h-5 w-5 text-primary" />
              {activeTopic.title}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <ol className="space-y-3">
              {activeTopic.steps.map((step, index) => (
                <li key={step} className="flex gap-3 rounded-xl border bg-muted/20 p-3 text-sm">
                  <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
                    {index + 1}
                  </span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
            <div>
              <div className="mb-2 flex items-center gap-2 text-sm font-medium">
                <Clipboard className="h-4 w-4" />
                复制命令，手动执行
              </div>
              <CommandBlock commands={activeTopic.commands} onCopied={() => showToast("已复制命令", "success")} />
            </div>
            {activeId === "new-agent" && (
              <div className="rounded-2xl border bg-muted/20 p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <div className="text-sm font-medium">新建 Agent 向导（只生成命令）</div>
                    <div className="text-xs text-muted-foreground">调整用途、profile 名称和入口平台；本页不会创建 profile，也不会写配置。</div>
                  </div>
                  <Badge tone={profileValid ? "success" : "destructive"}>{profileValid ? "name ok" : "name invalid"}</Badge>
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                  <div className="space-y-1.5">
                    <FieldLabel>Agent 类型</FieldLabel>
                    <NativeSelect value={agentType} onChange={setAgentType}>
                      {AGENT_TYPES.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
                    </NativeSelect>
                  </div>
                  <div className="space-y-1.5">
                    <FieldLabel>Profile 名称</FieldLabel>
                    <Input value={profileName} onChange={(event) => setProfileName(event.target.value)} placeholder="research-bot" />
                  </div>
                  <div className="space-y-1.5">
                    <FieldLabel>入口平台</FieldLabel>
                    <NativeSelect value={entryPlatform} onChange={setEntryPlatform}>
                      {PLATFORM_CHOICES.map((item) => <option key={item} value={item}>{item}</option>)}
                    </NativeSelect>
                  </div>
                </div>
                <div className="mt-3 rounded-xl border bg-background/60 p-3 text-xs text-muted-foreground">
                  建议权限模板：<span className="font-mono text-foreground">{selectedAgentType.preset}</span>。如果包含 terminal/messaging/cronjob，请先确认平台级 toolsets 和 owner allowlist。
                </div>
                <div className="mt-3">
                  <CommandBlock commands={generatedAgentCommands} onCopied={() => showToast("已复制向导命令", "success")} />
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                当前状态快照
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {state && (
                <>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Provider</span>
                    <Badge tone="secondary">{state.model.provider || "auto"}</Badge>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Model</span>
                    <span className="font-mono text-xs">{state.model.model || "未设置"}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Delegation</span>
                    <span>{state.delegation.configured ? "已配置" : "默认继承"}</span>
                  </div>
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-muted-foreground">Platform toolsets</span>
                    <span>{Object.keys(state.toolsets.platform_toolsets).length} platforms</span>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <KeyRound className="h-4 w-4 text-primary" />
                平台连接
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              {state && Object.entries(state.platforms).map(([id, platform]) => (
                <div key={id} className="rounded-xl border p-3">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{platformLabel(id)}</span>
                    <Badge tone={platform.configured ? "success" : "secondary"}>{yesNo(platform.configured)}</Badge>
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    credential: {yesNo(platform.credential_configured)} · runtime: {platform.runtime_state || "unknown"}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Settings2 className="h-4 w-4 text-primary" />
                安全边界
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <div className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-primary" /> 不返回 secret 原文</div>
              <div className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-primary" /> 不写 config.yaml / .env</div>
              <div className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-primary" /> 不执行 shell 命令</div>
              <div className="flex items-center gap-2"><ShieldCheck className="h-4 w-4 text-primary" /> 不自动重启 Gateway</div>
            </CardContent>
          </Card>
        </div>
      </div>
      <Toast toast={toast} />
    </div>
  );
}
