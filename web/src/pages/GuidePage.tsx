import { useEffect } from "react";
import {
  Activity,
  BarChart3,
  Bot,
  BookOpen,
  Clock,
  Cpu,
  FileText,
  KeyRound,
  MessageSquare,
  Package,
  Puzzle,
  Settings,
  Users,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { usePageHeader } from "@/contexts/usePageHeader";

type Tone = "success" | "warning" | "secondary" | "destructive" | "outline";

const navRows = [
  ["Sessions", "看历史会话和某次对话细节", "高", "success"],
  ["Analytics", "看整体 token、成本、模型使用趋势", "中", "secondary"],
  ["Models", "看模型池和模型分析", "中", "secondary"],
  ["Agents", "管理 Agent、模型、fallback、Console", "最高", "success"],
  ["Delegations", "看飞书背后的多 Agent 委托链路", "最高", "success"],
  ["Logs", "查报错、查没分工、查连接问题", "高", "success"],
  ["Cron", "管理定时任务", "中", "secondary"],
  ["Skills", "查看/启停技能", "中", "secondary"],
  ["Plugins", "插件页", "低", "outline"],
  ["Profiles", "多 profile 管理", "低到中", "outline"],
  ["Config", "系统配置", "谨慎", "destructive"],
  ["Keys", "API Key / 环境变量", "谨慎", "destructive"],
  ["Documentation", "文档入口", "低", "outline"],
] as const;

const agentRows = [
  ["Hermes 技术翻译官", "技术中间层、需求拆解、策略判断", "可以", "success"],
  ["DeepSeek 低成本快工", "小改、小测、日志初筛、轻量查证", "可以", "success"],
  ["Intelligence 情报研究员", "调研、竞品、资料核验", "可以", "success"],
  ["Pirlo 商业策划师", "商业方案、PPT、内容结构", "可以", "success"],
  ["TARS 桌面操作员", "桌面、浏览器、截图、视觉验证", "可以", "success"],
  ["Ambrosini 质量门卫", "高风险验收、最终质量门", "可以", "success"],
  ["Claude 主程执行官", "外部 Claude Code CLI / CC Switch", "只展示", "warning"],
  ["Codex 代码审查官", "外部 Codex CLI", "只展示", "warning"],
] as const;

const IconMap = {
  Sessions: MessageSquare,
  Analytics: BarChart3,
  Models: Cpu,
  Agents: Bot,
  Delegations: Activity,
  Logs: FileText,
  Cron: Clock,
  Skills: Package,
  Plugins: Puzzle,
  Profiles: Users,
  Config: Settings,
  Keys: KeyRound,
  Documentation: BookOpen,
};

function GuideTable({
  headers,
  rows,
}: {
  headers: string[];
  rows: readonly (readonly string[])[];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-xs text-muted-foreground">
            {headers.map((header) => (
              <th key={header} className="py-2 pr-4 text-left font-medium">
                {header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row[0]}-${index}`} className="border-b border-border/50 last:border-0">
              {row.map((cell, cellIndex) => (
                <td key={`${cell}-${cellIndex}`} className="py-2 pr-4 align-top">
                  {cellIndex === row.length - 1 && ["高", "中", "低", "低到中", "最高", "谨慎", "可以", "只展示"].includes(cell) ? (
                    <Badge tone={(row[cellIndex + 1] as Tone) || "secondary"}>{cell}</Badge>
                  ) : cellIndex === row.length - 1 && ["success", "warning", "secondary", "destructive", "outline"].includes(cell) ? null : (
                    <span className={cellIndex === 0 ? "font-medium" : "text-muted-foreground"}>{cell}</span>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StepList({ steps }: { steps: string[] }) {
  return (
    <ol className="space-y-2 text-sm">
      {steps.map((step, index) => (
        <li key={step} className="flex gap-2">
          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center border border-border bg-secondary/40 font-mono text-[10px] text-muted-foreground">
            {index + 1}
          </span>
          <span className="text-muted-foreground">{step}</span>
        </li>
      ))}
    </ol>
  );
}

function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="overflow-x-auto border border-border bg-background/70 p-3 font-mono text-xs text-muted-foreground">
      <code>{children}</code>
    </pre>
  );
}

function GuideCard({
  id,
  title,
  icon: Icon,
  children,
}: {
  id: string;
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  children: React.ReactNode;
}) {
  return (
    <Card id={id} className="scroll-mt-20">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted-foreground" />
          <CardTitle className="text-base">{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  );
}

export default function GuidePage({ embedded = false }: { embedded?: boolean }) {
  const { setTitle } = usePageHeader();

  useEffect(() => {
    if (embedded) return undefined;
    setTitle("Guide");
    return () => setTitle(null);
  }, [embedded, setTitle]);

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="space-y-4 py-5">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="max-w-3xl space-y-2">
              <div className="flex items-center gap-2">
                <Badge tone="secondary">9119</Badge>
                <Badge tone="outline">Operator Manual</Badge>
              </div>
              <h1 className="font-display text-2xl font-semibold tracking-tight">
                Hermes 9119 后台操作手册
              </h1>
              <p className="text-sm text-muted-foreground">
                飞书马尔蒂尼是日常入口。9119 是管理、观察和诊断后台：调模型看 Agents，分工看 Delegations，报错看 Logs，历史看 Sessions。
              </p>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
              {["Agents", "Delegations", "Logs", "Sessions"].map((item) => (
                <a key={item} href={`#${item.toLowerCase()}`} className="border border-border bg-secondary/20 px-3 py-2 hover:bg-secondary/40">
                  {item}
                </a>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[16rem_minmax(0,1fr)]">
        <Card className="h-fit xl:sticky xl:top-4">
          <CardHeader>
            <CardTitle className="text-sm">左侧菜单速查</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-1 text-sm">
            {Object.entries(IconMap).map(([name, Icon]) => (
              <a key={name} href={`#${name.toLowerCase()}`} className="flex items-center gap-2 px-2 py-1.5 text-muted-foreground hover:bg-secondary/30 hover:text-foreground">
                <Icon className="h-3.5 w-3.5" />
                {name}
              </a>
            ))}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <GuideCard id="overview" title="左侧导航总览" icon={BookOpen}>
            <GuideTable headers={["菜单", "一句话用途", "优先级", ""]} rows={navRows} />
          </GuideCard>

          <GuideCard id="sessions" title="Sessions：看历史会话" icon={MessageSquare}>
            <p className="text-sm text-muted-foreground">Sessions 用来倒查某次对话发生了什么。不要把它当主聊天入口，也不要随便删除历史会话。</p>
            <StepList steps={["进入 Sessions。", "找到最近的会话。", "点进去查看消息记录。", "如果怀疑没有分工，再去 Delegations 查同一时间段。"]} />
          </GuideCard>

          <GuideCard id="analytics" title="Analytics：看整体用量" icon={BarChart3}>
            <GuideTable
              headers={["现象", "可能含义"]}
              rows={[
                ["某模型调用突然升高", "某个 Agent 被频繁使用，或 fallback 经常触发"],
                ["output token 很高", "任务输出过长，可能需要限制总结长度"],
                ["skill 使用异常", "某类任务被频繁触发，可能需要优化路由"],
              ]}
            />
          </GuideCard>

          <GuideCard id="models" title="Models：模型池视角" icon={Cpu}>
            <p className="text-sm text-muted-foreground">Models 偏模型，不偏 Agent。如果要给某个 Agent 换模型，优先去 Agents。</p>
            <GuideTable headers={["页面", "看问题的角度"]} rows={[["Models", "这个模型本身怎么样、用了多少"], ["Agents", "这个 Agent 现在用哪个模型、fallback 怎么配"]]} />
          </GuideCard>

          <GuideCard id="agents" title="Agents：最重要的管理页" icon={Bot}>
            <GuideTable headers={["Agent", "定位", "是否可改", ""]} rows={agentRows} />
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="space-y-3">
                <h3 className="text-sm font-semibold">改模型操作</h3>
                <StepList steps={["进入 Agents。", "找到目标 Agent。", "在 Model / Strategy 区域选择模型。", "选择 fixed 或 fallback。", "点击保存。新配置只影响之后的新任务。"]} />
              </div>
              <div className="space-y-3">
                <h3 className="text-sm font-semibold">fixed / fallback</h3>
                <CodeBlock>{`mode = fixed\nprimary = deepseek_pro\n\nmode = fallback\nchain = deepseek_pro → opencode_go_deepseek_pro → deepseek_flash`}</CodeBlock>
              </div>
            </div>
            <GuideTable
              headers={["fallback_on", "含义", "建议"]}
              rows={[
                ["quota_exceeded", "额度不足", "建议勾选"],
                ["rate_limited", "限流", "建议勾选"],
                ["timeout", "超时", "建议勾选"],
                ["server_error", "服务端错误", "建议勾选"],
                ["empty_final_content", "模型空返回", "建议勾选"],
              ]}
            />
            <div className="border border-warning/40 bg-warning/10 p-3 text-sm text-muted-foreground">
              Console 是单 Agent 直聊/调试，不是飞书主入口。Enter 发送，Shift + Enter 换行。
            </div>
          </GuideCard>

          <GuideCard id="delegations" title="Delegations：看分工过程" icon={Activity}>
            <GuideTable headers={["字段", "含义"]} rows={[["agent_id", "被派活的 Agent"], ["status", "started / completed / failed / interrupted"], ["duration", "跑了多久"], ["api_calls / tokens", "模型调用和消耗"], ["fallback", "是否发生模型切换"], ["summary", "子 Agent 返回摘要"]]} />
            <StepList steps={["进入 Delegations。", "看最近时间段有没有记录。", "有记录说明有分工。", "没有记录就去 Logs 搜 delegate_task / subagent。", "Logs 也没有，大概率 Coordinator 自己做了。"]} />
          </GuideCard>

          <GuideCard id="logs" title="Logs：查问题" icon={FileText}>
            <CodeBlock>{`delegate_task\nsubagent\nagent_id\nmodel_ref\nfallback\ncooldown\nerror\nAPI_SERVER_KEY\nSQLite\nKanbanBridge\nBad credentials`}</CodeBlock>
            <GuideTable headers={["问题", "搜什么"]} rows={[["飞书没回复", "error / gateway / timeout"], ["没有分工", "delegate_task / subagent"], ["模型切换异常", "fallback / model_ref / cooldown"], ["API key 问题", "Bad credentials / 401 / API_SERVER_KEY"], ["数据库问题", "SQLite / events.db"]]} />
          </GuideCard>

          <GuideCard id="cron" title="Cron：定时任务" icon={Clock}>
            <GuideTable headers={["操作", "说明"]} rows={[["Create", "创建定时任务"], ["Pause", "暂停任务"], ["Resume", "恢复任务"], ["Trigger", "立刻运行一次"], ["Delete", "删除任务，谨慎"]]} />
          </GuideCard>

          <GuideCard id="skills" title="Skills：技能管理" icon={Package}>
            <p className="text-sm text-muted-foreground">Skills 是系统知道“怎么做某类事”的技能库，例如 html-anything、anysearch-lite、codebase-inspection、github-code-review。</p>
            <StepList steps={["进入 Skills。", "搜关键词，比如 html、search、github。", "看是否启用。", "不确定时不要乱关，因为可能影响 Agent 行为。"]} />
          </GuideCard>

          <GuideCard id="plugins" title="Plugins：插件" icon={Puzzle}>
            <p className="text-sm text-muted-foreground">
              Plugins 是 9119 的扩展机制。插件可以新增自己的页面，也可以把小组件插入现有页面的固定位置。
            </p>
            <GuideTable
              headers={["位置", "说明"]}
              rows={[
                ["独立插件页", "插件可以在左侧导航新增自己的 tab，例如某个外部工具的控制台。"],
                ["页面插槽", "插件也可以插入现有页面，例如 Sessions 顶部、Logs 底部、Docs 顶部或底部。"],
                ["docs:top / docs:bottom", "Documentation 页面里的插件区域就是预留插槽，给插件放文档卡片、帮助入口或诊断小组件。"],
              ]}
            />
            <div className="border border-border bg-secondary/20 p-3 text-sm text-muted-foreground">
              你日常不需要主动操作这些插槽。它们是给系统扩展用的：有插件接入时会自动出现，没有插件时就是空的。
            </div>
          </GuideCard>

          <GuideCard id="profiles" title="Profiles：Profile 管理" icon={Users}>
            <p className="text-sm text-muted-foreground">Profiles 是不同运行身份/配置档管理。现在已经收敛成单入口，所以日常不常用。不确定自己在切换 profile 时，不要随便改。</p>
          </GuideCard>

          <GuideCard id="config" title="Config：系统配置" icon={Settings}>
            <div className="border border-destructive/40 bg-destructive/10 p-3 text-sm text-muted-foreground">
              高风险页面。不要随便改 API server 监听地址、API_SERVER_KEY、gateway、memory、delegation 深度和并发、外部 CLI runtime。
            </div>
          </GuideCard>

          <GuideCard id="keys" title="Keys：API Key / 环境变量" icon={KeyRound}>
            <GuideTable headers={["操作", "说明"]} rows={[["is_set", "看 key 是否存在"], ["Update", "替换 key"], ["Delete", "删除 key，慎用"], ["Reveal", "查看明文，注意安全"]]} />
            <p className="text-sm text-muted-foreground">不要截图发给别人，不要把完整 key 写进聊天。替换 key 后要做最小真实调用验证。</p>
          </GuideCard>

          <GuideCard id="documentation" title="Documentation：文档入口" icon={BookOpen}>
            <p className="text-sm text-muted-foreground">
              Documentation 是文档中心，不是一篇混合文档。它把本地 9119 操作手册和 Hermes 官方文档分成两个独立入口。
            </p>
            <GuideTable
              headers={["区域", "用途"]}
              rows={[
                ["9119 操作手册", "面向你的系统实操说明，解释 9119 每个菜单怎么用。"],
                ["Hermes 官方文档", "Hermes Agent 的通用产品文档，适合查上游功能。"],
                ["插件区域", "docs:top 和 docs:bottom 是插件扩展位，有插件时会自动显示对应内容。"],
              ]}
            />
          </GuideCard>
        </div>
      </div>
    </div>
  );
}
