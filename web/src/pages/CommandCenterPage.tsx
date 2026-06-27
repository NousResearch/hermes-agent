import { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  Clock,
  Database,
  GitBranch,
  MessageSquare,
  Radio,
  RefreshCw,
  Terminal,
  Workflow,
  Zap,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { api, fetchJSON } from "@/lib/api";
import type {
  CommandCenterProcess,
  CommandCenterSnapshot,
  CronJob,
  PaginatedSessions,
  SessionInfo,
  SessionStoreStats,
  StatusResponse,
} from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";

type ProfileSessionInfo = SessionInfo & { profile?: string | null };

type AttentionItem = {
  label: string;
  detail: string;
  tone: "success" | "warning" | "destructive" | "secondary";
};

type CronBucket = {
  key: string;
  label: string;
  description: string;
  jobs: CronJob[];
};

function shortText(value: string | null | undefined, fallback: string, max = 120): string {
  const text = (value ?? "").replace(/\s+/g, " ").trim();
  if (!text) return fallback;
  return text.length > max ? `${text.slice(0, max - 3)}...` : text;
}

function compactModel(model: string | null): string {
  if (!model) return "モデル未確認";
  return model.split("/").pop() ?? model;
}

type CronDisplay = {
  title: string;
  description: string;
};

type AutomationTeam = {
  title: string;
  tag: string;
  description: string;
  tone: "success" | "warning" | "outline" | "secondary";
};

type AutomationTeamSummary = AutomationTeam & {
  key: string;
  jobs: CronJob[];
  running: number;
  nextRun: string | null | undefined;
};

type AutomationGuardrail = {
  label: string;
  detail: string;
  tone: "success" | "warning" | "outline" | "secondary";
};

type AutomationStatus = {
  label: string;
  detail: string;
  tone: "success" | "warning" | "outline" | "secondary" | "destructive";
};

type ProcessRole = {
  label: string;
  description: string;
  tone: "success" | "warning" | "outline" | "secondary";
  countsAsAgent: boolean;
};

function rawCronName(job: CronJob): string {
  return job.name || job.id || "Scheduled job";
}

function cronSearchText(job: CronJob): string {
  return [job.name, job.script, job.prompt, ...(job.skills ?? [])]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function cronDisplay(job: CronJob): CronDisplay {
  const text = cronSearchText(job);
  const rules: Array<[string[], CronDisplay]> = [
    [["screenpipe_pipeline_watchdog", "screenpipe pipeline health watchdog"], {
      title: "Screenpipe録画パイプライン監視",
      description: "画面ログ・録画・OCR取り込みが止まっていないかを短い間隔で確認する。",
    }],
    [["screenpipe_export_to_vault", "screenpipe -> obsidian"], {
      title: "Screenpipe→Obsidian書き出し",
      description: "録画/OCRログをObsidian側の読める日次ログへ反映する。",
    }],
    [["hermes_health_watchdog", "hermes health watchdog"], {
      title: "Hermes稼働チェック",
      description: "Hermes / Gateway / Slack連携が沈黙していないか確認する。健康なら基本silent。",
    }],
    [["morning executive briefing"], {
      title: "朝のCEOブリーフ",
      description: "今日の最初の3手、DJ優先事項、承認待ち、危険な再掲を朝にまとめる。",
    }],
    [["daily_todo_carryover", "daily todo carryover"], {
      title: "ToDo持ち越し整理",
      description: "ObsidianのToDo正典を日次で更新し、未完了タスクを次の日へ引き継ぐ。",
    }],
    [["generate_todo_next_actions", "daily todo next actions"], {
      title: "今日の次アクション抽出",
      description: "ToDoから今日動かす候補を抽出し、実行順に並べる。",
    }],
    [["generate_todo_top3", "tomorrow's top3", "tomorrows top3"], {
      title: "明日のTop3作成",
      description: "夜に翌日の重要3件を作る。Slackに出す数少ない定例通知。",
    }],
    [["sync_notion_to_vault", "notion -> obsidian"], {
      title: "Notion→Obsidian同期",
      description: "Notion側の情報をObsidian vaultへ取り込み、情報源を寄せる。",
    }],
    [["pip-audit", "run_pip_audit"], {
      title: "Python依存CVEチェック",
      description: "Python依存パッケージの脆弱性を月次で確認する。",
    }],
    [["vault-wiki-pending-enricher"], {
      title: "Vault Wiki未整理ノート補強",
      description: "未整理のwiki候補を補強し、後で知識化しやすくする。",
    }],
    [["stagnation_detector"], {
      title: "停滞検知",
      description: "古く残ったタスクや進んでいない論点を検出する。訂正台帳の禁止再掲を守る。",
    }],
    [["inbox_digest"], {
      title: "Inbox週次ダイジェスト",
      description: "Obsidian inboxの未整理情報を週次でまとめ、判断しやすくする。",
    }],
    [["freee_weekly_cashflow", "freee-weekly-cashflow"], {
      title: "freee週次キャッシュフロー確認",
      description: "freeeを読み取り照合し、Rino判断が必要な資金・会計論点だけを出す。更新はしない。",
    }],
    [["monthly_pl_report", "monthly-pl-report"], {
      title: "月次PLレポート",
      description: "月初にPL確認用のローカルレポートを作る。",
    }],
    [["regulatory_intelligence", "regulatory-intelligence"], {
      title: "規制インテリジェンス",
      description: "DJ/関連事業に影響しそうな規制・行政情報を週次で確認する。",
    }],
    [["freee_audit", "freee-audit"], {
      title: "freee読み取り監査",
      description: "freeeの読み取り監査。登録・紐付け・更新は承認なしに実行しない。",
    }],
    [["identity context freshness"], {
      title: "Rino文脈鮮度チェック",
      description: "identity/current-focus系の古さを定期確認し、古い前提を残さない。",
    }],
    [["ops_dashboard_snapshot", "ops dashboard snapshot"], {
      title: "Ops Dashboardスナップショット",
      description: "Hermes/cron/process/Obsidian鮮度をローカルに集計し、Command Centerの材料にする。",
    }],
    [["daily note v2"], {
      title: "Daily Note v2正本生成",
      description: "日次ログのAI用データ層を生成する。人間向けビューの元データ。",
    }],
    [["self-dialogue"], {
      title: "Rino自己対話ループ",
      description: "夜の振り返り・学習・翌日の仕込みをproposal中心で回す。",
    }],
    [["meeting-dossier"], {
      title: "MTG事前資料準備",
      description: "予定されている会議の前提・相手・論点を事前にまとめる。",
    }],
    [["post-meeting-action"], {
      title: "MTG後アクション抽出",
      description: "会議後の会話やメモから次アクションを抽出する。",
    }],
    [["research-digest", "global-standards"], {
      title: "研究・標準情報ダイジェスト",
      description: "研究/標準/技術情報を収集し、必要なものだけ後で読める形にする。",
    }],
    [["fieldy"], {
      title: "Fieldy→Obsidian取り込み",
      description: "Fieldy会話ログをObsidianへ取り込む。現在の有効/停止状態はcron詳細で確認。",
    }],
    [["la_flora", "la-flora"], {
      title: "La Flora機会ウォッチ",
      description: "La Floraは商機・規制変化だけ軽く監視。書類作成タスクは復活させない。",
    }],
  ];

  for (const [needles, display] of rules) {
    if (needles.some((needle) => text.includes(needle))) return display;
  }

  const raw = rawCronName(job);
  if (job.script) {
    return {
      title: shortText(raw, "自動化ジョブ", 56),
      description: `スクリプト実行: ${job.script}`,
    };
  }
  if (job.skills?.length) {
    return {
      title: shortText(raw, "AI自動化ジョブ", 56),
      description: `AIスキル実行: ${job.skills.join(", ")}`,
    };
  }
  return {
    title: shortText(raw, "AI自動化ジョブ", 56),
    description: shortText(job.prompt, "AIが定期実行する自動化。詳細はCron画面で確認。", 110),
  };
}

function cronLabel(job: CronJob): string {
  return cronDisplay(job).title;
}

function cronPurpose(job: CronJob): string {
  return cronDisplay(job).description;
}

function automationTeam(job: CronJob): AutomationTeam {
  const text = cronSearchText(job);
  if (["watchdog", "health", "ops_dashboard", "screenpipe_pipeline"].some((needle) => text.includes(needle))) {
    return {
      title: "監視クローン",
      tag: "watch",
      description: "止まると困る入口・基盤・鮮度を見張る。健康なら基本silent。",
      tone: "success",
    };
  }
  if (["research", "regulatory", "global-standards", "intelligence"].some((needle) => text.includes(needle))) {
    return {
      title: "調査クローン",
      tag: "research",
      description: "研究・規制・外部変化を拾い、必要なものだけ後で読める形にする。",
      tone: "outline",
    };
  }
  if (["freee", "cashflow", "pl_report", "monthly-pl", "revenue", "amazon", "vertosa", "dispensary japan"].some((needle) => text.includes(needle))) {
    return {
      title: "DJ事業クローン",
      tag: "dj",
      description: "売上・会計・事業判断の材料を作る。更新や送信は承認待ちへ回す。",
      tone: "warning",
    };
  }
  if (["la_flora", "la-flora"].some((needle) => text.includes(needle))) {
    return {
      title: "La Flora機会クローン",
      tag: "laflora",
      description: "機会・規制変化だけ軽く拾う。不要になった書類作成は復活させない。",
      tone: "outline",
    };
  }
  if (["todo", "daily note", "notion", "obsidian", "vault", "wiki", "fieldy", "inbox"].some((needle) => text.includes(needle))) {
    return {
      title: "整理クローン",
      tag: "organize",
      description: "Obsidian/ログ/ToDoを整え、次の判断に使える形へ寄せる。",
      tone: "secondary",
    };
  }
  if (["briefing", "top3", "self-dialogue", "reflection", "stagnation", "delight"].some((needle) => text.includes(needle))) {
    return {
      title: "思考クローン",
      tag: "think",
      description: "振り返り・仮説・明日の一手をproposal中心で作る。",
      tone: "secondary",
    };
  }
  return {
    title: "汎用AIクローン",
    tag: "auto",
    description: "定期実行されるAI/スクリプト。詳細はCron画面で確認。",
    tone: "outline",
  };
}

function automationGuardrail(job: CronJob): AutomationGuardrail {
  const text = cronSearchText(job);
  const deliver = (job.deliver || "").toLowerCase();
  if (text.includes("proposal") || text.includes("no-auto-promote")) {
    return {
      label: "proposal-only",
      detail: "提案作成まで。canon/cron/外部更新は承認待ち。",
      tone: "success",
    };
  }
  if (job.script) {
    return {
      label: "script-first",
      detail: "決定的スクリプト中心。成功時silent設計が望ましい。",
      tone: "secondary",
    };
  }
  if (deliver.includes("slack")) {
    return {
      label: "Slack visible",
      detail: "Slack通知あり。頻度・重複は監査対象。",
      tone: "warning",
    };
  }
  return {
    label: "approval-gated",
    detail: "詳細アクションは各ページへ。ここから直接実行しない。",
    tone: "outline",
  };
}

function automationStatus(job: CronJob): AutomationStatus {
  if (job.last_error) {
    return {
      label: "壊れている可能性",
      detail: shortText(job.last_error, "last_error", 140),
      tone: "destructive",
    };
  }
  if (job.state === "running") {
    return {
      label: "実行中",
      detail: "いま走っている自動化。",
      tone: "success",
    };
  }
  if (!job.last_run_at) {
    return {
      label: "未実行/待機",
      detail: "まだ実行履歴なし、または履歴未取得。",
      tone: "warning",
    };
  }
  return {
    label: "正常待機",
    detail: `last ${formatNextRun(job.last_run_at)}`,
    tone: "success",
  };
}

function teamStatus(team: AutomationTeamSummary): AutomationStatus {
  const broken = team.jobs.find((job) => job.last_error);
  if (broken) return automationStatus(broken);
  if (team.running > 0) {
    return { label: "実行中あり", detail: `${team.running}件が実行中`, tone: "success" };
  }
  const neverRun = team.jobs.filter((job) => !job.last_run_at).length;
  if (neverRun > 0) {
    return { label: "一部未実行", detail: `${neverRun}件はまだ実行履歴がない`, tone: "warning" };
  }
  return { label: "正常待機", detail: "有効な自動化に直近エラーはない", tone: "success" };
}

function formatNextRun(value: string | null | undefined): string {
  if (!value) return "次回未定";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString("ja-JP", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusLabel(status: StatusResponse | null): string {
  if (!status) return "確認中";
  if (status.gateway_state === "startup_failed") return "要確認";
  if (status.gateway_running) return "稼働中";
  return "停止中";
}

function isRecentSession(session: SessionInfo): boolean {
  return Date.now() / 1000 - session.last_active < 30 * 60;
}

type HumanWorkDisplay = {
  target: string;
  need: string;
  current: string;
};

function textForWork(...values: Array<string | null | undefined>): string {
  return values.filter(Boolean).join(" ").toLowerCase();
}

function humanTargetFromText(text: string): string {
  if (text.includes("command center") || text.includes("dashboard") || text.includes("ダッシュボード")) {
    return "Hermes Agent repo / Command Center";
  }
  if (text.includes("freee") || text.includes("cashflow") || text.includes("pl")) return "DJ / La Flora 会計・資金管理";
  if (text.includes("screenpipe") || text.includes("ocr")) return "Rino OS / Screenpipe観測基盤";
  if (text.includes("obsidian") || text.includes("vault") || text.includes("daily note")) return "Obsidian vault / Rino OS";
  if (text.includes("meeting") || text.includes("会議")) return "Rino OS / MTG準備・フォロー";
  if (text.includes("todo") || text.includes("top3")) return "Rino OS / ToDo・実行順";
  if (text.includes("la flora") || text.includes("laflora")) return "La Flora / 機会・会計ウォッチ";
  return "Hermes / 現在の作業";
}

function humanNeedFromText(text: string): string {
  if (text.includes("command center") || text.includes("dashboard") || text.includes("ダッシュボード")) {
    return "作戦ボードで、対象・必要性・現在作業が人間に伝わる必要がある。";
  }
  if (text.includes("freee") || text.includes("cashflow") || text.includes("pl")) return "会計・資金判断の材料を、更新なしで確認する必要がある。";
  if (text.includes("screenpipe") || text.includes("ocr")) return "画面ログ・OCR・日次ログの観測基盤が止まっていないか確認する必要がある。";
  if (text.includes("meeting") || text.includes("会議")) return "会議前後の判断材料と次アクションを取りこぼさないようにする必要がある。";
  if (text.includes("todo") || text.includes("top3")) return "今日/明日の実行順を整理し、次の一手に落とす必要がある。";
  return "進行中の会話・自動化を整理し、次に判断できる状態にする必要がある。";
}

function humanCurrentFromText(text: string, fallback: string): string {
  if (text.includes("command center") || text.includes("dashboard") || text.includes("ダッシュボード")) {
    return "Dashboardの表示を、人間が読む作業メモとして整えている。";
  }
  return shortText(fallback, "現在作業は未登録。詳細を開いて確認。", 150);
}

function humanWorkFromSession(session: ProfileSessionInfo): HumanWorkDisplay {
  const text = textForWork(session.title, session.preview, session.source);
  return {
    target: humanTargetFromText(text),
    need: humanNeedFromText(text),
    current: humanCurrentFromText(text, session.preview || session.title || session.id),
  };
}

function humanWorkFromProcess(process: CommandCenterProcess): HumanWorkDisplay {
  const text = textForWork(process.label, process.kind, process.mission, process.command);
  return {
    target: humanTargetFromText(text),
    need: humanNeedFromText(text),
    current: humanCurrentFromText(text, process.mission || process.label || process.command),
  };
}

function snapshotSourceLabel(source: string | null | undefined): string {
  if (source === "session") return "セッション由来";
  if (source === "agent_registry") return "作業登録由来";
  if (source === "recent_run") return "直近完了由来";
  if (source === "snapshot") return "スナップショット由来";
  return "スナップショット";
}

function waitingTypeLabel(type: string | null | undefined): string {
  if (type === "APPROVE") return "承認待ち";
  if (type === "CHOOSE") return "選択待ち";
  if (type === "REVIEW") return "確認待ち";
  if (type === "INFORM") return "情報";
  if (type === "SAFE_AUTO") return "安全側";
  return "確認";
}

function scheduleExpr(job: CronJob): string {
  return job.schedule_display || job.schedule?.display || job.schedule?.expr || "手動/未確認";
}

function cronBucketKey(job: CronJob): string {
  const expr = scheduleExpr(job);
  const parts = expr.split(/\s+/);
  if (expr.includes("*/") || expr.startsWith("every ")) return "high";
  if (parts.length === 5) {
    const [minute, hour, dayOfMonth, , dayOfWeek] = parts;
    if (dayOfMonth !== "*") return "monthly";
    if (dayOfWeek !== "*") return "weekly";
    const h = Number(hour.split(",")[0]);
    if (Number.isFinite(h)) {
      if (h >= 5 && h <= 10) return "morning";
      if (h >= 11 && h <= 18) return "day";
      if (h >= 19 || h <= 2) return "night";
    }
    if (minute.includes("/")) return "high";
  }
  return "other";
}

function buildCronBuckets(jobs: CronJob[]): CronBucket[] {
  const buckets: CronBucket[] = [
    { key: "high", label: "常時 / 高頻度", description: "30分〜毎時など。鮮度維持・監視系。", jobs: [] },
    { key: "morning", label: "朝", description: "朝ブリーフ、健康確認、今日の次アクション。", jobs: [] },
    { key: "day", label: "昼", description: "事業ブロック中の確認・監査・週次レビュー。", jobs: [] },
    { key: "night", label: "夜", description: "Top3、日報、学習・振り返り。", jobs: [] },
    { key: "weekly", label: "週次", description: "承認ダイジェスト、規制・財務・棚卸し。", jobs: [] },
    { key: "monthly", label: "月次 / 四半期", description: "月次PL、CVE、鮮度監査。", jobs: [] },
    { key: "other", label: "その他", description: "頻度未分類・手動に近いもの。", jobs: [] },
  ];
  const byKey = new Map(buckets.map((b) => [b.key, b]));
  for (const job of jobs) {
    byKey.get(cronBucketKey(job))?.jobs.push(job);
  }
  for (const bucket of buckets) {
    bucket.jobs.sort((a, b) => (a.next_run_at || "").localeCompare(b.next_run_at || ""));
  }
  return buckets.filter((bucket) => bucket.jobs.length > 0);
}

function buildAttentionQueue(status: StatusResponse | null, jobs: CronJob[], sessions: ProfileSessionInfo[]): AttentionItem[] {
  const items: AttentionItem[] = [];

  if (status?.gateway_state === "startup_failed") {
    items.push({
      label: "要確認 Gateway起動",
      detail: status.gateway_exit_reason || "Gatewayの起動に失敗した。",
      tone: "destructive",
    });
  }

  if (status) {
    for (const [name, platform] of Object.entries(status.gateway_platforms ?? {})) {
      if (platform.state === "fatal" || platform.state === "disconnected") {
        items.push({
          label: `要確認 ${name}`,
          detail: platform.error_message || `連携状態: ${platform.state}`,
          tone: platform.state === "fatal" ? "destructive" : "warning",
        });
      }
    }
  }

  for (const job of jobs) {
    if (job.enabled && job.last_error) {
      items.push({
        label: `要確認 ${cronLabel(job)}`,
        detail: shortText(job.last_error, "直近エラー", 180),
        tone: "warning",
      });
    }
  }

  const activeSessions = sessions.filter((session) => session.is_active).length;
  if (activeSessions > 0) {
    items.push({
      label: "情報 セッション稼働中",
      detail: `${activeSessions}件の会話/作業セッションが開いている。必要なら下のセッションカードから再開できる。`,
      tone: "secondary",
    });
  }

  if (items.length === 0) {
    items.push({
      label: "安全側 異常なし",
      detail: "Gateway・外部連携・実行中cronのエラーは見つからない。",
      tone: "success",
    });
  }

  return items.slice(0, 6);
}

function StatCard({ label, value, detail, href, icon: Icon, tone }: {
  label: string;
  value: string;
  detail: string;
  href: string;
  icon: typeof Activity;
  tone: string;
}) {
  return (
    <Link to={href} className="group border border-border bg-background-base/40 p-4 transition-colors hover:border-primary/50 hover:bg-primary/[0.04]">
      <div className="mb-4 flex items-center justify-between gap-3">
        <span className={`flex h-10 w-10 items-center justify-center rounded-full border border-current/25 ${tone}`}>
          <Icon className="h-5 w-5" />
        </span>
        <RefreshCw className="h-3.5 w-3.5 text-text-tertiary opacity-0 transition-opacity group-hover:opacity-100" />
      </div>
      <p className="text-3xl font-semibold tabular-nums text-foreground">{value}</p>
      <p className="text-sm font-medium text-text-secondary">{label}</p>
      <p className="mt-1 text-xs text-text-tertiary">{detail}</p>
    </Link>
  );
}

function processRole(process: CommandCenterProcess): ProcessRole {
  const kind = process.kind;
  const command = process.command.toLowerCase();
  if (kind === "hermes_cli") {
    return {
      label: "主エージェント",
      description: "Rinoと直接会話している実行主体。エージェント数に含める。",
      tone: "success",
      countsAsAgent: true,
    };
  }
  if (kind === "cron_worker") {
    return {
      label: "実行中クローン",
      description: "スケジュールで実行中の自動化。エージェント数に含める。",
      tone: "warning",
      countsAsAgent: true,
    };
  }
  if (kind === "claude") {
    return {
      label: "外部エージェント",
      description: "別AI coding agent。エージェント数に含める。",
      tone: "outline",
      countsAsAgent: true,
    };
  }
  if (kind === "codex") {
    const supportNeedles = [
      "mcp-server",
      "node_repl",
      "skycomputeruseclient",
      "skycomputeruseservice",
      "codex computer use.app",
      "gpu-process",
      "utility",
      "--type=renderer",
      "codex (renderer)",
      "bare-modifier-monitor",
      "app-server",
      "extension-host",
      "chrome-extension://",
      "sparkle",
      "autoupdate",
      "updater.app",
      "/helpers/codex",
      "antigravity/out/mcp-server",
    ];
    const isSupport = supportNeedles.some((needle) => command.includes(needle));
    return {
      label: isSupport ? "補助プロセス" : "外部エージェント",
      description: isSupport
        ? "Codex/Computer Use の補助プロセス。稼働数には見えるが、エージェント数からは除外。"
        : "別AI coding agent。エージェント数に含める。",
      tone: isSupport ? "secondary" : "outline",
      countsAsAgent: !isSupport,
    };
  }
  if (["hermes_gateway", "hermes_dashboard", "screenpipe", "lp_studio"].includes(kind)) {
    return {
      label: "基盤プロセス",
      description: "Gateway/Dashboard/収集基盤。必要だがエージェント数からは除外。",
      tone: "secondary",
      countsAsAgent: false,
    };
  }
  return {
    label: "補助プロセス",
    description: "役割未分類のローカル補助プロセス。エージェント数からは除外。",
    tone: "secondary",
    countsAsAgent: false,
  };
}

function ProcessCard({ process }: { process: CommandCenterProcess }) {
  const role = processRole(process);
  const work = humanWorkFromProcess(process);
  return (
    <div className="border border-border bg-background-base/35 p-3">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">{process.label}</p>
          <p className="font-mono-ui text-[0.68rem] text-text-tertiary">pid {process.pid} / {process.uptime}</p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <Badge tone={role.tone} className="text-[0.65rem]">{role.label}</Badge>
          <Badge tone="outline" className="text-[0.6rem]">{process.kind}</Badge>
        </div>
      </div>
      <div className="grid gap-2 text-xs leading-relaxed text-text-secondary">
        <p><span className="font-semibold text-text-tertiary">対象:</span> {work.target}</p>
        <p><span className="font-semibold text-text-tertiary">必要:</span> {work.need}</p>
        <p><span className="font-semibold text-text-tertiary">今:</span> {work.current}</p>
      </div>
      <p className="mt-2 text-[0.68rem] leading-relaxed text-text-tertiary">{role.description}</p>
      <p className="mt-2 truncate font-mono-ui text-[0.68rem] text-text-tertiary">{process.command}</p>
    </div>
  );
}

function SessionCard({ session }: { session: ProfileSessionInfo }) {
  const active = session.is_active;
  const warm = !active && isRecentSession(session);
  const work = humanWorkFromSession(session);
  return (
    <Link to={`/chat?resume=${encodeURIComponent(session.id)}`} className="block border border-border bg-background-base/35 p-3 transition-colors hover:border-primary/50 hover:bg-primary/[0.04]">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">{shortText(session.title, "無題のセッション", 64)}</p>
          <p className="font-mono-ui text-[0.68rem] text-text-tertiary">
            {session.profile || "default"} / {session.source || "出所未確認"}
          </p>
        </div>
        <Badge tone={active ? "success" : warm ? "warning" : "outline"} className="shrink-0 text-[0.65rem]">
          {active ? "作業中" : warm ? "直近" : "履歴"}
        </Badge>
      </div>
      <div className="grid gap-2 text-xs leading-relaxed text-text-secondary">
        <p><span className="font-semibold text-text-tertiary">対象:</span> {work.target}</p>
        <p><span className="font-semibold text-text-tertiary">必要:</span> {work.need}</p>
        <p><span className="font-semibold text-text-tertiary">今:</span> {work.current}</p>
      </div>
      <p className="mt-2 font-mono-ui text-[0.68rem] text-text-tertiary">
        {timeAgo(session.last_active)} / 発言 {session.message_count} / ツール {session.tool_call_count} / {compactModel(session.model)}
      </p>
    </Link>
  );
}

function CronRow({ job }: { job: CronJob }) {
  const running = job.state === "running";
  const team = automationTeam(job);
  const guardrail = automationGuardrail(job);
  const status = automationStatus(job);
  return (
    <Link to="/cron" className="block border border-border bg-background-base/35 p-3 transition-colors hover:border-primary/50 hover:bg-primary/[0.04]">
      <div className="mb-1 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">{cronLabel(job)}</p>
          <p className="font-mono-ui text-[0.68rem] text-text-tertiary">{scheduleExpr(job)} / 次回 {formatNextRun(job.next_run_at)}</p>
        </div>
        <Badge tone={running ? "success" : job.last_error ? "warning" : "outline"} className="shrink-0 text-[0.65rem]">
          {running ? "実行中" : job.last_error ? "要確認" : job.state === "paused" ? "停止中" : "待機中"}
        </Badge>
      </div>
      <p className="text-xs leading-relaxed text-text-secondary">{cronPurpose(job)}</p>
      <div className="mt-2 flex flex-wrap gap-1.5">
        <Badge tone={status.tone} className="text-[0.62rem]">{status.label}</Badge>
        <Badge tone={team.tone} className="text-[0.62rem]">{team.title}</Badge>
        <Badge tone={guardrail.tone} className="text-[0.62rem]">{guardrail.label}</Badge>
      </div>
      <p className="mt-1 truncate text-[0.68rem] text-text-tertiary">{status.detail}</p>
    </Link>
  );
}

function buildAutomationTeamSummaries(jobs: CronJob[]): AutomationTeamSummary[] {
  const byKey = new Map<string, AutomationTeamSummary>();
  for (const job of jobs) {
    const team = automationTeam(job);
    const existing = byKey.get(team.tag);
    if (existing) {
      existing.jobs.push(job);
      existing.running += job.state === "running" ? 1 : 0;
      if (!existing.nextRun || (job.next_run_at && job.next_run_at < existing.nextRun)) existing.nextRun = job.next_run_at;
    } else {
      byKey.set(team.tag, {
        ...team,
        key: team.tag,
        jobs: [job],
        running: job.state === "running" ? 1 : 0,
        nextRun: job.next_run_at,
      });
    }
  }
  return [...byKey.values()].sort((a, b) => {
    const priority: Record<string, number> = { watch: 0, dj: 1, laflora: 2, organize: 3, think: 4, research: 5, auto: 6 };
    return (priority[a.key] ?? 99) - (priority[b.key] ?? 99) || a.title.localeCompare(b.title);
  });
}

function AutomationTeamCard({ team }: { team: AutomationTeamSummary }) {
  const status = teamStatus(team);
  return (
    <details className="group border border-border bg-background-base/35 p-4" open={team.running > 0 || status.tone === "destructive"}>
      <summary className="flex cursor-pointer list-none items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-foreground">{team.title}</p>
          <p className="mt-1 text-xs leading-relaxed text-text-tertiary">{team.description}</p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          <Badge tone={status.tone} className="text-[0.65rem]">{status.label}</Badge>
          <Badge tone={team.tone} className="text-[0.65rem]">{team.jobs.length}件</Badge>
          {team.running > 0 && <Badge tone="success" className="text-[0.65rem]">実行中 {team.running}</Badge>}
        </div>
      </summary>
      <div className="mt-3 grid gap-2 border-t border-border pt-3">
        <div className="flex items-center justify-between gap-3 text-[0.68rem] text-text-tertiary">
          <span>状態</span>
          <span className="truncate text-right text-foreground">{status.detail}</span>
        </div>
        <div className="flex items-center justify-between gap-3 text-[0.68rem] text-text-tertiary">
          <span>次回</span>
          <span className="font-mono-ui text-foreground">{formatNextRun(team.nextRun)}</span>
        </div>
        {team.jobs.slice(0, 4).map((job) => (
          <Link key={job.id} to="/cron" className="block border border-border/70 bg-background-base/30 p-2 transition-colors hover:border-primary/40">
            <div className="flex items-center justify-between gap-2">
              <p className="min-w-0 truncate text-xs font-medium text-foreground">{cronLabel(job)}</p>
              <span className="shrink-0 font-mono-ui text-[0.62rem] text-text-tertiary">{formatNextRun(job.next_run_at)}</span>
            </div>
            <p className="mt-1 truncate text-[0.68rem] text-text-tertiary">{automationGuardrail(job).label} / {scheduleExpr(job)}</p>
          </Link>
        ))}
      </div>
    </details>
  );
}

export default function CommandCenterPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [sessions, setSessions] = useState<ProfileSessionInfo[]>([]);
  const [stats, setStats] = useState<SessionStoreStats | null>(null);
  const [jobs, setJobs] = useState<CronJob[]>([]);
  const [processes, setProcesses] = useState<CommandCenterProcess[]>([]);
  const [snapshot, setSnapshot] = useState<CommandCenterSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { setAfterTitle, setEnd } = usePageHeader();

  const load = useCallback((silent = false) => {
    if (silent) setRefreshing(true);
    else setLoading(true);
    setError(null);

    Promise.allSettled([
      api.getStatus(),
      fetchJSON<PaginatedSessions>("/api/profiles/sessions?limit=24&offset=0&order=recent&profile=all"),
      api.getSessionStats("default"),
      api.getCronJobs("all"),
      api.getCommandCenterProcesses(),
      api.getCommandCenterSnapshot(),
    ])
      .then(([statusResult, sessionsResult, statsResult, jobsResult, processesResult, snapshotResult]) => {
        if (statusResult.status === "fulfilled") setStatus(statusResult.value);
        if (sessionsResult.status === "fulfilled") setSessions(sessionsResult.value.sessions as ProfileSessionInfo[]);
        if (statsResult.status === "fulfilled") setStats(statsResult.value);
        if (jobsResult.status === "fulfilled") setJobs(jobsResult.value);
        if (processesResult.status === "fulfilled") setProcesses(processesResult.value.processes);
        else setProcesses([]);
        if (snapshotResult.status === "fulfilled") setSnapshot(snapshotResult.value);
        else setSnapshot(null);

        const failed = [statusResult, sessionsResult, jobsResult].some((result) => result.status === "rejected");
        if (failed) setError("Command Centerの一部データを読み込めなかった。");
      })
      .finally(() => {
        setLoading(false);
        setRefreshing(false);
      });
  }, []);

  useEffect(() => {
    void Promise.resolve().then(() => load(false));
    const id = setInterval(() => load(true), 5000);
    return () => clearInterval(id);
  }, [load]);

  useEffect(() => {
    setAfterTitle(
      <Badge tone={status?.gateway_running ? "success" : "outline"} className="text-xs">
        {statusLabel(status)}
      </Badge>,
    );
    setEnd(
      <Button outlined size="sm" onClick={() => load(true)} prefix={<RefreshCw />}>
        {refreshing ? "更新中" : "更新"}
      </Button>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [load, refreshing, setAfterTitle, setEnd, status]);

  const activeSessions = useMemo(
    () => sessions.filter((session) => session.is_active || isRecentSession(session)),
    [sessions],
  );
  const recentSessions = useMemo(() => sessions.slice(0, 8), [sessions]);
  const enabledJobs = useMemo(() => jobs.filter((job) => job.enabled), [jobs]);
  const runningJobs = useMemo(() => enabledJobs.filter((job) => job.state === "running"), [enabledJobs]);
  const nextJobs = useMemo(
    () => [...enabledJobs].sort((a, b) => (a.next_run_at || "9999").localeCompare(b.next_run_at || "9999")).slice(0, 8),
    [enabledJobs],
  );
  const cronBuckets = useMemo(() => buildCronBuckets(enabledJobs), [enabledJobs]);
  const automationTeams = useMemo(() => buildAutomationTeamSummaries(enabledJobs), [enabledJobs]);
  const attention = useMemo(() => buildAttentionQueue(status, enabledJobs, sessions), [enabledJobs, sessions, status]);
  const agentProcesses = useMemo(
    () => processes.filter((process) => processRole(process).countsAsAgent),
    [processes],
  );
  const serviceProcesses = useMemo(
    () => processes.filter((process) => !processRole(process).countsAsAgent && processRole(process).label === "基盤プロセス"),
    [processes],
  );
  const supportProcesses = useMemo(
    () => processes.filter((process) => !processRole(process).countsAsAgent && processRole(process).label !== "基盤プロセス"),
    [processes],
  );
  const agentDisplayProcesses = useMemo(
    () => [...agentProcesses, ...serviceProcesses, ...supportProcesses.slice(0, 4)],
    [agentProcesses, serviceProcesses, supportProcesses],
  );
  const nextJob = nextJobs[0];
  const snapshotAgents = snapshot?.agent_registry?.active ?? [];
  const snapshotRecentRuns = snapshot?.agent_registry?.recent_runs ?? [];
  const snapshotSessions = snapshot?.sessions?.active ?? [];
  const snapshotWaiting = snapshot?.rino_waiting ?? [];
  const humanState = snapshot?.human_state;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="mx-auto flex min-w-0 w-full max-w-[1560px] flex-col gap-5 px-3 py-4 sm:px-6 sm:py-6">
      {error && (
        <div className="flex items-start gap-3 border border-warning/30 bg-warning/[0.06] p-3 text-sm text-warning">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <section className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(22rem,0.8fr)]">
        <div className="border border-border bg-background-base/45 p-5 sm:p-6">
          <p className="mb-2 font-mono-ui text-xs uppercase tracking-[0.18em] text-text-tertiary">
            Rino OS / LP Command Center
          </p>
          <h1 className="max-w-4xl text-3xl font-semibold tracking-tight text-foreground sm:text-5xl">
            いま誰が動き、どのクローンが次に動くかを一画面で見る。
          </h1>
          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-text-secondary sm:text-base">
            冷たい監視盤ではなく、Rinoが「任せてよいもの / 判断が必要なもの / 次に走る自動化クローン」を上から順に読めるLP型の作戦ボード。
          </p>
          <div className="mt-5 flex flex-wrap gap-2">
            <Link to="/chat"><Button size="sm" prefix={<Terminal />}>チャットを開く</Button></Link>
            <Link to="/sessions"><Button outlined size="sm" prefix={<MessageSquare />}>セッション</Button></Link>
            <Link to="/cron"><Button outlined size="sm" prefix={<Workflow />}>自動化</Button></Link>
          </div>
        </div>

        <Card className="min-w-0 overflow-hidden">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" />
              <CardTitle className="text-base">Rino判断待ち / 注意</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="grid gap-2">
            {attention.map((item) => (
              <div key={`${item.label}:${item.detail}`} className="border border-border bg-background-base/35 p-3">
                <div className="mb-1 flex items-center gap-2">
                  {item.tone === "success" ? (
                    <CheckCircle2 className="h-4 w-4 text-success" />
                  ) : (
                    <AlertTriangle className={`h-4 w-4 ${item.tone === "destructive" ? "text-destructive" : "text-warning"}`} />
                  )}
                  <Badge tone={item.tone} className="text-[0.65rem]">{item.label}</Badge>
                </div>
                <p className="line-clamp-3 text-xs leading-relaxed text-text-secondary">{item.detail}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </section>

      <section>
        <Card className="min-w-0 overflow-hidden border-primary/25 bg-primary/[0.025]">
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-primary" />
                <CardTitle className="text-base">0. いま何をしているか / Snapshot SSOT</CardTitle>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge tone={snapshot?.exists === false || snapshot?.error ? "warning" : "success"} className="text-xs">
                  {snapshot?.generated_at ? `生成 ${snapshot.generated_at.replace("T", " ").slice(0, 16)}` : "snapshot待機中"}
                </Badge>
                <Badge tone="outline" className="text-xs tabular-nums">
                  作業登録 {snapshotAgents.length}
                </Badge>
                <Badge tone="outline" className="text-xs tabular-nums">
                  完了 {snapshotRecentRuns.length}
                </Badge>
                <Badge tone="outline" className="text-xs tabular-nums">
                  直近セッション {snapshotSessions.length}
                </Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="border border-border bg-background-base/35 p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-semibold text-foreground">人間向けの作業メモ</p>
                <Badge tone="outline" className="text-[0.65rem]">
                  {snapshotSourceLabel(humanState?.source)}
                </Badge>
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                <div className="border border-border bg-background-base/30 p-3">
                  <p className="mb-1 text-[0.68rem] font-semibold text-text-tertiary">どのリポジトリ / プロジェクト</p>
                  <p className="text-sm leading-relaxed text-foreground">{humanState?.target || "未確認"}</p>
                </div>
                <div className="border border-border bg-background-base/30 p-3">
                  <p className="mb-1 text-[0.68rem] font-semibold text-text-tertiary">何が必要か</p>
                  <p className="text-sm leading-relaxed text-foreground">{humanState?.need || "未確認"}</p>
                </div>
                <div className="border border-border bg-background-base/30 p-3">
                  <p className="mb-1 text-[0.68rem] font-semibold text-text-tertiary">今やっていること</p>
                  <p className="text-sm leading-relaxed text-foreground">{humanState?.current || snapshot?.one_line_state || "まだCommand Center snapshotが生成されていない。"}</p>
                </div>
              </div>
              {humanState?.next_automation && (
                <p className="mt-3 text-xs text-text-secondary">次の自動化: {humanState.next_automation}</p>
              )}
              {snapshot?.path && (
                <p className="mt-2 break-all font-mono-ui text-[0.7rem] text-text-tertiary">{snapshot.path}</p>
              )}
              {snapshot?.error && <p className="mt-2 text-xs text-destructive">{snapshot.error}</p>}
            </div>

            <div className="grid gap-3 xl:grid-cols-3">
              <div className="grid gap-2 xl:col-span-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-medium text-text-secondary">作業カード</p>
                  <Badge tone="outline" className="text-[0.65rem]">作業中 {snapshotAgents.length}</Badge>
                </div>
                {snapshotAgents.length ? (
                  snapshotAgents.slice(0, 5).map((agent) => (
                    <div key={agent.agent_id || agent.mission || agent.current_step || "agent"} className="border border-border bg-background-base/30 p-3">
                      <div className="mb-1 flex flex-wrap items-center gap-2">
                        <Badge tone={agent.status === "blocked" ? "warning" : "secondary"} className="text-[0.65rem]">
                          {agent.status === "blocked" ? "停止中" : agent.status === "running" ? "実行中" : agent.status === "waiting" ? "待機中" : agent.status === "registered" ? "登録済み" : "状態未確認"}
                        </Badge>
                        <span className="text-sm font-medium text-foreground">{agent.mission || agent.agent_id || "作業"}</span>
                      </div>
                      <div className="grid gap-2 text-xs leading-relaxed text-text-secondary sm:grid-cols-3">
                        <p><span className="font-semibold text-text-tertiary">対象:</span> {agent.workdir || agent.artifact || agent.team || "未登録"}</p>
                        <p><span className="font-semibold text-text-tertiary">必要:</span> {agent.mission || "未登録"}</p>
                        <p><span className="font-semibold text-text-tertiary">今:</span> {agent.current_step || "今の作業未登録"}</p>
                      </div>
                      {agent.artifact && <p className="mt-2 break-all font-mono-ui text-[0.68rem] text-text-tertiary">{agent.artifact}</p>}
                      {agent.blocker && <p className="mt-1 text-xs text-warning">詰まり: {agent.blocker}</p>}
                    </div>
                  ))
                ) : (
                  <div className="border border-border bg-background-base/30 p-3 text-xs text-muted-foreground">
                    作業登録はまだない。次は実行ラッパーから現在作業を登録すると「何をしているか」がここに出る。
                  </div>
                )}

                {snapshotRecentRuns.length > 0 && (
                  <div className="mt-2 grid gap-2 border-t border-border/70 pt-3">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium text-text-secondary">最近完了した作業</p>
                      <Badge tone="outline" className="text-[0.65rem]">{snapshotRecentRuns.length}</Badge>
                    </div>
                    {snapshotRecentRuns.slice(0, 4).map((agent) => (
                      <div key={`recent:${agent.agent_id || agent.run_id || agent.artifact || agent.mission}`} className="border border-border bg-background-base/20 p-3">
                        <div className="mb-1 flex flex-wrap items-center gap-2">
                          <Badge tone={agent.status === "failed" ? "warning" : "outline"} className="text-[0.65rem]">
                            {agent.status === "failed" ? "失敗" : agent.status === "completed" ? "完了" : "完了扱い"}
                          </Badge>
                          <span className="text-sm font-medium text-foreground">{agent.mission || agent.agent_id || "完了した作業"}</span>
                        </div>
                        <p className="text-xs leading-relaxed text-text-secondary">{agent.current_step || "完了"}</p>
                        {agent.artifact && <p className="mt-1 break-all font-mono-ui text-[0.68rem] text-text-tertiary">{agent.artifact}</p>}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="grid gap-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-medium text-text-secondary">Rino判断待ち</p>
                  <Badge tone="outline" className="text-[0.65rem]">{snapshotWaiting.length}</Badge>
                </div>
                {snapshotWaiting.slice(0, 5).map((item) => (
                  <div key={`${item.type}:${item.title}:${item.detail}`} className="border border-border bg-background-base/30 p-3">
                    <Badge tone={item.type === "APPROVE" ? "warning" : "secondary"} className="mb-2 text-[0.65rem]">
                      {waitingTypeLabel(item.type)}
                    </Badge>
                    <p className="text-xs font-medium text-foreground">{item.title || "判断待ち"}</p>
                    <p className="mt-1 text-xs leading-relaxed text-text-secondary">{item.detail || "詳細なし"}</p>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section>
        <Card className="min-w-0 overflow-hidden border-primary/25 bg-primary/[0.025]">
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Bot className="h-5 w-5 text-primary" />
                <CardTitle className="text-base">1. 今動いているエージェント</CardTitle>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <Badge tone="success" className="text-xs tabular-nums">作業主体 {agentProcesses.length}</Badge>
                <Badge tone="secondary" className="text-xs tabular-nums">基盤 {serviceProcesses.length}</Badge>
                <Badge tone="outline" className="text-xs tabular-nums">補助 {supportProcesses.length}</Badge>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="grid gap-3 md:grid-cols-3">
              <div className="border border-border bg-background-base/35 p-3">
                <p className="text-3xl font-semibold tabular-nums text-foreground">{agentProcesses.length}</p>
                <p className="text-sm font-medium text-text-secondary">エージェントとして数える</p>
                <p className="mt-1 text-xs text-text-tertiary">主会話・外部AI・実行中の自動化だけ。</p>
              </div>
              <div className="border border-border bg-background-base/35 p-3">
                <p className="text-3xl font-semibold tabular-nums text-foreground">{serviceProcesses.length}</p>
                <p className="text-sm font-medium text-text-secondary">基盤プロセス</p>
                <p className="mt-1 text-xs text-text-tertiary">Gateway / Dashboard / 収集基盤。必要だが数に混ぜない。</p>
              </div>
              <div className="border border-border bg-background-base/35 p-3">
                <p className="text-3xl font-semibold tabular-nums text-foreground">{supportProcesses.length}</p>
                <p className="text-sm font-medium text-text-secondary">補助プロセス</p>
                <p className="mt-1 text-xs text-text-tertiary">MCP/GUI helper等。膨らむのでカウント外。</p>
              </div>
            </div>

            <details className="border border-border bg-background-base/25 p-3">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-sm text-text-secondary">
                <span>プロセス詳細を開く</span>
                <Badge tone="outline" className="text-[0.65rem]">表示 {agentDisplayProcesses.length}</Badge>
              </summary>
              <div className="mt-3 grid gap-3 border-t border-border pt-3 md:grid-cols-2 2xl:grid-cols-3">
                {agentDisplayProcesses.length ? (
                  agentDisplayProcesses.map((process) => <ProcessCard key={`${process.pid}:${process.kind}`} process={process} />)
                ) : (
                  <div className="border border-border bg-background-base/35 p-4 text-sm text-muted-foreground md:col-span-2 2xl:col-span-3">
                    プロセス情報を取得できない、または作業主体として数えるプロセスが見つからない。セッションと自動化は下に表示する。
                  </div>
                )}
              </div>
            </details>
          </CardContent>
        </Card>
      </section>

      <section className="grid min-w-0 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="今動いているエージェント"
          value={String(agentProcesses.length)}
          detail={`取得 ${processes.length}件 / 補助は除外`}
          href="/system"
          icon={Bot}
          tone="text-primary"
        />
        <StatCard
          label="今動いているセッション"
          value={String(activeSessions.length)}
          detail={`表示できるセッション ${sessions.length || stats?.total || 0}件`}
          href="/sessions"
          icon={MessageSquare}
          tone="text-success"
        />
        <StatCard
          label="今動いているクローン"
          value={String(runningJobs.length)}
          detail={`${enabledJobs.length}件の自動化クローンが待機中`}
          href="/cron"
          icon={Workflow}
          tone={runningJobs.length ? "text-success" : "text-text-tertiary"}
        />
        <StatCard
          label="次に動くクローン"
          value={nextJob ? formatNextRun(nextJob.next_run_at) : "なし"}
          detail={nextJob ? cronLabel(nextJob) : "有効な自動化クローンなし"}
          href="/cron"
          icon={Clock}
          tone="text-warning"
        />
      </section>

      <section className="grid min-w-0 gap-4 xl:grid-cols-2">
        <Card className="min-w-0 overflow-hidden">
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-success" />
                <CardTitle className="text-base">2. 今動いているセッション</CardTitle>
              </div>
              <Badge tone="outline" className="text-xs tabular-nums">作業中/直近 {activeSessions.length}</Badge>
            </div>
          </CardHeader>
          <CardContent className="grid gap-3">
            {(activeSessions.length ? activeSessions : recentSessions).slice(0, 8).map((session) => (
              <SessionCard key={session.id} session={session} />
            ))}
            {sessions.length === 0 && (
              <div className="border border-border bg-background-base/35 p-4 text-sm text-muted-foreground">
                セッションが返っていない。バックエンド更新後に、default profileを含む会話履歴がここに出る。
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="min-w-0 overflow-hidden">
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Workflow className="h-5 w-5 text-warning" />
                <CardTitle className="text-base">3. 今動いているクローン</CardTitle>
              </div>
              <Badge tone={runningJobs.length ? "success" : "outline"} className="text-xs tabular-nums">実行中 {runningJobs.length}</Badge>
            </div>
          </CardHeader>
          <CardContent className="grid gap-3">
            {runningJobs.length ? (
              runningJobs.slice(0, 6).map((job) => <CronRow key={job.id} job={job} />)
            ) : (
              <div className="border border-border bg-background-base/35 p-4 text-sm text-muted-foreground">
                今この瞬間に実行中のクローンはない。待機中の自動化は次セクションで確認。
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      <section>
        <Card className="min-w-0 overflow-hidden">
          <details open>
            <summary className="cursor-pointer list-none">
              <CardHeader>
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Clock className="h-5 w-5 text-warning" />
                    <CardTitle className="text-base">4. 次に動くクローン</CardTitle>
                  </div>
                  <Badge tone="outline" className="text-xs tabular-nums">次回 {Math.min(nextJobs.length, 8)}</Badge>
                </div>
              </CardHeader>
            </summary>
            <CardContent className="grid gap-3 md:grid-cols-2 2xl:grid-cols-4">
              {nextJobs.slice(0, 8).map((job) => (
                <CronRow key={job.id} job={job} />
              ))}
              {nextJobs.length === 0 && (
                <div className="border border-border bg-background-base/35 p-4 text-sm text-muted-foreground md:col-span-2 2xl:col-span-4">
                  次に動くクローンはまだ見つかっていない。
                </div>
              )}
            </CardContent>
          </details>
        </Card>
      </section>

      <section>
        <Card className="min-w-0 overflow-hidden">
          <details>
            <summary className="cursor-pointer list-none">
              <CardHeader>
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Workflow className="h-5 w-5 text-warning" />
                    <CardTitle className="text-base">5. 自動化チーム編成</CardTitle>
                  </div>
                  <Badge tone="outline" className="text-xs tabular-nums">{automationTeams.length}チーム / {enabledJobs.length}件</Badge>
                </div>
              </CardHeader>
            </summary>
            <CardContent className="grid gap-3 md:grid-cols-2 2xl:grid-cols-3">
              {automationTeams.map((team) => (
                <AutomationTeamCard key={team.key} team={team} />
              ))}
              {automationTeams.length === 0 && (
                <div className="border border-border bg-background-base/35 p-4 text-sm text-muted-foreground md:col-span-2 2xl:col-span-3">
                  有効な自動化チームはまだ見つかっていない。
                </div>
              )}
            </CardContent>
          </details>
        </Card>
      </section>

      <section>
        <Card className="min-w-0 overflow-hidden">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Radio className="h-5 w-5 text-primary" />
              <CardTitle className="text-base">6. 自動化タイムテーブル — いつ、何が、何のために動くか</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
            {cronBuckets.map((bucket) => (
              <div key={bucket.key} className="border border-border bg-background-base/35 p-4">
                <div className="mb-3 flex items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-foreground">{bucket.label}</p>
                    <p className="mt-1 text-xs leading-relaxed text-text-tertiary">{bucket.description}</p>
                  </div>
                  <Badge tone="outline" className="text-[0.65rem]">{bucket.jobs.length}</Badge>
                </div>
                <div className="grid gap-2">
                  {bucket.jobs.slice(0, 5).map((job) => (
                    <Link key={job.id} to="/cron" className="block border border-border/70 bg-background-base/30 p-2 transition-colors hover:border-primary/40">
                      <div className="flex items-center justify-between gap-2">
                        <p className="min-w-0 truncate text-xs font-medium text-foreground">{cronLabel(job)}</p>
                        <span className="shrink-0 font-mono-ui text-[0.62rem] text-text-tertiary">{formatNextRun(job.next_run_at)}</span>
                      </div>
                      <p className="mt-1 truncate text-[0.68rem] text-text-tertiary">{scheduleExpr(job)} / {cronPurpose(job)}</p>
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </section>

      <section className="grid min-w-0 gap-3 border border-border bg-background-base/30 p-4 text-xs leading-relaxed text-text-tertiary md:grid-cols-3">
        <div>
          <p className="mb-1 font-semibold text-text-secondary"><Database className="mr-1 inline h-3.5 w-3.5" />データ元</p>
          <p>プロセス・セッション・cronの読み取り専用APIから表示している。</p>
        </div>
        <div>
          <p className="mb-1 font-semibold text-text-secondary"><GitBranch className="mr-1 inline h-3.5 w-3.5" />安全制約</p>
          <p>外部送信・本番更新・cron変更・canon/memory/skill昇格はここから直接実行しない。</p>
        </div>
        <div>
          <p className="mb-1 font-semibold text-text-secondary"><Activity className="mr-1 inline h-3.5 w-3.5" />更新</p>
          <p>5秒ごとにread-only更新。操作は各詳細ページへのリンクだけ。</p>
        </div>
      </section>
    </div>
  );
}
