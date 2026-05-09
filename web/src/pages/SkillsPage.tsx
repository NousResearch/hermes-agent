import { useEffect, useLayoutEffect, useState, useMemo } from "react";
import {
  Package,
  Search,
  Wrench,
  X,
  Cpu,
  Globe,
  Shield,
  Eye,
  Paintbrush,
  Brain,
  Blocks,
  Code,
  Zap,
  Filter,
} from "lucide-react";
import { api } from "@/lib/api";
import type { SkillInfo, ToolsetInfo } from "@/lib/api";
import { useToast } from "@/hooks/useToast";
import { Toast } from "@/components/Toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/i18n";
import { usePageHeader } from "@/contexts/usePageHeader";
import { PluginSlot } from "@/plugins";

/* ------------------------------------------------------------------ */
/*  Types & helpers                                                    */
/* ------------------------------------------------------------------ */

const CATEGORY_LABELS: Record<string, string> = {
  mlops: "MLOps",
  "mlops/cloud": "MLOps / Cloud",
  "mlops/evaluation": "MLOps / Evaluation",
  "mlops/inference": "MLOps / Inference",
  "mlops/models": "MLOps / Models",
  "mlops/training": "MLOps / Training",
  "mlops/vector-databases": "MLOps / Vector DBs",
  mcp: "MCP",
  "red-teaming": "Red Teaming",
  ocr: "OCR",
  p5js: "p5.js",
  ai: "AI",
  ux: "UX",
  ui: "UI",
};

function prettyCategory(
  raw: string | null | undefined,
  generalLabel: string,
): string {
  if (!raw) return generalLabel;
  if (CATEGORY_LABELS[raw]) return CATEGORY_LABELS[raw];
  return raw
    .split(/[-_/]/)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

const BUNDLED_SKILL_DESCRIPTION_JA: Record<string, string> = {
  "Delegate coding to Claude Code CLI (features, PRs).": "Claude Code CLI に実装作業を任せます（機能追加、PR 作成など）。",
  "Delegate coding to OpenAI Codex CLI (features, PRs).": "OpenAI Codex CLI に実装作業を任せます（機能追加、PR 作成など）。",
  "Configure, extend, or contribute to Hermes Agent.": "Hermes Agent の設定、拡張、開発に使います。",
  "Delegate coding to OpenCode CLI (features, PR review).": "OpenCode CLI に実装作業や PR レビューを任せます。",
  "Comprehensive Cloudflare platform skill covering Workers, Pages, R2, KV, D1, Durable Objects, Queues, AI, Images, Stream, Zero Trust, DNS, WAF, and developer tooling.": "Workers、Pages、KV/D1/R2、Workers AI、Tunnel、WAF など Cloudflare 全般を扱います。",
  "Build AI agents on Cloudflare Workers using the Agents SDK.": "Cloudflare Workers 上で、状態を持つエージェント、ワークフロー、WebSocket アプリ、MCP サーバー、チャット/音声エージェントを作ります。",
  "Send and receive transactional emails with Cloudflare Email Service.": "Cloudflare Email Service で取引メールの送受信、ルーティング、Workers 連携を扱います。",
  "Create and review Cloudflare Durable Objects. Use when building stateful serverless apps.": "Cloudflare Durable Objects で状態管理、SQLite ストレージ、アラーム、WebSocket を実装・レビューします。",
  "Build sandboxed applications for secure code execution.": "安全なコード実行、コードインタープリタ、CI/CD、対話型の開発環境を構築します。",
  "Reviews and authors Cloudflare Workers code against production best practices.": "Cloudflare Workers のコードを本番運用向けの作法に沿って作成・レビューします。",
  "Cloudflare Workers CLI for deploying, developing, and managing Cloudflare resources.": "Wrangler CLI で Workers、KV、R2、D1、Vectorize、Workers AI などをデプロイ・管理します。",
  "Analyzes web performance using Chrome DevTools MCP.": "Chrome DevTools MCP で Core Web Vitals と Web 性能を分析します。",
  "Dark-themed SVG architecture/cloud/infra diagrams as HTML.": "ダークテーマのアーキテクチャ図、クラウド構成図、インフラ図を SVG/HTML で作成します。",
  "ASCII art: pyfiglet, cowsay, boxes, image-to-ascii.": "pyfiglet、cowsay、boxes、image-to-ascii で ASCII アートを作成します。",
  "ASCII video: convert video/audio to colored ASCII MP4/GIF.": "動画や音声を、色付き ASCII の MP4/GIF に変換します。",
  "Knowledge comics (知识漫画): educational, biography, tutorial.": "教育、伝記、チュートリアル向けの知識漫画を作成します。",
  "Infographics: 21 layouts x 21 styles (信息图, 可视化).": "情報図や可視化インフォグラフィックを、21 種類の構成と 21 種類のスタイルで作成します。",
  "Design one-off HTML artifacts (landing, deck, prototype).": "ランディングページ、資料、プロトタイプなど、単発の HTML デザイン成果物を作成します。",
  "Generate images, video, and audio with ComfyUI — install, launch, manage nodes/models, run workflows with parameter injection. Uses the official comfy-cli for lifecycle and direct REST/WebSocket API for execution.": "ComfyUI の導入、起動、ワークフロー実行を行い、画像・動画・音声を生成します。",
  "Generate project ideas via creative constraints.": "制約条件を手がかりに、プロジェクト案を発想します。",
  "Author/validate/export Google's DESIGN.md token spec files.": "Google DESIGN.md のトークン仕様ファイルを作成、検証、書き出しします。",
  "Hand-drawn Excalidraw JSON diagrams (arch, flow, seq).": "アーキテクチャ、フロー、シーケンスなどの手描き風 Excalidraw JSON 図を作成します。",
  "Humanize text: strip AI-isms and add real voice.": "AI っぽさを取り除き、自然な文章に整えます。",
  "Manim CE animations: 3Blue1Brown math/algo videos.": "Manim CE で 3Blue1Brown 風の数学・アルゴリズム動画を作成します。",
  "p5.js sketches: gen art, shaders, interactive, 3D.": "p5.js で生成アート、シェーダー、対話型スケッチ、3D スケッチを作ります。",
  "Pixel art w/ era palettes (NES, Game Boy, PICO-8).": "NES、Game Boy、PICO-8 などの配色でピクセルアートを作ります。",
  "54 real design systems (Stripe, Linear, Vercel) as HTML/CSS.": "Stripe、Linear、Vercel など実在のデザインシステム風 HTML/CSS を作成します。",
  "Use when building creative browser demos with @chenglou/pretext — DOM-free text layout for ASCII art, typographic flow around obstacles, text-as-geometry games, kinetic typography, and text-powered generative art. Produces single-file HTML demos by default.": "@chenglou/pretext で ASCII や文字組みを使ったブラウザデモを構築します。",
  "Throwaway HTML mockups: 2-3 design variants to compare.": "比較用の使い捨て HTML モックアップを 2〜3 案作成します。",
  "Songwriting craft and Suno AI music prompts.": "作詞の技法と Suno 向け音楽生成プロンプトを扱います。",
  "Control a running TouchDesigner instance via twozero MCP — create operators, set parameters, wire connections, execute Python, build real-time visuals. 36 native tools.": "twozero MCP で TouchDesigner を操作し、リアルタイム映像を構築します。",
  "Iterative Python via live Jupyter kernel (hamelnb).": "稼働中の Jupyter カーネルを使い、Python を対話的に探索します。",
  "Use when generating, checking, registering, configuring, and maintaining domains.": "ドメインの命名、空き確認、DNS、Cloudflare、メールルーティング、到達性を扱います。",
  "Decomposition playbook + specialist-roster conventions + anti-temptation rules for an orchestrator profile routing work through Kanban. The \"don't do the work yourself\" rule and the basic lifecycle are auto-injected into every kanban worker's system prompt; this skill is the deeper playbook when you're specifically playing the orchestrator role.": "Kanban で作業を分解し、適切な専門プロファイルへ回すための手順です。",
  "Pitfalls, examples, and edge cases for Hermes Kanban workers. The lifecycle itself is auto-injected into every worker's system prompt as KANBAN_GUIDANCE (from agent/prompt_builder.py); this skill is what you load when you want deeper detail on specific scenarios.": "Hermes Kanban ワーカー向けに、落とし穴、引き継ぎ、再試行、例外処理を扱います。",
  "Use when administering or troubleshooting Linux desktop/workstation systems.": "Linux デスクトップ/ワークステーションの DNS、HTTPS 信頼、polkit、リモートデスクトップ、端末キー設定を管理・調査します。",
  "Deploy and operate LAN-first self-hosted services on home servers/VPS with Docker Compose and reverse proxies.": "セルフホストサービスを Docker Compose などで運用します。",
  "Webhook subscriptions: event-driven agent runs.": "Webhook 購読によるイベント駆動のエージェント実行を扱います。",
  "Exploratory QA of web apps: find bugs, evidence, reports.": "Web アプリを探索的に QA し、不具合、証拠、報告をまとめます。",
  "Himalaya CLI: IMAP/SMTP email from terminal.": "Himalaya CLI で IMAP/SMTP メールを端末から扱います。",
  "Host modded Minecraft servers (CurseForge, Modrinth).": "CurseForge/Modrinth などの Mod 入り Minecraft サーバーを構築・運用します。",
  "Play Pokemon via headless emulator + RAM reads.": "ヘッドレスエミュレータと RAM 読み取りで Pokemon をプレイします。",
  "Inspect codebases w/ pygount: LOC, languages, ratios.": "pygount でコードベースの行数、言語構成、比率を調査します。",
  "GitHub auth setup: HTTPS tokens, SSH keys, gh CLI login.": "GitHub の HTTPS トークン、SSH キー、gh CLI ログインを設定します。",
  "Review PRs: diffs, inline comments via gh or REST.": "PR の差分をレビューし、gh/REST でインラインコメントします。",
  "Create, triage, label, assign GitHub issues via gh or REST.": "GitHub Issue の作成、整理、ラベル付け、担当者設定を行います。",
  "GitHub PR lifecycle: branch, commit, open, CI, merge.": "GitHub PR のブランチ作成、コミット、オープン、CI、マージまでを扱います。",
  "Clone/create/fork repos; manage remotes, releases.": "リポジトリの clone/create/fork、remote、release を管理します。",
  "MCP client: connect servers, register tools (stdio/HTTP).": "MCP クライアントとして stdio/HTTP サーバーに接続し、ツール登録を扱います。",
  "Search/download GIFs from Tenor via curl + jq.": "Tenor から curl + jq で GIF を検索・取得します。",
  "HeartMuLa: Suno-like song generation from lyrics + tags.": "HeartMuLa で歌詞とタグから Suno 風の曲を生成します。",
  "Audio spectrograms/features (mel, chroma, MFCC) via CLI.": "音声のスペクトログラムや特徴量（mel、chroma、MFCC）を CLI で扱います。",
  "Spotify: play, search, queue, manage playlists and devices.": "Spotify の再生、検索、キュー、プレイリスト、デバイスを管理します。",
  "YouTube transcripts to summaries, threads, blogs.": "YouTube の文字起こしを要約、スレッド、ブログ記事に変換します。",
  "lm-eval-harness: benchmark LLMs (MMLU, GSM8K, etc.).": "lm-eval-harness で MMLU、GSM8K などの LLM ベンチマークを実行します。",
  "W&B: log ML experiments, sweeps, model registry, dashboards.": "W&B で実験、スイープ、モデル登録、ダッシュボードを管理します。",
  "HuggingFace hf CLI: search/download/upload models, datasets.": "Hugging Face の hf CLI でモデル/データセットを検索、取得、アップロードします。",
  "llama.cpp local GGUF inference + HF Hub model discovery.": "llama.cpp でローカル GGUF 推論と Hugging Face Hub のモデル探索を扱います。",
  "OBLITERATUS: abliterate LLM refusals (diff-in-means).": "diff-in-means による LLM の拒否傾向除去を扱います。",
  "Outlines: structured JSON/regex/Pydantic LLM generation.": "Outlines で JSON、正規表現、Pydantic に沿った構造化生成を行います。",
  "vLLM: high-throughput LLM serving, OpenAI API, quantization.": "vLLM で高スループットの LLM 配信、OpenAI API、量子化を扱います。",
  "AudioCraft: MusicGen text-to-music, AudioGen text-to-sound.": "AudioCraft/MusicGen/AudioGen でテキストから音楽や効果音を生成します。",
  "SAM: zero-shot image segmentation via points, boxes, masks.": "SAM で点、矩形、マスクを使ったゼロショット画像セグメンテーションを行います。",
  "DSPy: declarative LM programs, auto-optimize prompts, RAG.": "DSPy で宣言的な LM プログラム、プロンプト最適化、RAG を構築します。",
  "Axolotl: YAML LLM fine-tuning (LoRA, DPO, GRPO).": "Axolotl の YAML で LoRA、DPO、GRPO などの LLM ファインチューニングを行います。",
  "TRL: SFT, DPO, PPO, GRPO, reward modeling for LLM RLHF.": "TRL で SFT、DPO、PPO、GRPO、報酬モデル作成を行います。",
  "Unsloth: 2-5x faster LoRA/QLoRA fine-tuning, less VRAM.": "Unsloth で高速な LoRA/QLoRA ファインチューニングを行います。",
  "Use when the user asks to remember, memo, note, save, keep, or record information for later.": "『覚えておいて』『メモして』などの依頼を、明示的なメモとして保存します。",
  "Read, search, create, and edit notes in the Obsidian vault.": "Obsidian vault のノートを読み取り、検索し、新規作成します。",
  "Airtable REST API via curl. Records CRUD, filters, upserts.": "Airtable REST API でレコードの CRUD、絞り込み、upsert を行います。",
  "Gmail, Calendar, Drive, Docs, Sheets via gws CLI or Python.": "Gmail、Calendar、Drive、Docs、Sheets を gws CLI や Python で扱います。",
  "Linear: manage issues, projects, teams via GraphQL + curl.": "Linear の Issue、プロジェクト、チームを GraphQL/curl で管理します。",
  "Geocode, POIs, routes, timezones via OpenStreetMap/OSRM.": "OpenStreetMap/OSRM でジオコーディング、施設検索、経路、タイムゾーンを扱います。",
  "Edit PDF text/typos/titles via nano-pdf CLI (NL prompts).": "nano-pdf CLI で PDF の誤字、本文、タイトルを自然言語で編集します。",
  "Notion API via curl: pages, databases, blocks, search.": "Notion API でページ、データベース、ブロック、検索を扱います。",
  "Extract text from PDFs/scans (pymupdf, marker-pdf).": "PDF やスキャン画像から pymupdf や marker-pdf でテキストを抽出します。",
  "Create, read, edit .pptx decks, slides, notes, templates.": "PowerPoint 資料の作成、読み取り、編集、ノート、テンプレートを扱います。",
  "Evaluate, configure, or compare GUI voice dictation apps.": "ローカル音声認識とカスタム LLM/API バックエンドを使う GUI 音声入力アプリを評価・設定します。",
  "Jailbreak LLMs: Parseltongue, GODMODE, ULTRAPLINIAN.": "Parseltongue、GODMODE、ULTRAPLINIAN などの LLM jailbreak を扱います。",
  "Search arXiv papers by keyword, author, category, or ID.": "arXiv 論文をキーワード、著者、カテゴリ、ID で検索します。",
  "Monitor blogs and RSS/Atom feeds via blogwatcher-cli tool.": "blogwatcher-cli でブログや RSS/Atom フィードを監視します。",
  "Karpathy's LLM Wiki: build/query interlinked markdown KB.": "Karpathy の LLM Wiki を Markdown 知識ベースとして構築・検索します。",
  "Query Polymarket: markets, prices, orderbooks, history.": "Polymarket のマーケット、価格、板、履歴を照会します。",
  "Write ML papers for NeurIPS/ICML/ICLR: design→submit.": "NeurIPS/ICML/ICLR 向け ML 論文の設計から投稿までを支援します。",
  "Control Philips Hue lights, scenes, rooms via OpenHue CLI.": "OpenHue CLI で Philips Hue のライト、シーン、部屋を制御します。",
  "X/Twitter via xurl CLI: post, search, DM, media, v2 API.": "xurl CLI で X/Twitter の投稿、検索、DM、メディア、v2 API を扱います。",
  "Debug Hermes TUI slash commands: Python, gateway, Ink UI.": "Hermes TUI の slash command を Python、gateway、Ink UI の経路でデバッグします。",
  "Use when the user asks to update documentation and then commit/push changes safely.": "ドキュメント更新後の確認、差分確認、コミット、プッシュを安全に実行します。",
  "Use when developing, debugging, or restoring Hermes Agent.": "Hermes Agent 本体の開発、デバッグ、UI ビルド成果物、実行時検証を扱います。",
  "Author in-repo SKILL.md: frontmatter, validator, structure.": "リポジトリ内 SKILL.md の frontmatter、検証、構造を作成します。",
  "Debug Node.js via --inspect + Chrome DevTools Protocol CLI.": "Node.js を --inspect と Chrome DevTools Protocol CLI でデバッグします。",
  "Plan mode: write markdown plan to .hermes/plans/, no exec.": "実行せず、Markdown の計画を .hermes/plans/ に書く計画モードです。",
  "Debug Python: pdb REPL + debugpy remote (DAP).": "pdb REPL と debugpy リモート（DAP）で Python をデバッグします。",
  "Pre-commit review: security scan, quality gates, auto-fix.": "コミット前レビュー、セキュリティ確認、品質ゲート、自動修正を行います。",
  "Throwaway experiments to validate an idea before build.": "実装前に使い捨ての実験でアイデアを検証します。",
  "Execute plans via delegate_task subagents (2-stage review).": "delegate_task サブエージェントで、2 段階レビュー付きの実装を行います。",
  "4-phase root cause debugging: understand bugs before fixing.": "原因理解を先に行う 4 段階の系統的デバッグです。",
  "TDD: enforce RED-GREEN-REFACTOR, tests before code.": "RED-GREEN-REFACTOR に沿ったテスト駆動開発を行います。",
  "Build, debug, package, and configure VS Code extensions.": "VS Code 拡張機能のビルド、デバッグ、パッケージ化、設定を扱います。",
  "Write implementation plans: bite-sized tasks, paths, code.": "小さなタスク、対象パス、コード方針を含む実装計画を書きます。",
  "Yuanbao (元宝) groups: @mention users, query info/members.": "Yuanbao（元宝）のグループで @mention、情報照会、メンバー照会を行います。",
};

function skillDescription(skill: SkillInfo, locale: string): string {
  if (locale !== "ja") return skill.description;
  return BUNDLED_SKILL_DESCRIPTION_JA[skill.description] || skill.description;
}

const TOOLSET_ICONS: Record<
  string,
  React.ComponentType<{ className?: string }>
> = {
  computer: Cpu,
  web: Globe,
  security: Shield,
  vision: Eye,
  design: Paintbrush,
  ai: Brain,
  integration: Blocks,
  code: Code,
  automation: Zap,
};

function toolsetIcon(
  name: string,
): React.ComponentType<{ className?: string }> {
  const lower = name.toLowerCase();
  for (const [key, icon] of Object.entries(TOOLSET_ICONS)) {
    if (lower.includes(key)) return icon;
  }
  return Wrench;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */

export default function SkillsPage() {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [toolsets, setToolsets] = useState<ToolsetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [view, setView] = useState<"skills" | "toolsets">("skills");
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [togglingSkills, setTogglingSkills] = useState<Set<string>>(new Set());
  const { toast, showToast } = useToast();
  const { t, locale } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

  useEffect(() => {
    Promise.all([api.getSkills(), api.getToolsets()])
      .then(([s, tsets]) => {
        setSkills(s);
        setToolsets(tsets);
      })
      .catch(() => showToast(t.common.loading, "error"))
      .finally(() => setLoading(false));
  }, []);

  /* ---- Toggle skill ---- */
  const handleToggleSkill = async (skill: SkillInfo) => {
    setTogglingSkills((prev) => new Set(prev).add(skill.name));
    try {
      await api.toggleSkill(skill.name, !skill.enabled);
      setSkills((prev) =>
        prev.map((s) =>
          s.name === skill.name ? { ...s, enabled: !s.enabled } : s,
        ),
      );
      showToast(
        `${skill.name} ${skill.enabled ? t.common.disabled : t.common.enabled}`,
        "success",
      );
    } catch {
      showToast(`${t.common.failedToToggle} ${skill.name}`, "error");
    } finally {
      setTogglingSkills((prev) => {
        const next = new Set(prev);
        next.delete(skill.name);
        return next;
      });
    }
  };

  /* ---- Derived data ---- */
  const lowerSearch = search.toLowerCase();
  const isSearching = search.trim().length > 0;

  const searchMatchedSkills = useMemo(() => {
    if (!isSearching) return [];
    return skills.filter(
      (s) =>
        s.name.toLowerCase().includes(lowerSearch) ||
        s.description.toLowerCase().includes(lowerSearch) ||
        skillDescription(s, locale).toLowerCase().includes(lowerSearch) ||
        (s.category ?? "").toLowerCase().includes(lowerSearch),
    );
  }, [skills, isSearching, lowerSearch, locale]);

  const activeSkills = useMemo(() => {
    if (isSearching) return [];
    if (!activeCategory)
      return [...skills].sort((a, b) => a.name.localeCompare(b.name));
    return skills
      .filter((s) =>
        activeCategory === "__none__"
          ? !s.category
          : s.category === activeCategory,
      )
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [skills, activeCategory, isSearching]);

  const allCategories = useMemo(() => {
    const cats = new Map<string, number>();
    for (const s of skills) {
      const key = s.category || "__none__";
      cats.set(key, (cats.get(key) || 0) + 1);
    }
    return [...cats.entries()]
      .sort((a, b) => {
        if (a[0] === "__none__") return -1;
        if (b[0] === "__none__") return 1;
        return a[0].localeCompare(b[0]);
      })
      .map(([key, count]) => ({
        key,
        name: prettyCategory(key === "__none__" ? null : key, t.common.general),
        count,
      }));
  }, [skills, t]);

  const enabledCount = skills.filter((s) => s.enabled).length;

  useLayoutEffect(() => {
    if (loading) {
      setAfterTitle(null);
      setEnd(null);
      return;
    }
    setAfterTitle(
      <span className="whitespace-nowrap text-xs text-muted-foreground">
        {t.skills.enabledOf
          .replace("{enabled}", String(enabledCount))
          .replace("{total}", String(skills.length))}
      </span>,
    );
    setEnd(
      <div className="relative w-full min-w-0 sm:max-w-xs">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
        <Input
          className="h-8 pl-8 pr-7 text-xs"
          placeholder={t.common.search}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        {search && (
          <Button
            ghost
            size="xs"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            onClick={() => setSearch("")}
            aria-label={t.common.clear}
          >
            <X />
          </Button>
        )}
      </div>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [enabledCount, loading, search, setAfterTitle, setEnd, skills.length, t]);

  const filteredToolsets = useMemo(() => {
    return toolsets.filter(
      (ts) =>
        !search ||
        ts.name.toLowerCase().includes(lowerSearch) ||
        ts.label.toLowerCase().includes(lowerSearch) ||
        ts.description.toLowerCase().includes(lowerSearch),
    );
  }, [toolsets, search, lowerSearch]);

  /* ---- Loading ---- */
  if (loading) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <PluginSlot name="skills:top" />
      <Toast toast={toast} />

      <div className="flex flex-col sm:flex-row sm:items-start gap-4">
        <aside aria-label={t.skills.title} className="sm:w-56 sm:shrink-0">
          <div className="sm:sticky sm:top-0">
            <div
              className={`
                flex flex-col
                border border-border bg-muted/20
              `}
            >
              <div className="hidden sm:flex items-center gap-2 px-3 py-2 border-b border-border">
                <Filter className="h-3 w-3 text-muted-foreground" />
                <span className="font-mondwest text-[0.65rem] tracking-[0.12em] uppercase text-muted-foreground">
                  {t.skills.filters}
                </span>
              </div>

              <div className="flex sm:flex-col gap-1 overflow-x-auto sm:overflow-x-visible scrollbar-none p-2">
                <PanelItem
                  icon={Package}
                  label={`${t.skills.all} (${skills.length})`}
                  active={view === "skills" && !isSearching}
                  onClick={() => {
                    setView("skills");
                    setActiveCategory(null);
                    setSearch("");
                  }}
                />
                <PanelItem
                  icon={Wrench}
                  label={`${t.skills.toolsets} (${toolsets.length})`}
                  active={view === "toolsets"}
                  onClick={() => {
                    setView("toolsets");
                    setSearch("");
                  }}
                />
              </div>

              {view === "skills" &&
                !isSearching &&
                allCategories.length > 0 && (
                  <div className="hidden sm:flex flex-col border-t border-border">
                    <div className="px-3 pt-2 pb-1 font-mondwest text-[0.6rem] tracking-[0.12em] uppercase text-muted-foreground/70">
                      {t.skills.categories}
                    </div>
                    <div className="flex flex-col p-2 pt-1 gap-px max-h-[calc(100vh-340px)] overflow-y-auto">
                      {allCategories.map(({ key, name, count }) => {
                        const isActive = activeCategory === key;

                        return (
                          <ListItem
                            key={key}
                            active={isActive}
                            onClick={() =>
                              setActiveCategory(isActive ? null : key)
                            }
                            className="rounded-sm px-2 py-1 text-[11px]"
                          >
                            <span className="flex-1 truncate">{name}</span>
                            <span
                              className={`text-[10px] tabular-nums ${
                                isActive
                                  ? "text-foreground/60"
                                  : "text-muted-foreground/50"
                              }`}
                            >
                              {count}
                            </span>
                          </ListItem>
                        );
                      })}
                    </div>
                  </div>
                )}
            </div>
          </div>
        </aside>

        <div className="flex-1 min-w-0">
          {isSearching ? (
            <Card>
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Search className="h-4 w-4" />
                    {t.skills.title}
                  </CardTitle>
                  <Badge tone="secondary" className="text-[10px]">
                    {t.skills.resultCount
                      .replace("{count}", String(searchMatchedSkills.length))
                      .replace(
                        "{s}",
                        searchMatchedSkills.length !== 1 ? "s" : "",
                      )}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {searchMatchedSkills.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    {t.skills.noSkillsMatch}
                  </p>
                ) : (
                  <div className="grid gap-1">
                    {searchMatchedSkills.map((skill) => (
                      <SkillRow
                        key={skill.name}
                        skill={skill}
                        toggling={togglingSkills.has(skill.name)}
                        onToggle={() => handleToggleSkill(skill)}
                        noDescriptionLabel={t.skills.noDescription}
                        locale={locale}
                      />
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          ) : view === "skills" ? (
            /* Skills list */
            <Card>
              <CardHeader className="py-3 px-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Package className="h-4 w-4" />
                    {activeCategory
                      ? prettyCategory(
                          activeCategory === "__none__" ? null : activeCategory,
                          t.common.general,
                        )
                      : t.skills.all}
                  </CardTitle>
                  <Badge tone="secondary" className="text-[10px]">
                    {t.skills.skillCount
                      .replace("{count}", String(activeSkills.length))
                      .replace("{s}", activeSkills.length !== 1 ? "s" : "")}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="px-4 pb-4">
                {activeSkills.length === 0 ? (
                  <p className="text-sm text-muted-foreground text-center py-8">
                    {skills.length === 0
                      ? t.skills.noSkills
                      : t.skills.noSkillsMatch}
                  </p>
                ) : (
                  <div className="grid gap-1">
                    {activeSkills.map((skill) => (
                      <SkillRow
                        key={skill.name}
                        skill={skill}
                        toggling={togglingSkills.has(skill.name)}
                        onToggle={() => handleToggleSkill(skill)}
                        noDescriptionLabel={t.skills.noDescription}
                        locale={locale}
                      />
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          ) : (
            /* Toolsets grid */
            <>
              {filteredToolsets.length === 0 ? (
                <Card>
                  <CardContent className="py-8 text-center text-sm text-muted-foreground">
                    {t.skills.noToolsetsMatch}
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {filteredToolsets.map((ts) => {
                    const TsIcon = toolsetIcon(ts.name);
                    const labelText =
                      ts.label.replace(/^[\p{Emoji}\s]+/u, "").trim() ||
                      ts.name;

                    return (
                      <Card key={ts.name} className="relative">
                        <CardContent className="py-4">
                          <div className="flex items-start gap-3">
                            <TsIcon className="h-5 w-5 text-muted-foreground shrink-0 mt-0.5" />
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-medium text-sm">
                                  {labelText}
                                </span>
                                <Badge
                                  tone={ts.enabled ? "success" : "outline"}
                                  className="text-[10px]"
                                >
                                  {ts.enabled
                                    ? t.common.active
                                    : t.common.inactive}
                                </Badge>
                              </div>
                              <p className="text-xs text-muted-foreground mb-2">
                                {ts.description}
                              </p>
                              {ts.enabled && !ts.configured && (
                                <p className="text-[10px] text-amber-300/80 mb-2">
                                  {t.skills.setupNeeded}
                                </p>
                              )}
                              {ts.tools.length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {ts.tools.map((tool) => (
                                    <Badge
                                      key={tool}
                                      tone="secondary"
                                      className="text-[10px] font-mono"
                                    >
                                      {tool}
                                    </Badge>
                                  ))}
                                </div>
                              )}
                              {ts.tools.length === 0 && (
                                <span className="text-[10px] text-muted-foreground/60">
                                  {ts.enabled
                                    ? t.skills.toolsetLabel.replace(
                                        "{name}",
                                        ts.name,
                                      )
                                    : t.skills.disabledForCli}
                                </span>
                              )}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>
      </div>
      <PluginSlot name="skills:bottom" />
    </div>
  );
}

function SkillRow({
  skill,
  toggling,
  onToggle,
  noDescriptionLabel,
  locale,
}: SkillRowProps) {
  return (
    <div className="group flex items-start gap-3 px-3 py-2.5 transition-colors hover:bg-muted/40">
      <div className="pt-0.5 shrink-0">
        <Switch
          checked={skill.enabled}
          onCheckedChange={onToggle}
          disabled={toggling}
        />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className={`font-mono-ui text-sm ${
              skill.enabled ? "text-foreground" : "text-muted-foreground"
            }`}
          >
            {skill.name}
          </span>
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
          {skillDescription(skill, locale) || noDescriptionLabel}
        </p>
      </div>
    </div>
  );
}

function PanelItem({ active, icon: Icon, label, onClick }: PanelItemProps) {
  return (
    <ListItem
      active={active}
      onClick={onClick}
      className={cn(
        "rounded-sm whitespace-nowrap px-2.5 py-1.5",
        "font-mondwest text-[0.7rem] tracking-[0.08em] uppercase",
        active && "bg-foreground/90 text-background hover:text-background",
      )}
    >
      <Icon className="h-3.5 w-3.5 shrink-0" />
      <span className="flex-1 truncate">{label}</span>
    </ListItem>
  );
}

interface PanelItemProps {
  active: boolean;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick: () => void;
}

interface SkillRowProps {
  noDescriptionLabel: string;
  locale: string;
  onToggle: () => void;
  skill: SkillInfo;
  toggling: boolean;
}
