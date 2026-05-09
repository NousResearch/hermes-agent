import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Switch } from "@nous-research/ui/ui/components/switch";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useI18n } from "@/i18n";

const JA_FIELD_LABELS: Record<string, string> = {
  "model": "既定モデル",
  "model_context_length": "コンテキスト長上書き",
  "fallback_providers": "フォールバック provider",
  "toolsets": "ツールセット",

  "agent.max_turns": "最大ターン数",
  "agent.gateway_timeout": "Gateway タイムアウト",
  "agent.restart_drain_timeout": "再起動ドレイン待機時間",
  "agent.api_max_retries": "API 最大リトライ回数",
  "agent.service_tier": "サービス tier",
  "agent.tool_use_enforcement": "ツール使用強制",
  "agent.gateway_timeout_warning": "Gateway タイムアウト警告",
  "agent.gateway_notify_interval": "Gateway 通知間隔",
  "agent.gateway_auto_continue_freshness": "Gateway 自動継続の鮮度",
  "agent.image_input_mode": "画像入力モード",
  "agent.disabled_toolsets": "無効化するツールセット",

  "compression.enabled": "有効化",
  "compression.threshold": "しきい値",
  "compression.target_ratio": "目標圧縮率",
  "compression.protect_last_n": "保護する末尾メッセージ数",
  "compression.hygiene_hard_message_limit": "Hygiene 強制メッセージ上限",

  "kanban.enabled": "有効化",
  "kanban.db_path": "DB パス",
  "kanban.failure_limit": "失敗上限",
};

const JA_FIELD_DESCRIPTIONS: Record<string, string> = {
  "model": "通常起動時に使う既定モデル。",
  "model_context_length": "モデルのコンテキスト長を手動指定する。0 はモデル情報から自動判定。",
  "fallback_providers": "主 provider が失敗した時に試す fallback provider の一覧。",
  "toolsets": "有効にする組み込みツールセット。",

  "agent.max_turns": "1回の会話で Agent が自律的に進める最大ターン数。暴走防止の上限。",
  "agent.gateway_timeout": "Gateway 経由の実行が完了するまで待つ最大秒数。長いタスク用。",
  "agent.restart_drain_timeout": "再起動時、実行中タスクの終了を待つ秒数。",
  "agent.api_max_retries": "LLM/API 呼び出し失敗時の最大リトライ回数。",
  "agent.service_tier": "OpenAI/Anthropic 系 API の service tier。通常は auto または default。",
  "agent.tool_use_enforcement": "ツール利用ルールの強さ。auto は通常の自動判定。",
  "agent.gateway_timeout_warning": "Gateway 実行中に警告を出し始める秒数。",
  "agent.gateway_notify_interval": "長時間実行中の進捗通知間隔。",
  "agent.gateway_auto_continue_freshness": "自動継続が有効とみなす直近状態の鮮度。",
  "agent.image_input_mode": "画像入力をモデルへ渡す方法。",
  "agent.disabled_toolsets": "この環境で無効化するツールセット名の一覧。",

  "compression.enabled": "会話履歴の自動圧縮を有効にする。",
  "compression.threshold": "圧縮を開始するしきい値。小さいほど早めに圧縮する。",
  "compression.target_ratio": "圧縮後に目指す履歴サイズの比率。",
  "compression.protect_last_n": "直近 N 件のメッセージを圧縮対象から守る。",
  "compression.hygiene_hard_message_limit": "履歴 hygiene が強制的に働く最大メッセージ数。",

  "kanban.enabled": "Kanban タスク管理機能を有効にする。",
  "kanban.db_path": "Kanban SQLite DB の保存先。空なら Hermes home の既定値を使う。",
  "kanban.failure_limit": "同じタスクが連続失敗した時に自動停止する上限。",
};

const JA_CATEGORY_LABELS: Record<string, string> = {
  general: "一般",
  agent: "エージェント",
  terminal: "ターミナル",
  display: "表示",
  delegation: "委任",
  memory: "メモリ",
  compression: "圧縮",
  security: "セキュリティ",
  browser: "ブラウザ",
  voice: "音声",
  tts: "音声合成",
  stt: "音声認識",
  logging: "ログ",
  discord: "Discord",
  auxiliary: "補助",
  bedrock: "Bedrock",
  curator: "Curator",
  kanban: "Kanban",
  model_catalog: "Model catalog",
  openrouter: "OpenRouter",
  sessions: "Sessions",
  tool_loop_guardrails: "ツールループ防止",
  tool_output: "ツール出力",
  updates: "更新",
};

const JA_TERM_LABELS: Record<string, string> = {
  api: "API",
  auto: "自動",
  browser: "ブラウザ",
  cache: "キャッシュ",
  catalog: "カタログ",
  channels: "チャンネル",
  context: "コンテキスト",
  disabled: "無効化",
  enabled: "有効化",
  enforcement: "強制",
  gateway: "Gateway",
  hard: "強制",
  hygiene: "Hygiene",
  image: "画像",
  input: "入力",
  interval: "間隔",
  limit: "上限",
  logging: "ログ",
  max: "最大",
  memory: "メモリ",
  message: "メッセージ",
  mode: "モード",
  notify: "通知",
  output: "出力",
  path: "パス",
  profiles: "プロファイル",
  protect: "保護",
  ratio: "比率",
  retries: "リトライ回数",
  service: "サービス",
  sessions: "セッション",
  tier: "tier",
  timeout: "タイムアウト",
  tool: "ツール",
  toolsets: "ツールセット",
  turns: "ターン数",
  warning: "警告",
};

function titleCase(raw: string): string {
  return raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function localizeKeySegment(raw: string): string {
  const words = raw.split("_").filter(Boolean);
  const translated = words.map((word) => JA_TERM_LABELS[word.toLowerCase()] ?? titleCase(word));
  return translated.join("");
}

function localizeFieldLabel(schemaKey: string, locale: string): string {
  if (locale === "ja") {
    if (JA_FIELD_LABELS[schemaKey]) return JA_FIELD_LABELS[schemaKey];
    const rawLabel = schemaKey.split(".").pop() ?? schemaKey;
    return localizeKeySegment(rawLabel);
  }
  const rawLabel = schemaKey.split(".").pop() ?? schemaKey;
  return titleCase(rawLabel);
}

function localizeFieldDescription(
  schema: Record<string, unknown>,
  schemaKey: string,
  locale: string,
): string {
  if (locale === "ja") {
    if (JA_FIELD_DESCRIPTIONS[schemaKey]) return JA_FIELD_DESCRIPTIONS[schemaKey];
    const category = String(schema.category ?? schemaKey.split(".")[0] ?? "general");
    const catLabel = JA_CATEGORY_LABELS[category] ?? localizeKeySegment(category);
    return `${catLabel} → ${localizeFieldLabel(schemaKey, locale)}`;
  }
  return schema.description ? String(schema.description) : "";
}

function FieldHint({
  schema,
  schemaKey,
  locale,
}: {
  schema: Record<string, unknown>;
  schemaKey: string;
  locale: string;
}) {
  const keyPath = schemaKey.includes(".") ? schemaKey : "";
  const description = localizeFieldDescription(schema, schemaKey, locale);

  if (!keyPath && !description) return null;

  return (
    <div className="flex flex-col gap-0.5">
      {keyPath && <span className="text-[10px] font-mono text-muted-foreground/50">{keyPath}</span>}
      {description && <span className="text-xs text-muted-foreground/70">{description}</span>}
    </div>
  );
}

export function AutoField({
  schemaKey,
  schema,
  value,
  onChange,
}: AutoFieldProps) {
  const { locale } = useI18n();
  const label = localizeFieldLabel(schemaKey, locale);

  if (schema.type === "boolean") {
    return (
      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-col gap-0.5">
          <Label className="text-sm">{label}</Label>
          <FieldHint schema={schema} schemaKey={schemaKey} locale={locale} />
        </div>
        <Switch checked={!!value} onCheckedChange={onChange} />
      </div>
    );
  }

  if (schema.type === "select") {
    const options = (schema.options as string[]) ?? [];
    return (
      <div className="grid gap-1.5">
        <Label className="text-sm">{label}</Label>
        <FieldHint schema={schema} schemaKey={schemaKey} locale={locale} />
        <Select value={String(value ?? "")} onValueChange={(v) => onChange(v)}>
          {options.map((opt) => (
            <SelectOption key={opt} value={opt}>
              {opt || "(none)"}
            </SelectOption>
          ))}
        </Select>
      </div>
    );
  }

  if (schema.type === "number") {
    return (
      <div className="grid gap-1.5">
        <Label className="text-sm">{label}</Label>
        <FieldHint schema={schema} schemaKey={schemaKey} locale={locale} />
        <Input
          type="number"
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") {
              onChange(0);
              return;
            }
            const n = Number(raw);
            if (!Number.isNaN(n)) {
              onChange(n);
            }
          }}
        />
      </div>
    );
  }

  if (schema.type === "text") {
    return (
      <div className="grid gap-1.5">
        <Label className="text-sm">{label}</Label>
        <FieldHint schema={schema} schemaKey={schemaKey} locale={locale} />
        <textarea
          className="flex min-h-[80px] w-full border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    );
  }

  if (schema.type === "list") {
    return (
      <div className="grid gap-1.5">
        <Label className="text-sm">{label}</Label>
        <FieldHint schema={schema} schemaKey={schemaKey} locale={locale} />
        <Input
          value={Array.isArray(value) ? value.join(", ") : String(value ?? "")}
          onChange={(e) =>
            onChange(
              e.target.value
                .split(",")
                .map((s) => s.trim())
                .filter(Boolean),
            )
          }
          placeholder="comma-separated values"
        />
      </div>
    );
  }

  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    const obj = value as Record<string, unknown>;
    return (
      <div className="grid gap-3 border border-border p-3">
        <Label className="text-xs font-medium">{label}</Label>
        <FieldHint schema={schema} schemaKey={schemaKey} locale={locale} />
        {Object.entries(obj).map(([subKey, subVal]) => (
          <div key={subKey} className="grid gap-1">
            <Label className="text-xs text-muted-foreground">{subKey}</Label>
            <Input
              value={String(subVal ?? "")}
              onChange={(e) => onChange({ ...obj, [subKey]: e.target.value })}
              className="text-xs"
            />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid gap-1.5">
      <Label className="text-sm">{label}</Label>
      <FieldHint schema={schema} schemaKey={schemaKey} locale={locale} />
      <Input value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

interface AutoFieldProps {
  schemaKey: string;
  schema: Record<string, unknown>;
  value: unknown;
  onChange: (v: unknown) => void;
}
