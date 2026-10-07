---
title: 認証情報プール
description: プロバイダーごとに複数の API キーまたは OAuth トークンをプールし、自動ローテーションとレート制限からの回復を実現します。
sidebar_label: 認証情報プール
sidebar_position: 9
---

# 認証情報プール

認証情報プールを使うと、同じプロバイダーに対して複数のAPIキーまたはOAuthトークンを登録できます。あるキーがレート制限や課金クォータに達すると、Hermesは自動的に次の正常なキーへローテーションします — プロバイダーを切り替えることなくセッションを生かし続けます。

これは[フォールバックプロバイダー](./fallback-providers.md)とは異なります。フォールバックプロバイダーは完全に*別の*プロバイダーに切り替えます。認証情報プールは同一プロバイダー内でのローテーションであり、フォールバックプロバイダーはプロバイダーをまたいだフェイルオーバーです。プールが先に試され — すべてのプールキーが尽きた場合に*はじめて*フォールバックプロバイダーが起動します。

## 仕組み

```
あなたのリクエスト
  → プールからキーを選択（round_robin / least_used / fill_first / random）
  → プロバイダーへ送信
  → 429 レート制限?
      → 同じキーを1回リトライ（一時的な瞬断）
      → 2回目の 429 → 次のプールキーへローテーション
      → すべてのキーが尽きた → fallback_model（別のプロバイダー）
  → 402 課金エラー?
      → 即座に次のプールキーへローテーション（24時間のクールダウン）
  → 401 認証期限切れ?
      → トークンのリフレッシュを試みる（OAuth）
      → リフレッシュ失敗 → 次のプールキーへローテーション
  → 成功 → 通常どおり続行
```

## クイックスタート

`.env` にすでにAPIキーが設定されている場合、Hermesはそれを1キーのプールとして自動検出します。プールの恩恵を受けるには、さらにキーを追加します。

```bash
# 2つ目の OpenRouter キーを追加
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# 2つ目の Anthropic キーを追加
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# Anthropic の OAuth 認証情報を追加（Claude Max プラン + 追加の利用クレジットが必要）
hermes auth add anthropic --type oauth
# OAuth ログイン用にブラウザが開く
```

プールを確認します。

```bash
hermes auth list
```

出力:
```
openrouter (2 credentials):
  #1  OPENROUTER_API_KEY   api_key env:OPENROUTER_API_KEY ←
  #2  backup-key           api_key manual

anthropic (3 credentials):
  #1  hermes_pkce          oauth   hermes_pkce ←
  #2  claude_code          oauth   claude_code
  #3  ANTHROPIC_API_KEY    api_key env:ANTHROPIC_API_KEY
```

`←` は現在選択されている認証情報を示します。

## インタラクティブな管理

サブコマンドなしで `hermes auth` を実行すると、インタラクティブウィザードが起動します。

```bash
hermes auth
```

これはプールの完全なステータスを表示し、メニューを提示します。

```
What would you like to do?
  1. Add a credential
  2. Remove a credential
  3. Reset cooldowns for a provider
  4. Set rotation strategy for a provider
  5. Exit
```

APIキーとOAuthの両方をサポートするプロバイダー（Anthropic、Nous、Codex）では、追加フローでどちらのタイプかを尋ねられます。

```
anthropic supports both API keys and OAuth login.
  1. API key (paste a key from the provider dashboard)
  2. OAuth login (authenticate via browser)
Type [1/2]:
```

## CLIコマンド

| コマンド | 説明 |
|---------|-------------|
| `hermes auth` | インタラクティブなプール管理ウィザード |
| `hermes auth list` | すべてのプールと認証情報を表示 |
| `hermes auth list <provider>` | 特定のプロバイダーのプールを表示 |
| `hermes auth add <provider>` | 認証情報を追加（タイプとキーを尋ねる） |
| `hermes auth add <provider> --type api-key --api-key <key>` | APIキーを非インタラクティブに追加 |
| `hermes auth add <provider> --type oauth` | ブラウザログイン経由でOAuth認証情報を追加 |
| `hermes auth remove <provider> <index>` | 1始まりのインデックスで認証情報を削除 |
| `hermes auth reset <provider>` | すべてのクールダウン/枯渇ステータスをクリア |

## ローテーション戦略

`hermes auth` →「Set rotation strategy」または `config.yaml` で設定します。

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| 戦略 | 挙動 |
|----------|----------|
| `fill_first`（デフォルト） | 最初の正常なキーを枯渇するまで使い、その後次へ移る |
| `round_robin` | キーを均等に巡回し、選択するたびにローテーション |
| `least_used` | 常にリクエスト数が最も少ないキーを選ぶ |
| `random` | 正常なキーの中からランダムに選択 |

## エラーからの回復

プールはエラーごとに異なる対応をします。

| エラー | 挙動 | クールダウン |
|-------|----------|----------|
| **429 レート制限** | 同じキーを1回リトライ（一時的）。2回連続の429で次のキーへローテーション | 1時間 |
| **402 課金/クォータ** | 即座に次のキーへローテーション | 24時間 |
| **401 認証期限切れ** | まずOAuthトークンのリフレッシュを試み、失敗した場合のみローテーション | — |
| **すべてのキーが枯渇** | 設定されていれば `fallback_model` にフォールスルー | — |

`has_retried_429` フラグは、API呼び出しが成功するたびにリセットされるため、一時的な429が1回起きてもローテーションはトリガーされません。

## カスタムエンドポイントのプール

カスタムのOpenAI互換エンドポイント（Together.ai、RunPod、ローカルサーバー）は、config.yamlの `custom_providers` のエンドポイント名をキーとして、独自のプールを持ちます。

`hermes model` でカスタムエンドポイントをセットアップすると、「Together.ai」や「Local (localhost:8080)」のような名前が自動生成されます。この名前がプールのキーになります。

```bash
# hermes model でカスタムエンドポイントをセットアップした後:
hermes auth list
# 表示:
#   Together.ai (1 credential):
#     #1  config key    api_key config:Together.ai ←

# 同じエンドポイントに2つ目のキーを追加:
hermes auth add Together.ai --api-key sk-together-second-key
```

カスタムエンドポイントのプールは、`auth.json` の `credential_pool` 配下に `custom:` 接頭辞付きで保存されます。

```json
{
  "credential_pool": {
    "openrouter": [...],
    "custom:together.ai": [...]
  }
}
```

## 自動検出

Hermesは、複数のソースから認証情報を自動的に検出し、起動時にプールへシードします。

| ソース | 例 | 自動シード? |
|--------|---------|-------------|
| 環境変数 | `OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` | はい |
| OAuthトークン (auth.json) | Codexデバイスコード、Nousデバイスコード | はい |
| Claude Code認証情報 | `~/.claude/.credentials.json` | はい（Anthropic） |
| Hermes PKCE OAuth | `~/.hermes/auth.json` | はい（Anthropic） |
| カスタムエンドポイント設定 | config.yamlの `model.api_key` | はい（カスタムエンドポイント） |
| 手動エントリ | `hermes auth add` で追加 | auth.jsonに永続化 |

自動シードされたエントリは、プールが読み込まれるたびに更新されます — 環境変数を削除すると、そのプールエントリも自動的に間引かれます。手動エントリ（`hermes auth add` で追加したもの）は決して自動で間引かれません。

## 委譲とサブエージェントの共有

エージェントが `delegate_task` でサブエージェントを起動すると、親の認証情報プールが自動的に子と共有されます。

- **同じプロバイダー** — 子は親の完全なプールを受け取り、レート制限時のキーローテーションが可能になります
- **異なるプロバイダー** — 子はそのプロバイダー自身のプールを読み込みます（設定されている場合）
- **プールが未設定** — 子は継承した単一のAPIキーにフォールバックします

これは、追加設定なしで、サブエージェントが親と同じレート制限への耐性の恩恵を受けることを意味します。タスクごとの認証情報リースにより、子が同時にキーをローテーションする際に互いに競合しないようにします。

## スレッドセーフティ

認証情報プールは、すべての状態変更（`select()`、`mark_exhausted_and_rotate()`、`try_refresh_current()`、`mark_used()`）にスレッディングロックを使用します。これにより、ゲートウェイが複数のチャットセッションを同時に処理する際の安全な並行アクセスが保証されます。

## アーキテクチャ

完全なデータフロー図については、リポジトリ内の [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g) を参照してください。

認証情報プールは、プロバイダー解決レイヤーに統合されています。

1. **`agent/credential_pool.py`** — プールマネージャー: ストレージ、選択、ローテーション、クールダウン
2. **`hermes_cli/auth_commands.py`** — CLIコマンドとインタラクティブウィザード
3. **`hermes_cli/runtime_provider.py`** — プールを認識する認証情報解決
4. **`run_agent.py`** — エラーからの回復: 429/402/401 → プールローテーション → フォールバック

## ストレージ

プールの状態は、`~/.hermes/auth.json` の `credential_pool` キー配下に保存されます。

```json
{
  "version": 1,
  "credential_pool": {
    "openrouter": [
      {
        "id": "abc123",
        "label": "OPENROUTER_API_KEY",
        "auth_type": "api_key",
        "priority": 0,
        "source": "env:OPENROUTER_API_KEY",
        "access_token": "sk-or-v1-...",
        "last_status": "ok",
        "request_count": 142
      }
    ]
  },
}
```

戦略は（`auth.json` ではなく）`config.yaml` に保存されます。

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```
