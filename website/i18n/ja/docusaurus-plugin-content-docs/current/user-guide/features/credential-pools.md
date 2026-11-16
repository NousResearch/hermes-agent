---
title: 認証情報プール
description: プロバイダーごとに複数の API キーまたは OAuth トークンをプールし、自動ローテーションとレート制限からの回復を実現します。
sidebar_label: 認証情報プール
sidebar_position: 9
---

# 認証情報プール

認証情報プールを使うと、同じプロバイダーに対して複数のAPIキーまたはOAuthトークンを登録できます。あるキーがレート制限や課金クォータに達すると、Hermesは自動的に次の正常なキーへローテーションします — プロバイダーを切り替えることなくセッションを生かし続けます。

これは[フォールバックプロバイダー](./fallback-providers.md)とは異なります。フォールバックプロバイダーは完全に*別の*プロバイダーに切り替えます。認証情報プールは同一プロバイダー内でのローテーションであり、フォールバックプロバイダーはプロバイダーをまたいだフェイルオーバーです。プールが先に試され — すべてのプールキーが尽きた場合に*はじめて*フォールバックプロバイダーが起動します。

:::warning キーのローテーションはプロンプトキャッシュをリセットします
プロバイダー側のプロンプトキャッシュ（Anthropic、OpenAI、OpenRouter）は、リクエストを行ったアカウント/APIキーにスコープされます。セッションの途中でプールが別のキーにローテーションすると、新しいキーには会話のキャッシュ済みプレフィックスがないため、次のリクエストは割引なしの入力価格で履歴全体を読み直します。後で元のキーに戻すときも、以前のキーのキャッシュTTLがまだ生きていない限り、再び全体の読み直しになります。ローテーションはセッションを継続させるためのものであり、それが目的ですが、長い会話ではローテーションのたびにコンテキスト全体を定価で1回処理するコストがかかります。
:::

:::tip
認証情報プールは主にAPIキー型のプロバイダー（OpenRouter、Anthropic）向けです。[Nous Portal](/docs/integrations/nous-portal) のOAuth 1つで300以上のモデルをカバーできるため、Portalを使っているほとんどのユーザーにはプールは不要です。
:::

## 仕組み

```
Your request
  → Pick key from pool (round_robin / least_used / fill_first / random)
  → Send to provider
  → 429 rate limit?
      → Plan/usage limit reached (e.g. ChatGPT/Codex "usage limit reached")?
          → Rotate to next pool key immediately (no retry — the cap won't clear on retry)
      → Generic / transient 429?
          → Retry same key once (transient blip)
          → Second 429 → rotate to next pool key
      → All keys exhausted → fallback_model (different provider)
  → 402 billing error?
      → Immediately rotate to next pool key (1h cooldown)
  → 401 auth expired?
      → Try refreshing the token (OAuth)
      → Refresh failed → rotate to next pool key
  → 400 "model is not supported when using Codex with a ChatGPT account"?
      → Bench this key for that model only, rotate to the next key (other models stay usable)
      → Every key rejects the model → fallback_model; the model is skipped for the session
  → HTTP 200 but `response.status: failed` (ChatGPT/Codex reports usage limits this way)?
      → Same rules as above, keyed on the embedded error code/message:
        quota/billing/auth → pool rotation first, provider fallback only once the pool is exhausted;
        content-policy and other failures → no rotation
  → Success → continue normally
```

## クイックスタート

`.env` にすでにAPIキーが設定されている場合、Hermesはそれを1キーのプールとして自動検出します。プールの恩恵を受けるには、さらにキーを追加します。

```bash
# Add a second OpenRouter key
hermes auth add openrouter --api-key sk-or-v1-your-second-key

# ...or let a browser login mint one (OpenRouter OAuth PKCE; stored as a plain API key)
hermes auth add openrouter --type oauth

# Add a second Anthropic key
hermes auth add anthropic --type api-key --api-key sk-ant-api03-your-second-key

# Add an Anthropic OAuth credential (requires Claude Max plan + extra usage credits)
hermes auth add anthropic --type oauth
# Opens browser for OAuth login
```

プールを確認します。

```bash
hermes auth list
```

出力:
```
openrouter (2 credentials):
  #1  OPENROUTER_API_KEY   api_key id=ab12cd34 priority=0 env:OPENROUTER_API_KEY ←
  #2  backup-key           api_key id=ef56gh78 priority=1 manual

anthropic (3 credentials):
  #1  hermes_pkce          oauth   id=ab12cd34 priority=0 hermes_pkce ←
  #2  claude_code          oauth   id=cd34ef56 priority=1 claude_code
  #3  ANTHROPIC_API_KEY    api_key id=ef56gh78 priority=2 env:ANTHROPIC_API_KEY
```

`←` は現在選択されている認証情報を示します。`id=` は、ラベルがあいまいな場合に
`hermes auth remove <provider> <target>` で指定できるエントリIDで、`priority=` は
`fill_first` 戦略でプールが認証情報を試す順序です。

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

`hermes auth add openai-codex` でログインするたびにそれぞれ独立したプールエントリになりますが、独立してローテーションできるのは**異なる** OpenAI アカウントだけです。同じアカウントで2回ログインすると、上流では1つのトークンファミリーを共有するため、OpenAI は古い方を失効させ、2つ目のエントリはクォータを増やしません。Hermes は追加時に警告します（`warning: this login is the same OpenAI account as openai-codex credential #N`）。別のアカウントでログインするか、1つだけを残してください。

## CLIコマンド

| コマンド | 説明 |
|---------|-------------|
| `hermes auth` | インタラクティブなプール管理ウィザード |
| `hermes auth list` | すべてのプールと認証情報を表示 |
| `hermes auth list <provider>` | 特定のプロバイダーのプールを表示 |
| `hermes auth add <provider>` | 認証情報を追加（タイプとキーを尋ねる） |
| `hermes auth add <provider> --type api-key --api-key <key>` | APIキーを非インタラクティブに追加 |
| `hermes auth add <provider> --type oauth` | ブラウザログイン経由でOAuth認証情報を追加 |
| `hermes auth add openai-codex --browser` | Codex のみ: デフォルトのデバイスコードの代わりに、`localhost:1455` でのブラウザ認可コード + PKCE フローでサインインします（デバイスコードグラントを無効にしている組織向け）。ポートが使用中の場合はデバイスコードにフォールバックします。`auth.codex_login_flow: browser` ですべての Codex ログインのデフォルトにできます |
| `hermes auth add <provider> --priority 0` | 認証情報を追加し、`fill_first` の順序で先頭に配置 |
| `hermes auth priority <provider> <target> <n>` | 認証情報を優先度 `n` に移動（0 = 最初に試行）。残りは番号が振り直されます |
| `hermes auth remove <provider> <index>` | 1始まりのインデックスで認証情報を削除 |
| `hermes auth reset <provider>` | すべてのクールダウン/枯渇ステータスをクリア（実行中のセッションにも適用されます: 稼働中のゲートウェイやチャットは、古いクールダウンを書き戻す代わりに次のリクエストでリセットを取り込みます） |
| `hermes auth reset <provider> <target>` | インデックス、ID、ラベルで指定した1つの認証情報のクールダウンをクリア |
| `hermes auth refresh <provider> [target]` | 1つのOAuth認証情報のトークンをリフレッシュしてローテーションに戻す（グラントが有効であることを確認します。次のリクエストでクォータが再確認されます） |

Nous の場合、`auth refresh` はログインの `device_code` シングルトンのみをサポートします。
独立した Nous プールアカウントはリフレッシュ前に拒否され、そのトークンと
クールダウンは保持されます。シングルトンを更新するには `hermes auth add nous --type oauth`
で再認証してください。これは独立したアカウントはリフレッシュしません。その他の
プロバイダーは、既存のソース固有のリフレッシュサポートを維持します。

## ローテーション戦略

優先度の位置は0始まりで、プールの両端にクランプされます。表示されるターゲットは
1始まりのインデックス、エントリID、またはあいまいでない完全一致のラベルです。`auth add --priority`
は、再認証によって更新された既存エントリの配置にも使われます。Anthropic は手動の
認証情報をシードされた認証情報より前に保つため、このルールで位置が変わった場合、コマンドは実際の
位置を報告します。他の戦略は優先度を上書きすることがあり、
並べ替えても実行中のセッションがすでに保持している認証情報は再バインドされません。

プールでの選択が成功するたびに、戦略にかかわらず `request_count` が増加します。
リフレッシュのみの参照やピーク（peek）はカウントされません。ステータスの読み取り（`hermes doctor`、`/model`
ピッカーのプロバイダー行、ダッシュボードの認証カード）はピークです。これらはプールの認証情報を
リフレッシュ、ローテーション、待機させることがないため、ピッカーを開いている間にトークンエンドポイントが一時的に不調でも、
まだリクエストを処理しているプロバイダーが隠れることはありません。これらは選択カウンターであり、
課金合計や推論リクエストの総数ではありません: キャッシュされた認証情報は
複数のリクエストを処理できます。カウントは次の既存のプール書き込み
（例: ローテーション、枯渇、リフレッシュ、管理上の変更）までメモリ上に保持されます。選択ごとに
ディスク書き込みが増えることはありません。

`hermes auth` →「Set rotation strategy」または `config.yaml` で設定します。

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```

| 戦略 | 挙動 |
|----------|----------|
| `fill_first`（デフォルト） | 最初の正常なキーを枯渇するまで使い、その後次へ移る。順序は各認証情報の `priority`（`hermes auth priority` で変更可能） |
| `round_robin` | キーを均等に巡回し、選択するたびにローテーション |
| `least_used` | 常にリクエスト数が最も少ないキーを選ぶ |
| `random` | 正常なキーの中からランダムに選択 |

### 正常な認証情報の降格 {#demoting-a-healthy-credential}

プールが認証情報を待機させるのは、プロバイダーがそれを拒否した*後*（429/402/401）だけです。他の場所で
積極的に使っている認証情報 — たとえば週次ウィンドウを対話的な作業のために残しておきたい Codex ログイン — を、
枯渇する*前*にゲートウェイの最初の選択から外すには、削除する代わりに
`fill_first` の順序の最後に移動します。

```bash
hermes auth list openai-codex                 # find the index, id or label
hermes auth priority openai-codex 1 99        # 1-based index, entry id, or exact label; large n = last
hermes auth priority openai-codex work-seat 0 # ...and back to the front later
```

`hermes auth priority <provider> <target> <priority>` は1つの認証情報を並べ替えて他の認証情報の番号を
振り直し、新しい順序を `auth.json` に永続化します。降格されたエントリは正常なままです: 枯渇して
いないため、Codex のクォータリセットプローブに触れられることはなく、`hermes auth reset` でクリアするものも
ありません。タイマーでリフレッシュされることもなく、それより前のすべての認証情報が待機状態になれば引き続き使われます。
すでに認証情報を保持しているセッションは、ローテーションするまでそれを使い続けます。新しいセッション（および次回の
ゲートウェイ起動）は新しい順序に従います。

## エラーからの回復

プールはエラーごとに異なる対応をします。

| エラー | 挙動 | クールダウン |
|-------|----------|----------|
| **429 レート制限** | 同じキーを1回リトライ（一時的）。2回連続の429で次のキーへローテーション | 1時間 |
| **402 課金/クォータ** | 即座に次のキーへローテーション | 1時間 |
| **401 認証期限切れ** | まずOAuthトークンのリフレッシュを試み、失敗した場合のみローテーション | 5分 |
| **400 Codex モデル利用権**（`The '<model>' model is not supported when using Codex with a ChatGPT account.`） | 拒否されたモデルについてのみこのキーを待機させ、次のキーへローテーション。他のモデルは引き続きこのキーを使用。その他の400ではローテーションしない | `hermes auth reset` まで（モデルごと。利用権はプランの属性であり、時間枠ではありません） |
| **すべてのキーが枯渇** | 設定されていれば `fallback_model` にフォールスルー | — |

プロバイダーが提供する `reset_at` タイムスタンプは、これらのデフォルトのクールダウンより優先されます。

`has_retried_429` フラグは、API呼び出しが成功するたびにリセットされるため、一時的な429が1回起きてもローテーションはトリガーされません。

**クォータによる待機は稼働中のセッションでも一時的です。** 429/402 によってセッションがある認証情報から
ローテーションした場合、そのセッションは各ターンの開始時に、待機中の認証情報がローテーションに戻ったかを確認し、
クールダウンが解除されしだいそれに戻ります — 新しいセッションが行うのと同じ選択です。
そのため、長時間稼働するチャット（ゲートウェイはエージェントをキャッシュし続けます）は、ウィンドウが再び開けば
サブスクリプションの席に戻り、残りの期間ずっと従量課金のフォールバックに請求し続けることはありません。`401` による待機
ではこれは発生しません。明示的な `/model` の切り替えは、保留中の切り戻しをキャンセルします。

**Anthropic の 429 はモデルごとです。** Anthropic はレート制限をモデルごとに適用するため、ある Claude モデルに対する
一般的な429は、その認証情報を*そのモデルについてのみ*クールダウンさせます — 同じキーは他のすべての
Claude モデルを引き続き処理し、`ANTHROPIC_API_KEY` や借用した Claude Code トークンも同じモデルごとの
クールダウンに従います。課金（`402`、利用上限）と認証（`401`）の失敗は、引き続き認証情報全体を待機させます。

**無効になった OAuth ログインは待機ではなく報告されます。** リフレッシュトークンが恒久的に拒否された場合
（`invalid_grant`、`invalid_token`、`refresh_token_reused` — トークンが失効したか、同じログインを保持する別のプログラムが
先にローテーションした場合 — または Nous の場合、プロファイルにリフレッシュ用の Portal ログインやトークン
ペアがない場合）、プールはエントリ名と修復コマンド（`hermes auth add <provider>`）を示す WARNING を1回ログに出力し、
認証情報はローテーションから外れます — `dead` としてマークされるか、プールがクリアしたばかりのトークンファイルを
ミラーしていただけの場合は削除されます — 再度サインインするまでこの状態が続きます。これは Anthropic、Codex、xAI、
Nous の OAuth ログインすべてに当てはまります。無効になった認証情報がタイマーでローテーションに戻ることはないため、失われたログインは
1時間ごとに静かに失敗し続けるのではなく、ログに1回だけ現れます。

**常時稼働するホームのすべての Codex ログインが無効になったら、採用を待たずに再度サインインしてください。** Hermes が
Codex CLI の `~/.codex/auth.json` を自動的にインポートするのは、*すでに持っているログインを修復する*場合だけです
— つまり、自身による `openai-codex` エントリのリフレッシュが失敗し、かつ `auth.adopt_external_logins` が有効な場合です
（[借用した CLI ログイン](../security.md#borrowed-cli-logins)を参照）。Codex エントリが
すべて `dead`（または削除済み）のプールには修復するものが残っていないため、ヘッドレスのゲートウェイや cron プロファイルは、
*その*ホームで `hermes auth add openai-codex` を実行するまで Codex の認証情報がないままになります
（名前付きプロファイルの場合は `hermes -p <profile> auth add openai-codex`）。このコマンドは Codex CLI のインポートを
対話的に提案します。1つのログインを共有すべきプロファイルは、使い捨てのリフレッシュトークンのコピーを
それぞれ持つ代わりに、同じ `HERMES_HOME` を指すようにできます。

**クールダウン中または無効な認証情報は、空のインストールとは異なります。** 設定済みのプロファイルが、唯一の
認証情報が待機中または隔離されている状態で CLI を起動すると、起動時に失敗内容と、待機の場合は
残りのクールダウン（無効な場合は `hermes auth add <provider>` による再ログイン）が表示されます — 初回起動時の
「No inference provider is configured yet」ウィザードが提示されるのは、リゾルバーが何も設定されていないと
判断した場合だけです。

## カスタムエンドポイントのプール

カスタムのOpenAI互換エンドポイント（Together.ai、RunPod、ローカルサーバー）は、config.yamlの `providers:` 辞書（または自動移行されるレガシーの `custom_providers` リスト）のエンドポイント名をキーとして、独自のプールを持ちます。

`hermes model` でカスタムエンドポイントをセットアップすると、「Together.ai」や「Local (localhost:8080)」のような名前が自動生成されます。この名前がプールのキーになります。

```bash
# After setting up a custom endpoint via hermes model:
hermes auth list
# Shows:
#   Together.ai (1 credential):
#     #1  config key    api_key config:Together.ai ←

# Add a second key for the same endpoint:
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

カスタムエンドポイント用に追加されたキー（`hermes auth add`、またはダッシュボードの認証情報プール）は、入力されたときのエンドポイントのアドレスを記録し、そのオリジン（スキーム、ホスト、ポート）にのみ送信されます。エンドポイントのアドレスを変更すると、そのプールのキーは置き去りになります: Desktop とダッシュボードのエディターは、それらを再利用するのではなく、新しいアドレス用のキー（または明示的に空欄にしてキーなし）を尋ねます。アドレスが記録されるようになる前に追加されたキーは、最初に使われたときにエンドポイントのアドレスにバインドされます。組み込みプロバイダーのキー（たとえば `providers.deepseek` エントリに対する `DEEPSEEK_API_KEY`）は、エントリがその変数を `key_env` として指定していない限り（例: `key_env: DEEPSEEK_API_KEY`）、そのプロバイダー自身のアドレス上でのみエントリに使われます。それらを使わなかった場合、Hermes はエントリごとに1回警告をログに出力します。これは委譲されたサブエージェント（`delegation.provider: deepseek`）や、そのエントリ上の補助タスクにも適用されます。アドレスなしで組み込みプロバイダーのプールに追加されたキーは、そのプロバイダー自身のアドレスに属し、同名のエントリのものになることはありません。オリジンはスキーム、ホスト、ポートのみです: 同じホスト上でパスだけが異なる2つのエンドポイント（`https://gw.example/tenantA/v1` と `https://gw.example/tenantB/v1`）は1つのオリジンとみなされるため、それらのプールのキーは互いに使われ得ます。

## 自動検出 {#auto-discovery}

Hermesは、複数のソースから認証情報を自動的に検出し、起動時にプールへシードします。

| ソース | 例 | 自動シード? |
|--------|---------|-------------|
| 環境変数 | `OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY` | はい |
| 番号付きの兄弟環境変数 | `OPENROUTER_API_KEY_2`、`OPENROUTER_API_KEY_3`、… | はい（下記参照） |
| OAuthトークン (auth.json) | Codexデバイスコード、Nousデバイスコード | はい |
| Claude Code認証情報 | `~/.claude/.credentials.json` | はい（Anthropic） |
| Hermes PKCE OAuth | `~/.hermes/auth.json` | はい（Anthropic） |
| カスタムエンドポイント設定 | config.yamlの `model.api_key` | はい（カスタムエンドポイント） |
| 手動エントリ | `hermes auth add` で追加 | auth.jsonに永続化 |

自動シードされたエントリは、プールが読み込まれるたびに更新されます — 環境変数を削除すると、そのプールエントリも自動的に間引かれます。手動エントリ（`hermes auth add` で追加したもの）は決して自動で間引かれません。

### 環境変数からの複数キー

`auth.json` にキーを一切保存せずに、1つのプロバイダーに複数のキーを使いたい場合は、番号を付けます。`NVIDIA_API_KEY` に加えて、シェル、`.env`、またはシークレットマネージャー（Bitwarden Secrets、Vault など）で `NVIDIA_API_KEY_2`、`NVIDIA_API_KEY_3`、… を設定すると、次回の読み込み時にそれぞれが独立したプールエントリになります — コマンドも設定も不要です。検出は最初に欠けた番号で止まるため、`_4` がないのに `_5` があっても無視されます。ローテーションさせるには `credential_pool_strategies` と組み合わせます。

```yaml
credential_pool_strategies:
  nvidia: round_robin
```

借用したランタイムシークレット（たとえば環境変数、Bitwarden/Vault/keyring/systemd の参照、カスタム設定値）は、`auth.json` の境界では参照のみとして扱われます。Hermes は解決した値を現在の実行中にメモリ上で使用できますが、永続化するのはソース参照、ラベル、ステータス、リクエストカウンター、不可逆なフィンガープリントなどのメタデータだけです。手動エントリと Hermes が所有する OAuth/デバイスコードの状態は、リフレッシュに必要な永続トークンを保持します。

## 委譲とサブエージェントの共有

エージェントが `delegate_task` でサブエージェントを起動すると、親の認証情報プールが自動的に子と共有されます。

- **同じプロバイダー** — 子は親の完全なプールを受け取り、レート制限時のキーローテーションが可能になります
- **異なるプロバイダー** — 子はそのプロバイダー自身のプールを読み込みます（設定されている場合）
- **プールが未設定** — 子は継承した単一のAPIキーにフォールバックします

これは、追加設定なしで、サブエージェントが親と同じレート制限への耐性の恩恵を受けることを意味します。タスクごとの認証情報リースにより、子が同時にキーをローテーションする際に互いに競合しないようにします。

## スレッドセーフティ

認証情報プールは、すべての状態変更（`select()`、`mark_exhausted_and_rotate()`、`try_refresh_current()`、`mark_used()`）にスレッディングロックを使用します。これにより、ゲートウェイが複数のチャットセッションを同時に処理する際の安全な並行アクセスが保証されます。

プロセス間（多数のサブエージェント、ゲートウェイと CLI、cron ジョブ）では、OAuth のリフレッシュは `auth.json` のファイルロックによって直列化されます。1つの共有 OAuth グラントが多数の並行プロセスのもとで期限切れになった場合、リフレッシュを行うのはちょうど1つのプロセスだけです。他のプロセスは、ディスク上のトークンが失敗したトークンと一致しなくなったことを検出し、使い捨てのリフレッシュトークンを再びローテーションする代わりにそれを採用します。ロック競争に負けたプロセスはエントリを正常なまま保ってリトライします — ロックの競合が認証情報の失敗として記録されることはありません。プールの古いコピーを保持しているセッションやプロセスが、その後に別のプロセスがローテーションしたペアの上に、そのコピーのトークンを書き戻すことはありません。ディスク上の新しいペアを保持し、それを採用します。

## アーキテクチャ

完全なデータフロー図については、リポジトリ内の [`docs/credential-pool-flow.excalidraw`](https://excalidraw.com/#json=2Ycqhqpi6f12E_3ITyiwh,c7u9jSt5BwrmiVzHGbm87g) を参照してください。

認証情報プールは、プロバイダー解決レイヤーに統合されています。

1. **`agent/credential_pool.py`** — プールマネージャー: ストレージ、選択、ローテーション、クールダウン。**`agent/credential_pool_admin.py`** はロック下でのターゲット解決、リセット、追加、削除、優先度の変更を担当し、**`agent/credential_pool_model_cooldowns.py`** は Anthropic のモデルごとの429クールダウンを担当します
2. **`hermes_cli/auth_commands.py`** — CLIコマンドとインタラクティブウィザード
3. **`hermes_cli/runtime_provider.py`** — プールを認識する認証情報解決
4. **`agent/turn_api_error.py`** — エラーからの回復: 429/402/401 → プールローテーション → フォールバック

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
        "secret_source": "bitwarden",
        "secret_fingerprint": "sha256:12ab34cd56ef7890",
        "last_status": "ok",
        "request_count": 142
      }
    ],
    "anthropic": [
      {
        "id": "manual1",
        "label": "personal-api-key",
        "auth_type": "api_key",
        "priority": 0,
        "source": "manual",
        "access_token": "sk-ant-api03-..."
      }
    ]
  }
}
```

上の OpenRouter エントリは外部ソースから借用されたものなので、生のキーは `auth.json` に保存されません。手動の Anthropic エントリは意図的に Hermes の認証情報ストアに追加されたものなので、そのトークンは永続化可能なままです。

`env:` の行は読み込みのたびに環境から再構成され、変数名は Hermes がそのプロバイダー用に宣言しているものである必要はありません: 番号付きの兄弟変数（`OPENROUTER_API_KEY_2`、[自動検出](#auto-discovery)を参照）は自動的にここに現れ、他の任意の変数を指す手書きの行も同じ方法で埋められます。その際、シークレットが `auth.json` に書き込まれることはありません。

戦略は（`auth.json` ではなく）`config.yaml` に保存されます。

```yaml
credential_pool_strategies:
  openrouter: round_robin
  anthropic: least_used
```
