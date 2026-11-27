---
title: Home Assistant
description: プラグインカタログの Home Assistant プラグインを通じて Hermes Agent でスマートホームを制御します。
sidebar_label: Home Assistant
sidebar_position: 5
---

# Home Assistant 連携

Hermes Agent は、[プラグインカタログ](../features/plugins.md)の公式 **`homeassistant` プラグイン**を通じて [Home Assistant](https://www.home-assistant.io/) と連携します。このプラグインは Nous Research が [NousResearch/hermes-official-plugins](https://github.com/NousResearch/hermes-official-plugins/tree/main/homeassistant) でメンテナンスしており、Hermes コアの一部ではありません。プラグインは次の2つを提供します。

1. **ゲートウェイプラットフォーム** — WebSocket を介してリアルタイムの状態変化を購読し、イベントに応答します
2. **スマートホームツール** — REST API を介してデバイスを照会・制御する、LLMから呼び出し可能な4つのツール（`homeassistant` ツールセット）

## インストール

```bash
hermes plugins install homeassistant
```

プラグインはプロファイルごとにインストールされます。別のプロファイルで Home Assistant を使うには、そちらにもインストールします。

```bash
hermes -p <profile> plugins install homeassistant
```

プラグインは独自の Python 依存関係（`aiohttp`）を宣言しているため、pip のエクストラをインストールする必要はありません。旧来の `hermes-agent[homeassistant]` エクストラは削除されました。

:::info Home Assistant を同梱していたリリースからのアップグレード
何もする必要はありません。すでに Home Assistant を使っていたすべてのプロファイル — `.env` に `HASS_TOKEN` がある、`config.yaml` で `platforms.homeassistant` が有効（または `token` が指定されている）、あるいは `platform_toolsets` に `homeassistant` ツールセットが記載されている — には、`hermes update` によってカタログからプラグインが自動的にインストールされます（そのインストールを共有するすべてのプロファイルが対象）。このステップを実行できなかった場合、Hermes はそのプロファイルの初回起動時（エージェントまたはゲートウェイの起動時。`security.allow_lazy_installs` に従います）にインストールします。失敗した場合（オフライン、カタログに到達できないなど）、起動時のリトライは最大で1時間に1回です。`hermes update` は常にリトライします。結果はターミナル、Desktop アプリ、チャットで報告されます。これはプロファイルごとに1回だけ行われます: 後でプラグインを削除した場合（`hermes plugins remove homeassistant`）、削除されたままになります。

設定は変更なしで引き継がれます: 同じ `HASS_TOKEN` / `HASS_URL` 変数、同じ `homeassistant` プラットフォーム名と `platforms.homeassistant` のキー、同じ `homeassistant` ツールセットとツール名、同じ cron の `deliver: homeassistant:<notify target>` 構文です。違いが1つあります: 他のすべてのプラグインツールと同様に、[Tool Search](../features/tools.md)（`tool_search` / `tool_call`）が有効な場合、`ha_*` ツールは直接一覧に載るのではなく、その背後に置かれます。
:::

## セットアップ

### 1. 長期アクセストークンを作成する

1. Home Assistant インスタンスを開きます
2. **プロフィール**（サイドバーの自分の名前をクリック）に移動します
3. **長期アクセストークン（Long-Lived Access Tokens）** までスクロールします
4. **トークンを作成（Create Token）** をクリックし、「Hermes Agent」のような名前を付けます
5. トークンをコピーします

### 2. 環境変数を設定する

```bash
# Add to ~/.hermes/.env

# Required: your Long-Lived Access Token
HASS_TOKEN=your-long-lived-access-token

# Optional: HA URL (default: http://homeassistant.local:8123)
HASS_URL=http://192.168.1.100:8123

# Optional: default notify target for a bare `deliver: homeassistant`
HASS_HOME_CHANNEL=mobile_app_my_phone
```

:::info
プラグインがインストールされていれば、`homeassistant` ツールセットは `HASS_TOKEN` が設定されると自動的に有効になります。ゲートウェイプラットフォームとデバイス制御ツールの両方が、この1つのトークンから有効化されます。
:::

### 3. ゲートウェイを起動する

```bash
hermes gateway
```

Home Assistant は、他のメッセージングプラットフォーム（Telegram、Discord など）と並んで、接続済みのプラットフォームとして表示されます。

## 利用可能なツール

プラグインは、スマートホーム制御のために `homeassistant` ツールセットに4つのツールを登録します。

### `ha_list_entities`

Home Assistant のエンティティを一覧表示します。任意でドメインやエリアでフィルタできます。

**パラメータ:**
- `domain` *(任意)* — エンティティドメインでフィルタ: `light`、`switch`、`climate`、`sensor`、`binary_sensor`、`cover`、`fan`、`media_player` など
- `area` *(任意)* — エリア／部屋名でフィルタ（フレンドリー名と照合）: `living room`、`kitchen`、`bedroom` など

**例:**
```
List all lights in the living room
```

エンティティID、状態、フレンドリー名を返します。

### `ha_get_state`

単一エンティティの詳細な状態を取得します。すべての属性（明るさ、色、温度設定値、センサーの測定値など）を含みます。

**パラメータ:**
- `entity_id` *(必須)* — 照会するエンティティ。例: `light.living_room`、`climate.thermostat`、`sensor.temperature`

**例:**
```
What's the current state of climate.thermostat?
```

返すもの: 状態、すべての属性、最終変更／更新のタイムスタンプ。

### `ha_list_services`

デバイス制御に利用可能なサービス（アクション）を一覧表示します。各デバイスタイプで実行できるアクションと、それらが受け取るパラメータを表示します。

**パラメータ:**
- `domain` *(任意)* — ドメインでフィルタ。例: `light`、`climate`、`switch`

**例:**
```
What services are available for climate devices?
```

### `ha_call_service`

Home Assistant のサービスを呼び出してデバイスを制御します。

**パラメータ:**
- `domain` *(必須)* — サービスドメイン: `light`、`switch`、`climate`、`cover`、`media_player`、`fan`、`scene`、`script`
- `service` *(必須)* — サービス名: `turn_on`、`turn_off`、`toggle`、`set_temperature`、`set_hvac_mode`、`open_cover`、`close_cover`、`set_volume_level`
- `entity_id` *(任意)* — 対象エンティティ。例: `light.living_room`
- `data` *(任意)* — JSONオブジェクトとしての追加パラメータ

**例:**

```
Turn on the living room lights
→ ha_call_service(domain="light", service="turn_on", entity_id="light.living_room")
```

```
Set the thermostat to 22 degrees in heat mode
→ ha_call_service(domain="climate", service="set_temperature",
    entity_id="climate.thermostat", data={"temperature": 22, "hvac_mode": "heat"})
```

```
Set living room lights to blue at 50% brightness
→ ha_call_service(domain="light", service="turn_on",
    entity_id="light.living_room", data={"brightness": 128, "color_name": "blue"})
```

## ゲートウェイプラットフォーム: リアルタイムイベント

Home Assistant ゲートウェイアダプターは、WebSocket を介して接続し、`state_changed` イベントを購読します。デバイスの状態が変化し、フィルタに一致すると、それがメッセージとしてエージェントに転送されます。

### イベントのフィルタリング

:::warning 必須の設定
デフォルトでは、**イベントは転送されません**。イベントを受信するには、`watch_domains`、`watch_entities`、`watch_all` のうち少なくとも1つを設定する必要があります。フィルタがない場合、起動時に警告がログに記録され、すべての状態変化は黙って破棄されます。
:::

エージェントが見るイベントは、`~/.hermes/config.yaml` の Home Assistant プラットフォームの `extra` セクションで設定します。

```yaml
platforms:
  homeassistant:
    enabled: true
    extra:
      url: http://192.168.1.100:8123   # optional; same as HASS_URL
      watch_domains:
        - climate
        - binary_sensor
        - alarm_control_panel
        - light
      watch_entities:
        - sensor.front_door_battery
      ignore_entities:
        - sensor.uptime
        - sensor.cpu_usage
        - sensor.memory_usage
      cooldown_seconds: 30
```

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `url` | `HASS_URL`、なければ `http://homeassistant.local:8123` | Home Assistant のベース URL |
| `watch_domains` | *(なし)* | これらのエンティティドメインのみを監視（例: `climate`、`light`、`binary_sensor`） |
| `watch_entities` | *(なし)* | これらの特定のエンティティIDのみを監視 |
| `watch_all` | `false` | **すべての** 状態変化を受信するには `true` に設定（ほとんどの構成では非推奨） |
| `ignore_entities` | *(なし)* | これらのエンティティを常に無視（ドメイン／エンティティフィルタの前に適用） |
| `cooldown_seconds` | `30` | 同じエンティティのイベント間の最小秒数 |

:::tip
焦点を絞ったドメインのセットから始めてください。`climate`、`binary_sensor`、`alarm_control_panel` が最も有用な自動化をカバーします。必要に応じて追加してください。CPU温度や稼働時間カウンターのようなノイズの多いセンサーを抑制するには `ignore_entities` を使用してください。
:::

### イベントの整形

状態変化は、ドメインに基づいて人間が読みやすいメッセージに整形されます。

| ドメイン | フォーマット |
|--------|--------|
| `climate` | "HVAC mode changed from 'off' to 'heat' (current: 21, target: 23)" |
| `sensor` | "changed from 21°C to 22°C" |
| `binary_sensor` | "triggered" / "cleared" |
| `light`, `switch`, `fan` | "turned on" / "turned off" |
| `alarm_control_panel` | "alarm state changed from 'armed_away' to 'triggered'" |
| *(その他)* | "changed from 'old' to 'new'" |

### エージェントの応答

エージェントからの送信メッセージは、**Home Assistant の永続通知**（`persistent_notification.create` 経由）として配信されます。これらは「Hermes Agent」というタイトルで HA の通知パネルに表示されます。

このプラットフォームは `minimal` の表示デフォルトを使います（通知にツールの進捗やストリーミングのおしゃべりは含まれません）。より多くを表示したい場合は、`config.yaml` の `display.platforms.homeassistant` で上書きしてください。

### Cron と Webhook の配信

スケジュールされたジョブと Webhook ルートは、Home Assistant に配信できます。

```yaml
deliver: homeassistant:mobile_app_my_phone   # explicit notify target
deliver: homeassistant                       # uses HASS_HOME_CHANNEL
```

単なる `homeassistant` 形式を使うには、`HASS_HOME_CHANNEL` にデフォルトの通知ターゲットを設定する必要があります。[スケジュールされたタスク](../features/cron.md)と [Webhooks](webhooks.md) を参照してください。

### 接続管理

- リアルタイムイベント用の **WebSocket**（30秒のハートビート付き）
- バックオフ付きの **自動再接続**: 5s → 10s → 30s → 60s
- 送信通知用の **REST API**（WebSocket の競合を避けるための別セッション）
- **認可** — HA のイベントは常に認可されます（ユーザー許可リストやペアリングは不要: `HASS_TOKEN` が接続を認証し、人間の送信者は存在しません）

## セキュリティ

Home Assistant ツールはセキュリティ制限を強制します。

:::warning ブロックされるドメイン
HA ホスト上での任意コード実行を防ぐため、以下のサービスドメインは **ブロック** されています。

- `shell_command` — 任意のシェルコマンド
- `command_line` — コマンドを実行するセンサー／スイッチ
- `python_script` — スクリプト化された Python の実行
- `pyscript` — より広範なスクリプト連携
- `hassio` — アドオン制御、ホストのシャットダウン／再起動
- `rest_command` — HA サーバーからの HTTP リクエスト（SSRF ベクトル）

これらのドメインのサービスを呼び出そうとするとエラーを返します。
:::

エンティティIDは、インジェクション攻撃を防ぐため、パターン `^[a-z_][a-z0-9_]*\.[a-z0-9_]+$` に対して検証されます。

## 自動化の例

### モーニングルーティン

```
User: Start my morning routine

Agent:
1. ha_call_service(domain="light", service="turn_on",
     entity_id="light.bedroom", data={"brightness": 128})
2. ha_call_service(domain="climate", service="set_temperature",
     entity_id="climate.thermostat", data={"temperature": 22})
3. ha_call_service(domain="media_player", service="turn_on",
     entity_id="media_player.kitchen_speaker")
```

### セキュリティチェック

```
User: Is the house secure?

Agent:
1. ha_list_entities(domain="binary_sensor")
     → checks door/window sensors
2. ha_get_state(entity_id="alarm_control_panel.home")
     → checks alarm status
3. ha_list_entities(domain="lock")
     → checks lock states
4. Reports: "All doors closed, alarm is armed_away, all locks engaged."
```

### リアクティブな自動化（ゲートウェイイベント経由）

ゲートウェイプラットフォームとして接続している場合、エージェントはイベントに反応できます。

```
[Home Assistant] Front Door: triggered (was cleared)

Agent automatically:
1. ha_get_state(entity_id="binary_sensor.front_door")
2. ha_call_service(domain="light", service="turn_on",
     entity_id="light.hallway")
3. Sends notification: "Front door opened. Hallway lights turned on."
```

## トラブルシューティング

**プラットフォームやツールが見つからない。**
`hermes plugins list` で、プラグインがアクティブなプロファイルにインストールされ有効になっているか確認してください。
見つからない場合は、`hermes plugins install homeassistant`
（または `hermes -p <profile> plugins install homeassistant`）を実行してゲートウェイを再起動します。
`security.allow_lazy_installs` がオフの場合、初回起動時の自動インストールは
スキップされるため、自分でプラグインをインストールする必要があります。

**環境変数が読み込まれない。**
アダプターは、`~/.hermes/.env`（起動時に自動マージ）または
`config.yaml` から認証情報を読み込みます。ファイルがアクティブな Hermes プロファイルの
ホーム配下にあること、URL/トークンの周りに余計な引用符がないことを再確認してください。編集後は
ゲートウェイを再起動してください — 環境変数の変更はプロセス起動時にのみ適用されます。

**REST 認証が失敗する（`401 Unauthorized`）。**
トークンは、HA のユーザープロフィールページ（**Profile → Security → Long-lived access tokens**）で
作成した*長期アクセストークン*でなければなりません。短命な UI
セッショントークンは機能しません。また、ベース URL にスキームと
ポートが含まれていること（例: `http://homeassistant.local:8123`）、Hermes を実行しているホストから
到達可能であることも確認してください — `curl -H "Authorization: Bearer <token>" <url>/api/` が
`{"message": "API running."}` を返すはずです。
