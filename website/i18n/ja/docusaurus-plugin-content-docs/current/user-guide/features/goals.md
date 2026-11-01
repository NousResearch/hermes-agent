---
sidebar_position: 16
title: "永続ゴール"
description: "標準ゴールを設定し、完了するまで Hermes にターンをまたいで作業を続けさせます。Ralph ループに対する私たちの解釈です。"
---

# 永続ゴール（`/goal`）

`/goal` は、ターンをまたいで存続する標準的な目標を Hermes に与えます。各ターンの後、軽量なジャッジモデルが、アシスタントの最後の応答によってゴールが満たされたかどうかをチェックします。満たされていなければ、Hermes は継続プロンプトを同じセッションに自動的にフィードバックし、作業を続けます — ゴールが達成されるか、あなたが一時停止/クリアするか、ターンの予算を使い切るまで。

これは **Ralph ループ**に対する私たちの解釈で、OpenAI の Eric Traut による [Codex CLI 0.128.0 の `/goal`](https://github.com/openai/codex) から直接着想を得ています。中心的なアイデア — ターンをまたいでゴールを生かし続け、達成するまで止めない — は彼らのものです。ここでの実装は独立しており、Hermes のアーキテクチャに合わせて適応されています。

## いつ使うか

`/goal` は、毎ターン再プロンプトせずに Hermes に自律的に反復してほしいタスクに使います:

- 「`src/` のすべての lint エラーを修正し、`ruff check` が通ることを確認して」
- 「機能 X をリポジトリ Y から、テストも含めて移植し、CI をグリーンにして」
- 「実行途中の圧縮でセッション ID がときどきずれる理由を調査し、レポートを書いて」
- 「EXIF の日付でファイルをリネームする小さな CLI を作り、photos/ フォルダで試して」

エージェントが 1 ターンで止まるタスクに `/goal` は必要ありません。*そうでなければ「続けて」を 3 回言わなければならない*ようなタスクこそ、これが真価を発揮する場面です。

## クイックスタート

```
/goal Fix every failing test in tests/hermes_cli/ and make sure scripts/run_tests.sh passes for that directory
```

表示されるもの:

1. **ゴール受理** — `⊙ Goal set (20-turn budget): <あなたのゴール>`
2. **ターン 1 実行** — Hermes は、あなたがゴールを通常のメッセージとして送ったかのように作業を開始します。
3. **ジャッジ実行** — ターンの後、ジャッジモデルが `done` か `continue` かを判断します。
4. **必要ならループ発火** — `continue` の場合、`↻ Continuing toward goal (1/20): <ジャッジの理由>` が表示され、Hermes が自動的に次のステップを実行します。
5. **終了** — 最終的に `✓ Goal achieved: <理由>` または `⏸ Goal paused — N/20 turns used` のいずれかが表示されます。

## コマンド

| コマンド | 機能 |
|---|---|
| `/goal <text>` | 標準ゴールを設定（または置換）します。最初のターンを即座に開始するので、別途メッセージを送る必要はありません。 |
| `/goal` または `/goal status` | 現在のゴール、そのステータス、使用ターン数を表示します。 |
| `/goal pause` | ゴールをクリアせずに自動継続ループを停止します。 |
| `/goal resume` | ループを再開します（ターンカウンターをゼロにリセット）。 |
| `/goal clear` | ゴールを完全に破棄します。 |

CLI とすべてのゲートウェイプラットフォーム（Telegram、Discord、Slack、Matrix、Signal、WhatsApp、SMS、iMessage、Webhook、API サーバー、Web ダッシュボード）で同一に動作します。

## 動作の詳細

### ジャッジ

各ターンの後、Hermes は補助モデルを次の内容で呼び出します:

- 標準ゴールのテキスト
- エージェントの最新の最終応答（テキストの最後の約 4 KB）
- 厳格な JSON で応答するようジャッジに指示するシステムプロンプト: `{"done": <bool>, "reason": "<一文の根拠>"}`

ジャッジは意図的に保守的です: 応答がゴールの完了を**明示的に**確認しているとき、最終成果物が明確に生成されているとき、またはゴールが達成不可能/ブロックされているとき（不可能なタスクに予算を使わないよう、ブロック理由付きで DONE として扱う）にのみ、ゴールを `done` とマークします。

### フェイルオープンのセマンティクス

ジャッジがエラーになった場合（ネットワークの瞬断、不正な形式の応答、補助クライアントの利用不可）、Hermes は判定を `continue` として扱います — 壊れたジャッジが進行を止めることは決してありません。**ターン予算**が真の最後の砦です。

### ターン予算

デフォルトは 20 継続ターン（`config.yaml` の `goals.max_turns`）です。予算に達すると、Hermes は自動的に一時停止し、どう進めるか正確に伝えます:

```
⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.
```

`/goal resume` はカウンターをゼロにリセットするので、測られたチャンク単位で続行できます。

### ユーザーメッセージは常に優先される

ゴールがアクティブな間にあなたが送る実際のメッセージは、継続ループより優先されます。CLI ではあなたのメッセージがキューに入った継続より前に `_pending_input` に着地し、ゲートウェイでは同じようにアダプターの FIFO を通ります。あなたのターンの後にジャッジが再び実行されます — そのため、あなたのメッセージがたまたまゴールを完了させると、ジャッジがそれを捉えて停止します。

### 実行途中の安全性（ゲートウェイ）

エージェントが既に実行中の間、`/goal status`、`/goal pause`、`/goal clear` は安全に実行できます — これらは制御プレーンの状態だけに触れ、現在のターンを中断しません。実行途中で**新しい**ゴールを設定する（`/goal <new text>`）と、まず `/stop` するよう伝えるメッセージとともに拒否されます。これにより、古い継続が新しいものと競合できないようにします。

### 永続化

ゴールの状態は、`goal:<session_id>` をキーとして `SessionDB.state_meta` に存在します。つまり `/resume` は中断したところから正確に再開します — ゴールを設定し、ラップトップを閉じ、明日戻ってきて `/resume` すれば、ゴールはあなたが残したとおり（アクティブ、一時停止、または完了）にまだ立っています。

### プロンプトキャッシュ

継続プロンプトは履歴に追加される単純なユーザーロールのメッセージです。システムプロンプトを変更したり、ツールセットを入れ替えたり、Hermes のプロンプトキャッシュを無効化するような形で会話に触れたりは**しません**。20 ターンのゴールを実行しても、キャッシュ的には 20 ターンの通常会話と同じコストです。

## 設定

`~/.hermes/config.yaml` に追加します:

```yaml
goals:
  # Hermes が自動一時停止して /goal resume を求めるまでの最大継続ターン数。
  # デフォルト 20。より緊密なループが欲しければ下げ、長時間のリファクタリングには
  # 上げてください。
  max_turns: 20
```

### ジャッジモデルの選択

ジャッジは `goal_judge` 補助タスクを使います。デフォルトではメインモデルに解決されます（[補助モデル](/docs/user-guide/configuration#auxiliary-models)を参照）。コストを抑えるためにジャッジを安価で高速なモデルにルーティングしたい場合は、オーバーライドを追加します:

```yaml
auxiliary:
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

ジャッジの呼び出しは小さく（出力約 200 トークン）、ターンごとに 1 回実行されるだけなので、安価で高速なモデルが通常は適切な選択です。

## ウォークスルー例

```
You: /goal Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

  ⊙ Goal set (20-turn budget): Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

Hermes: Creating /tmp/note_1.txt now.
  💻 echo "1" > /tmp/note_1.txt   (0.1s)
  I've created /tmp/note_1.txt with the content "1". I'll continue with the remaining files on the next turn as you specified.

  ↻ Continuing toward goal (1/20): Only 1 of 4 files has been created; 3 files remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "2" > /tmp/note_2.txt   (0.1s)
  Created /tmp/note_2.txt. Two more to go.

  ↻ Continuing toward goal (2/20): 2 of 4 files created; 2 remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "3" > /tmp/note_3.txt   (0.1s)
  Created /tmp/note_3.txt.

  ↻ Continuing toward goal (3/20): 3 of 4 files created; 1 remains.

Hermes: [Continuing toward your standing goal]
  💻 echo "4" > /tmp/note_4.txt   (0.1s)
  All four files have been created: /tmp/note_1.txt through /tmp/note_4.txt, each containing its number.

  ✓ Goal achieved: All four files were created with the specified content, completing the goal.

You: _
```

4 ターン、1 回の `/goal` 呼び出し、あなたからの「続けて」プロンプトはゼロです。

## ジャッジが間違えるとき

完璧なジャッジはありません。注意すべき 2 つの失敗モード:

**偽陰性 — ゴールが実際には完了しているのにジャッジが continue と言う。** これはターン予算が捉えます。`⏸ Goal paused` が表示され、`/goal clear` するか、単に新しいメッセージを送れます。

**偽陽性 — 作業が残っているのにジャッジが done と言う。** `✓ Goal achieved` が表示されますが、あなたはそうでないと分かっています。フォローアップメッセージを送って続行するか、ゴールをより正確に設定し直してください: `/goal <より具体的なテキスト>`。ジャッジのシステムプロンプトは意図的に保守的で、偽陽性が偽陰性より稀になるようにしています。

ジャッジの判定に納得できない場合、`↻ Continuing toward goal` または `✓ Goal achieved` の行にある理由テキストが、ジャッジが何を見たのかを正確に教えてくれます。それで通常、ゴールのテキストが曖昧だったのか、モデルの応答が曖昧だったのかを診断するのに十分です。

## 帰属

`/goal` は **Ralph ループ**パターンに対する Hermes の解釈です。ユーザー向けの設計 — ターンをまたいでゴールを生かし続け、達成するまで止めず、作成/一時停止/再開/クリアの制御を備える — は、OpenAI の Codex チームの Eric Traut によって [Codex CLI 0.128.0](https://github.com/openai/codex) で広められ出荷されました。私たちの実装は独立しています（中央の `CommandDef` レジストリ、`SessionDB.state_meta` 永続化、補助クライアントジャッジ、ゲートウェイ側のアダプター FIFO 継続）が、アイデアは彼らのものです。功績は功績に帰すべきです。
