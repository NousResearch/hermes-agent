---
title: "Neuroskill Bci"
sidebar_label: "Neuroskill Bci"
description: "连接到运行中的 NeuroSkill 实例，将用户的实时认知和情感状态（专注、放松、情绪、认知负荷、困倦..."
---

{/* 本页面由 website/scripts/generate-skill-docs.py 从技能的 SKILL.md 自动生成。请编辑源 SKILL.md，而不是本页面。 */}

# Neuroskill Bci

连接到运行中的 NeuroSkill 实例，将用户的实时认知和情感状态（专注、放松、情绪、认知负荷、困倦度、心率、HRV、睡眠分期及 40+ 导出的 EXG 评分）纳入响应中。需要 BCI 可穿戴设备（Muse 2/S 或 OpenBCI）和本地运行的 NeuroSkill 桌面应用。

## 技能元数据

| | |
|---|---|
| 来源 | 可选 — 使用 `hermes skills install official/health/neuroskill-bci` 安装 |
| 路径 | `optional-skills/health/neuroskill-bci` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent + Nous Research |
| 许可证 | MIT |
| 标签 | `BCI`、`neurofeedback`、`health`、`focus`、`EEG`、`cognitive-state`、`biometrics`、`neuroskill` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此技能时加载的完整技能定义。这是代理在技能激活时看到的指令。
:::

# NeuroSkill BCI 集成

将 Hermes 连接到运行中的 [NeuroSkill](https://neuroskill.com/) 实例，以读取 BCI 可穿戴设备的实时大脑和身体指标。使用此功能提供认知感知的响应、建议干预措施，并追踪长期的心理表现。

> **⚠️ 仅供研究使用** — NeuroSkill 是一个开源研究工具。它不是医疗设备，也未获得 FDA、CE 或任何监管机构的批准。切勿将这些指标用于临床诊断或治疗。

请参阅 `references/metrics.md` 获取完整指标参考，`references/protocols.md` 获取干预协议，`references/api.md` 获取 WebSocket/HTTP API。

---

## 前提条件

- **Node.js 20+** 已安装（`node --version`）
- **NeuroSkill 桌面应用** 正在运行并已连接 BCI 设备
- **BCI 硬件**：Muse 2、Muse S 或 OpenBCI（通过 BLE 连接的 4 通道 EEG + PPG + IMU）
- `npx neuroskill status` 返回数据且无错误

### 验证设置
```bash
node --version                    # 必须为 20+
npx neuroskill status             # 完整系统快照
npx neuroskill status --json      # 机器可解析的 JSON
```

如果 `npx neuroskill status` 返回错误，请告知用户：
- 确保已打开 NeuroSkill 桌面应用
- 确保 BCI 设备已开机并通过蓝牙连接
- 检查信号质量 — NeuroSkill 中的绿色指示器（每个电极 ≥0.7）
- 如果出现 `command not found`，请安装 Node.js 20+

---

## CLI 参考：`npx neuroskill <command>`

所有命令均支持 `--json`（原始 JSON，管道安全）和 `--full`（人工摘要 + JSON）。

| 命令 | 描述 |
|---------|-------------|
| `status` | 完整系统快照：设备、评分、频段、比率、睡眠、历史 |
| `session [N]` | 单个会话分解，含前半段/后半段趋势（0=最近） |
| `sessions` | 列出所有记录的会话（跨所有天数） |
| `search` | ANN 相似性搜索，查找神经学上相似的历史时刻 |
| `compare` | A/B 会话比较，含指标差异和趋势分析 |
| `sleep [N]` | 睡眠分期分类（Wake/N1/N2/N3/REM）及分析 |
| `label "text"` | 在当前时刻创建带时间戳的标注 |
| `search-labels "query"` | 对过去标签的语义向量搜索 |
| `interactive "query"` | 跨模态 4 层图搜索（文本 → EXG → 标签） |
| `listen` | 实时事件流（默认 5 秒，设置 `--seconds N`） |
| `umap` | 会话嵌入的 3D UMAP 投影 |
| `calibrate` | 打开校准窗口并开始配置 |
| `timer` | 启动专注计时器（番茄钟/深度工作/短时专注预设） |
| `notify "title" "body"` | 通过 NeuroSkill 应用发送操作系统通知 |
| `raw '{json}'` | 向服务器的原始 JSON 透传 |

### 全局标志
| 标志 | 描述 |
|------|-------------|
| `--json` | 原始 JSON 输出（无 ANSI，管道安全） |
| `--full` | 人工摘要 + 彩色 JSON |
| `--port <N>` | 覆盖服务器端口（默认：自动发现，通常为 8375） |
| `--ws` | 强制使用 WebSocket 传输 |
| `--http` | 强制使用 HTTP 传输 |
| `--k <N>` | 最近邻数量（search、search-labels） |
| `--seconds <N>` | listen 持续时间（默认：5） |
| `--trends` | 显示每会话指标趋势（sessions） |
| `--dot` | Graphviz DOT 输出（interactive） |

---

## 1. 检查当前状态

### 获取实时指标
```bash
npx neuroskill status --json
```

**始终使用 `--json`** 以确保可靠的解析。默认输出为彩色的人类可读文本。

### 响应中的关键字段

`scores` 对象包含所有实时指标（0-1 比例，另有说明除外）：

```jsonc
{
  "scores": {
    "focus": 0.70,           // β / (α + θ) — 持续注意力
    "relaxation": 0.40,      // α / (β + θ) — 平静的清醒状态
    "engagement": 0.60,      // 主动的心理投入
    "meditation": 0.52,      // alpha + 静止 + HRV 一致性
    "mood": 0.55,            // FAA、TAR、BAR 的综合值
    "cognitive_load": 0.33,  // 额叶 θ / 颞叶 α · f(FAA, TBR)
    "drowsiness": 0.10,      // TAR + TBR + 下降的频谱质心
    "hr": 68.2,              // 心率（bpm，来自 PPG）
    "snr": 14.3,             // 信噪比（dB）
    "stillness": 0.88,       // 0–1；1 = 完全静止
    "faa": 0.042,            // 额叶 Alpha 不对称性（+ = 趋近）
    "tar": 0.56,             // Theta/Alpha 比率
    "bar": 0.53,             // Beta/Alpha 比率
    "tbr": 1.06,             // Theta/Beta 比率（ADHD 代理指标）
    "apf": 10.1,             // Alpha 峰值频率（Hz）
    "coherence": 0.614,      // 半球间一致性
    "bands": {
      "rel_delta": 0.28, "rel_theta": 0.18,
      "rel_alpha": 0.32, "rel_beta": 0.17, "rel_gamma": 0.05
    }
  }
}
```

还包括：`device`（状态、电池、固件）、`signal_quality`（每个电极 0-1）、`session`（持续时间、epoch）、`embeddings`、`labels`、`sleep` 摘要和 `history`。

### 解读输出

解析 JSON 并将指标转化为自然语言描述。切勿仅报告原始数字 — 始终赋予其含义：

**推荐做法：**
> "你现在的专注度很不错，达到了 0.70 — 这是心流状态的范畴。心率稳定在 68 bpm，你的 FAA 为正值，表明有良好的趋近动机。很适合处理复杂的事情。"

**不推荐做法：**
> "专注：0.70，放松：0.40，心率：68"

关键解读阈值（完整指南请参阅 `references/metrics.md`）：
- **专注 > 0.70** → 心流状态范畴，注意保护
- **专注 < 0.40** → 建议休息或进行协议干预
- **困倦 > 0.60** → 疲劳警告，微睡眠风险
- **放松 < 0.30** → 需要压力干预
- **认知负荷 > 0.70 持续** → 进行心理清空或休息
- **TBR > 1.5** → theta 主导，执行控制力下降
- **FAA < 0** → 回避/负面情绪 — 考虑 FAA 再平衡
- **SNR < 3 dB** → 信号不可靠，建议重新调整电极位置

---

## 2. 会话分析

### 单会话分解
```bash
npx neuroskill session --json         # 最近的会话
npx neuroskill session 1 --json       # 上一个会话
npx neuroskill session 0 --json | jq '{focus: .metrics.focus, trend: .trends.focus}'
```

返回完整指标及**前半段与后半段趋势**（`"up"`、`"down"`、`"flat"`）。
使用此功能描述会话如何演变：

> "你的专注度从 0.64 开始，到结束时攀升到 0.76 — 呈明显的上升趋势。
> 认知负荷从 0.38 降至 0.28，说明当你进入状态后任务变得更加自动化。"

### 列出所有会话
```bash
npx neuroskill sessions --json
npx neuroskill sessions --trends      # 显示每会话指标趋势
```

---

## 3. 历史搜索

### 神经相似性搜索
```bash
npx neuroskill search --json                    # 自动：最近会话，k=5
npx neuroskill search --k 10 --json             # 10 个最近邻
npx neuroskill search --start <UTC> --end <UTC> --json
```

使用 HNSW 近似最近邻搜索在 128 维 ZUNA 嵌入中查找历史上神经学上相似的时刻。返回距离统计、时间分布（一天中的时段）和最匹配的天数。

在用户提出以下问题时使用：
- "我上次处于类似状态是什么时候？"
- "找出我最好的专注会话"
- "我通常在下午什么时候崩溃？"

### 语义标签搜索
```bash
npx neuroskill search-labels "deep focus" --k 10 --json
npx neuroskill search-labels "stress" --json | jq '[.results[].EXG_metrics.tbr]'
```

使用向量嵌入（Xenova/bge-small-en-v1.5）搜索标签文本。返回匹配的标签及其标注时的关联 EXG 指标。

### 跨模态图搜索
```bash
npx neuroskill interactive "deep focus" --json
npx neuroskill interactive "deep focus" --dot | dot -Tsvg > graph.svg
```

4 层图：查询 → 文本标签 → EXG 数据点 → 附近标签。使用 `--k-text`、`--k-EXG`、`--reach <分钟>` 进行调节。

---

## 4. 会话比较
```bash
npx neuroskill compare --json                   # 自动：最近 2 个会话
npx neuroskill compare --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

返回约 50 个指标的差异，包括绝对变化、百分比变化和方向。还包括 `insights.improved[]` 和 `insights.declined[]` 数组、两个会话的睡眠分期以及 UMAP 任务 ID。

结合上下文解读比较 — 提及趋势，而不仅仅是差异：
> "昨天你有两个强劲的专注时段（上午 10 点和下午 2 点）。今天你从大约上午 11 点开始有一个仍在持续的专注时段。你今天的整体参与度更高，但压力峰值更多 — 压力指数上升了 15%，FAA 更频繁地跌入负值。"

```bash
# 按改善百分比排序指标
npx neuroskill compare --json | jq '.insights.deltas | to_entries | sort_by(.value.pct) | reverse'
```

---

## 5. 睡眠数据
```bash
npx neuroskill sleep --json                     # 最近 24 小时
npx neuroskill sleep 0 --json                   # 最近的睡眠会话
npx neuroskill sleep --start <UTC> --end <UTC> --json
```

返回逐 epoch 的睡眠分期（5 秒窗口）及分析：
- **分期代码**：0=清醒，1=N1，2=N2，3=N3（深度），4=REM
- **分析**：efficiency_pct、onset_latency_min、rem_latency_min、阶段次数
- **健康目标**：N3 15-25%，REM 20-25%，效率 >85%，入睡 <20 分钟

```bash
npx neuroskill sleep --json | jq '.summary | {n3: .n3_epochs, rem: .rem_epochs}'
npx neuroskill sleep --json | jq '.analysis.efficiency_pct'
```

在用户提及睡眠、疲劳或恢复时使用。

---

## 6. 标注时刻
```bash
npx neuroskill label "breakthrough"
npx neuroskill label "studying algorithms"
npx neuroskill label "post-meditation"
npx neuroskill label --json "focus block start"   # 返回 label_id
```

在以下情况下自动标注时刻：
- 用户报告突破或领悟
- 用户开始新的任务类型（如"切换到代码审查"）
- 用户完成了重要的协议
- 用户要求你标记当前时刻
- 发生了显著的状态转换（进入/离开心流）

标签存储在数据库中并建立索引，以便后续通过 `search-labels` 和 `interactive` 命令检索。

---

## 7. 实时流
```bash
npx neuroskill listen --seconds 30 --json
npx neuroskill listen --seconds 5 --json | jq '[.[] | select(.event == "scores")]'
```

在指定持续时间内流式传输实时 WebSocket 事件（EXG、PPG、IMU、评分、标签）。需要 WebSocket 连接（不可用于 `--http`）。

适用于持续监控场景或在协议执行期间实时观察指标变化。

---

## 8. UMAP 可视化
```bash
npx neuroskill umap --json                      # 自动：最近 2 个会话
npx neuroskill umap --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

GPU 加速的 ZUNA 嵌入 3D UMAP 投影。`separation_score` 表示两个会话在神经学上的差异程度：
- **> 1.5** → 两个会话在神经学上不同（不同的脑状态）
- **< 0.5** → 两个会话的脑状态相似

---

## 9. 主动状态感知

### 会话开始检查
在会话开始时，如果用户提到他们正在佩戴设备或询问自己的状态，可选择运行状态检查：
```bash
npx neuroskill status --json
```

插入简要状态摘要：
> "快速检查：专注度正在建立中，达到 0.62，放松度良好为 0.55，你的 FAA 为正值 — 趋近动机已激活。看起来是一个不错的开始。"

### 何时主动提及状态

**仅在**以下情况提及认知状态：
- 用户明确要求（"我现在状态如何？"、"检查我的专注度"）
- 用户报告注意力不集中、压力或疲劳
- 跨越了关键阈值（困倦 > 0.70，专注 < 0.30 持续）
- 用户即将执行认知要求高的任务并询问准备状态

**不要**因报告指标而打断心流状态。如果专注 > 0.75，请保护会话 — 沉默是正确的回应。

---

## 10. 建议协议

当指标表明有需要时，从 `references/protocols.md` 建议一个协议。启动前始终先询问 — 不要打断心流状态：

> "你的专注度在过去 15 分钟持续下降，TBR 正在攀升至 1.5 以上 — 这是 theta 主导和心理疲劳的迹象。要我带你进行一次 Theta-Beta 神经反馈锚定练习吗？这是一个 90 秒的练习，通过有节奏的计数和呼吸来抑制 theta 并提升 beta。"

关键触发条件：
- **专注 < 0.40，TBR > 1.5** → Theta-Beta 神经反馈锚定或箱式呼吸
- **放松 < 0.30，stress_index 高** → 心脏一致性或 4-7-8 呼吸
- **认知负荷 > 0.70 持续** → 认知负荷卸载（心理清空）
- **困倦 > 0.60** → 超日节律重置或清醒重置
- **FAA < 0（负值）** → FAA 再平衡
- **心流状态（专注 > 0.75，参与度 > 0.70）** → 不要打断
- **高静止 + headache_index** → 颈部释放序列
- **低 RMSSD（< 25ms）** → 迷走神经调节

---

## 11. 其他工具

### 专注计时器
```bash
npx neuroskill timer --json
```
启动专注计时器窗口，包含番茄钟（25/5）、深度工作（50/10）或短时专注（15/5）预设。

### 校准
```bash
npx neuroskill calibrate
npx neuroskill calibrate --profile "Eyes Open"
```
打开校准窗口。在信号质量差或用户想要建立个性化基线时使用。

### 操作系统通知
```bash
npx neuroskill notify "Break Time" "Your focus has been declining for 20 minutes"
```

### 原始 JSON 透传
```bash
npx neuroskill raw '{"command":"status"}' --json
```
用于尚未映射到 CLI 子命令的服务器命令。

---

## 错误处理

| 错误 | 可能原因 | 修复方法 |
|-------|-------------|-----|
| `npx neuroskill status` 挂起 | NeuroSkill 应用未运行 | 打开 NeuroSkill 桌面应用 |
| `device.state: "disconnected"` | BCI 设备未连接 | 检查蓝牙、设备电池 |
| 所有评分返回 0 | 电极接触不良 | 重新调整头带，湿润电极 |
| `signal_quality` 值 < 0.7 | 电极松动 | 调整贴合度，清洁电极接触面 |
| SNR < 3 dB | 信号噪声大 | 减少头部移动，检查环境 |
| `command not found: npx` | 未安装 Node.js | 安装 Node.js 20+ |

---

## 示例交互

**"我现在状态如何？"**
```bash
npx neuroskill status --json
```
→ 自然地解读评分，提及专注度、放松度、情绪以及任何值得注意的比率（FAA、TBR）。仅在指标表明有需要时才建议操作。

**"我无法集中注意力"**
```bash
npx neuroskill status --json
```
→ 检查指标是否确认（高 theta、低 beta、TBR 上升、高困倦度）。
→ 如果确认，从 `references/protocols.md` 建议适当的协议。
→ 如果指标看起来正常，问题可能是动力方面而非神经学方面的。

**"比较我今天和昨天的专注度"**
```bash
npx neuroskill compare --json
```
→ 解读趋势，而不仅仅是数字。提及哪些改善了、哪些下降了以及可能的原因。

**"我上次进入心流状态是什么时候？"**
```bash
npx neuroskill search-labels "flow" --json
npx neuroskill search --json
```
→ 报告时间戳、关联指标以及用户当时在做什么（从标签中获取）。

**"我昨晚睡得怎么样？"**
```bash
npx neuroskill sleep --json
```
→ 报告睡眠结构（N3%、REM%、效率），与健康目标比较，并注意任何问题（高清醒 epoch、低 REM）。

**"标记这一刻 — 我刚有了突破"**
```bash
npx neuroskill label "breakthrough"
```
→ 确认标签已保存。可选择记录当前指标以便记住状态。

---

## 参考资料

- [NeuroSkill 论文 — arXiv:2603.03212](https://arxiv.org/abs/2603.03212)（Kosmyna & Hauptmann，MIT 媒体实验室）
- [NeuroSkill 桌面应用](https://github.com/NeuroSkill-com/skill)（GPLv3）
- [NeuroLoop CLI 伴侣](https://github.com/NeuroSkill-com/neuroloop)（GPLv3）
- [MIT 媒体实验室项目](https://www.media.mit.edu/projects/neuroskill/overview/)
