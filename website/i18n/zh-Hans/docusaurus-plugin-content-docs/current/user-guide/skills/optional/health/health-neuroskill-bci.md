---
title: "Neuroskill Bci"
sidebar_label: "Neuroskill Bci"
description: "连接到运行中的 NeuroSkill 实例，将用户的实时认知与情绪状态（专注度、放松度、情绪、认知负荷、困倦度、心率、HRV、睡眠分期及 40+ 项衍生 EXG 评分）融入响应中..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Neuroskill Bci

连接到运行中的 NeuroSkill 实例，将用户的实时认知与情绪状态（专注度、放松度、情绪、认知负荷、困倦度、心率、HRV、睡眠分期及 40+ 项衍生 EXG 评分）融入响应中。需要 BCI 可穿戴设备（Muse 2/S 或 OpenBCI）以及在本地运行的 NeuroSkill 桌面应用。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/health/neuroskill-bci` 安装 |
| 路径 | `optional-skills/health/neuroskill-bci` |
| 版本 | `1.0.0` |
| 作者 | Hermes Agent + Nous Research |
| 许可证 | MIT |
| 平台 | linux, macos, windows |
| 标签 | `BCI`, `neurofeedback`, `health`, `focus`, `EEG`, `cognitive-state`, `biometrics`, `neuroskill` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# NeuroSkill BCI 集成

将 Hermes 连接到运行中的 [NeuroSkill](https://neuroskill.com/) 实例，从 BCI 可穿戴设备读取实时脑部与身体指标。用于提供具有认知感知能力的响应、建议干预措施，并随时间追踪心理表现。

> **⚠️ 仅供研究使用** — NeuroSkill 是一款开源研究工具。它**不是**医疗设备，**未**经 FDA、CE 或任何监管机构批准。切勿将这些指标用于临床诊断或治疗。

完整指标参考见 `references/metrics.md`，干预协议见 `references/protocols.md`，WebSocket/HTTP API 见 `references/api.md`。

---

## 前提条件

- 已安装 **Node.js 20+**（`node --version`）
- **NeuroSkill 桌面应用**正在运行，且已连接 BCI 设备
- **BCI 硬件**：Muse 2、Muse S 或 OpenBCI（通过 BLE 连接的 4 通道 EEG + PPG + IMU）
- `npx neuroskill status` 无错误返回数据

### 验证设置
```bash
node --version                    # Must be 20+
npx neuroskill status             # Full system snapshot
npx neuroskill status --json      # Machine-parseable JSON
```

如果 `npx neuroskill status` 返回错误，请告知用户：
- 确保 NeuroSkill 桌面应用已打开
- 确保 BCI 设备已开机并通过蓝牙连接
- 检查信号质量 — NeuroSkill 中显示绿色指示（每个电极 ≥0.7）
- 如提示 `command not found`，请安装 Node.js 20+

---

## CLI 参考：`npx neuroskill <command>`

所有命令均支持 `--json`（原始 JSON，适合管道传输）和 `--full`（人类可读摘要 + JSON）。

| 命令 | 描述 |
|---------|-------------|
| `status` | 完整系统快照：设备、评分、频段、比率、睡眠、历史记录 |
| `session [N]` | 单次会话详情，含前半段/后半段趋势（0=最近一次） |
| `sessions` | 列出所有日期的所有已记录会话 |
| `search` | 基于 ANN 的神经相似历史时刻搜索 |
| `compare` | A/B 会话对比，含指标差值与趋势分析 |
| `sleep [N]` | 睡眠分期分类（Wake/N1/N2/N3/REM）及分析 |
| `label "text"` | 在当前时刻创建带时间戳的注释 |
| `search-labels "query"` | 对历史标签进行语义向量搜索 |
| `interactive "query"` | 跨模态 4 层图搜索（文本 → EXG → 标签） |
| `listen` | 实时事件流（默认 5 秒，可通过 `--seconds N` 设置） |
| `umap` | 会话嵌入的 3D UMAP 投影 |
| `calibrate` | 打开校准窗口并启动配置文件 |
| `timer` | 启动专注计时器（Pomodoro/深度工作/短时专注预设） |
| `notify "title" "body"` | 通过 NeuroSkill 应用发送系统通知 |
| `raw '{json}'` | 原始 JSON 直通至服务器 |

### 全局标志
| 标志 | 描述 |
|------|-------------|
| `--json` | 原始 JSON 输出（无 ANSI，适合管道传输） |
| `--full` | 人类可读摘要 + 彩色 JSON |
| `--port <N>` | 覆盖服务器端口（默认：自动发现，通常为 8375） |
| `--ws` | 强制使用 WebSocket 传输 |
| `--http` | 强制使用 HTTP 传输 |
| `--k <N>` | 最近邻数量（search、search-labels） |
| `--seconds <N>` | listen 持续时长（默认：5） |
| `--trends` | 显示每会话指标趋势（sessions） |
| `--dot` | Graphviz DOT 输出（interactive） |

---

## 1. 检查当前状态

### 获取实时指标
```bash
npx neuroskill status --json
```

**始终使用 `--json`** 以确保可靠解析。默认输出为带颜色的人类可读文本。

### 响应中的关键字段

`scores` 对象包含所有实时指标（除特别说明外，均为 0–1 范围）：

```jsonc
{
  "scores": {
    "focus": 0.70,           // β / (α + θ) — 持续注意力
    "relaxation": 0.40,      // α / (β + θ) — 平静清醒状态
    "engagement": 0.60,      // 主动心理投入
    "meditation": 0.52,      // alpha + 静止 + HRV 相干性
    "mood": 0.55,            // 由 FAA、TAR、BAR 综合计算
    "cognitive_load": 0.33,  // 额叶 θ / 颞叶 α · f(FAA, TBR)
    "drowsiness": 0.10,      // TAR + TBR + 频谱质心下降
    "hr": 68.2,              // 心率（bpm，来自 PPG）
    "snr": 14.3,             // 信噪比（dB）
    "stillness": 0.88,       // 0–1；1 = 完全静止
    "faa": 0.042,            // 额叶 Alpha 不对称性（正值 = 趋近动机）
    "tar": 0.56,             // Theta/Alpha 比率
    "bar": 0.53,             // Beta/Alpha 比率
    "tbr": 1.06,             // Theta/Beta 比率（ADHD 代理指标）
    "apf": 10.1,             // Alpha 峰值频率（Hz）
    "coherence": 0.614,      // 半球间相干性
    "bands": {
      "rel_delta": 0.28, "rel_theta": 0.18,
      "rel_alpha": 0.32, "rel_beta": 0.17, "rel_gamma": 0.05
    }
  }
}
```

还包括：`device`（状态、电量、固件）、`signal_quality`（每电极 0–1）、`session`（时长、epoch 数）、`embeddings`、`labels`、`sleep` 摘要及 `history`。

### 解读输出

解析 JSON 并将指标转化为自然语言。切勿单独报告原始数字 — 始终赋予其含义：

**应该这样做：**
> "您目前的专注度相当不错，达到 0.70 — 这已进入心流状态区间。心率稳定在 68 bpm，FAA 为正值，表明趋近动机良好。现在是处理复杂任务的好时机。"

**不应该这样做：**
> "专注度：0.70，放松度：0.40，心率：68"

关键解读阈值（完整指南见 `references/metrics.md`）：
- **专注度 > 0.70** → 心流状态区间，注意保护
- **专注度 &lt; 0.40** → 建议休息或执行协议
- **困倦度 > 0.60** → 疲劳警告，存在微睡眠风险
- **放松度 &lt; 0.30** → 需要压力干预
- **认知负荷 > 0.70 持续** → 建议思维倾倒或休息
- **TBR > 1.5** → theta 主导，执行控制减弱
- **FAA &lt; 0** → 回避/负面情绪 — 考虑 FAA 再平衡
- **SNR &lt; 3 dB** → 信号不可靠，建议重新定位电极

---

## 2. 会话分析

### 单次会话详情
```bash
npx neuroskill session --json         # most recent session
npx neuroskill session 1 --json       # previous session
npx neuroskill session 0 --json | jq '{focus: .metrics.focus, trend: .trends.focus}'
```

返回完整指标及**前半段与后半段趋势**（`"up"`、`"down"`、`"flat"`）。用于描述会话的演变过程：

> "您的专注度从 0.64 开始，到结束时上升至 0.76 — 呈明显上升趋势。认知负荷从 0.38 降至 0.28，表明随着您逐渐进入状态，任务变得更加自动化。"

### 列出所有会话
```bash
npx neuroskill sessions --json
npx neuroskill sessions --trends      # show per-session metric trends
```

---

## 3. 历史搜索

### 神经相似性搜索
```bash
npx neuroskill search --json                    # auto: last session, k=5
npx neuroskill search --k 10 --json             # 10 nearest neighbors
npx neuroskill search --start <UTC> --end <UTC> --json
```

使用基于 128 维 ZUNA 嵌入的 HNSW 近似最近邻搜索，在历史记录中查找神经状态相似的时刻。返回距离统计、时间分布（一天中的小时）及最匹配的日期。

在用户提问以下问题时使用：
- "我上次处于这种状态是什么时候？"
- "找出我最佳的专注会话"
- "我通常在下午什么时候状态下滑？"

### 语义标签搜索
```bash
npx neuroskill search-labels "deep focus" --k 10 --json
npx neuroskill search-labels "stress" --json | jq '[.results[].EXG_metrics.tbr]'
```

使用向量嵌入（Xenova/bge-small-en-v1.5）搜索标签文本。返回匹配标签及其标注时刻的关联 EXG 指标。

### 跨模态图搜索
```bash
npx neuroskill interactive "deep focus" --json
npx neuroskill interactive "deep focus" --dot | dot -Tsvg > graph.svg
```

4 层图：查询 → 文本标签 → EXG 点 → 附近标签。使用 `--k-text`、`--k-EXG`、`--reach <minutes>` 进行调整。

---

## 4. 会话对比
```bash
npx neuroskill compare --json                   # auto: last 2 sessions
npx neuroskill compare --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

返回约 50 项指标的差值，包含绝对变化量、百分比变化及方向。还包括 `insights.improved[]` 和 `insights.declined[]` 数组、两次会话的睡眠分期及 UMAP 任务 ID。

解读对比时需结合上下文 — 强调趋势而非单纯数字：
> "昨天您有两个强专注时段（上午 10 点和下午 2 点）。今天从上午 11 点左右开始了一个仍在持续的专注时段。您今天的整体投入度更高，但压力峰值更多 — 压力指数上升了 15%，FAA 更频繁地出现负值。"

```bash
# Sort metrics by improvement percentage
npx neuroskill compare --json | jq '.insights.deltas | to_entries | sort_by(.value.pct) | reverse'
```

---

## 5. 睡眠数据
```bash
npx neuroskill sleep --json                     # last 24 hours
npx neuroskill sleep 0 --json                   # most recent sleep session
npx neuroskill sleep --start <UTC> --end <UTC> --json
```

返回逐 epoch 的睡眠分期（5 秒窗口）及分析：
- **分期代码**：0=清醒，1=N1，2=N2，3=N3（深睡），4=REM
- **分析**：efficiency_pct、onset_latency_min、rem_latency_min、bout 计数
- **健康目标**：N3 占 15–25%，REM 占 20–25%，效率 >85%，入睡潜伏期 &lt;20 分钟

```bash
npx neuroskill sleep --json | jq '.summary | {n3: .n3_epochs, rem: .rem_epochs}'
npx neuroskill sleep --json | jq '.analysis.efficiency_pct'
```

当用户提及睡眠、疲倦或恢复时使用此命令。

---

## 6. 标注时刻
```bash
npx neuroskill label "breakthrough"
npx neuroskill label "studying algorithms"
npx neuroskill label "post-meditation"
npx neuroskill label --json "focus block start"   # returns label_id
```

在以下情况下自动标注时刻：
- 用户报告突破或洞见
- 用户开始新的任务类型（例如"切换到代码审查"）
- 用户完成重要协议
- 用户要求标记当前时刻
- 发生显著的状态转变（进入/离开心流）

标签存储在数据库中，并通过 `search-labels` 和 `interactive` 命令建立索引以供后续检索。

---

## 7. 实时流式传输
```bash
npx neuroskill listen --seconds 30 --json
npx neuroskill listen --seconds 5 --json | jq '[.[] | select(.event == "scores")]'
```

在指定时长内流式传输实时 WebSocket 事件（EXG、PPG、IMU、评分、标签）。需要 WebSocket 连接（`--http` 模式下不可用）。

适用于持续监控场景，或在协议执行期间实时观察指标变化。

---

## 8. UMAP 可视化
```bash
npx neuroskill umap --json                      # auto: last 2 sessions
npx neuroskill umap --a-start <UTC> --a-end <UTC> --b-start <UTC> --b-end <UTC> --json
```

对 ZUNA 嵌入进行 GPU 加速的 3D UMAP 投影。`separation_score` 表示两次会话在神经层面的差异程度：
- **> 1.5** → 会话在神经层面存在显著差异（不同脑状态）
- **&lt; 0.5** → 两次会话的脑状态相似

---

## 9. 主动状态感知

### 会话开始检查
在会话开始时，如果用户提到正在佩戴设备或询问自身状态，可选择性地执行状态检查：
```bash
npx neuroskill status --json
```

注入简短的状态摘要：
> "快速检查：专注度正在上升至 0.62，放松度良好为 0.55，FAA 为正值 — 趋近动机已激活。看起来是个不错的开始。"

### 何时主动提及状态

**仅在以下情况下**提及认知状态：
- 用户明确询问（"我状态怎么样？"、"检查一下我的专注度"）
- 用户反映难以集中注意力、感到压力或疲劳
- 超过关键阈值（困倦度 > 0.70，专注度 &lt; 0.30 持续）
- 用户即将进行认知要求较高的任务并询问准备情况

**切勿**打断心流状态来报告指标。如果专注度 > 0.75，请保护该会话 — 沉默是正确的响应。

---

## 10. 建议协议

当指标表明有需要时，从 `references/protocols.md` 中建议相应协议。始终在开始前征得同意 — 切勿打断心流状态：

> "您的专注度在过去 15 分钟持续下降，TBR 已超过 1.5 — 这是 theta 主导和心理疲劳的迹象。需要我带您做一个 Theta-Beta 神经反馈锚定练习吗？这是一个 90 秒的练习，通过有节奏的计数和呼吸来抑制 theta 并提升 beta。"

关键触发条件：
- **专注度 &lt; 0.40，TBR > 1.5** → Theta-Beta 神经反馈锚定或箱式呼吸
- **放松度 &lt; 0.30，stress_index 高** → 心脏相干性或 4-7-8 呼吸法
- **认知负荷 > 0.70 持续** → 认知负荷卸载（思维倾倒）
- **困倦度 > 0.60** → 超日节律重置或清醒重置
- **FAA &lt; 0（负值）** → FAA 再平衡
- **心流状态（专注度 > 0.75，投入度 > 0.70）** → 切勿打断
- **高静止度 + headache_index** → 颈部放松序列
- **低 RMSSD（&lt; 25ms）** → 迷走神经调节

---

## 11. 附加工具

### 专注计时器
```bash
npx neuroskill timer --json
```
启动专注计时器窗口，提供 Pomodoro（25/5）、深度工作（50/10）或短时专注（15/5）预设。

### 校准
```bash
npx neuroskill calibrate
npx neuroskill calibrate --profile "Eyes Open"
```
打开校准窗口。适用于信号质量较差或用户希望建立个性化基线时。

### 系统通知
```bash
npx neuroskill notify "Break Time" "Your focus has been declining for 20 minutes"
```

### 原始 JSON 直通
```bash
npx neuroskill raw '{"command":"status"}' --json
```
用于尚未映射到 CLI 子命令的任何服务器命令。

---

## 错误处理

| 错误 | 可能原因 | 解决方法 |
|-------|-------------|-----|
| `npx neuroskill status` 挂起 | NeuroSkill 应用未运行 | 打开 NeuroSkill 桌面应用 |
| `device.state: "disconnected"` | BCI 设备未连接 | 检查蓝牙及设备电量 |
| 所有评分返回 0 | 电极接触不良 | 重新定位头带，润湿电极 |
| `signal_quality` 值 &lt; 0.7 | 电极松动 | 调整佩戴位置，清洁电极触点 |
| SNR &lt; 3 dB | 信号噪声过大 | 减少头部移动，检查环境干扰 |
| `command not found: npx` | 未安装 Node.js | 安装 Node.js 20+ |

---

## 交互示例

**"我现在状态怎么样？"**
```bash
npx neuroskill status --json
```
→ 自然地解读评分，提及专注度、放松度、情绪及任何值得关注的比率（FAA、TBR）。仅在指标表明有需要时才建议采取行动。

**"我无法集中注意力"**
```bash
npx neuroskill status --json
```
→ 检查指标是否印证（高 theta、低 beta、TBR 上升、困倦度高）。
→ 如果得到印证，从 `references/protocols.md` 中建议适当的协议。
→ 如果指标看起来正常，问题可能是动机层面而非神经层面。

**"对比我今天和昨天的专注度"**
```bash
npx neuroskill compare --json
```
→ 解读趋势而非单纯数字。提及哪些方面有所改善、哪些有所下降，以及可能的原因。

**"我上次处于心流状态是什么时候？"**
```bash
npx neuroskill search-labels "flow" --json
npx neuroskill search --json
```
→ 报告时间戳、关联指标及用户当时正在做的事情（来自标签）。

**"我睡得怎么样？"**
```bash
npx neuroskill sleep --json
```
→ 报告睡眠结构（N3%、REM%、效率），与健康目标对比，并指出任何问题（清醒 epoch 过多、REM 不足）。

**"标记这个时刻 — 我刚有了一个突破"**
```bash
npx neuroskill label "breakthrough"
```
→ 确认标签已保存。可选择性地记录当前指标以留存该状态的记忆。

---

## 参考资料

- [NeuroSkill 论文 — arXiv:2603.03212](https://arxiv.org/abs/2603.03212)（Kosmyna & Hauptmann，MIT Media Lab）
- [NeuroSkill 桌面应用](https://github.com/NeuroSkill-com/skill)（GPLv3）
- [NeuroLoop CLI 伴侣](https://github.com/NeuroSkill-com/neuroloop)（GPLv3）
- [MIT Media Lab 项目](https://www.media.mit.edu/projects/neuroskill/overview/)