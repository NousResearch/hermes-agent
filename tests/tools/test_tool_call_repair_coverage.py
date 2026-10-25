"""FIX-2066 regression fixtures: real stringified tool_call payloads that the
rescue stack previously rejected, captured from state.db (10-24/10-25 windows,
all Qwen3.8-27B task-card sessions).

Each payload is the EXACT ``calls`` value the model emitted. The registry
fixture below mirrors the production schemas of the tools the payloads
reference (ekko-studio task-card MCP tools as the watchdog mocks them; core
tools trimmed to the properties each payload exercises) — the same
registration mechanism the gateway uses, so the no-guess gates run against
real schemas.
"""

import json

import pytest

from tools.registry import registry
from tools.tool_search_validation import normalize_tool_call_entries

_UPDATE_PLAN = "mcp__ekko_studio_interaction__ekko_studio_update_plan"
_UPDATE_TASKS = "mcp__ekko_studio_interaction__ekko_studio_update_session_tasks"
_LIST_TASKS = "mcp__ekko_studio_interaction__ekko_studio_list_session_tasks"
_EKKO = "mcp-ekko-studio-interaction"

_SPECS = [
    (_UPDATE_PLAN, ["context_id", "plan"],
     {"token": {"type": "string"}, "profile": {"type": "string"},
      "context_id": {"type": "string"}, "explanation": {"type": "string"},
      "plan": {"type": "array"}}, _EKKO),
    (_UPDATE_TASKS, ["context_id", "revision", "changes"],
     {"token": {"type": "string"}, "profile": {"type": "string"},
      "context_id": {"type": "string"}, "revision": {"type": "integer"},
      "changes": {"type": "array"}}, _EKKO),
    (_LIST_TASKS, ["context_id"],
     {"token": {"type": "string"}, "profile": {"type": "string"},
      "context_id": {"type": "string"}}, _EKKO),
    ("read_file", ["path"],
     {"path": {"type": "string"}, "offset": {"type": "integer"},
      "limit": {"type": "integer"}}, "core"),
    ("terminal", ["command"],
     {"command": {"type": "string"}, "background": {"type": "boolean"},
      "timeout": {"type": "integer"}, "workdir": {"type": "string"},
      "pty": {"type": "boolean"}}, "core"),
    ("process_manage", ["action"],
     {"action": {"type": "string"}, "session_id": {"type": "string"},
      "data": {"type": "string"}, "timeout": {"type": "integer"},
      "limit": {"type": "integer"}}, "core"),
    ("cronjob_manage", ["action"],
     {"action": {"type": "string"}, "job_id": {"type": "string"},
      "name": {"type": "string"}, "prompt": {"type": "string"},
      "schedule": {"type": "string"}, "deliver": {"type": "string"},
      "enabled_toolsets": {"type": "array"}, "paused": {"type": "boolean"}}, "core"),
    ("session_search", [],
     {"query": {"type": "string"}, "limit": {"type": "integer"},
      "sort": {"type": "string"}, "session_id": {"type": "string"},
      "around_message_id": {"type": "integer"}, "window": {"type": "integer"}}, "core"),
    ("x_search", ["query"],
     {"query": {"type": "string"}, "from_date": {"type": "string"},
      "to_date": {"type": "string"}}, "core"),
    # 10-25 adversarial-review guards: a {name}-key fingerprint must be able to
    # be AMBIGUOUS (two registered tools with a "name" property) so the
    # lift-vs-fingerprint conflict test can hold. skill_view's real schema has
    # required=["name"]; the second is a synthetic no-required {name,note} tool
    # mirroring the real registry's no-required "name" property class.
    ("skill_view", ["name"],
     {"name": {"type": "string"}, "file_path": {"type": "string"}}, "core"),
    ("fixture_nameprop_tool", [],
     {"name": {"type": "string"}, "note": {"type": "string"}}, "core"),
]


@pytest.fixture
def tool_registry():
    for name, required, props, toolset in _SPECS:
        registry.register(
            name=name, toolset=toolset, handler=lambda a, **kw: "{}",
            schema={"name": name, "description": "fixture",
                    "parameters": {"type": "object", "properties": props,
                                   "required": required}})
    try:
        yield
    finally:
        for name, _r, _p, _t in _SPECS:
            registry.deregister(name)


def _naive_parse_fails(payload):
    import pytest as _pt
    with _pt.raises(json.JSONDecodeError):
        json.loads(payload)


# ---------------------------------------------------------------------------
# The six measured 10-25 failures (acceptance: all six rescued)
# ---------------------------------------------------------------------------


def test_fixture_612161_dangling_name_at_array_level_with_stray_arg(tool_registry):
    """rowid 612161 (sess local-bee8b9): the third entry kept "limit" at ENTRY
    level (it belongs to read_file's arguments), closed the entry early, then
    emitted the name at array level followed by a stray closer. Rescue =
    dangle re-attachment; the stray folds into arguments because read_file's
    schema accepts path+limit."""
    payload = '[{"arguments": {"session_id": "mvmhx7yp1mwmjc", "around_message_id": 612100, "window": 15}, "name": "session_search"}, {"arguments": {"session_id": "mvmke8qwsvw3if"}, "name": "session_search"}, {"arguments": {"path": "/Users/xiawenhua/Wiki/02-洞察/投资/验证_跌60%反弹30-50%与月线量200%逃顶信号.md"}, "limit": 30}, "name": "read_file"}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == ["session_search", "session_search", "read_file"]
    assert entries[2]["arguments"]["limit"] == 30
    assert entries[2]["arguments"]["path"].endswith("验证_跌60%反弹30-50%与月线量200%逃顶信号.md")


def test_fixture_612929_entry_close_drift_two_nameless_entries(tool_registry):
    """rowid 612929 (sess mvmrbqo0zad4): the last entry's closer drifted to a
    bracket and the tail carried a surplus closer; BOTH entries are nameless
    and resolve by unique schema fingerprint (plan vs changes shape)."""
    payload = '[{"arguments": {"context_id": "310c3242-346f-411b-9efc-b4f19e2b8a9b", "explanation": "新信息（传统IT+AI本地化交付+烟草跨市+4名员工）使上轮“收割模式”结论失效，需以公司为主轴重出计划", "plan": [{"id": "recalc", "step": "重算公司成本结构与500万战备金分配", "status": "in_progress"}, {"id": "deliver", "step": "交付以公司为主轴的修正版经营+财务计划", "status": "pending"}, {"id": "ledger", "step": "更新会话账本（重开原目标并记录待补信息）", "status": "pending"}]}}, {"arguments": {"context_id": "310c3242-346f-411b-9efc-b4f19e2b8a9b", "revision": 2, "changes": [{"id": "78da77f8-3be4-4a02-b114-c99e618ea1bd", "status": "open", "summary": "新信息：传统IT公司正向AI转型（低成本本地化AI交付），客户烟草（已跨市）+高校，服务占比高，4名员工月开销3.2万。上轮“收割模式”作废，重出以公司为主轴的计划。待补：80万现金vs应收、AI成果客户集中度、跨市机制。"}]}]}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == [_UPDATE_PLAN, _UPDATE_TASKS]
    assert entries[1]["arguments"]["revision"] == 2


def test_fixture_612965_dangling_name_after_batch_close(tool_registry):
    """rowid 612965 (sess mvmrbqo0zad4): the batch closed before the last
    entry's name was emitted — the dangle sits at ROOT level. The dangle
    (update_plan) fits the FIRST entry's arguments and not the second's
    (revision/changes): schema-fit arbitration picks the unique target, the
    other entry resolves by fingerprint."""
    payload = '[{"arguments": {"context_id": "9440715c-24d8-4c81-9bca-7b51b1de7451", "explanation": "80万是纯利润（此前误当营收），在手合同160万，公司盈利结构与悬崖时间轴均需重算", "plan": [{"status": "in_progress", "id": "recalc2", "step": "重算公司盈利结构与160万在手合同时间轴"}, {"status": "pending", "id": "redeliver", "step": "交付修正版计划（500万再分配+悬崖后移）"}, {"status": "pending", "id": "ledger2", "step": "更新会话账本"}]}}, {"arguments": {"context_id": "9440715c-24d8-4c81-9bca-7b51b1de7451", "revision": 3, "changes": [{"id": "78da77f8-3be4-4a02-b114-c99e618ea1bd", "status": "open", "summary": "新信息：80万/年是纯利润（非营收），公司在手合同160万。公司远比预估健康（利润率~54%），悬崖时间轴后移。需重算500万分配与验证节点。待确认：160万是合同额还是利润额；AI成果客户集中度；跨市机制。", "title": "为对话中的人物出具体财务/经营计划"}]}], "name": "mcp__ekko_studio_interaction__ekko_studio_update_plan"}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == [_UPDATE_PLAN, _UPDATE_TASKS]
    assert entries[1]["arguments"]["revision"] == 3


def test_fixture_613401_escape_drift_nested_quotes(tool_registry):
    """rowid 613401 (sess mvkt75f75xav): the summary value quotes nested text;
    the model opened the value with an ESCAPED quote and kept escaping every
    structural quote to the tail. Rescue = the earliest boundary whose
    candidate re-parses to a fully resolvable batch; the later boundary
    (value swallowing the over-escaped title pair) is a strictly larger
    deviation and must not win."""
    payload = '[{"arguments": {"context_id": "e8a50224-40ff-43cd-b625-2df10abe434e", "plan": [{"id": "ledger", "status": "completed", "step": "读账本 + 重开目标（上下文假设验证）"}, {"id": "bench2", "status": "completed", "step": "基准扩展：V3 next-3 / V4 历史 / V5 两者 + 消融 V6/V7（只塞 state 不改指令）"}, {"id": "run", "status": "completed", "step": "同种子 5 局×5 变体，对照 V2 基线 142块/45行"}, {"id": "verdict", "status": "completed", "step": "数据判定：上下文越多越差（142→88/79/65/56/48），假设证伪"}, {"id": "ship", "status": "completed", "step": "不 ship（全负结果）；demo 维持 V2 最优档"}]}, "name": "mcp__ekko_studio_interaction__ekko_studio_update_plan"}, {"arguments": {"changes": [{"completion_evidence": "5 变体×同种子 5 局受控实验：加上下文（next-3/20步历史/两者）全负，均值 142→88/79/65/56/48 块；消融 V6/V7（只塞 state 不改指令）同样变差，排除提示词混淆。上下文假设被数据证伪，demo 维持 V2。", "id": "3b20c7c2-63c6-4b5b-aae2-7274b92b18e4", "status": "completed", "summary": \\"已用受控实验证伪\\\\\\"clef 靠上下文判断\\\\\\"：加 next-3/对局历史到 state，表现单调变差（142→48 块）。长上下文≠长程判断，单步决策头被上下文分散。V2(45行)仅为休闲水平，非世界冠军。\\", \\"title\\": \\"clef 图片决策通路（俄罗斯方块玩法）\\"}], \\"context_id\\": \\"e8a50224-40ff-43cd-b625-2df10abe434e\\", \\"revision\\": 7}, "name": "mcp__ekko_studio_interaction__ekko_studio_update_session_tasks"}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == [_UPDATE_PLAN, _UPDATE_TASKS]
    change = entries[1]["arguments"]["changes"][0]
    assert change["summary"].startswith("已用受控实验证伪")
    assert "\\" in change["summary"] or '"' in change["summary"]
    assert change["title"] == "clef 图片决策通路（俄罗斯方块玩法）"
    assert entries[1]["arguments"]["context_id"] == "e8a50224-40ff-43cd-b625-2df10abe434e"


def test_fixture_613854_name_trapped_in_arguments(tool_registry):
    """rowid 613854 (sess local-9d776c): the second entry dropped its
    arguments closer, trapping the tool name INSIDE arguments; the tail
    bracket errors. Rescue = k-run (drop the drifted closer, re-close
    entry+batch) plus lifting the trapped name (its reduced args fit
    terminal's schema)."""
    payload = '[{"arguments": {"action": "poll", "session_id": "proc_92bbb1e6c7e5"}, "name": "process_manage"}, {"arguments": {"command": "cd ~/projects/hermes-agent && ls tests/e2e/ | head -20; echo \\"=== playwright? ===\\"; grep -rln \\"playwright\\" tests/e2e/ 2>/dev/null | head -5; cat playwright.config.* 2>/dev/null | head -15", "name": "terminal"}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == ["process_manage", "terminal"]
    assert "name" not in entries[1]["arguments"]
    assert entries[1]["arguments"]["command"].startswith("cd ~/projects/hermes-agent")


def test_fixture_613927_name_trapped_in_arguments_second_slip(tool_registry):
    """rowid 613927 (sess local-9d776c, three minutes after 613854): same
    shape, different command — the repeated slip the regression demands."""
    payload = '[{"arguments": {"action": "kill", "session_id": "proc_92bbb1e6c7e5"}, "name": "process_manage"}, {"arguments": {"command": "cd ~/projects/hermes-agent && ls tests/tools/ | grep -iE \\"env|terminal|local\\" | head -20; echo \\"=== pm files with PYTHONPATH assertions ===\\"; grep -rln \\"PYTHONPATH\\" tests/pm/ | head", "name": "terminal"}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == ["process_manage", "terminal"]
    assert "name" not in entries[1]["arguments"]

# ---------------------------------------------------------------------------
# Backlog form samples (10-24, 18-day blind window; a few per measured form)
# ---------------------------------------------------------------------------

def test_fixture_609650_backlog_nameless_plan_entry_drifted_tail(tool_registry):
    """rowid 609650 (sess mvm57ofwgwwvvu — 8 identical failures in the blind
    window): single nameless update_plan entry whose closer drifted to a
    bracket with a surplus batch closer. k-run + unique fingerprint."""
    payload = '[{"arguments": {"context_id": "c952bbe4-1424-405c-bfcf-e45ae636df34", "explanation": "Phase 1a 完成：main 对齐 0.7.39+7、6 分支 rebase（1 冲突已合）、local-fixes 重建 5 fixes", "plan": [{"id": "preflight", "status": "completed", "step": "Pre-flight：remotes/工作区/HFC 5.0.7 fixtures 定 Agent 目标"}, {"id": "webui-sync", "status": "in_progress", "step": "Phase 1a Web UI：main→upstream、rebase fix/*、重建 local-fixes、push fork"}, {"id": "webui-deploy", "status": "pending", "step": "Phase 1b Web UI：npm ci+build、部署验证、重启 :81"}, {"id": "agent-sync", "status": "pending", "step": "Phase 2a Agent：对齐 HFC 钉 v0.22.4、rebase、重建 local-fixes"}, {"id": "agent-hfc", "status": "pending", "step": "Phase 2b Agent：HFC 5.0.7 三环境对齐 + hooks 重注入 + sidecar 验证"}, {"id": "verify", "status": "pending", "step": "升级后验证矩阵（版本/HTTP/doctor/平台连接）"}, {"id": "gateway-restart", "status": "pending", "step": "Phase 3：gateway 重启（等用户确认）"}]}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == [_UPDATE_PLAN]
    assert len(entries[0]["arguments"]["plan"]) == 7


def test_fixture_594780_backlog_named_entry_stray_closer_run(tool_registry):
    """rowid 594780: a NAMED entry followed by a mangled run of three surplus
    closers with the batch never closed. k-run drops the run and re-closes;
    every surviving completion reads the same batch, so the rescue is unique."""
    payload = '[{"name": "mcp__ekko_studio_interaction__ekko_studio_update_plan", "arguments": {"context_id": "1e85a33a-c7e5-430c-964b-58bfa6b8d6c6", "plan": [{"id": "survey", "step": "摸清本地 3 个 fix 分支与上游 #54 重叠 + 部署形态", "status": "in_progress"}, {"id": "backup", "step": "备份当前 local-fixes 到 backup 分支", "status": "pending"}, {"id": "rebase", "step": "rebase 本地 fix 分支到 origin/main（解决 #54 冲突，保留本地完整版）", "status": "pending"}, {"id": "compose", "step": "重建 local-fixes = origin/main + 本地 fix，清理已合并分支", "status": "pending"}, {"id": "push", "step": "推送 fork 镜像（main/local-fixes/fix 分支）", "status": "pending"}, {"id": "build", "step": "npm 构建（env -u NODE_ENV）", "status": "pending"}, {"id": "verify", "step": "重启服务 + 健康验证（port 83 / onboarding status）", "status": "pending"}]}}}]}'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == [_UPDATE_PLAN]


def test_fixture_557545_backlog_surplus_close_missing_batch(tool_registry):
    """rowid 557545: entry closed correctly, one surplus closer after it, and
    the batch closer never emitted. k-run: drop the stray, append the batch."""
    payload = '[{"name": "mcp__ekko_studio_interaction__ekko_studio_update_plan", "arguments": {"context_id": "92bdec2c-662d-407c-bbdf-403793f146fb", "plan": [{"id": "upstream", "status": "completed", "step": "查上游: 命中 LMCache#4154 — 原生C传输kernel对多格式hybrid拓扑报CUDA invalid argument, 官方确认pure-torch fallback可用"}, {"id": "decide", "status": "in_progress", "step": "定位 0.5.5 中 native kernel 的开关, 决定强制 fallback 方式"}, {"id": "apply", "status": "pending", "step": "应用开关(纯Python/env, 不动二进制) 重启 store"}, {"id": "verify", "status": "pending", "step": "重跑 8003 写入+读取, 确认 invalid argument 消失、retrieve 命中"}]}}}'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == [_UPDATE_PLAN]


def test_fixture_587063_backlog_cronjob_job_name_is_a_real_parameter(tool_registry):
    """rowid 587063: single nameless entry whose arguments carry a "name" key —
    here the CRON JOB's name, a legitimate cronjob_manage property. The direct
    fingerprint must win over the tool-name lift, so "name" stays in
    arguments. Tail = args closer + one drifted bracket, entry/batch closers
    missing (k-run append)."""
    payload = '[{"arguments": {"action": "create", "deliver": "qqbot", "enabled_toolsets": ["terminal"], "name": "10-16午评下午剧本验证(一次性)", "prompt": "你是A股行情验证 agent。现在拉取实时行情并输出一条简短验证报告（人话，≤200字）。\\n\\n步骤1：执行 python3 脚本拉腾讯行情（GBK解码）：http://qt.gtimg.cn/q=sz300308,sh515880,sh000852,sh000905,sz300498,sh600760,sh600893,sz002714 —— 字段：f[3]=现价 f[4]=昨收 f[5]=今开 f[32]=涨跌% f[33]=最高 f[34]=最低。\\n\\n步骤2：对照 10-16 博主午盘口播的下午剧本，逐项判定：\\n① 方向：13:00-13:30 科技/通信ETF(515880)是加速涨（高举高打→剧本判定=无买入机会）还是下跌？\\n② 触发：若下跌——中际旭创(300308)现价是否跌破 671（博主称的黄色日均线，口径存疑，照报数字即可）；中证1000/中证500 是否一路加速跌到 14:30 前后？两者都满足=触发『买回通信ETF 2-3成仓位』信号。\\n③ 个股：温氏(300498)午后是否突破 15.3 前高（上午高15.27差0.02）；中航沈飞(600760)是否挑战 46.6 前压（上午高46.30）；航发动力(600893)在 40.2 压力上方是否站稳（上午高40.40已破）；牧原(002714) 表现。\\n\\n步骤3：报告格式——第一行『10-16下午剧本验证 HH:MM』，然后①②③各一句，最后给结论：『未触发，继续等周一』或『触发，通信ETF 2-3成买入信号』或『部分触发，观察尾盘』。不要给投资建议，只报事实+博主剧本对照。", "schedule": "2026-10-16T14:35:00"}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == ["cronjob_manage"]
    assert entries[0]["arguments"]["name"] == "10-16午评下午剧本验证(一次性)"
    assert entries[0]["arguments"]["schedule"] == "2026-10-16T14:35:00"


def test_fixture_577515_backlog_inner_arguments_surplus_close(tool_registry):
    """rowid 577515: the ENVELOPE parses — it is the entry's stringified
    arguments that carries one surplus tail closer. The per-entry k-run (dict
    target) recovers the unique reading."""
    calls = [{'arguments': '{"context_id": "c5359dc5-4db8-4edc-9755-773fd5ba01db", "plan": [{"id": "recon", "status": "completed", "step": "摸底: VR 仅 vite.config.ts 一冲突; vibe-astock 195提交=架构重写不可机械rebase"}, {"id": "decide", "status": "completed", "step": "定案: VR 用 VRA_API_PORT 环境变量收编直接升; vibe-astock main对齐运行栈不动"}, {"id": "upg-vr", "status": "completed", "step": "Vibe-Research: rebase到v2.2.1 + 测试(零回归) + plist加VRA_API_PORT + 部署验证(8766/5930/LAN/llm全通)"}, {"id": "upg-astock", "status": "in_progress", "step": "vibe-astock: main ff对齐d5bf80d, local-fixes/8910不动"}]}}', 'name': 'mcp__ekko_studio_interaction__ekko_studio_update_plan'}]
    inner = calls[0]["arguments"]
    _naive_parse_fails(inner)
    entries, err = normalize_tool_call_entries({"calls": calls})
    assert err is None, err
    assert [e["name"] for e in entries] == [_UPDATE_PLAN]
    assert len(entries[0]["arguments"]["plan"]) == 4


def test_fixture_571114_backlog_query_fingerprint_ambiguous_not_guessed(tool_registry):
    """rowid 571114: a nameless single-key entry. The tail IS repairable
    (k-run parses it), but the fingerprint is genuinely ambiguous —
    session_search (no required parameters) and x_search (required [query])
    both fit. no-guess: reject rather than pick one."""
    payload = '[{"arguments": {"query": "看板巡检 建议"}]}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert entries == []
    assert err is not None


def test_fixture_599244_backlog_double_defect_wrong_property_rejected(tool_registry):
    """rowid 599244: the missing tail closer is repairable, but the third
    entry's arguments use "queries" — a property session_search does not have
    (the model confused it with tool_search). Lifting the trapped name would
    certify a call schema validation must refuse, so the rescue declines."""
    payload = '[{"arguments": {"session_id": "mvgpf8cfwcmo2k"}, "name": "session_search"}, {"arguments": {"session_id": "mvgmylmrlgl2nn"}, "name": "session_search"}, {"arguments": {"queries": ["upgrade 升级", "决策 方案 部署", "错误 失败 修复"], "name": "session_search"}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert entries == []
    assert err is not None


def test_fixture_582641_backlog_degenerate_runaway_dangle_rejected(tool_registry):
    """rowid 582641 (93KB): the model looped the same plan entry 174 times and
    leaked a TWO-key root dangle (explanation + name) — X would fit every one
    of the 174 identical entries. Synthetic mini of the identical shape:
    dangle recognized (teaching error) but not guessed."""
    e = ('{"arguments": {"context_id": "c-1", "plan": '
         '[{"id": "s", "status": "pending", "step": "x"}]}}')
    payload = ("[" + e + ", " + e + '], "explanation": "创建升级计划"}, '
               '"name": "' + _UPDATE_PLAN + '"}]')
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert entries == []
    assert "landed OUTSIDE its entry" in (err or "")


def test_fixture_561153_backlog_placeholder_stub_stays_rejected(tool_registry):
    """rowid 561153: the model stubbed the call with arguments =
    "placeholder" — not a bracket slip, a call it did not finish. No repair
    may invent content for it."""
    entries, err = normalize_tool_call_entries(
        {"calls": [{"arguments": "placeholder", "name": "noop"}]})
    assert entries == []
    assert "arguments is not valid JSON" in (err or "")

# ---------------------------------------------------------------------------
# No-guess guard tests (synthetic shapes)
# ---------------------------------------------------------------------------

def test_krun_uniqueness_dedupes_equivalent_completions(tool_registry):
    """Different k/suffix completions that read the SAME batch dedupe to one
    canonical result — the uniqueness gate counts readings, not edits."""
    from tools.tool_search_validation import _collect_tail_closer_run
    text = '[{"name": "session_search", "arguments": {"query": "q"}}'
    found = _collect_tail_closer_run(text + "]}")
    assert len(found) == 1


def test_dangling_name_fits_two_entries_rejected(tool_registry):
    """A root dangle whose name fits BOTH nameless entries is a guess —
    reject with the teaching error, not a coin flip."""
    e = ('{"arguments": {"context_id": "c-1", "plan": '
         '[{"id": "s", "status": "pending", "step": "x"}]}}')
    payload = "[" + e + ", " + e + '], "name": "' + _UPDATE_PLAN + '"}]'
    _naive_parse_fails(payload)
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert entries == []
    assert "landed OUTSIDE its entry" in (err or "")


def test_lifted_name_out_of_arguments_rescued(tool_registry):
    """args carry a "name" key whose value names a registered tool and whose
    reduced args fit that tool: lift it to entry level (the 613854/599244
    family without the bracket damage)."""
    payload = json.dumps([{"arguments": {"query": "q", "name": "x_search"}}])
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert err is None, err
    assert [e["name"] for e in entries] == ["x_search"]
    assert "name" not in entries[0]["arguments"]


def test_lift_vs_ambiguous_fingerprint_rejected(tool_registry):
    """10-25 adversarial review, defect 1: a trapped "name" value must NOT win
    the lift when the argument fingerprint is itself ambiguous — {name} fits
    skill_view (required=["name"]) and fixture_nameprop_tool (no required),
    so {"name": "session_search"} has >=3 valid readings. The old code lifted
    unconditionally in this case (clean JSON, no corruption)."""
    payload = json.dumps([{"arguments": {"name": "session_search"}}])
    entries, err = normalize_tool_call_entries({"calls": payload})
    assert entries == []
    assert "requires a 'name'" in (err or "")
    # ...while a fingerprint with ZERO candidates still lifts (shape E):
    payload2 = json.dumps([{"arguments": {"command": "ls", "name": "terminal"}}])
    entries2, err2 = normalize_tool_call_entries({"calls": payload2})
    assert err2 is None, err2
    assert [e["name"] for e in entries2] == ["terminal"]


def test_dangling_name_typo_refused_by_dangle_strategy(tool_registry):
    """10-25 adversarial review, defect 2: the dangle repair must not attach a
    name that no registered tool carries (typo). _repair_dangling_name returns
    (None, None) — 'shape does not apply' — so no false 'recovered' log and no
    fake name riding into a sibling-valid batch. (The pre-existing
    single-char-edit strategy may still repair the bracket and pass the
    explicit name through to the dispatch gate, which rejects unknown tools —
    that fail-open is the original contract, not the dangle path's.)"""
    from tools.tool_search_validation import _repair_dangling_name
    text = '[{"arguments": {"command": "ls"}}, "name": "terminall"}]'
    assert _repair_dangling_name(text) == (None, None)
    # a real registered name still repairs (the measured shape):
    text2 = '[{"arguments": {"path": "/x"}}, "name": "read_file"}]'
    recovered, teach = _repair_dangling_name(text2)
    assert recovered is not None and teach is None
    assert recovered[0]["name"] == "read_file"

