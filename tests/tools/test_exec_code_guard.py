"""Regression tests for the execute_code approval guard (#65592).

Pins the helper contracts added by PR #65592 and extended after the
2026-08-25 andrexibiza review:

1. Sensitive file writes (#49578): builtin ``open()`` write modes and
   pathlib ``Path.write_text`` / ``write_bytes`` / ``Path.open`` must be
   flagged; read-only forms must pass.
2. Process-kill hard block: direct, import-aliased, assignment-aliased
   (incl. chains), star-imported, ``getattr`` and ``__dict__`` dynamic
   forms must be caught; ordinary os usage must pass.
3. Conversation-loop user-denial halt: plain-text and JSON-wrapped
   BLOCKED tool results must be recognized.

The denial tests fail on the pre-review implementation (which skipped
star imports, assignment aliases, and pathlib writes).
"""

import pytest

from tools.approval import (
    _execute_code_has_dangerous_ops,
    _execute_code_has_self_destructive_ops,
    check_execute_code_guard,
)
from tools.exec_code_policy import (
    _execute_code_has_sensitive_write,
    _execute_code_has_package_acquisition,
    _execute_code_touches_sensitive_path,
)


# ─────────────────────────────────────────────────────────────────────
# Blocker 1: #49578 sensitive file writes (open + pathlib)
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    # raw open() write modes (the #49578 reproducer shape)
    'open("/path/.hermes/config.yaml", "a").write("injected")',
    'open("/path/.hermes/config.yaml", "w").write("injected")',
    'open("/path/x", "x").write("injected")',
    'open("/path/x", "r+").write("injected")',
    'with open("/path/.hermes/config.yaml", "a") as f:\n    f.write("injected")',
    # open() with non-literal mode must fail closed (conservative)
    'mode = "a"\nopen("/path/x", mode).write("x")',
    # pathlib write surfaces (review: not recognized before)
    'from pathlib import Path\nPath("/path/.hermes/config.yaml").write_text("injected")',
    'from pathlib import Path\nPath("/path/.hermes/config.yaml").write_bytes(b"injected")',
    'from pathlib import Path\nwith Path("/path/.hermes/config.yaml").open("a") as f:\n    f.write("x")',
    'import pathlib\npathlib.Path("/path/x").write_text("x")',
    # aliased pathlib constructor
    'from pathlib import Path as P\nP("/path/x").write_text("x")',
])
def test_sensitive_file_write_is_flagged(code):
    assert _execute_code_has_dangerous_ops(code) == "open-write"


@pytest.mark.parametrize("code", [
    'open("/path/x", "r").read()',
    'open("/path/x").read()',  # default mode is read
    'with open("/path/x", "rb") as f:\n    data = f.read()',
    'from pathlib import Path\nPath("/path/x").read_text()',
    'from pathlib import Path\nwith Path("/path/x").open() as f:\n    print(f.read())',
    'from pathlib import Path\nwith Path("/path/x").open("r") as f:\n    print(f.read())',
])
def test_read_only_file_access_passes(code):
    assert _execute_code_has_dangerous_ops(code) is None


# ─────────────────────────────────────────────────────────────────────
# Blocker 2: process-kill hard block — alias / dynamic forms
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    # direct (baseline)
    'import os\nos.kill(os.getpid(), 15)',
    'import os\nos.killpg(0, 9)',
    # import aliases (baseline from July review)
    'import os as o\no.kill(os.getpid(), 15)',
    'from os import kill\nkill(os.getpid(), 15)',
    # assignment alias (review: not caught before)
    'import os\nkiller = os.kill\nkiller(os.getpid(), 15)',
    'import os\nk = os.killpg\nk(0, 9)',
    # chained assignment alias
    'import os\na = os\nb = a.kill\nb(os.getpid(), 15)',
    # star import (review: skipped before)
    'from os import *\nkill(os.getpid(), 15)',
    # getattr dynamic (review-adjacent)
    'import os\ngetattr(os, "kill")(os.getpid(), 15)',
    'import os\ngetattr(os, "killpg")(0, 9)',
    # __dict__ access
    'import os\nos.__dict__["kill"](os.getpid(), 15)',
])
def test_process_kill_forms_are_hard_blocked(code):
    assert _execute_code_has_self_destructive_ops(code) is not None


@pytest.mark.parametrize("code", [
    'import os\nprint(os.getpid())',
    'import os\nprint(os.environ.get("HOME", ""))',
    'import os\np = os.path.join("/tmp", "x")\nprint(p)',
    'import os\nprint(getattr(os, "environ").get("HOME", ""))',
    'import sys\nprint(sys.version)',
    'import math\nprint(math.sqrt(16))',
    'import json\nprint(json.dumps({"a": 1}))',
    'import subprocess\nprint("no call")',  # import alone is not a call
])
def test_benign_os_sys_usage_passes(code):
    assert _execute_code_has_self_destructive_ops(code) is None
    assert _execute_code_has_dangerous_ops(code) is None


# ─────────────────────────────────────────────────────────────────────
# Dangerous ops: alias / star / exec* forms
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,reason", [
    ('import subprocess as sp\nsp.run(["rm", "-rf", "/"], capture_output=True)', "command-exec"),
    ('import subprocess as sp\nrun = sp.run\nrun(["ls"])', "command-exec"),
    ('from shutil import *\nrmtree("/important")', "file-delete"),
    ('import os\nos.execv("/bin/sh", ["sh"])', "command-exec"),
    ('import os\nos.system("rm -rf /")', "command-exec"),
    ('from os import remove\nremove("/path/x")', "file-delete"),
])
def test_dangerous_call_aliases_are_flagged(code, reason):
    assert _execute_code_has_dangerous_ops(code) == reason


# ─────────────────────────────────────────────────────────────────────
# Guard end-to-end: hard block returns outcome=hard_blocked
# ─────────────────────────────────────────────────────────────────────

def test_guard_returns_hard_blocked_outcome():
    result = check_execute_code_guard(
        'import os\nos.kill(os.getpid(), 15)', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "HARD BLOCKED" in result["message"]


def test_guard_hard_blocks_alias_after_yolo_check():
    # Even with yolo, hard block must fire (it runs before the mode gate).
    result = check_execute_code_guard(
        'import os\nkiller = os.kill\nkiller(os.getpid(), 15)', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


# ─────────────────────────────────────────────────────────────────────
# Re-review (2026-08-25, andrexibiza) Blocker 1: sensitive-write invariant
# ─────────────────────────────────────────────────────────────────────

SENSITIVE_WRITE_CASES = [
    # literal protected targets
    'open("/root/.hermes/config.yaml", "a").write("injected")',
    'open("/root/.ssh/authorized_keys", "a").write("key")',
    'with open("/root/.hermes/config.yaml", "a") as f:\n    f.write("injected")',
    # expanduser forms (the #49578 reproducer shape)
    'import os\ntarget = os.path.expanduser("~/.hermes/config.yaml")\nwith open(target, "a") as f:\n    f.write("injected")',
    'import os\nopen(os.path.expanduser("~/.hermes/config.yaml"), "a").write("x")',
    # pathlib to protected targets
    'from pathlib import Path\nPath("/root/.ssh/authorized_keys").write_text("key")',
    'from pathlib import Path\nwith Path("/etc/passwd").open("a") as f:\n    f.write("x")',
    # simple variable alias
    'target = "/root/.hermes/config.yaml"\nopen(target, "a").write("x")',
]


@pytest.mark.parametrize("code", SENSITIVE_WRITE_CASES)
def test_sensitive_write_target_detected(code):
    assert _execute_code_has_sensitive_write(code) is not None


@pytest.mark.parametrize("code", [
    'open("/tmp/x.txt", "w").write("x")',
    'open("/home/user/project/a.py", "w").write("x")',
    'open("/root/.hermes/config.yaml", "r").read()',
    'from pathlib import Path\nPath("/root/.hermes/config.yaml").read_text()',
])
def test_non_sensitive_write_passes(code):
    assert _execute_code_has_sensitive_write(code) is None


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_sensitive_write_hard_blocked_in_all_modes(monkeypatch, mode_gate):
    """The #49578 destination invariant must hold even when approval is
    turned off — the sensitive-write check runs before the yolo/mode-off
    bypass gates (re-review Blocker 1)."""
    code = ('import os\ntarget = os.path.expanduser("~/.hermes/config.yaml")\n'
            'with open(target, "a") as f:\n    f.write("injected")')

    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )
    # normal: nothing to patch — env_type="local" goes through the guard

    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


# ─────────────────────────────────────────────────────────────────────
# Re-review (2026-08-25) Blocker 2: eval/exec dynamic-exec detection
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    'import os\neval("os.kill")(os.getpid(), 15)',
    'exec("os.kill(os.getpid(), 15)")',
    'import os\ncompile("os.kill(1,9)", "<s>", "exec")',
])
def test_eval_exec_dynamic_code_flagged(code):
    """eval("os.kill")(...) escapes the hard-block resolver (outer callee
    is eval) but must not fall through to auto-approve — it is flagged as
    dynamic-exec and requires approval."""
    assert _execute_code_has_dangerous_ops(code) == "dynamic-exec"


def test_eval_exec_guard_not_auto_approved(monkeypatch):
    """dynamic-exec must fall through to the approval prompt rather than
    the danger_reason-is-None auto-approve path."""
    import tools.approval as approval_module
    # Force CLI-like path: not gateway, not ask, not yolo, not off.
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    result = check_execute_code_guard(
        'import os\neval("os.kill")(os.getpid(), 15)', env_type="local"
    )
    # It must NOT be auto-approved; it should be blocked/ask for approval.
    assert result["approved"] is False or result.get("outcome") in (
        "blocked", "denied", "pending",
    )


# ─────────────────────────────────────────────────────────────────────
# Conversation-loop user-denial halt (plain-text + JSON-wrapped BLOCKED)
# ─────────────────────────────────────────────────────────────────────

from agent.conversation_loop import (
    _tool_results_contain_user_blocked,
    _user_blocked_halt_response,
)


def _tool_msgs(*contents):
    """Build trailing tool messages (all role=tool) around a user ask."""
    messages = [{"role": "user", "content": "do it"}]
    for c in contents:
        messages.append({"role": "tool", "content": c})
    return messages


def test_plain_text_blocked_detected():
    assert _tool_results_contain_user_blocked(
        _tool_msgs("BLOCKED: User denied dangerous command")
    ) is True


def test_bracket_blocked_list_detected():
    assert _tool_results_contain_user_blocked(
        _tool_msgs('["BLOCKED: User denied", "more context"]')
    ) is True


def test_json_wrapped_blocked_detected():
    # execute_code denial format: {"status":"error","error":"BLOCKED: ..."}
    assert _tool_results_contain_user_blocked(
        _tool_msgs('{"status": "error", "error": "BLOCKED: User denied"}')
    ) is True


def test_non_blocked_tool_result_passes():
    assert _tool_results_contain_user_blocked(
        _tool_msgs("command output ok", '{"status": "error", "error": "boom"}')
    ) is False


def test_only_trailing_tool_messages_scanned():
    # A BLOCKED in an OLD tool message (followed by a newer non-BLOCKED
    # tool message) must not halt the turn — only the trailing batch is
    # the current tool execution's output.
    messages = [
        {"role": "user", "content": "go"},
        {"role": "tool", "content": "BLOCKED: old denial"},
        {"role": "assistant", "content": "trying again"},
        {"role": "tool", "content": "all good"},
    ]
    assert _tool_results_contain_user_blocked(messages) is False


# ─────────────────────────────────────────────────────────────────────
# Loop-boundary control flow: exit reason + no-retry semantics
# (2026-08-25 re-review: parser recognition alone does not prove
# termination semantics)
# ─────────────────────────────────────────────────────────────────────

class _FakeAgent:
    """Minimal agent stub with the halt side-effect surface."""

    def __init__(self):
        self.emitted = []
        self.printed = []
        self.streamed = []
        self.stream_delta_callback = self._stream

    def _emit_status(self, text):
        self.emitted.append(text)

    def _safe_print(self, text):
        self.printed.append(text)

    def _stream(self, text):
        self.streamed.append(text)


def test_user_blocked_halt_sets_exit_reason_and_appends_response():
    agent = _FakeAgent()
    messages = _tool_msgs('{"status": "error", "error": "BLOCKED: User denied"}')
    result = _user_blocked_halt_response(agent, messages)

    assert result == ("user_blocked", "操作被拒绝。请指示下一步。")
    # Side effects: status emitted, response appended, stream flushed.
    assert agent.emitted and "用户拒绝了危险操作" in agent.emitted[0]
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["content"] == "操作被拒绝。请指示下一步。"
    assert agent.printed and "操作被拒绝" in agent.printed[0]


def test_user_blocked_halt_returns_none_without_denial():
    agent = _FakeAgent()
    messages = _tool_msgs("command output ok")
    assert _user_blocked_halt_response(agent, messages) is None
    # No side effects when there is no denial.
    assert agent.emitted == []
    assert agent.printed == []
    assert messages[-1]["role"] == "tool"


# ─────────────────────────────────────────────────────────────────────
# Builtin 别名绕过回归（2026-08-26 复现发现）
# ``op = open`` / ``e = eval`` / ``from builtins import open as op`` 曾绕过
# open/eval/exec 检测分支（只匹配裸名字 func.id），在 CLI 交互模式下
# auto-approve；``op = open`` + expanduser 函数别名组合还击穿了 #49578
# 敏感写不变量。修复：_resolve_alias_value 将 builtin 名解析为
# ("builtins", name)，open/eval/exec 检测统一走 _resolve_call_target。
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("code,reason", [
    # open 赋值别名 → 写模式必须触发 open-write
    ('op = open\nwith op("/tmp/t.txt", "w") as f:\n    f.write("x")', "open-write"),
    # 链式赋值别名 g = f; f = open
    ('f = open\ng = f\nwith g("/tmp/t.txt", "w") as fh:\n    fh.write("x")', "open-write"),
    # builtins import 别名
    ('from builtins import open as op\nop("/tmp/t.txt", "w").write("x")', "open-write"),
    # eval/exec 赋值别名 → dynamic-exec
    ('e = eval\ne("os.kill")(1, 15)', "dynamic-exec"),
    ('x = exec\nx("import os; os.kill(1, 15)")', "dynamic-exec"),
])
def test_builtin_alias_forms_flagged(code, reason):
    """builtin 别名（open/eval/exec）必须被危险扫描识别（曾绕过）。"""
    assert _execute_code_has_dangerous_ops(code) == reason


@pytest.mark.parametrize("code", [
    # 只读别名不误报
    'op = open\nwith op("data.txt", "r") as f:\n    print(f.read())',
    # 非危险 builtin 别名不误报
    'p = print\np("hi")',
    'l = len\nprint(l([1, 2, 3]))',
])
def test_builtin_alias_benign_passes(code):
    """无害 builtin 别名不得触发危险扫描。"""
    assert _execute_code_has_dangerous_ops(code) is None


def test_builtin_open_alias_not_auto_approved(monkeypatch):
    """op = open 写文件必须落到审批弹窗，不能 auto-approve（2026-08-26 漏洞）。"""
    import tools.approval as approval_module
    monkeypatch.setattr(approval_module, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "manual")
    result = check_execute_code_guard(
        "op = open\nwith op('/tmp/t.txt', 'w') as f:\n    f.write('x')",
        env_type="local",
    )
    assert result["approved"] is False or result.get("outcome") in (
        "blocked", "denied", "pending",
    )


@pytest.mark.parametrize("code", [
    # op = open + expanduser 变量目标（#49578 原始形状的别名版）
    ('import os\nop = open\ntarget = os.path.expanduser("~/.hermes/config.yaml")\n'
     'with op(target, "a") as f:\n    f.write("x")'),
    # op = open + expanduser 函数别名组合（曾完整击穿不变量）
    ('import os\nop = open\nh = os.path.expanduser\n'
     'with op(h("~/.hermes/config.yaml"), "a") as f:\n    f.write("x")'),
    # builtins import 别名 + 敏感目标
    ('from builtins import open as op\n'
     'op("~/.ssh/authorized_keys", "a").write("x")'),
])
def test_builtin_open_alias_sensitive_write_hard_blocked(code):
    """builtin 别名写敏感目标必须命中 #49578 不变量（曾 auto-approved）。"""
    assert _execute_code_has_sensitive_write(code) is not None


def test_expanduser_function_alias_sensitive_write():
    """h = os.path.expanduser 函数引用别名写敏感目标必须硬阻断（曾退化为审批）。"""
    code = ('import os\nh = os.path.expanduser\n'
            'with open(h("~/.hermes/config.yaml"), "a") as f:\n    f.write("x")')
    assert _execute_code_has_sensitive_write(code) is not None


# ─────────────────────────────────────────────────────────────────────
# 2026-08-26: #49578 残余面 — 库写方法（pandas/numpy）敏感路径参数
# ─────────────────────────────────────────────────────────────────────
# pd.to_csv('/root/.ssh/authorized_keys') 等调用接受任意路径字符串，绕过
# open()/Path() AST 形状检测（复现：曾直接放行）。新增通用兜底
# _execute_code_touches_sensitive_path：非只读方法调用携带静态可解析的
# 敏感路径参数 → hard_blocked；只读/查询（os.path、存在性、目录列举、
# read/load/read_*）保持放行。

LIB_WRITE_SENSITIVE_CASES = [
    # pandas 写方法 → 敏感目标（曾全部放行的复现形状）
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_csv("/root/.ssh/authorized_keys")',
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_json("/etc/hosts")',
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_pickle("~/.hermes/config.yaml")',
    # numpy 写方法 → 敏感目标
    'import numpy as np\nnp.savetxt("/root/.ssh/authorized_keys", np.array([1]))',
    'import numpy as np\nnp.save("/root/.hermes/config.yaml", np.array([1]))',
    # expanduser / 关键字参数 / 简单别名 变体
    ('import pandas as pd, os\n'
     'pd.DataFrame({"a": [1]}).to_csv(os.path.expanduser("~/.ssh/authorized_keys"))'),
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_csv(path="/root/.ssh/authorized_keys")',
    'import pandas as pd\ntarget = "/root/.hermes/config.yaml"\npd.DataFrame({"a": [1]}).to_csv(target)',
    # 写方法 + 别名模块
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_feather("/var/run/x")',
]


@pytest.mark.parametrize("code", LIB_WRITE_SENSITIVE_CASES)
def test_library_writer_sensitive_path_detected(code):
    """库写方法的敏感路径参数必须命中 #49578 不变量（曾直接放行）。"""
    assert _execute_code_touches_sensitive_path(code) is not None


@pytest.mark.parametrize("code", [
    # 只读/查询敏感路径 — 不得误伤
    'import pandas as pd\npd.read_csv("/root/.ssh/config.csv")',
    'import numpy as np\nnp.load("/root/.hermes/data.npy")',
    'import os\nprint(os.path.exists("/root/.ssh"))',
    'import os\nprint(os.path.join("/root/.ssh", "x"))',
    'from pathlib import Path\nprint(Path("/root/.hermes").exists())',
    'import glob\nprint(glob.glob("/root/.ssh/*"))',
    'import json\nprint(json.load(open("/root/.hermes/config.json")))',
    'print(open("/etc/hosts").read())',
    # 普通路径写 — 不属于敏感不变量（危险面由 AST-dangerous-ops 单独管辖）
    'import pandas as pd\npd.DataFrame({"a": [1]}).to_csv("/tmp/ok.csv")',
])
def test_library_readonly_sensitive_path_passes(code):
    """只读/查询方法携带敏感路径参数必须放行（与 open() 只读行为一致）。"""
    assert _execute_code_touches_sensitive_path(code) is None


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_library_writer_sensitive_hard_blocked_in_all_modes(monkeypatch, mode_gate):
    """库写方法敏感路径在 yolo/approvals=off 下同样不可覆盖（#49578 不变量）。"""
    code = ('import pandas as pd\n'
            'pd.DataFrame({"a": [1]}).to_csv("/root/.ssh/authorized_keys")')

    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )

    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


def test_library_writer_guard_returns_hard_blocked_outcome():
    """guard 集成：CLI 路径下库写敏感返回 hard_blocked（复现用例）。"""
    result = check_execute_code_guard(
        'import pandas as pd\npd.DataFrame({"a": [1]}).to_csv("/root/.ssh/authorized_keys")',
        env_type="local",
    )
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


# ─────────────────────────────────────────────────────────────────────
# 2026-08-26 re-review (andrexibiza, head 828e13e35): 剩余静态绑定缺口
# ─────────────────────────────────────────────────────────────────────
# Blocker 1: 进程kill硬阻断仍可被普通赋值形式绕过
#   - 多重赋值 a = b = os.kill（_collect_exec_code_bindings 只记录
#     len(targets)==1，多重赋值整个逃出绑定图）
#   - __dict__ 动态访问经赋值别名 o = os（__dict__ 分支只查 imports）
# Blocker 2: #49578 敏感目标仍有别名/对象缺口
#   - 组合属性别名 p = os.path; h = p.expanduser（base 已带属性时返回
#     ('os','path') 而非组合 ('os.path','expanduser')）
#   - Path 对象存变量 p = Path(...); p.write_text（call-valued RHS 不在
#     抽象绑定图）
# 全部要求通过 check_execute_code_guard() 端到端回归（helper 级不够）。

MULTI_TARGET_KILL_CASES = [
    'import os\na = b = os.kill\na(os.getpid(), 15)',
    'import os\na = b = c = os.killpg\na(0, 9)',
    # 链式 + 多重赋值组合
    'import os\nk = os\na = b = k.kill\na(os.getpid(), 15)',
]

@pytest.mark.parametrize("code", MULTI_TARGET_KILL_CASES)
def test_multi_target_kill_hard_blocked(code):
    """多重赋值 a = b = os.kill 必须命中 hard block（曾逃出绑定图）。"""
    assert _execute_code_has_self_destructive_ops(code) is not None


@pytest.mark.parametrize("code", [
    'import os\no = os\no.__dict__["kill"](os.getpid(), 15)',
    'import os\no = os\nx = o.__dict__["killpg"]\nx(0, 9)',
])
def test_dict_access_via_alias_hard_blocked(code):
    """o = os; o.__dict__["kill"] 必须命中 hard block（__dict__ 分支曾只查 imports）。"""
    assert _execute_code_has_self_destructive_ops(code) is not None


def test_multi_target_kill_guard_hard_blocked():
    """guard 端到端：多重赋值 kill 返回 hard_blocked（re-review 要求）。"""
    result = check_execute_code_guard(
        'import os\na = b = os.kill\na(os.getpid(), 15)', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "HARD BLOCKED" in result["message"]


def test_dict_alias_kill_guard_hard_blocked():
    """guard 端到端：别名 __dict__ kill 返回 hard_blocked。"""
    result = check_execute_code_guard(
        'import os\no = os\no.__dict__["kill"](os.getpid(), 15)', env_type="local"
    )
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


COMPOSED_ATTR_SENSITIVE_CASES = [
    # p = os.path; h = p.expanduser（组合属性别名）
    ('import os\np = os.path\nh = p.expanduser\n'
     'with open(h("~/.hermes/config.yaml"), "a") as f:\n    f.write("x")'),
    # p = os.path 后直接 p.expanduser(...)（resolve_call_target 组合链）
    ('import os\np = os.path\n'
     'open(p.expanduser("~/.hermes/config.yaml"), "a").write("x")'),
]

@pytest.mark.parametrize("code", COMPOSED_ATTR_SENSITIVE_CASES)
def test_composed_attr_sensitive_write_detected(code):
    """组合属性别名必须恢复敏感写目标（曾解析成 ('os','path') 丢失目标）。"""
    assert _execute_code_has_sensitive_write(code) is not None


PATH_OBJECT_VAR_SENSITIVE_CASES = [
    'from pathlib import Path\np = Path("~/.hermes/config.yaml")\np.write_text("x")',
    'from pathlib import Path\np = Path("/root/.ssh/authorized_keys")\np.write_bytes(b"key")',
    'from pathlib import Path\np = Path("/root/.ssh/authorized_keys")\nwith p.open("a") as f:\n    f.write("x")',
    'import pathlib\np = pathlib.Path("~/.hermes/config.yaml")\np.write_text("x")',
    # 链式对象别名 q = p
    'from pathlib import Path\np = Path("~/.hermes/config.yaml")\nq = p\nq.write_text("x")',
    # Path 构造别名 P
    'from pathlib import Path as P\np = P("~/.hermes/config.yaml")\np.write_text("x")',
]

@pytest.mark.parametrize("code", PATH_OBJECT_VAR_SENSITIVE_CASES)
def test_path_object_var_sensitive_write_detected(code):
    """Path 对象存变量后的写方法必须命中 #49578 不变量（曾完全逃过检测）。"""
    assert _execute_code_has_sensitive_write(code) is not None


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_composed_attr_sensitive_write_hard_blocked_in_all_modes(monkeypatch, mode_gate):
    """组合属性敏感写在 yolo/approvals=off 下同样不可覆盖（#49578 不变量）。"""
    code = ('import os\np = os.path\nh = p.expanduser\n'
            'with open(h("~/.hermes/config.yaml"), "a") as f:\n    f.write("x")')

    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )

    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_path_object_var_sensitive_write_hard_blocked_in_all_modes(monkeypatch, mode_gate):
    """Path 对象变量敏感写在 yolo/approvals=off 下同样不可覆盖。"""
    code = ('from pathlib import Path\np = Path("~/.hermes/config.yaml")\n'
            'p.write_text("x")')

    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )

    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


# 良性对照：新解析能力不得引入误报

@pytest.mark.parametrize("code", [
    # 多重赋值但目标无害
    'import os\na = b = os.path.join\nprint(a("/tmp", "x"))',
    # 组合属性但操作无害
    'import os\np = os.path\nprint(p.join("/tmp", "x"))',
    'import os\np = os.path\nprint(p.exists("/tmp"))',
    # Path 对象变量：非敏感写放行、敏感只读放行、构造本身放行
    'from pathlib import Path\np = Path("/tmp/x")\np.write_text("ok")',
    'from pathlib import Path\np = Path("/root/.hermes/config.yaml")\nprint(p.exists())',
    'from pathlib import Path\np = Path("/root/.hermes/config.yaml")\nprint(p.read_text())',
    'from pathlib import Path\nprint(Path("/root/.ssh"))',
    # 元组解包不是多重赋值别名
    'a, b = 1, 2\nprint(a + b)',
])
def test_new_binding_shapes_benign_passes(code):
    """新解析能力只影响危险形状，良性代码零误伤。"""
    assert _execute_code_has_self_destructive_ops(code) is None
    assert _execute_code_has_sensitive_write(code) is None


# ─────────────────────────────────────────────────────────────────────
# 2026-08-28 re-review (andrexibiza) Blocker 1/2/3 regressions
# ─────────────────────────────────────────────────────────────────────
# 三个 blocker 的复现（修复前 head d73f538529 上全部失败）：
#   1. 兄弟局部作用域 import 互相污染 → os.kill 解析成 math.kill 放行
#   2. open(file=..., mode="w") 关键字形式 → 敏感写目标无法恢复，
#      yolo/approvals-off 击穿 #49578 不变量
#   3. asyncio.create_subprocess_* / os.startfile / pty.spawn 不在
#      命令执行精确表 → 本地静默放行

# ── Blocker 1: sibling-scope import contamination ─────────────────────

SIBLING_SCOPE_KILL = '''
def dangerous():
    import os
    os.kill(os.getppid(), 15)

def harmless():
    import math as os
    return os.sqrt(4)

dangerous()
'''

def test_sibling_scope_import_does_not_shadow_dangerous_binding():
    """两个兄弟函数同名 import：harmless 的 ``import math as os`` 不得
    把 dangerous 的 ``import os`` 从绑定图中擦除（修复前 os.kill 被解析
    成 math.kill → hard block 漏拦，approved=True）。"""
    reason = _execute_code_has_self_destructive_ops(SIBLING_SCOPE_KILL)
    assert reason is not None
    assert "os.kill" in reason


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_sibling_scope_kill_hard_blocked_in_all_modes(monkeypatch, mode_gate):
    """兄弟作用域污染的 os.kill 在普通/yolo/approvals-off 下全部
    hard_blocked（硬阻断在 yolo/off 门之前执行）。"""
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )
    result = check_execute_code_guard(SIBLING_SCOPE_KILL, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


def test_sibling_scope_safe_usage_still_passes():
    """候选保留模型不得误伤：harmless 函数的 ``os.sqrt(4)`` 与模块级
    安全 os 用法仍放行。"""
    code = '''
def dangerous():
    import os
    os.kill(os.getppid(), 15)

def harmless():
    import math as os
    return os.sqrt(4)
'''
    # 只调 harmless（不调 dangerous）→ dangerous 体内的调用不执行，但
    # 静态层仍会看到它——候选保留使 os.kill 命中硬阻断；而单独的
    # harmless 用法必须放行：
    harmless_only = '''
def harmless():
    import math as os
    return os.sqrt(4)

print(harmless())
'''
    assert _execute_code_has_self_destructive_ops(harmless_only) is None
    assert _execute_code_has_dangerous_ops(harmless_only) is None


def test_nested_sibling_import_no_cross_contamination():
    """更深的兄弟污染形状：嵌套函数里同名 import + 模块级调用。"""
    code = '''
import os

def shadow():
    import math as os

os.kill(os.getpid(), 15)
'''
    reason = _execute_code_has_self_destructive_ops(code)
    assert reason is not None
    assert "os.kill" in reason


# ── Blocker 2: builtin-open keyword file= bypass (#49578) ─────────────

@pytest.mark.parametrize("code", [
    'open(file="/root/.hermes/config.yaml", mode="w").write("x")',
    'open(file="/root/.ssh/authorized_keys", mode="a").write("x")',
    # mode 缺省但 file 关键字（缺省 r，只读，不应命中 sensitive-write；
    # 但 open-write 判定由 _open_mode_is_write 负责，此处验证目标恢复）
    'import os\nopen(file=os.path.expanduser("~/.hermes/config.yaml"), mode="w").write("x")',
    # io.open 与内置 open 同签名（Blocker 2 等效面）
    'import io\nio.open(file="/root/.hermes/config.yaml", mode="w").write("x")',
    'import io\nio.open("/root/.hermes/config.yaml", mode="w").write("x")',
    # codecs.open 用 filename 关键字（Blocker 2 等效面）
    'import codecs\ncodecs.open(filename="/root/.hermes/config.yaml", mode="w").write("x")',
    'import codecs\ncodecs.open("/root/.hermes/config.yaml", mode="w").write("x")',
])
def test_open_keyword_file_sensitive_write_detected(code):
    """open 的 file 参数关键字形式必须恢复写目标（修复前
    _resolve_static_write_target 只读位置参数 → 目标为 None →
    #49578 不变量降级为可恢复审批）。"""
    assert _execute_code_has_sensitive_write(code) is not None


@pytest.mark.parametrize("code", [
    'open(file="/root/.hermes/config.yaml", mode="w").write("x")',
    'import io\nio.open(file="/root/.hermes/config.yaml", mode="w").write("x")',
    'import codecs\ncodecs.open(filename="/root/.hermes/config.yaml", mode="w").write("x")',
])
@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_open_keyword_sensitive_write_hard_blocked_in_all_modes(
        monkeypatch, mode_gate, code):
    """open 关键字形式在 yolo/approvals-off 下同样不可覆盖
    （#49578 目标不变量，修复前 yolo/off 击穿 approved=True）。"""
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )
    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"
    assert "protected path" in result["message"]


@pytest.mark.parametrize("code", [
    # file= 关键字只读形态不误报
    'open(file="/root/.hermes/config.yaml", mode="r").read()',
    'open(file="/root/.hermes/config.yaml").read()',
    'import io\nio.open(file="/root/.hermes/config.yaml").read()',
    'import codecs\ncodecs.open(filename="/root/.hermes/config.yaml").read()',
    # 非敏感目标关键字写 → 只落 open-write 审批（不硬阻断）
    'open(file="/tmp/x.txt", mode="w").write("x")',
])
def test_open_keyword_read_or_non_sensitive_passes(code):
    assert _execute_code_has_sensitive_write(code) is None


# ── Blocker 3: process-launch family (asyncio / os.startfile / pty) ───

@pytest.mark.parametrize("code", [
    'import asyncio\nasyncio.create_subprocess_exec("rm", "-rf", "/")',
    'import asyncio\nasyncio.create_subprocess_shell("rm -rf /")',
    'import os\nos.startfile("evil.bat")',
    'import pty\npty.spawn("/bin/sh")',
    # 家族前缀规则（不逐名枚举）：asyncio.create_subprocess_* 前缀、
    # os/posix spawn*/exec* 前缀、subprocess 模块级
    'import asyncio\nasyncio.create_subprocess_exec2("x")',
    'import posix\nposix.spawnv(0, "/bin/sh", ["/bin/sh"])',
    'import posix\nposix.execl("/bin/sh", "/bin/sh")',
])
def test_process_launch_family_flagged_command_exec(code):
    """asyncio/os.startfile/pty.spawn 与 subprocess 同等级：进入审批链
    （修复前本地路径静默放行）。"""
    assert _execute_code_has_dangerous_ops(code) == "command-exec"


@pytest.mark.parametrize("code", [
    # asyncio 非进程用法不误报
    'import asyncio\nasyncio.sleep(1)',
    'import asyncio\nasyncio.create_task(asyncio.sleep(1))',
    # os 无害用法不误报
    'import os\nprint(os.path.join("/tmp", "x"))',
    'import os\nprint(os.getcwd())',
    # pty 非 spawn 方法不误报
    'import pty\nprint(pty.openpty())',
])
def test_process_launch_family_benign_passes(code):
    assert _execute_code_has_dangerous_ops(code) is None
    assert _execute_code_touches_sensitive_path(code) is None


# ─────────────────────────────────────────────────────────────────────
# 2026-08-29 re-review (andrexibiza) Blocker: comprehension targets
# ─────────────────────────────────────────────────────────────────────
# 推导式目标逃出绑定图（修复前 head fc97920507 上全部失败）：
#   [k(os.getpid(), 15) for k in [os.kill]] — 运行时真调用 os.kill，
#   但 callee Name('k') 无法解析 → hard block / danger pass 双漏，
#   本地路径 auto-approve。与语句级 ast.For 同一能力（字面量可迭代
#   目标绑定），只是不同的 AST 形状（ast.comprehension）。

COMPREHENSION_KILL_CASES = [
    'import os\n[k(os.getpid(), 15) for k in [os.kill]]',
    'import os\nnext(k(os.getpid(), 15) for k in [os.kill])',
    'import os\n{k(os.getpid(), 15) for k in [os.kill]}',
    'import os\n{k: k(os.getpid(), 15) for k in [os.kill]}',
    # 嵌套 generators（内层 comp 由通用递归触达）
    'import os\n[[k(os.getpid(), 15) for k in [os.kill]] for _ in [1]]',
]


@pytest.mark.parametrize("code", COMPREHENSION_KILL_CASES)
def test_comprehension_target_kill_detected(code):
    """comprehension 目标 k 绑定到字面量可迭代候选 → os.kill 可解析
    （修复前 approved=True，硬阻断漏拦）。"""
    reason = _execute_code_has_self_destructive_ops(code)
    assert reason is not None
    assert "os.kill" in reason


@pytest.mark.parametrize("code", COMPREHENSION_KILL_CASES)
@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_comprehension_target_kill_hard_blocked_in_all_modes(
        monkeypatch, mode_gate, code):
    """推导式里的 os.kill 在普通/yolo/approvals-off 下全部 hard_blocked。"""
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )
    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


def test_comprehension_target_command_exec_detected():
    """danger pass 同样受益：推导式里的 subprocess.run 别名进入审批链
    （review 指出 ordinary danger pass 有同样的 missing alias）。"""
    code = 'import subprocess\n[k("rm -rf /", shell=True) for k in [subprocess.run]]'
    assert _execute_code_has_dangerous_ops(code) == "command-exec"


@pytest.mark.parametrize("code", [
    # 纯数值字面量（无模块语义）
    '[x * 2 for x in [1, 2, 3]]',
    '[x.upper() for x in ["a", "b"]]',
    # 非字面量 iterable（range/call）→ 不绑定，保守不误报
    'import os\n[os.path.join("/tmp", str(x)) for x in range(3)]',
    # 安全模块调用
    'import os\n[os.getcwd() for _ in [1]]',
    'import os\n[os.path.exists(p) for p in ["/tmp"]]',
    # 生成器安全求和
    'sum(x * x for x in range(10))',
    # 目标绑定安全函数（os.path.join 不在危险集）
    'import os\n[f("/tmp", "x") for f in [os.path.join]]',
])
def test_comprehension_benign_passes(code):
    """保守候选模型不得把每个局部推导式变成安全终端状态。"""
    assert _execute_code_has_self_destructive_ops(code) is None
    assert _execute_code_has_dangerous_ops(code) is None
    assert _execute_code_touches_sensitive_path(code) is None


# ─────────────────────────────────────────────────────────────────────
# Package acquisition invariant — execute_code side (#97657 BLOCKER 2)
# ─────────────────────────────────────────────────────────────────────
# #97657 (dandckr-ops) makes package acquisition owner-gated in the
# terminal layer; andrexibiza's review assigned the execute_code side to
# #65592. The same invariant must hold for process-launch calls inside
# execute_code scripts (subprocess/os.system/...), BEFORE the
# container/YOLO/off short-circuits — isolated backends included.
# All detection tests below FAIL on fc97920507 (no package detection).

PACKAGE_ACQUISITION_CASES = [
    # andrexibiza #97657 review 的三个例子
    'import subprocess\nsubprocess.run(["apk", "add", "openssh"])',
    'import subprocess\nsubprocess.run(["npm", "add", "plausible-vendor-sdk"])',
    'import subprocess\nsubprocess.run(["uv", "run", "--with", "plausible-vendor-sdk", "python", "-c", "import x"])',
    # 常见形状
    'import subprocess\nsubprocess.run(["pip", "install", "pkg"], check=True)',
    'import subprocess\nsubprocess.run("pip install pkg", shell=True)',
    'import os\nos.system("pip install pkg")',
    'import subprocess\nsubprocess.Popen(["npm", "install", "pkg"])',
    'import asyncio\nasyncio.create_subprocess_exec("pip", "install", "pkg")',
    'import subprocess\ncmd = ["pip", "install", "pkg"]\nsubprocess.run(cmd)',
    'import subprocess\nsubprocess.run(["uv", "sync"])',
]


@pytest.mark.parametrize("code", PACKAGE_ACQUISITION_CASES)
def test_package_acquisition_detected(code):
    """execute_code 里的包获取调用必须被识别（返回包管理器名）。"""
    assert _execute_code_has_package_acquisition(code) is not None


@pytest.mark.parametrize("code", PACKAGE_ACQUISITION_CASES)
@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_package_acquisition_owner_gated_in_all_modes(monkeypatch, mode_gate, code):
    """包获取在普通/yolo/approvals-off 下全部拒绝（owner-gated，
    检查位于 YOLO/off 短路之前）。"""
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(
            approval_module, "_get_approval_mode", lambda: "off"
        )
    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "package_acquisition"


@pytest.mark.parametrize("env_type", ["vercel_sandbox", "docker"])
def test_package_acquisition_owner_gated_in_isolated_backends(env_type):
    """隔离后端同样 owner-gated：容器/沙箱 skip 在包获取检查之后。"""
    code = 'import subprocess\nsubprocess.run(["pip", "install", "pkg"])'
    result = check_execute_code_guard(code, env_type=env_type,
                                      has_host_access=False)
    assert result["approved"] is False
    assert result["outcome"] == "package_acquisition"


@pytest.mark.parametrize("code", [
    # 非获取命令不误报（确认式检测）
    'import subprocess\nsubprocess.run(["npm", "run", "build"])',
    'import subprocess\nsubprocess.run(["pip", "list"])',
    'import subprocess\nsubprocess.run(["apt", "search", "openssh"])',
    'import subprocess\nsubprocess.run(["apk", "info", "openssh"])',
    'import subprocess\nsubprocess.run(["git", "clone", "https://example.com/x"])',
    'import subprocess\nsubprocess.run(["python", "-c", "print(1+1)"])',
    'import subprocess\nsubprocess.run("echo \'pip install x\'", shell=True)',
    # 非命令执行调用（读文件等）
    'import subprocess\nsubprocess.run(["cat", "/etc/hostname"])',
])
def test_package_acquisition_benign_passes(code):
    assert _execute_code_has_package_acquisition(code) is None


# fail-closed（P0-3，对齐 #98138 bounded 设计）：command-exec 调用存在但
# argv 静态不可解析 → 无法判定是否包获取 → 返回 _PACKAGE_UNRESOLVABLE
# 要求 owner 审批，不再放行（旧语义「不确认 → 放行」已废弃，见 #97657
# owner-gate + yolo/off 不可绕过）。
@pytest.mark.parametrize("code", [
    'import subprocess\nsubprocess.run(cmd, shell=True)',
    'import os\nos.system(os.environ["CMD"])',
    'import subprocess\nargs = get_args()\nsubprocess.run(args)',
])
def test_package_acquisition_unresolvable_fails_closed(code):
    from tools.exec_code_policy import _PACKAGE_UNRESOLVABLE
    assert _execute_code_has_package_acquisition(code) == _PACKAGE_UNRESOLVABLE


# ─────────────────────────────────────────────────────────────────────
# exec*/spawn* 签名感知提取（P1 修复，andrexibiza 2026-08-29 P1）
# ─────────────────────────────────────────────────────────────────────
# 此前 node.args[-1] 一刀切把 argv 当最后位置参数：execve/spawnve/
# posix_spawn 的最后参数是 env → 全部漏检。现按各家族真实签名提取
# argv 槽位，并以 path 参数（真实可执行文件）做权威判定，argv[0]
# 仅作辅助（防 argv[0] 伪造绕过）。所有检测用例在 c8e072d01c 上
# 漏检（None），修复后全部返回包管理器名。

EXEC_SPAWN_ACQUISITION_CASES = [
    # *e 家族：最后参数是 env，不是 argv（P1 三个复现例）
    "import os\nos.execve(\"/usr/bin/pip\", [\"pip\", \"install\", \"pkg\"], os.environ)",
    "import os\nos.spawnve(os.P_WAIT, \"/usr/bin/pip\", [\"pip\", \"install\", \"pkg\"], os.environ)",
    "import os\nos.posix_spawn(\"/usr/bin/pip\", [\"pip\", \"install\", \"pkg\"], os.environ)",
    "import os\nos.posix_spawnp(\"/usr/bin/pip\", [\"pip\", \"install\", \"pkg\"], os.environ)",
    "import os\nos.execvpe(\"/usr/bin/pip\", [\"pip\", \"install\", \"pkg\"], os.environ)",
    "import os\nos.spawnvpe(os.P_WAIT, \"/usr/bin/pip\", [\"pip\", \"install\", \"pkg\"], os.environ)",
    # argv[0] 伪造：真实执行文件在 path 参数（P1 权威源问题）
    "import os\nos.execve(\"/usr/bin/pip\", [\"harmless-name\", \"install\", \"pkg\"], os.environ)",
    # *l 家族：变长位置尾部
    "import os\nos.execl(\"/usr/bin/pip\", \"pip\", \"install\", \"pkg\")",
    "import os\nos.execle(\"/usr/bin/pip\", \"pip\", \"install\", \"pkg\", os.environ)",
    "import os\nos.execlp(\"/usr/bin/pip\", \"pip\", \"install\", \"pkg\")",
    "import os\nos.spawnl(os.P_WAIT, \"/usr/bin/pip\", \"pip\", \"install\", \"pkg\")",
    "import os\nos.spawnle(os.P_WAIT, \"/usr/bin/pip\", \"pip\", \"install\", \"pkg\", os.environ)",
    "import os\nos.spawnlp(os.P_WAIT, \"/usr/bin/pip\", \"pip\", \"install\", \"pkg\")",
    # v 家族常规形状（回归确认不退化）
    "import os\nos.execv(\"/usr/bin/pip\", [\"pip\", \"install\", \"pkg\"])",
    "import os\nos.spawnv(os.P_WAIT, \"/usr/bin/pip\", [\"pip\", \"install\", \"pkg\"])",
]

@pytest.mark.parametrize("code", EXEC_SPAWN_ACQUISITION_CASES)
def test_exec_spawn_signature_aware_detection(code):
    """exec*/spawn*/posix_spawn 各家族按真实签名提取 argv，包获取
    必须检出（P1 修复；c8e072d01c 上这些形状全部漏检）。"""
    assert _execute_code_has_package_acquisition(code) is not None


# 良性 exec/spawn 对照（签名感知不得引入误报）
@pytest.mark.parametrize("code", [
    "import os\nos.execv(\"/usr/bin/true\", [\"true\"])",
    "import os\nos.execl(\"/bin/date\", \"date\", \"-u\")",
    "import os\nos.spawnl(os.P_WAIT, \"/usr/bin/pip\", \"pip\", \"list\")",
    "import os\nos.execve(\"/bin/date\", [\"date\"], os.environ)",
    "import os\nos.posix_spawn(\"/usr/bin/true\", [\"true\"], os.environ)",
    "import os\nos.execlp(\"/usr/bin/git\", \"git\", \"status\")",
])
def test_exec_spawn_benign_passes(code):
    assert _execute_code_has_package_acquisition(code) is None


# ─────────────────────────────────────────────────────────────────────
# 2026-08-31 举一反三：attrgetter / eval-字符串 / env-VAR=val / docker 子命令
# 四个静态可见形状曾漏检（attrgetter 全层放行；eval 字符串只在审批层；
# env 赋值与容器子命令在 yolo/off 下放行包获取）——现全部归位。
# ─────────────────────────────────────────────────────────────────────

# P1: operator.attrgetter 是 getattr 的函数式等价物——attrgetter("kill")(os)
# 与已拦截的 getattr(os, "kill") 运行时等价，但能力名只以字符串字面量出现，
# 曾全层放行（含 yolo/off）。attrgetter(lit) 应用于 os/sys 即硬阻断。
@pytest.mark.parametrize("code", [
    'import os\nfrom operator import attrgetter\nattrgetter("kill")(os)(1, 9)',
    'import os\nimport operator\noperator.attrgetter("kill")(os)(1, 9)',
    'import os as o\nfrom operator import attrgetter\nattrgetter("kill")(o)(1, 9)',
    'from operator import attrgetter\nattrgetter("kill")(__import__("os"))(1, 9)',
    'import os\nfrom operator import attrgetter\nattrgetter("killpg")(os)(0, 9)',
])
def test_attrgetter_on_os_hard_blocked(code):
    assert _execute_code_has_self_destructive_ops(code) is not None


def test_attrgetter_on_benign_object_passes():
    """attrgetter 应用于非 os/sys 对象不误伤。"""
    code = ('from operator import attrgetter\n'
            'class A:\n    x = 1\n'
            'a = A()\nattrgetter("x")(a)')
    assert _execute_code_has_self_destructive_ops(code) is None


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_attrgetter_kill_hard_blocked_in_all_modes(monkeypatch, mode_gate):
    code = 'import os\nfrom operator import attrgetter\nattrgetter("kill")(os)(1, 9)'
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "off")
    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


# P2: eval/exec/compile 字符串字面量含 kill 家族 → 硬阻断。docstring 声称
# 的覆盖此前缺失（只落审批层，yolo/off 放行）。
@pytest.mark.parametrize("code", [
    'eval("os.kill(1, 9)")',
    'eval("os.kill")(1, 9)',
    'exec("os.kill(1, 9)")',
    'compile("os.kill(1, 9)", "x", "exec")',
    'from os import kill\nexec("kill(1, 9)")',
    'e = eval\ne("os.kill(1, 9)")',
])
def test_eval_exec_string_kill_hard_blocked(code):
    assert _execute_code_has_self_destructive_ops(code) is not None


def test_eval_benign_string_not_hard_blocked():
    assert _execute_code_has_self_destructive_ops('eval("os.getpid()")') is None


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
def test_eval_string_kill_hard_blocked_in_all_modes(monkeypatch, mode_gate):
    code = 'eval("os.kill(1, 9)")'
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "off")
    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "hard_blocked"


# P3/P4: env VAR=val 前缀 + 容器运行时子命令的包获取判定（曾漏检，
# yolo/off 下放行 pip install）。
@pytest.mark.parametrize("code", [
    'import os\nos.system("env PATH=/x pip install x")',
    'import os\nos.system("FOO=bar pip install y")',
    'import subprocess\nsubprocess.run(["env", "PATH=/x", "pip", "install", "x"])',
    'import subprocess\nsubprocess.run(["docker", "run", "img", "pip", "install", "x"])',
    'import subprocess\nsubprocess.run(["docker", "exec", "c1", "pip", "install", "x"])',
    'import subprocess\nsubprocess.run(["docker", "run", "--rm", "img", "pip", "install", "x"])',
    'import subprocess\nsubprocess.run(["podman", "run", "img", "uv", "add", "x"])',
])
def test_wrapper_variant_package_acquisition_detected(code):
    assert _execute_code_has_package_acquisition(code) is not None


@pytest.mark.parametrize("code", [
    'import os\nos.system("VAR=x echo hi")',
    'import subprocess\nsubprocess.run(["docker", "run", "img", "echo", "hi"])',
    'import subprocess\nsubprocess.run(["docker", "run", "img", "npm", "run", "build"])',
    'import subprocess\nsubprocess.run(["docker", "pull", "img"])',
    'import subprocess\nsubprocess.run(["docker", "images"])',
])
def test_wrapper_variant_benign_passes(code):
    assert _execute_code_has_package_acquisition(code) is None


@pytest.mark.parametrize("mode_gate", ["normal", "yolo", "off"])
@pytest.mark.parametrize("code", [
    'import os\nos.system("env PATH=/x pip install x")',
    'import subprocess\nsubprocess.run(["docker", "exec", "c1", "pip", "install", "x"])',
])
def test_wrapper_variant_owner_gated_in_all_modes(monkeypatch, mode_gate, code):
    import tools.approval as approval_module
    if mode_gate == "yolo":
        monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)
    elif mode_gate == "off":
        monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "off")
    result = check_execute_code_guard(code, env_type="local")
    assert result["approved"] is False
    assert result["outcome"] == "package_acquisition"
