"""Static policy analysis for execute_code scripts (PR #65592).

Extracted from ``tools/approval.py`` into its own sub-2K module per the
andrexibiza re-review (2026-08-25): the execute-code static analyzer —
AST binding resolver, call-target resolver, write-mode analysis,
process-kill policy, import classifier, sensitive-write target analysis —
lives here; ``tools/approval.py`` retains only the orchestration seam
(``check_execute_code_guard``) that calls into this module.

Scope honesty (documented limitation, #65592 review): everything in this
module is a *static* AST layer.  It blocks every statically-detectable
form — direct calls, import aliases, assignment aliases (incl. chains,
walrus, tuple-unpack, for-targets), container subscripts (dict/list
literals), star imports, ``getattr`` / ``__dict__`` dynamic access,
``sys.modules`` / ``globals()`` / ``vars()`` / ``__import__`` chains,
``functools.partial``, process-kill equivalents (``signal.kill`` /
``psutil.kill``), pathlib write methods, literal/expanduser/expandvars/
f-string/join sensitive targets, and capability leak (hard-blocked
capabilities referenced as values — return/assignment/container/
argument — so they cannot escape per-cell scanning, #94647).  Code that
builds calls at runtime (string-concatenated ``exec``, dynamic f-string
path interpolation, lambda-body *calls* like ``lambda: os.kill()``,
non-literal for-iterables) is not statically visible and belongs to the
runtime/sandbox boundary.  Callers must not present ``hard_blocked`` as
an unbypassable syscall-level property.
"""

import ast
import logging
import os
import posixpath
import re
import shlex

logger = logging.getLogger(__name__)

# Dangerous Python operations that bypass terminal() approval when used
# inside execute_code scripts.  Detected via AST walk with import tracking
# so both ``os.remove(x)`` and ``from os import remove; remove(x)`` are
# caught.  ctypes is listed as a whole-module gate. reason key 见
# _EXEC_CODE_DANGER_DETAILS。
_EXEC_CODE_DANGEROUS_CALLS = {
    # ── 能力类别 1：文件/目录删除 ─────────────────────────────
    ("os", "remove"): "file-delete",
    ("os", "unlink"): "file-delete",
    ("os", "rmdir"): "file-delete",
    ("os", "removedirs"): "file-delete",
    ("shutil", "rmtree"): "file-delete",
    ("pathlib", "unlink"): "file-delete",
    ("pathlib", "rmdir"): "file-delete",
    # ── 能力类别 2：文件移动/复制/重命名/创建（#49578 config write 面）──
    ("shutil", "copy"): "file-mutate",
    ("shutil", "copy2"): "file-mutate",
    ("shutil", "copyfile"): "file-mutate",
    ("shutil", "copyfileobj"): "file-mutate",
    ("shutil", "copytree"): "file-mutate",
    ("shutil", "move"): "file-mutate",
    ("shutil", "copystat"): "file-mutate",
    ("os", "rename"): "file-mutate",
    ("os", "replace"): "file-mutate",
    ("os", "link"): "file-mutate",
    ("os", "symlink"): "file-mutate",
    ("os", "mkdir"): "file-mutate",
    ("os", "makedirs"): "file-mutate",
    ("os", "mkfifo"): "file-mutate",
    ("os", "mknod"): "file-mutate",
    # 文件属性/内容篡改（无参数路径的 receiver 或目标参数均可达敏感区）
    ("os", "chmod"): "file-mutate",
    ("os", "chown"): "file-mutate",
    ("os", "lchmod"): "file-mutate",
    ("os", "lchown"): "file-mutate",
    ("os", "utime"): "file-mutate",
    ("os", "truncate"): "file-mutate",
    ("pathlib", "touch"): "file-mutate",
    ("pathlib", "chmod"): "file-mutate",
    ("pathlib", "chown"): "file-mutate",
    ("pathlib", "rename"): "file-mutate",
    ("pathlib", "replace"): "file-mutate",
    ("pathlib", "mkdir"): "file-mutate",
    ("pathlib", "symlink_to"): "file-mutate",
    ("pathlib", "hardlink_to"): "file-mutate",
    # ── 能力类别 3：任意命令执行（绕过 terminal() DANGEROUS_PATTERNS）──
    ("os", "system"): "command-exec",
    ("os", "popen"): "command-exec",
    ("os", "spawnl"): "command-exec",
    ("os", "spawnle"): "command-exec",
    ("os", "spawnlp"): "command-exec",
    ("os", "spawnlpe"): "command-exec",
    ("os", "spawnv"): "command-exec",
    ("os", "spawnve"): "command-exec",
    ("os", "spawnvp"): "command-exec",
    ("os", "spawnvpe"): "command-exec",
    ("subprocess", "run"): "command-exec",
    ("subprocess", "call"): "command-exec",
    ("subprocess", "Popen"): "command-exec",
    ("subprocess", "check_output"): "command-exec",
    ("subprocess", "check_call"): "command-exec",
    ("subprocess", "getoutput"): "command-exec",
    ("subprocess", "getstatusoutput"): "command-exec",
    # 进程替换（exec* 系列）——脚本把自己换成本机程序，绕过后续全部
    # Python 级检查（#65592 review 举一反三补充）
    ("os", "execv"): "command-exec",
    ("os", "execve"): "command-exec",
    ("os", "execvp"): "command-exec",
    ("os", "execvpe"): "command-exec",
    ("os", "execl"): "command-exec",
    ("os", "execle"): "command-exec",
    ("os", "execlp"): "command-exec",
    ("os", "execlpe"): "command-exec",
    ("os", "posix_spawn"): "command-exec",
    ("os", "posix_spawnp"): "command-exec",
    # ── 能力类别 3b：进程启动新入口（2026-08-28 re-review Blocker 3）──
    # asyncio.create_subprocess_exec/shell 是 async 版 subprocess；
    # os.startfile 是 Windows 的进程启动（用关联程序打开/执行文件）；
    # pty.spawn 在伪终端里执行命令。三者都是「创建进程执行外部程序」
    # 的同一能力，曾不在精确名字表中 → 本地 yolo/approvals-off 路径
    # 静默执行。列在这里 + 家族前缀规则（见 _match_command_exec_family）
    # 双保险：精确表给可读 reason，家族规则兜底 stdlib 演进。
    ("os", "startfile"): "command-exec",
    ("pty", "spawn"): "command-exec",
    ("asyncio", "create_subprocess_exec"): "command-exec",
    ("asyncio", "create_subprocess_shell"): "command-exec",
    # ── 能力类别 4：文件内容写入（pathlib，#49578 等效面）──
    ("pathlib", "write_text"): "open-write",
    ("pathlib", "write_bytes"): "open-write",
    ("pathlib", "open"): "open-write",
}

# 模块的整体导入即触发 guard（即使不调用具体函数）。ctypes 符合——
# ``ctypes.CDLL(None).unlink(...)`` 无需 os.remove 即可绕过所有检查。
_EXEC_CODE_SUSPICIOUS_IMPORTS = frozenset({"ctypes"})

# Builtin 危险名称（2026-08-25 复现发现）：``op = open``、``e = eval``、
# ``from builtins import open as op`` 等别名形式曾绕过基于 ``func.id`` 的
# 直接检测（open/eval/exec 检测分支只匹配裸名字）。解析为
# ``("builtins", name)`` 后与模块属性检测统一走 ``_resolve_call_target``。
_EXEC_CODE_DANGEROUS_BUILTINS = frozenset({
    "open", "eval", "exec", "compile", "__import__",
})

# 命令执行家族的**家族级**规则（2026-08-28 re-review Blocker 3 根因修复）。
# 精确名字表（_EXEC_CODE_DANGEROUS_CALLS 的 command-exec 项）永远追不上
# stdlib 演进——asyncio.create_subprocess_*、os.startfile、pty.spawn 都是
# 逐轮 review 才发现的新入口。家族模式按「模块 × 方法前缀」定义能力，
# 与平台无关（静态 AST 分析），新方法名自动落入同一能力族：
#
#   ("module", "prefix")  → 该模块下所有以 prefix 开头的方法 = 命令执行
#   ("module", None)      → 该模块**任意**方法 = 命令执行（subprocess 全模块）
#
# 边界说明：multiprocessing.Process / os.fork 创建进程但**不执行外部
# 程序**（target 是脚本内函数），且 multiprocessing.Pool 数据并行是合法
# 用途——不归入命令执行族（避免误报），由运行时边界管控。
_EXEC_CODE_COMMAND_EXEC_PREFIXES = (
    ("os", "spawn"),    # os.spawnl/le/lp/lpe/v/ve/vp/vpe
    ("os", "exec"),     # os.execv/ve/vp/vpe/l/le/lp/lpe（进程替换）
    ("posix", "spawn"),  # os 的底层实现模块（直接 import posix 时）
    ("posix", "exec"),
    ("asyncio", "create_subprocess_"),  # exec + shell 两入口
)
_EXEC_CODE_COMMAND_EXEC_MODULES = frozenset({"subprocess"})

# open 形状（2026-08-28 re-review Blocker 2 根因修复的等效面）：内置 open
# 不是唯一接受路径+写模式的 stdlib writer——io.open 与内置同签名（file），
# codecs.open 用 filename。统一按签名解析目标与 mode，不再假设 builtin
# open 是全部能力。值为 file 参数的关键字名（位置 0 或该关键字）。
_EXEC_CODE_OPEN_SHAPES = {
    ("builtins", "open"): "file",
    ("io", "open"): "file",
    ("codecs", "open"): "filename",
}


def _match_command_exec_family(m, a) -> str | None:
    """精确名字表之外的命令执行家族匹配（前缀 / 模块级规则）。

    2026-08-28 re-review Blocker 3：asyncio.create_subprocess_exec、
    os.startfile、pty.spawn 曾因不在精确表而静默放行。家族规则使任何
    落入「模块 × 前缀」能力族的方法都返回 command-exec，不依赖逐名枚举。
    """
    for mod, prefix in _EXEC_CODE_COMMAND_EXEC_PREFIXES:
        if m == mod and a is not None and a.startswith(prefix):
            return "command-exec"
    if m in _EXEC_CODE_COMMAND_EXEC_MODULES:
        return "command-exec"
    return None

# =========================================================================
# Layer 3 — Hard Block: Self-Destructive / Process-Killing Operations
# =========================================================================
# These operations can destroy the Hermes parent process or kill arbitrary
# system processes. They NEVER enter the approval chain — no user consent,
# yolo mode, smart approval, or session persistence can override them.
# Design principle: Linux seccomp / macOS SIP — if the operation is
# fundamentally incompatible with agent operation, no bypass exists.
# (from PR #65592 commit 4, 66e423e4)
#
# NOTE (2026-08-25 re-review): the *static* scan can only match
# statically-visible call shapes; ``eval("os.kill")(...)`` with a literal
# string is caught by the eval guard below, but runtime-built call names
# are not statically visible.  The returned message therefore does NOT
# claim an absolute "no bypass exists" guarantee — the residual surface
# belongs to the runtime/sandbox boundary.

_HARD_BLOCKED_CALLS = frozenset({
    # Process killing — can target the Hermes parent process (os.getppid())
    # or any arbitrary system process.
    ("os", "kill"),
    ("os", "killpg"),
    # os.kill 的等价物（2026-08-25 复测发现：signal.kill 是 POSIX os.kill，
    # psutil.kill 是跨平台 os.kill；psutil.Process(...).kill() 经
    # _resolve_call_target 的 psutil 方法链解析到 ("psutil", "kill")）。
    # 同类能力必须同样不可审批——否则 signal.kill(getppid(), 9) 绕过硬阻断。
    ("signal", "kill"),
    ("signal", "pthread_kill"),
    ("psutil", "kill"),
    # psutil.Process 进程终止能力（2026-08-25 re-review Blocker 2）：
    # terminate()（SIGTERM）与 send_signal(SIGKILL) 是 os.kill 的
    # 普通静态等价物，曾因 _HARD_BLOCKED_CALLS 只列 ("psutil","kill")
    # 而全部自动放行。按进程终止能力定义硬阻断，而非单个方法名。
    ("psutil", "terminate"),
    ("psutil", "send_signal"),
    # psutil.Process(...).suspend() 是 SIGSTOP——可冻结 Hermes 父进程，
    # 同为进程控制能力（2026-08-25 举一反三）。
    ("psutil", "suspend"),
})

# eval/exec/compile 字符串字面量中的裸进程终止调用形态（2026-08-31 举一反三：
# ``from os import kill; exec("kill(1, 9)")`` 的字符串无模块前缀——按调用名
# 保守匹配；``kill -l`` 之类查询不在 execute_code 字符串语境出现，误伤可忽略。
_KILL_CALL_IN_STR_RE = re.compile(
    r"\b(kill|killpg|pthread_kill|terminate|send_signal|suspend)\s*\(")


def _resolve_binding_expr(expr, imports, raw_aliases, seen=None, known=None):
    """解析赋值 RHS 表达式为规范化的 (module, attr) 元组。

    2026-08-25 复测重构（#65592）：原 _resolve_alias_value 只收 name 字符串，
    容器下标 / vars / globals / sys.modules / __import__ / partial 等
    绑定形状无法表达。现统一由本函数解析任意 RHS 表达式，覆盖：
      - ``a = os``        → ('os', None)        （模块级别名）
      - ``killer = os.kill`` → ('os', 'kill')   （函数别名）
      - ``b = a.kill``    → ('os', 'kill')      （链式别名）
      - ``x = getattr(os, 'kill')`` → ('os', 'kill')
      - ``h = os.path.expanduser`` → ('os.path', 'expanduser')
      - ``op = open`` / ``e = eval`` → ('builtins', name)
      - ``(k := os.kill)`` → ('os', 'kill')     （walrus，NamedExpr）
      - ``k = {'kill': os.kill}['kill']`` → ('os', 'kill')（dict 字面量下标）
      - ``k = [os.kill][0]`` → ('os', 'kill')   （list 字面量下标）
      - ``x = vars(os)['kill']`` / ``x = globals()['os']`` / ``sys.modules['os']``
      - ``k = partial(os.kill, 1, 15)`` → ('os', 'kill')
      - ``k = __import__('os')`` → ('os', None)
    解析不了返回 None（调用方决定是否保守拦截）。

    ``known``（可选）：危险目标集合。同名字多次赋值保留全部候选时，
    **危险优先**——任一候选解析结果命中 ``known`` 即返回该结果（而非
    取第一个可解析），保证 ``killer = os.path.join; if c: killer =
    os.kill`` 这类分支 join 形状不会被安全候选遮蔽危险候选
    （2026-08-28 拆解：review Blocker 1 要求 conservatively retain
    every possible dangerous target）。known 为 None 时维持第一个可解析。
    """
    if seen is None:
        seen = set()

    # (k := os.kill) — walrus 表达式本身（2026-08-25 复测：NamedExpr 不是
    # Assign，整个逃出绑定图 → (k := os.kill)(1, 15) 曾直接放行）
    if isinstance(expr, ast.NamedExpr):
        return _resolve_binding_expr(expr.value, imports, raw_aliases, seen, known)

    # a = b（别名链）/ a = os（模块级别名）
    if isinstance(expr, ast.Name):
        if expr.id in _EXEC_CODE_DANGEROUS_BUILTINS:
            return ("builtins", expr.id)
        # 候选保留模型（2026-08-28）：import 候选可能多个，known 非 None
        # 时危险优先（任一候选命中 known 即返回），否则取第一个候选。
        _r = _resolve_import(expr.id, imports, known)
        if _r is not None:
            return _r
        if expr.id in seen:
            return None  # 循环别名（a=b; b=a）— 放弃，保守不误报
        seen.add(expr.id)
        if expr.id in raw_aliases:
            # 同名多次赋值保留全部候选（review Blocker 1：分支 join 时
            # 任一候选都可能生效）。**危险优先**（2026-08-28 拆解）：
            # 先扫描全部候选，任一命中 known 危险集合即返回——防止
            # ``killer = os.path.join; if c: killer = os.kill`` 中安全
            # 候选遮蔽危险候选；全部候选都不危险时才取第一个可解析。
            # known 扫描用 seen 副本：builtins 命中不在 known 集合（如
            # ("builtins","open") 不属 DANGEROUS_CALLS 键）时，不能把
            # 别名链上的名字污染进 seen 导致 fallback 短路（f = open;
            # g = f; g(...) 曾因此返回 None）。
            cands = raw_aliases[expr.id]
            if known:
                for cand in cands:
                    resolved = _resolve_binding_expr(
                        cand, imports, raw_aliases, set(seen), known)
                    if resolved is not None and resolved in known:
                        return resolved
            for cand in cands:
                resolved = _resolve_binding_expr(cand, imports, raw_aliases, seen, known)
                if resolved is not None:
                    return resolved
        return None

    # killer = os.kill / b = a.kill / h = p.expanduser（p = os.path）
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        base = _resolve_import(expr.value.id, imports, known)
        if base is None:
            base = _resolve_binding_expr(expr.value, imports, raw_aliases, seen, known)
        if base:
            m, a = base
            # 组合属性链（2026-08-25 re-review）：base 已带属性时（如
            # p = os.path → ('os', 'path')），下一层属性必须组合成
            # ('os.path', attr)，否则 p.expanduser 会被错误解析成
            # ('os', 'path')，敏感写目标解析彻底丢失。
            if a is None:
                return (m, expr.attr)
            return (f"{m}.{a}", expr.attr)

    # h = os.path.expanduser — os.path 子模块属性链（2026-08-25：敏感写
    # 目标解析曾漏掉此函数引用别名，退化为审批而非硬阻断）
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Attribute):
        inner = expr.value
        if (inner.attr == "path" and isinstance(inner.value, ast.Name)
                and _import_is_module(inner.value.id, "os", imports)):
            return ("os.path", expr.attr)
        if (inner.attr == "path" and isinstance(inner.value, ast.Name)
                and _resolve_binding_expr(inner.value, imports, raw_aliases, seen, known)
                == ("os", None)):
            return ("os.path", expr.attr)

    # x = getattr(os, 'kill') / getattr(o, 'kill')（别名 base）/ 关键字形式
    # （2026-08-25 复测：getattr(os, name='kill') 关键字 attr 曾完全逃逸）
    if isinstance(expr, ast.Call) and getattr(expr.func, "id", None) == "getattr":
        obj_expr, attr_expr = _getattr_args(expr)
        if obj_expr is not None and isinstance(obj_expr, ast.Name):
            m = None
            _r = _resolve_import(obj_expr.id, imports, known)
            if _r is not None:
                m = _r[0]
            else:
                base = _resolve_binding_expr(obj_expr, imports, raw_aliases, seen, known)
                if base:
                    m = base[0]
            if m is not None:
                if (isinstance(attr_expr, ast.Constant)
                        and isinstance(attr_expr.value, str)):
                    return (m, attr_expr.value)
                return (m, None)  # 动态属性名 — 调用方决定是否保守拦截
        return None

    # k = partial(os.kill, 1, 15) — functools.partial 首参即被调用目标
    # （2026-08-25 复测举一反三：partial 包装曾完全逃过解析）
    if (isinstance(expr, ast.Call) and expr.args
            and _resolve_binding_expr(expr.func, imports, raw_aliases, seen, known)
            in (("functools", "partial"), ("builtins", "partial"))):
        return _resolve_binding_expr(expr.args[0], imports, raw_aliases, seen, known)

    # k = __import__('os') — 返回模块对象
    if (isinstance(expr, ast.Call) and getattr(expr.func, "id", None) == "__import__"
            and expr.args and isinstance(expr.args[0], ast.Constant)
            and isinstance(expr.args[0].value, str)):
        return (expr.args[0].value, None)

    # x = o.__dict__["killpg"] / x = {'kill': os.kill}['kill'] /
    #     x = [os.kill][0] / x = sys.modules['os'] / x = globals()['os'] /
    #     x = vars(os)['kill'] — 下标取值形状
    if isinstance(expr, ast.Subscript):
        return _resolve_subscript_expr(expr, imports, raw_aliases, seen, known)
    return None


def _fold_str_expr(e, raw_aliases=None):
    """静态折叠字符串表达式（字面量 + 常量拼接 + 简单别名）。

    2026-08-25 复测举一反三：``os.__dict__["ki" + "ll"](...)`` /
    ``os.__dict__[name](...)``（name = "kill"）曾因 slice 不是 Constant
    而逃过所有下标分支。折叠失败返回 None。
    """
    if isinstance(e, ast.Constant) and isinstance(e.value, str):
        return e.value
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
        left = _fold_str_expr(e.left, raw_aliases)
        right = _fold_str_expr(e.right, raw_aliases)
        if left is not None and right is not None:
            return left + right
        return None
    if raw_aliases is not None and isinstance(e, ast.Name) and e.id in raw_aliases:
        for cand in raw_aliases[e.id]:
            folded = _fold_str_expr(cand, raw_aliases)
            if folded is not None:
                return folded
    return None


def _getattr_args(call_node):
    """提取 getattr(object, name[, default]) 的位置/关键字参数对。

    2026-08-25 复测：``getattr(os, name='kill')`` 关键字形式曾绕过——
    _resolve_call_target 的 getattr 分支只读位置参数。
    """
    obj_expr, attr_expr = None, None
    if call_node.args:
        obj_expr = call_node.args[0]
        if len(call_node.args) >= 2:
            attr_expr = call_node.args[1]
    if attr_expr is None:
        for kw in call_node.keywords:
            if kw.arg == "name":
                attr_expr = kw.value
    if obj_expr is None:
        for kw in call_node.keywords:
            if kw.arg == "object":
                obj_expr = kw.value
    return obj_expr, attr_expr


def _resolve_subscript_expr(expr, imports, raw_aliases, seen=None, known=None):
    """解析下标表达式 x[slice] 为规范化的 (module, attr)（调用位与赋值 RHS 共用）。

    覆盖（2026-08-25 复测举一反三）：
      - ``o.__dict__['kill']`` / ``os.__dict__['kill']``（含别名 base）
      - ``{'kill': os.kill}['kill']`` — dict 字面量下标
      - ``[os.kill][0]`` — list/tuple 字面量下标
      - ``sys.modules['os']`` — 运行时模块表（'os' 为字面量）
      - ``globals()['os']`` — 仅当名字是脚本显式 import 的
      - ``vars(os)['kill']`` — vars(x) 等价 x.__dict__
    """
    val = expr.value
    # slice 常量折叠：字面量 / "ki"+"ll" 拼接 / name = "kill" 别名
    # （2026-08-25 复测举一反三：非常量 slice 曾逃过全部下标分支）
    slice_str = _fold_str_expr(expr.slice, raw_aliases)
    # __dict__ 动态访问（原逻辑 + 别名 base）
    if (isinstance(val, ast.Attribute) and val.attr == "__dict__"
            and isinstance(val.value, ast.Name)):
        m = None
        _r = _resolve_import(val.value.id, imports, known)
        if _r is not None:
            m = _r[0]
        else:
            base = _resolve_binding_expr(val.value, imports, raw_aliases, seen, known)
            if base:
                m = base[0]
        if m is not None and slice_str is not None:
            return (m, slice_str)
    # {'kill': os.kill}['kill'] — dict 字面量
    if (isinstance(val, ast.Dict) and slice_str is not None):
        for key_n, value_n in zip(val.keys, val.values):
            key_s = _fold_str_expr(key_n, raw_aliases)
            if key_s is not None and key_s == slice_str:
                return _resolve_binding_expr(value_n, imports, raw_aliases, seen, known)
    # [os.kill][0] / (os.kill,)[0] — list/tuple 字面量（slice 为整型常量）
    if (isinstance(val, (ast.List, ast.Tuple))
            and isinstance(expr.slice, ast.Constant)
            and isinstance(expr.slice.value, int)):
        idx = expr.slice.value
        if 0 <= idx < len(val.elts):
            return _resolve_binding_expr(val.elts[idx], imports, raw_aliases, seen, known)
    # sys.modules['os'] — 模块表查询（折叠后字符串 → 该模块）
    if (isinstance(val, ast.Attribute) and val.attr == "modules"
            and isinstance(val.value, ast.Name)
            and _import_is_module(val.value.id, "sys", imports)
            and slice_str is not None):
        return (slice_str, None)
    # globals()['os'] — 仅当名字是脚本显式 import（否则无法静态判定内容）
    if (isinstance(val, ast.Call) and getattr(val.func, "id", None) == "globals"
            and not val.args and slice_str is not None
            and slice_str in imports):
        return _resolve_import(slice_str, imports, known)
    # vars(os)['kill'] — vars(x) 等价 x.__dict__
    if (isinstance(val, ast.Call) and getattr(val.func, "id", None) == "vars"
            and val.args and isinstance(val.args[0], ast.Name)):
        m = None
        _r = _resolve_import(val.args[0].id, imports, known)
        if _r is not None:
            m = _r[0]
        else:
            base = _resolve_binding_expr(val.args[0], imports, raw_aliases, seen, known)
            if base:
                m = base[0]
        if m is not None and slice_str is not None:
            return (m, slice_str)
    return None


def _resolve_alias_value(name, imports, raw_aliases, seen=None, known=None):
    """解析赋值别名链为规范化的 (module, attr) 元组（name 字符串入口）。

    直接调用（open(...)）与别名 RHS（op = open）都先检查 builtin 名；
    其余交给 _resolve_binding_expr 做表达式级解析。解析不了返回 None。
    """
    if name in _EXEC_CODE_DANGEROUS_BUILTINS:
        return ("builtins", name)
    if name not in raw_aliases:
        return None
    if seen is None:
        seen = set()
    if name in seen:
        return None  # 循环别名（a=b; b=a）— 放弃，保守不误报
    seen.add(name)
    cands = raw_aliases[name]
    # 危险优先（2026-08-28 拆解）：任一候选命中 known 即返回，防止
    # 安全候选遮蔽危险候选（killer = os.path.join; if c: killer = os.kill）。
    # known 扫描用 seen 副本，避免污染 fallback（同 _resolve_binding_expr）。
    if known:
        for cand in cands:
            resolved = _resolve_binding_expr(cand, imports, raw_aliases, set(seen), known)
            if resolved is not None and resolved in known:
                return resolved
    for cand in cands:
        resolved = _resolve_binding_expr(cand, imports, raw_aliases, seen, known)
        if resolved is not None:
            return resolved
    return None


def _resolve_attribute_chain(expr, imports, raw_aliases, seen=None, known=None):
    """把嵌套属性表达式解析为规范化 (module, attr)。

    2026-08-25 re-review 补充：``os.path.expanduser`` 的 func 是三层
    Attribute（expanduser → os.path → os），``_resolve_call_target`` 的
    Attribute 分支只认 ``base`` 为 Name 的形状，嵌套属性 base 曾直接
    return None——导致组合属性调用（``p = os.path; p.expanduser(...)``
    之外还有 ``os.path.expanduser(...)`` 内联形式）解析失败。
    """
    if isinstance(expr, ast.Name):
        _r = _resolve_import(expr.id, imports, known)
        if _r is not None:
            return _r
        return _resolve_alias_value(expr.id, imports, raw_aliases, seen, known)
    if isinstance(expr, ast.Attribute):
        base = _resolve_attribute_chain(expr.value, imports, raw_aliases, seen, known)
        if base:
            m, a = base
            if a is None:
                return (m, expr.attr)
            return (f"{m}.{a}", expr.attr)
    return None


def _expr_is_path_constructor(expr, imports, raw_aliases):
    """Path(...) / pathlib.Path(...) / P(...)（P 为 Path 的别名）→ True。"""
    if not isinstance(expr, ast.Call):
        return False
    f = expr.func
    if isinstance(f, ast.Name):
        if f.id == "Path":
            return True
        if _import_is_module(f.id, "pathlib", imports):
            return True
        return _resolve_alias_value(f.id, imports, raw_aliases) == ("pathlib", "Path")
    if isinstance(f, ast.Attribute) and f.attr == "Path" and isinstance(f.value, ast.Name):
        if _import_is_module(f.value.id, "pathlib", imports):
            return True
        return _resolve_alias_value(f.value.id, imports, raw_aliases) == ("pathlib", None)
    return False


def _resolve_path_constructor(expr, raw_aliases, imports):
    """把 Path 构造调用解析出来：直接 ``Path("...")`` / ``pathlib.Path("...")``
    或存进变量的对象 ``p = Path("...")``（含链式 ``q = p``）。

    2026-08-25 re-review（andrexibiza Blocker 2b）：call-valued RHS 不在
    抽象绑定图里，``p = Path("~/.hermes/config.yaml"); p.write_text("x")``
    曾完全逃过敏感写形状检测。返回 ast.Call 节点（可继续取构造参数），
    解析不了返回 None。
    """
    if isinstance(expr, ast.Name):
        for cand in raw_aliases.get(expr.id, []):
            resolved = _resolve_path_constructor(cand, raw_aliases, imports)
            if resolved is not None:
                return resolved
        return None
    if _expr_is_path_constructor(expr, imports, raw_aliases):
        return expr
    return None


def _collect_exec_code_bindings(code):
    """Pass 1：收集脚本的 import / star-import / 赋值别名绑定。

    返回 ``(imports, star_modules, raw_aliases)``：
      - imports:      {local_name: list[(module, attr_or_None)]} — **候选
                      列表**。同名多次绑定（含不同作用域）永不覆盖，全部
                      候选入列；解析时任一候选命中危险目标即拦截。
      - star_modules: set[str] — ``from X import *`` 的模块名集合
      - raw_aliases:  {local_name: list[ast.expr]} — 赋值 RHS 原始表达式
                      列表。同名多次赋值保留**全部候选**（而非
                      last-write-wins 覆盖）——分支 join 时任一候选都可能
                      生效，解析时任一候选命中危险目标即拦截。

    2026-08-25 复测扩展（#65592）——普通 Python 绑定形式补全：
      - walrus ``(k := os.kill)``（ast.NamedExpr，曾完全逃出绑定图）
      - 元组解包 ``k1, k2 = os.kill, os.killpg``（Tuple 目标按位置配对）
      - for 循环目标 ``for f in [os.kill]``（字面量可迭代时取首元素——
        第一次迭代即执行，取第一个可解析元素是正确且保守的）

    2026-08-25 re-review（andrexibiza Blocker 1）——作用域与控制流修正：
      - **静态不可达分支跳过**：``if False:`` / ``while False:`` 的 body
        不再写入绑定（``if False: killer = print`` 曾覆盖 killer =
        os.kill，使可达的 ``killer(...)`` 丢失危险身份）。
      - **同名多次赋值保留全部候选**：顺序赋值/分支赋值一律 append，
        解析时取第一个可解析候选（保守方向：任一候选危险即拦截）。

    2026-08-28 re-review（andrexibiza Blocker 1 根因修复）——import 绑定
    从「单值 + depth 覆盖」改为「候选保留」：
      - depth（数值）不是词法作用域身份：``def dangerous(): import os``
        与 ``def harmless(): import math as os`` 同 depth，后写覆盖先写
        把 dangerous 的 os.kill 解析成 math.kill → 硬阻断漏拦。任何
        ``depth <= prev`` 覆盖规则都无法区分兄弟作用域——根因是「名字→
        单值」的丢失性模型。现在同名绑定全部入列（去重），解析时任一
        候选命中危险目标即拦截（conservatively retain every possible
        dangerous target）。嵌套作用域绑定不再覆盖模块级绑定（两者都是
        候选，模块级的危险候选依然可见），同时函数内真实危险导入也不会
        因模块级同名安全导入而被遮蔽。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}, set(), {}
    imports: dict = {}   # local_name -> list[(module, attr)] 候选列表
    star_modules: set = set()
    raw_aliases: dict = {}

    def _const_truthiness(node):
        """字面量常量的真值（False/0/''/None/[] 等）；无法静态判定返回 None。"""
        if isinstance(node, ast.Constant):
            return bool(node.value)
        return None

    def _record_import(name, module, attr, depth):
        # 候选保留模型（2026-08-28 re-review Blocker 1 根因修复）：同名绑定
        # **永不覆盖**——同深度兄弟作用域（``def dangerous(): import os`` +
        # ``def harmless(): import math as os``）曾因后写覆盖先写而把
        # dangerous 里的 os.kill 解析成 math.kill 放行。depth 是数值不是
        # 词法作用域身份，任何 ``depth <= prev`` 覆盖规则都无法区分兄弟
        # 作用域。现在全部候选入列，解析时任一候选命中危险目标即拦截
        # （conservatively retain every possible dangerous target）。
        # 去重：同一 (name, module, attr) 只保留一个候选。
        cand = (module, attr)
        existing = imports.setdefault(name, [])
        if cand not in existing:
            existing.append(cand)

    def _record_star(module, depth):
        top = module.split(".")[0]
        star_modules.add(top)

    def _visit(node, depth):
        # ── 函数/类/lambda 体是独立作用域：body 内绑定 depth+1 ──────
        # （review Blocker 1：``def shadow(): import math as os`` 的函数体
        # import 曾以模块级深度覆盖顶层 ``os``。iter_child_nodes 会把
        # 函数体语句当作普通子节点，必须在这里把 body 提升一层。）
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Lambda)):
            if isinstance(node, ast.Lambda):
                _visit(node.body, depth + 1)
                return
            for s in node.body:
                _visit(s, depth + 1)
            return

        # ── 控制流：跳过静态不可达分支 ────────────────────────────
        if isinstance(node, ast.If):
            _visit(node.test, depth)
            t = _const_truthiness(node.test)
            if t is True:
                for s in node.body:
                    _visit(s, depth)
            elif t is False:
                for s in node.orelse:
                    _visit(s, depth)
            else:
                for s in node.body:
                    _visit(s, depth)
                for s in node.orelse:
                    _visit(s, depth)
            return
        if isinstance(node, ast.While):
            t = _const_truthiness(node.test)
            if t is False:
                # while False 的 body 不执行；orelse 执行一次
                for s in node.orelse:
                    _visit(s, depth)
                return
            _visit(node.test, depth)
            for s in node.body:
                _visit(s, depth)
            for s in node.orelse:
                _visit(s, depth)
            return

        # ── 绑定节点 ──────────────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                _record_import(name, alias.name, None, depth)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    _record_star(module, depth)
                    continue
                name = alias.asname or alias.name
                _record_import(name, module, alias.name, depth)
        elif isinstance(node, ast.Assign):
            # 多重赋值 a = b = os.kill → targets=[a, b]，每个名字都绑定同一 RHS
            # （2026-08-25 re-review：曾只记录 len(targets)==1，多重赋值完全
            # 逃出绑定图 → a = b = os.kill 绕过 hard block）
            for t in node.targets:
                if isinstance(t, ast.Name):
                    raw_aliases.setdefault(t.id, []).append(node.value)
                elif isinstance(t, ast.Tuple) and isinstance(
                        node.value, (ast.Tuple, ast.List)):
                    # 元组解包 k1, k2 = os.kill, os.killpg → 按位置配对
                    # （2026-08-25 复测：Tuple 目标曾整段跳过 → k1(...) 放行）
                    for target_elt, value_elt in zip(t.elts, node.value.elts):
                        if isinstance(target_elt, ast.Name):
                            raw_aliases.setdefault(target_elt.id, []).append(value_elt)
        elif isinstance(node, ast.NamedExpr):
            # (k := os.kill) — walrus 绑定（2026-08-25 复测发现）
            if isinstance(node.target, ast.Name):
                raw_aliases.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.For):
            # for f in [os.kill]: f(...) — 字面量可迭代的目标绑定
            # （2026-08-25 复测：for 目标曾完全逃逸）
            if (isinstance(node.target, ast.Name)
                    and isinstance(node.iter, (ast.List, ast.Tuple))
                    and node.iter.elts):
                first = node.iter.elts[0]
                if isinstance(first, (ast.Name, ast.Attribute,
                                      ast.Subscript, ast.Call, ast.NamedExpr)):
                    raw_aliases.setdefault(node.target.id, []).append(first)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                               ast.GeneratorExp)):
            # comprehension 目标绑定（2026-08-29 re-review Blocker 1）：
            # ``[k(os.getpid(), 15) for k in [os.kill]]`` 的 k 曾完全逃出
            # 绑定图（_visit 只处理语句级 ast.For，推导式的 ast.comprehension
            # 节点是表达式级循环目标——同一能力的不同 AST 形状）。运行时
            # 该推导式真的调用 os.kill，而 callee Name('k') 无法解析 →
            # hard block / danger pass 双漏，本地路径 auto-approve。
            # 与 ast.For 同一保守字面量可迭代规则：每个 generator 的
            # target（Name）绑定 iterable 首元素候选；含嵌套 generators
            # （[[k(...) for k in [os.kill]] for _ in [1]] 的内层 comp
            # 由通用递归触达同一分支）。
            for gen in node.generators:
                if (isinstance(gen.target, ast.Name)
                        and isinstance(gen.iter, (ast.List, ast.Tuple))
                        and gen.iter.elts):
                    first = gen.iter.elts[0]
                    if isinstance(first, (ast.Name, ast.Attribute,
                                          ast.Subscript, ast.Call,
                                          ast.NamedExpr)):
                        raw_aliases.setdefault(gen.target.id, []).append(first)

        # ── 递归子节点（作用域深度已由入口的 FunctionDef/ClassDef/Lambda
        #    分支处理，这里保持 depth 传递即可）──────────────────────
        for child in ast.iter_child_nodes(node):
            _visit(child, depth)

    for stmt in tree.body:
        _visit(stmt, 0)
    return imports, star_modules, raw_aliases


def _import_candidates(name, imports):
    """返回 *name* 的全部 import 候选 ``[(module, attr)]``（候选保留模型）。

    2026-08-28 re-review Blocker 1 根因修复：imports 从「名字→单值」改为
    「名字→候选列表」，解析时必须考虑**全部**候选——任一候选命中危险
    目标即拦截，防止兄弟/嵌套作用域的同名绑定互相遮蔽。
    """
    return imports.get(name, ()) if isinstance(imports, dict) else ()


def _resolve_import(name, imports, known=None):
    """把 *name* 的 import 候选列表解析为单个 ``(module, attr)``。

    ``known`` 非 None 时**危险优先**：任一候选命中 ``known`` 即返回（防止
    安全候选遮蔽危险候选——与 raw_aliases 的候选解析语义一致）；否则返回
    第一个候选。无候选返回 None。
    """
    cands = _import_candidates(name, imports)
    if not cands:
        return None
    if known:
        for c in cands:
            if c in known:
                return c
    return cands[0]


def _import_is_module(name, module, imports):
    """*name* 的任一 import 候选的首模块是 *module*（多候选任一命中即真）。"""
    return any(c[0] == module for c in _import_candidates(name, imports))


def _resolve_call_target(func, imports, star_modules, raw_aliases, known):
    """把 ast.Call 的 func 解析为规范化 (module, attr)，失败返回 None。

    覆盖（#65592 review Blocker 2 + 举一反三）：
      - 直接调用: ``os.kill(x)``
      - import 别名: ``from os import kill; kill(x)``、``import os as o; o.kill(x)``
      - 赋值别名: ``killer = os.kill; killer(x)``（含链式 ``a=os; b=a.kill``）
      - star import: ``from os import *; kill(x)``
      - 动态调用: ``getattr(os, 'kill')(x)``、``os.__dict__['kill'](x)``
      - pathlib: ``Path(x).write_text(...)``、``pathlib.Path(x).open('a')``

    ``known`` 是调用方持有的 (module, attr) 集合，用于 star import 场景下
    判断 ``kill`` 是否确实来自 ``os``（避免把任意名字都当 os 的函数）。
    """
    # ── ast.NamedExpr: (k := os.kill)(...) — walrus 作调用目标
    #    （2026-08-25 复测：NamedExpr func 曾直接 return None → 放行）
    if isinstance(func, ast.NamedExpr):
        return _resolve_call_target(func.value, imports, star_modules, raw_aliases, known)

    # ── ast.Name: kill(...) ──
    if isinstance(func, ast.Name):
        name = func.id
        # imports 绑定也是候选（2026-08-28 拆解：from os import kill;
        # if cond: kill = os.path.join; kill(...) 的 kill 同时有 import
        # 绑定 ("os","kill") 和赋值候选 os.path.join——先查 imports 危险
        # 命中，防止赋值别名把 import 危险绑定遮蔽掉）。候选保留模型
        # （2026-08-28 re-review Blocker 1）：遍历**全部** import 候选，
        # 任一命中 known 即返回——兄弟/嵌套作用域的同名绑定不再互相遮蔽。
        for m, a in _import_candidates(name, imports):
            if a is not None and (m, a) in known:
                return (m, a)
        alias = _resolve_alias_value(name, imports, raw_aliases, None, known)
        if alias is not None:
            return alias
        _r = _resolve_import(name, imports, known)
        if _r is not None:
            m, a = _r
            return (m, a) if a is not None else None
        for mod in star_modules:  # from os import *; kill(...)
            if (mod, name) in known:
                return (mod, name)
        return None

    # ── ast.Attribute: os.kill(...) / o.kill(...) / os.path.expanduser(...)
    #    / Path(x).write_text(...) / p.write_text(...) ──
    if isinstance(func, ast.Attribute):
        attr = func.attr
        base = func.value
        # Path 构造（直接 / pathlib.Path / P 别名 / 变量存对象 p = Path(...)）
        if _resolve_path_constructor(base, raw_aliases, imports) is not None:
            return ("pathlib", attr)
        resolved_base = None
        if isinstance(base, ast.Name):
            if base.id == "Path":
                return ("pathlib", attr)
            # 候选保留模型：任一 import 候选命中 known 即优先（危险优先），
            # 否则取第一个候选；都没有再走赋值别名。
            resolved_base = _resolve_import(base.id, imports, known)
            if resolved_base is None:
                # 赋值别名：killer = os.kill 的 base 是 os；o.kill 的 base 是 o；
                # p.expanduser 的 base 是 p（p = os.path，组合属性）
                resolved_base = _resolve_alias_value(
                    base.id, imports, raw_aliases, None, known)
        elif isinstance(base, ast.Attribute):
            # 嵌套属性 base：os.path.expanduser 的 base 是 os.path
            # （2026-08-25 re-review：此前只认 Name base，直接 return None）
            resolved_base = _resolve_attribute_chain(
                base, imports, raw_aliases, None, known)
        elif isinstance(base, ast.Call):
            # psutil.Process(1).kill() — 实例方法链（2026-08-25 复测：
            # psutil.Process(...).kill() 曾完全逃逸）。base 是可调用构造，
            # 先解析构造器本身；psutil 家族的方法 kill 与 os.kill 同能力。
            inner = _resolve_call_target(base.func, imports, star_modules,
                                         raw_aliases, known)
            if inner is not None and inner[0] == "psutil":
                return ("psutil", attr)
            # __import__('os').kill(...) — base 是模块导入调用
            resolved_base = _resolve_binding_expr(
                base, imports, raw_aliases, None, known)
        elif isinstance(base, ast.Subscript):
            # sys.modules['os'].kill(...) / vars(os)['kill'](...) /
            # {'kill': os.kill}['kill'](...) — base 是下标取值
            resolved_base = _resolve_binding_expr(
                base, imports, raw_aliases, None, known)
        elif isinstance(base, ast.NamedExpr):
            # (k := os).kill(...) — walrus base
            resolved_base = _resolve_binding_expr(
                base, imports, raw_aliases, None, known)
        if resolved_base is not None:
            m, a = resolved_base
            # 组合属性链：o = os.path 后 o.expanduser(...) 必须解析成
            # ('os.path', 'expanduser') 而非 ('os', 'path')。
            if a is None:
                return (m, attr)
            return (f"{m}.{a}", attr)
        return None

    # ── ast.Call: getattr(os, 'kill')(...) / os.__dict__.get('kill')(...) ──
    if isinstance(func, ast.Call):
        # getattr（2026-08-25 复测：关键字形式 getattr(os, name='kill')
        # 曾绕过——分支只读位置参数；别名 base o=os 一并支持）
        if getattr(func.func, "id", None) == "getattr":
            obj_expr, attr_expr = _getattr_args(func)
            if obj_expr is not None and isinstance(obj_expr, ast.Name):
                m = None
                _r = _resolve_import(obj_expr.id, imports, known)
                if _r is not None:
                    m = _r[0]
                else:
                    alias = _resolve_alias_value(
                        obj_expr.id, imports, raw_aliases, None, known)
                    if alias is not None:
                        m = alias[0]
                if m is not None:
                    if (isinstance(attr_expr, ast.Constant)
                            and isinstance(attr_expr.value, str)):
                        return (m, attr_expr.value)
                    return (m, None)  # 动态属性名 — 由调用方决定是否保守拦截
            return None
        # X.__dict__.get('kill')(...) — .get 方法链（2026-08-25 复测：
        # os.__dict__.get('kill')(1, 15) 曾完全逃逸）
        if isinstance(func.func, ast.Attribute) and func.func.attr == "get":
            key = None
            if func.args:
                key = _fold_str_expr(func.args[0], raw_aliases)
            if key is not None:
                base = func.func.value
                if (isinstance(base, ast.Attribute) and base.attr == "__dict__"
                        and isinstance(base.value, ast.Name)):
                    m = None
                    _r = _resolve_import(base.value.id, imports, known)
                    if _r is not None:
                        m = _r[0]
                    else:
                        alias = _resolve_alias_value(
                            base.value.id, imports, raw_aliases, None, known)
                        if alias is not None:
                            m = alias[0]
                    if m is not None:
                        return (m, key)
                if isinstance(base, ast.Dict):
                    # {'kill': os.kill}.get('kill')(...) — dict 字面量 .get
                    for key_n, value_n in zip(base.keys, base.values):
                        if (isinstance(key_n, ast.Constant)
                                and key_n.value == key):
                            return _resolve_binding_expr(
                                value_n, imports, raw_aliases, None, known)
            return None
        # partial(os.kill, 1, 15)(...) — functools.partial 首参即被调用目标
        if (func.args and _resolve_binding_expr(
                func.func, imports, raw_aliases, None, known)
                in (("functools", "partial"), ("builtins", "partial"))):
            return _resolve_binding_expr(
                func.args[0], imports, raw_aliases, None, known)
        return None

    # ── ast.Subscript: os.__dict__['kill'](...) / o.__dict__['kill'](...) /
    #    sys.modules['os'].kill(...) / vars(os)['kill'](...) /
    #    {'kill': os.kill}['kill'](...) ──
    # （2026-08-25 复测：统一委托 _resolve_subscript_expr——下标取值形状
    #   全家族覆盖：__dict__、dict/list 字面量、sys.modules、globals()、
    #   vars()；调用位与赋值 RHS 共用同一解析）
    if isinstance(func, ast.Subscript):
        return _resolve_subscript_expr(func, imports, raw_aliases, None, known)

    return None


def _execute_code_has_self_destructive_ops(code: str) -> str | None:
    """Return a human-readable reason if *code* contains operations that
    can destroy the Hermes process or kill arbitrary processes, or None
    if the code is free of self-destructive operations.

    These operations are HARD BLOCKED — they never enter the approval
    chain and cannot be bypassed via yolo, smart mode, or session
    persistence.

    Scope honesty (#65592 review, andrexibiza): this is a *static* AST
    layer.  It blocks every statically-detectable form — direct calls,
    import aliases, assignment aliases (incl. chains, walrus, tuple-unpack,
    for-targets), container subscripts, star imports, ``getattr`` /
    ``__dict__`` dynamic access, ``sys.modules`` / ``globals()`` /
    ``vars()`` / ``__import__`` chains, ``functools.partial``,
    ``operator.attrgetter`` chains applied to os/sys (2026-08-31:
    ``attrgetter("kill")(os)(1, 9)`` is runtime-equivalent to the blocked
    ``getattr(os, "kill")`` form), the os.kill-equivalents ``signal.kill`` /
    ``signal.pthread_kill`` / ``psutil.kill`` /
    ``psutil.Process(...).kill()``, and ``eval``/``exec``/``compile`` with a
    string literal containing a process-killing call.  Code that builds the
    call at runtime (string concatenation into ``exec``, calls routed
    through user-defined functions/lambdas, non-literal for-iterables,
    ``attrgetter`` results bound to variables before application) is not
    statically visible.  That residual surface belongs to the
    runtime/sandbox boundary, not to this heuristic; the message returned
    to the model therefore does NOT claim an absolute "no bypass exists"
    guarantee.  Design follows Linux seccomp / macOS SIP only in spirit:
    if the operation is fundamentally incompatible with the agent's
    continued operation, no user consent can make it safe.
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)

    # ── Pass 2: walk call nodes ──────────────────────────────
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_target(
            node.func, imports, star_modules, raw_aliases, _HARD_BLOCKED_CALLS
        )
        if resolved is None:
            # ── operator.attrgetter 链（2026-08-31 举一反三）──────────
            # ``attrgetter("kill")(os)(1, 9)`` 与已拦截的
            # ``getattr(os, "kill")(1, 9)`` 运行时完全等价，但能力名只以
            # 字符串字面量出现，Name/Attribute 解析不可见 → 曾全层放行
            # （含 yolo/off）。所有成分静态可见（字面量 + 模块名）→ 与
            # getattr(os, ...) 保守拦截对齐：attrgetter(lit) 应用于
            # os/sys 模块即拦（getter 是 operator.attrgetter，base 是
            # os/sys 时等价动态属性访问，可能取到 kill/killpg）。
            if isinstance(node.func, ast.Call):
                outer = node.func  # attrgetter("kill")(os)
                if isinstance(outer.func, ast.Call):
                    getter = _resolve_call_target(
                        outer.func.func, imports, star_modules,
                        raw_aliases, _HARD_BLOCKED_CALLS)
                    if getter == ("operator", "attrgetter"):
                        for base_expr in outer.args:
                            b = _resolve_binding_expr(
                                base_expr, imports, raw_aliases, None,
                                _HARD_BLOCKED_CALLS)
                            if b is not None and b[0] in ("os", "sys"):
                                return (
                                    f"operator.attrgetter(...) applied to "
                                    f"{b[0]} — dynamic attribute access may "
                                    "resolve to a process-killing function "
                                    "(HARD BLOCKED, no approval path; "
                                    "use explicit calls instead)"
                                )
            continue
        m, a = resolved
        if (m, a) in _HARD_BLOCKED_CALLS:
            return (
                f"{m}.{a}() — "
                f"process-killing operation (HARD BLOCKED, no approval path)"
            )
        # ── eval/exec/compile 字符串字面量含 kill 家族（2026-08-31 复现）──
        # docstring 声称 eval("os.kill") string literal 是 HARD 覆盖形态，
        # 但实现此前缺失——eval/exec 只落 DANG 层 dynamic-exec 审批，
        # yolo/off 下放行。字面量字符串静态可见：含硬阻断能力调用即拦。
        if (m, a) in (("builtins", "eval"), ("builtins", "exec"),
                      ("builtins", "compile")):
            if node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                src = node.args[0].value
                for _m2, _a2 in _HARD_BLOCKED_CALLS:
                    if f"{_m2}.{_a2}" in src:
                        return (
                            f"{a} string literal contains {_m2}.{_a2}() — "
                            "process-killing operation (HARD BLOCKED, "
                            "no approval path)"
                        )
                # 裸名调用形态（from os import kill 后 exec("kill(1, 9)")）
                if (_KILL_CALL_IN_STR_RE.search(src)):
                    return (
                        f"{a} string literal contains a process-killing "
                        "call (HARD BLOCKED, no approval path)"
                    )
        if a is None and m in ("os", "sys"):
            # getattr(os, dynamic_name)(...) — 动态属性名，静态无法判定具体
            # 函数，但 os/sys 的动态属性访问可能取到 kill/killpg。保守拦截：
            # 该类别本身允许用户改用显式调用（os.kill 会被上面精确拦截并
            # 给出明确原因），误伤面极小。
            return (
                f"getattr({m}, ...) — dynamic attribute access on {m} "
                f"(HARD BLOCKED: may resolve to a process-killing function; "
                f"use explicit {m}.<name> calls instead)"
            )
    return None


# reason key → (用户可读的中文原因, 建议改用方式)。用于 execute_code
# 拦截提示区分具体原因，避免 Agent 被拦后无从判断该换什么工具。
_EXEC_CODE_DANGER_DETAILS = {
    "open-write": ("文件写入（open 的 w/a/x 或 + 模式）",
                   "改用 write_file 或 patch 工具"),
    "file-delete": ("文件/目录删除（os.remove / shutil.rmtree 等）",
                    "先确认目标路径，或改用 terminal 走正常审批"),
    "file-mutate": ("文件移动/复制/重命名（shutil.copy / os.rename 等）",
                    "先确认目标路径"),
    "command-exec": ("任意命令执行（subprocess / os.system 等）",
                     "改用 terminal 工具，走正常命令审批"),
    "ctypes-import": ("ctypes 模块导入（可绕过所有 Python 级检查）",
                      "确认确实需要 syscall 级访问"),
    "dynamic-exec": ("eval/exec/compile 动态代码执行（可绕过静态分析）",
                     "改用显式函数调用，或改用 terminal 工具走正常审批"),
    "sensitive-write": ("写入安全敏感路径（config / .ssh / 系统目录）",
                        "该路径受保护，禁止通过 execute_code 修改"),
}


def _exec_code_reason_text(reason: str) -> str:
    """把 reason key 转成用户可读的拦截说明（含建议改用方式）。"""
    detail = _EXEC_CODE_DANGER_DETAILS.get(reason)
    if detail is None:
        return f"危险操作（{reason}）"
    why, remedy = detail
    return f"{why}；建议：{remedy}"


def _log_blocked_exec_code(code: str, reason: str) -> None:
    """Log a blocked execute_code script with redacted content."""
    from agent.redact import redact_sensitive_text
    truncated = code[:4000]
    if len(code) > 4000:
        truncated += f"\n... [truncated, {len(code)} total chars]"
    logger.warning(
        "execute_code BLOCKED (%s). Script (%d chars):\n%s",
        reason, len(code), redact_sensitive_text(truncated),
    )


def _open_mode_is_write(call_node: ast.Call) -> bool:
    """判断 open(...) 调用的 mode 参数是否为写模式。

    open(file, mode='r', ...) — mode 是第二个位置参数或 keyword 参数。
    mode 缺省或明确只读（r/rb/rt）→ False；含写标志（w/a/x/+）→ True；
    mode 是变量/表达式无法静态判定 → True（保守拦截）。
    """
    mode_arg = None
    if len(call_node.args) >= 2:
        mode_arg = call_node.args[1]  # 第二个位置参数
    else:
        for kw in call_node.keywords:
            if kw.arg == "mode":
                mode_arg = kw.value
                break
    if mode_arg is None:
        return False  # 无 mode → 默认 'r'，只读
    if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
        return any(c in mode_arg.value for c in "wax+")
    return True  # 变量/表达式，无法静态判定 → 保守拦截


def _path_open_mode_is_write(call_node: ast.Call) -> bool:
    """判断 pathlib ``Path.open(mode, ...)`` 的 mode 是否为写模式。

    Path.open 的签名是 ``open(mode='r', buffering=-1, ...)`` —— mode 是
    第一个位置参数（与内置 open(file, mode='r') 不同，见 #65592 review）。
    mode 缺省或只读 → False；含写标志（w/a/x/+）→ True；无法静态判定 → True。
    """
    mode_arg = None
    if call_node.args:
        mode_arg = call_node.args[0]
    else:
        for kw in call_node.keywords:
            if kw.arg == "mode":
                mode_arg = kw.value
                break
    if mode_arg is None:
        return False  # 缺省 'r'，只读
    if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
        return any(c in mode_arg.value for c in "wax+")
    return True  # 变量/表达式，无法静态判定 → 保守拦截


def _execute_code_has_capability_leak(code: str) -> str | None:
    """检测硬阻断能力被「作为值存储/传递」而非直接调用（能力泄漏）。

    #94647 session-kernel 跨 cell 绕过的 cell-1 源头：``def f(): return
    os.kill`` 这类形状把能力以值的形式逃出当前 cell 的调用点扫描——
    cell 1 只存能力不调用，cell 2 用不透明名称间接调用，每 cell 单独
    扫描都判定安全。本函数解析每个 Name/Attribute 到硬阻断能力
    （_HARD_BLOCKED_CALLS）；若其出现在**非调用位置**（return 值 /
    赋值右值 / 容器元素 / 参数 / lambda 外的闭包捕获）→ 返回泄漏
    描述——能力存不进去，跨 cell 调用链在源头断开。

    调用位置（Call.func）由 _execute_code_has_self_destructive_ops
    处理，这里不重复。运行时构造（eval("os.kill") 字符串、lambda 内
    调用）超出静态边界 → XFAIL（sandbox/runtime 边界）。
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    parent_map = {}
    for p in ast.walk(tree):
        for child in ast.iter_child_nodes(p):
            parent_map[id(child)] = p
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        resolved = _resolve_call_target(
            node, imports, star_modules, raw_aliases, _HARD_BLOCKED_CALLS)
        if resolved is None or resolved not in _HARD_BLOCKED_CALLS:
            continue
        parent = parent_map.get(id(node))
        # 调用位置（Call.func）不算泄漏——由调用点检测处理
        if isinstance(parent, ast.Call) and parent.func is node:
            continue
        return (
            f"capability leak: {resolved[0]}.{resolved[1]} referenced as a "
            f"value (returned/stored/passed) instead of called — cross-cell "
            f"capability persistence vector (#94647); HARD BLOCKED so the "
            f"capability cannot leave the current cell"
        )
    return None


def _execute_code_has_dangerous_ops(code: str):
    """返回 execute_code 脚本中首个危险操作的 reason key（见
    ``_EXEC_CODE_DANGER_DETAILS``），无危险操作返回 None。

    Two-pass scan:
    1. Collect imports + assignment aliases via
       ``_collect_exec_code_bindings``
    2. Walk call nodes, resolving every call target to a canonical
       ``(module, attr)`` pair (direct calls, import aliases, assignment
       aliases, star imports, ``getattr`` / ``__dict__`` dynamic access,
       pathlib methods) before checking the denylist.

    Immune to whitespace / comments / string literals (``ast.parse``).
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)

    # ── 可疑模块整体导入（ctypes）────────────────────────────
    # 候选保留模型：imports[name] 是候选列表，遍历全部候选。
    for _cands in imports.values():
        for (_module, _attr) in _cands:
            if _module in _EXEC_CODE_SUSPICIOUS_IMPORTS:
                return "ctypes-import"

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    # ── Pass 2: walk call nodes ──────────────────────────────
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        resolved = _resolve_call_target(
            func, imports, star_modules, raw_aliases, _EXEC_CODE_DANGEROUS_CALLS
        )

        # open 形状（builtin open + io.open + codecs.open，2026-08-28
        # Blocker 2 等效面）——区分读写模式（只读放行，写拦截）。resolved
        # 覆盖直接调用、``op = open`` 赋值别名、``from builtins import
        # open as op``（2026-08-25 复现：赋值别名曾绕过此检测）。
        if resolved in _EXEC_CODE_OPEN_SHAPES:
            if _open_mode_is_write(node):
                return "open-write"
            continue  # 只读 open，放行

        # eval/exec/compile — 动态代码执行（review Blocker 2 举一反三：
        # ``eval("os.kill")(...)`` 的外层 callee 是 eval 调用，普通解析器
        # 不处理；字面量 eval 是静态可检测的，动态字符串拼接无法检测。
        # 别名形式（``e = eval``）同样解析为 ("builtins", "eval")）
        if resolved in (
            ("builtins", "eval"), ("builtins", "exec"), ("builtins", "compile"),
        ):
            return "dynamic-exec"

        if resolved is None:
            continue
        m, a = resolved
        reason = _EXEC_CODE_DANGEROUS_CALLS.get((m, a))
        if reason is None:
            # 命令执行家族模式匹配（2026-08-28 Blocker 3 根因修复）：
            # asyncio.create_subprocess_* / os.startfile / pty.spawn 曾
            # 不在精确名字表 → 本地 yolo/approvals-off 路径静默执行。
            reason = _match_command_exec_family(m, a)
        if reason is not None:
            if (m, a) == ("pathlib", "open"):
                # Path.open 的 mode 是第一个位置参数，需单独判断读写
                if _path_open_mode_is_write(node):
                    return "open-write"
                continue  # 只读 Path.open，放行
            return reason

    return None


# =========================================================================
# Layer 4 — Sensitive-Write Target Analysis (review Blocker 1)
# =========================================================================
# #49578 is an *effect/destination* invariant: security-sensitive files
# (Hermes config, ~/.ssh, system dirs) are hard-refused by the file-tool
# path (`_check_sensitive_path` in tools/file_tools.py) REGARDLESS of
# approval mode.  execute_code must preserve that invariant: a statically
# resolvable write to a protected target is hard-blocked BEFORE any
# recoverable approval bypass (--yolo / approvals.mode=off), so the
# invariant cannot be traded away by turning approvals off.
#
# Static limitation (honest): only literal / expanduser / simple-alias
# target strings are resolvable here.  A target computed at runtime
# (`os.path.join(home, name)`) is not statically visible — that residual
# surface belongs to the runtime/sandbox boundary.

# Sensitive prefixes mirrored from tools/file_tools._SENSITIVE_PATH_PREFIXES
# plus the Hermes/SSH trees (same class of protected destination #49578
# names).  Matched after ~/env expansion and normpath.
_EXEC_CODE_SENSITIVE_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/", "/private/etc/",
    "/private/var/db/", "/private/var/root/",
    "/run/", "/var/run/",
)
_EXEC_CODE_SENSITIVE_HOME_TREES = (".ssh", ".hermes", ".aws", ".gnupg")
_EXEC_CODE_SENSITIVE_EXACT = {"/var/run/docker.sock", "/run/docker.sock"}

# 只读/查询方法白名单（2026-08-25 补：#49578 残余面——pandas/numpy 等库
# 写方法的路径参数绕过 open()/Path() 形状检测）。带敏感路径参数的方法调用
# 若不在本集合中，一律 hard-block。集合覆盖：纯路径操作（无 IO）、存在性/
# 元数据查询、目录列举、文件内容读取（read/load/read_* 系列）。
_EXEC_CODE_READONLY_QUERY_NAMES = frozenset({
    # --- 纯路径操作（os.path.* / pathlib 属性，无文件 IO）---
    "join", "basename", "dirname", "split", "splitext", "abspath", "realpath",
    "normpath", "normcase", "expanduser", "expandvars", "commonpath",
    "commonprefix", "relpath", "samefile", "sameopenfile", "samestat",
    "name", "suffix", "suffixes", "stem", "anchor", "parent", "parents",
    "parts", "as_posix", "as_uri", "cwd", "home",
    # --- 存在性 / 元数据查询（不读取内容）---
    "exists", "isfile", "isdir", "islink", "ismount", "lexists",
    "is_file", "is_dir", "is_symlink", "is_socket", "is_fifo",
    "is_block_device", "is_char_device", "is_absolute", "is_relative_to",
    "stat", "lstat", "fstat", "getsize", "getmtime", "getctime", "getatime",
    "access", "walk", "scandir", "listdir", "glob", "iglob", "rglob",
    "iterdir", "absolute", "resolve",
    # --- 文件内容读取（读敏感目标 = #46900 的 secret 读取面，单独管控；
    #   与 open() 只读放行行为保持一致）---
    "read", "read_text", "read_bytes", "readlines", "readline",
    "load", "loads", "loadtxt", "loadmat", "load_npy", "fromfile",
    "fromstring", "frombuffer", "memmap", "imread", "imdecode", "mmap",
    "read_csv", "read_json", "read_excel", "read_parquet", "read_pickle",
    "read_hdf", "read_sql", "read_html", "read_xml", "read_fwf",
    "read_table", "read_sas", "read_spss", "read_clipboard", "read_feather",
    "read_orc", "read_stata", "read_gbq", "read_sql_table", "read_sql_query",
})

# 无文件系统 I/O 的容器/字符串方法白名单（2026-08-25 re-review Blocker 3b）：
# ``paths.append("/etc/passwd")`` 曾因 append 不在只读白名单 + 参数敏感
# 而被误报为 protected-path 硬阻断——append 只是把字符串加进内存列表，
# 不触碰文件系统。这些方法只操作内存对象（list/dict/set/str），携带
# 敏感字符串参数不构成文件 I/O，必须放行。
_EXEC_CODE_NO_IO_METHODS = frozenset({
    # --- list / deque / 序列容器 ---
    "append", "extend", "insert", "pop", "remove", "clear", "copy",
    "count", "index", "sort", "reverse", "popleft", "appendleft",
    # --- dict ---
    "get", "setdefault", "update", "keys", "values", "items", "popitem",
    "fromkeys",
    # --- set / frozenset ---
    "add", "discard", "union", "intersection", "difference",
    "symmetric_difference", "issubset", "issuperset", "isdisjoint",
    # --- str / bytes ---
    "format", "split", "join", "strip", "rstrip", "lstrip", "upper",
    "lower", "title", "capitalize", "casefold", "replace", "find",
    "rfind", "index", "rindex", "startswith", "endswith", "encode",
    "decode", "zfill", "ljust", "rjust", "center", "expandtabs",
    "partition", "rpartition", "splitlines", "rsplit", "removeprefix",
    "removesuffix", "isalnum", "isalpha", "isascii", "isdigit",
    "islower", "isupper", "isspace", "istitle", "isnumeric",
    "isdecimal", "isidentifier", "isprintable",
    # --- 数值/通用对象方法（无 I/O） ---
    "bit_length", "to_bytes", "from_bytes", "hex", "real", "imag",
})

# pathlib 变异方法（2026-08-25 re-review Blocker 3a receiver-bound）：
# 这些方法的目标路径在 **receiver**（Path 构造参数）而不是调用参数里，
# 且多数无参数（unlink/touch/chmod...）——参数型敏感检测看不到它们。
# rename/replace 的语义：receiver 是源，参数是目标——两个方向都要检查。
_EXEC_CODE_PATHLIB_MUTATORS = frozenset({
    "unlink", "rmdir", "touch", "chmod", "chown", "mkdir",
    "rename", "replace", "symlink_to", "hardlink_to",
})

_STRING_CONSTANT_EVAL_RE = re.compile(
    r"os\.path\.expanduser\((['\"])(.*?)\1\)", re.DOTALL
)


def _resolve_expr_path(expr, raw_aliases, imports) -> str | None:
    """Try to resolve *expr* (any AST expression) to a literal path string.

    Handles: string literal, ``os.path.expanduser("...")`` with a literal,
    a simple variable alias whose RHS is one of those, and (2026-08-25) a
    function-reference alias ``h = os.path.expanduser; h("...")``.  Returns
    None when the target is not statically resolvable.
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    # bytes 字面量（2026-08-28 拆解伪装面：``open(b"/root/.hermes/config.yaml",
    # "w")`` 的路径是 bytes 而非 str，曾完全漏过目标解析 → 只落可恢复审批）
    if isinstance(expr, ast.Constant) and isinstance(expr.value, bytes):
        try:
            return expr.value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    # 字符串拼接 BinOp（2026-08-28 拆解伪装面：``open("/root/.hermes/" +
    # "config.yaml", "w")`` 是 BinOp Add，曾漏过 → 只落可恢复审批）。
    # 与 _fold_str_expr 同语义：左右两侧都是静态可解析字符串才折叠。
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = _resolve_expr_path(expr.left, raw_aliases, imports)
        right = _resolve_expr_path(expr.right, raw_aliases, imports)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        return None
    # bytes.decode() 方法调用（2026-08-28 拆解伪装面：
    # ``b'config.yaml'.decode()`` 是 bytes→str 常见转换，静态可解析）。
    if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
            and expr.func.attr == "decode"
            and isinstance(expr.func.value, ast.Constant)
            and isinstance(expr.func.value.value, bytes)
            and not expr.args):
        try:
            return expr.func.value.value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    # f-string（2026-08-25 复测：#49578 不变量曾漏掉——f'/root/.hermes/...'
    # 是 JoinedStr，不是 Constant；全字面量/可解析插值拼接，否则视为运行时）
    if isinstance(expr, ast.JoinedStr):
        parts = []
        for v in expr.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                r = _resolve_expr_path(v.value, raw_aliases, imports)
                if isinstance(r, str):
                    parts.append(r)
                else:
                    return None
            else:
                return None
        return "".join(parts)
    if isinstance(expr, ast.Name) and expr.id in raw_aliases:
        for rhs in raw_aliases[expr.id]:
            # os.path.expanduser("~/.hermes/config.yaml")
            if (isinstance(rhs, ast.Constant) and isinstance(rhs.value, str)):
                return rhs.value
            if (isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute)
                    and isinstance(rhs.func.value, ast.Attribute)
                    and isinstance(rhs.func.value.value, ast.Name)
                    and rhs.func.value.value.id == "os"
                    and rhs.func.value.attr == "path"
                    and rhs.func.attr == "expanduser"
                    and rhs.args and isinstance(rhs.args[0], ast.Constant)
                    and isinstance(rhs.args[0].value, str)):
                return os.path.expanduser(rhs.args[0].value)
    # 任意可解析的 os.path 函数调用 — 统一走 _resolve_call_target：
    #   h = os.path.expanduser; h("...")（函数引用别名）
    #   p = os.path; p.expanduser("...")（组合属性别名，2026-08-25 re-review）
    #   os.path.expanduser("...")（直接调用）
    # 曾各自写死 AST 形状，组合属性链漏掉导致敏感写目标解析丢失。
    # 2026-08-25 复测扩展：expandvars（$HOME/...）与 join（全字面量参数）
    # 同为静态可解析形状，曾降级为可恢复审批 → yolo 下击穿 #49578 不变量。
    if (isinstance(expr, ast.Call) and expr.args
            and isinstance(expr.args[0], ast.Constant)
            and isinstance(expr.args[0].value, str)
            and _resolve_call_target(
                expr.func, imports, set(), raw_aliases, frozenset()
            ) == ("os.path", "expanduser")):
        return os.path.expanduser(expr.args[0].value)
    if (isinstance(expr, ast.Call) and expr.args
            and isinstance(expr.args[0], ast.Constant)
            and isinstance(expr.args[0].value, str)
            and _resolve_call_target(
                expr.func, imports, set(), raw_aliases, frozenset()
            ) == ("os.path", "expandvars")):
        return os.path.expandvars(expr.args[0].value)
    if (isinstance(expr, ast.Call) and expr.args
            and _resolve_call_target(
                expr.func, imports, set(), raw_aliases, frozenset()
            ) == ("os.path", "join")):
        # os.path.join("lit", "lit", ...) — 每个参数递归解析（expanduser/
        # expandvars/字面量/简单别名），任一不可解析 → 视为运行时拼接
        parts = []
        for a in expr.args:
            r = _resolve_expr_path(a, raw_aliases, imports)
            if isinstance(r, str):
                parts.append(r)
            else:
                return None
        if parts:
            return posixpath.join(*parts)
    return None


def _resolve_static_write_target(node: ast.Call, raw_aliases, imports,
                                 file_kw: str = "file") -> str | None:
    """解析 open 形状调用的写目标路径。

    按签名解析（2026-08-28 re-review Blocker 2 根因修复）：内置 open 的
    ``file`` 是签名参数——位置 0 **或关键字 file=**。此前只解析位置参数，
    ``open(file=\"/root/.hermes/config.yaml\", mode=\"w\")`` 的目标无法
    恢复 → 敏感写不变量降级为可恢复审批（yolo/approvals-off 击穿）。
    ``file_kw`` 是 open 形状的 file 参数关键字名（builtins/io 是 ``file``，
    codecs.open 是 ``filename``，见 ``_EXEC_CODE_OPEN_SHAPES``）。
    """
    if node.args:
        return _resolve_expr_path(node.args[0], raw_aliases, imports)
    for kw in node.keywords:
        if kw.arg == file_kw:
            return _resolve_expr_path(kw.value, raw_aliases, imports)
    return None


def _resolve_path_ctor_target(ctor: ast.Call, raw_aliases, imports) -> str | None:
    """Path 构造调用的完整目标：``Path("/root", ".hermes", "config.yaml")``
    的路径是全部参数按 posixpath.join 拼接（2026-08-28 拆解伪装面：
    之前只取 args[0]，多参构造的敏感路径被截断 → 只落可恢复审批）。
    任一参数静态不可解析 → 返回 None（保守降级，不误报）。
    """
    if not ctor.args:
        return None
    parts = []
    for a in ctor.args:
        r = _resolve_expr_path(a, raw_aliases, imports)
        if isinstance(r, str):
            parts.append(r)
        else:
            return None
    if len(parts) == 1:
        return parts[0]
    return posixpath.join(*parts)


def _write_target_is_sensitive(path: str) -> bool:
    """True if *path* targets a protected destination (mirrors the file-tool
    sensitive-path invariant from #49578)."""
    if not path:
        return False
    expanded = os.path.expanduser(os.path.expandvars(path))
    normalized = posixpath.normpath(expanded.replace("\\", "/"))
    # POSIX 规范：路径以 // 开头是实现定义，Linux 下 // == /；但
    # posixpath.normpath 会保留开头的双斜杠前缀，导致后续 startswith
    # 比较失败——``open("//root/.hermes/config.yaml", "w")`` 曾逃过不变量
    # （2026-08-25 复测）。折叠开头多斜杠为单斜杠再比较。
    if normalized.startswith("//"):
        normalized = "/" + normalized.lstrip("/")
    if normalized in _EXEC_CODE_SENSITIVE_EXACT:
        return True
    for prefix in _EXEC_CODE_SENSITIVE_PREFIXES:
        if normalized.startswith(prefix):
            return True
    # Hermes home tree (config/env live here — approvals.mode etc.)
    hermes_home = os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")
    hermes_norm = posixpath.normpath(hermes_home.replace("\\", "/"))
    if normalized == hermes_norm or normalized.startswith(hermes_norm + "/"):
        return True
    # Home trees that gate agent security: .ssh, .aws, .gnupg
    home = os.path.expanduser("~")
    home_norm = posixpath.normpath(home.replace("\\", "/"))
    for tree in _EXEC_CODE_SENSITIVE_HOME_TREES:
        target = home_norm + "/" + tree
        if normalized == target or normalized.startswith(target + "/"):
            return True
    return False


def _execute_code_has_sensitive_write(code: str) -> str | None:
    """Return the protected target path if *code* statically writes a
    sensitive destination via open()/Path(), else None.

    Runs BEFORE the yolo/approvals-off bypass gates in
    ``check_execute_code_guard`` so the #49578 destination invariant is
    enforced even when approval is turned off.  Only statically
    resolvable targets are caught; runtime-computed paths are a
    documented static limitation.
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        resolved = _resolve_call_target(
            func, imports, star_modules, raw_aliases, _EXEC_CODE_DANGEROUS_CALLS
        )

        # builtin open(...) / open(file, mode) with write mode
        # （resolved 覆盖 op = open 别名 / from builtins import open as op；
        # 2026-08-28 Blocker 2 根因修复：io.open / codecs.open 同属 open
        # 形状——file 参数按签名解析，位置 0 或关键字 file=/filename=）
        if resolved in _EXEC_CODE_OPEN_SHAPES:
            if _open_mode_is_write(node):
                file_kw = _EXEC_CODE_OPEN_SHAPES[resolved]
                target = _resolve_static_write_target(node, raw_aliases, imports,
                                                      file_kw)
                if target and _write_target_is_sensitive(target):
                    return target
            continue

        # Path(...).write_text / write_bytes / open(write) — 含变量存对象
        # p = Path("..."); p.write_text("x")（2026-08-25 re-review Blocker 2b）
        if resolved in (("pathlib", "write_text"), ("pathlib", "write_bytes")):
            if not isinstance(func, ast.Attribute):
                continue
            ctor = _resolve_path_constructor(func.value, raw_aliases, imports)
            if ctor is not None and ctor.args:
                target = _resolve_path_ctor_target(ctor, raw_aliases, imports)
                if target and _write_target_is_sensitive(target):
                    return target
        elif resolved == ("pathlib", "open"):
            if _path_open_mode_is_write(node):
                if not isinstance(func, ast.Attribute):
                    continue
                ctor = _resolve_path_constructor(func.value, raw_aliases, imports)
                if ctor is not None and ctor.args:
                    target = _resolve_path_ctor_target(ctor, raw_aliases, imports)
                    if target and _write_target_is_sensitive(target):
                        return target
        # receiver-bound mutations（2026-08-25 re-review Blocker 3a）：
        # Path("/root/.ssh/authorized_keys").unlink() / touch() / chmod()
        # 等——目标路径在 **receiver**（Path 构造参数）里且方法无参数，
        # 参数型检测完全看不到。unlink/rmdir 删除敏感文件、touch/chmod
        # 篡改敏感文件、symlink_to/hardlink_to 在敏感区建链接，都是
        # #49578 目标不变量的变异侧。
        elif (resolved is not None and resolved[0] == "pathlib"
              and resolved[1] in _EXEC_CODE_PATHLIB_MUTATORS):
            if not isinstance(func, ast.Attribute):
                continue
            ctor = _resolve_path_constructor(func.value, raw_aliases, imports)
            if ctor is not None and ctor.args:
                target = _resolve_path_ctor_target(ctor, raw_aliases, imports)
                if target and _write_target_is_sensitive(target):
                    return target
    return None


def _execute_code_touches_sensitive_path(code: str) -> str | None:
    """Return the protected target if any *library* call's path argument
    statically references a sensitive destination.

    Closes the #49578 residual surface found 2026-08-25: pandas/numpy and
    other third-party writers (``to_csv``/``save``/``dump``/...) accept
    arbitrary path strings that never match the open()/Path() AST shapes
    checked by ``_execute_code_has_sensitive_write`` — so
    ``pd.DataFrame(...).to_csv('/root/.ssh/authorized_keys')`` sailed
    straight through.  Any call whose method is NOT in the read-only
    query whitelist and whose positional/keyword arguments statically
    resolve to a sensitive path is hard-blocked.  Read-only access
    (os.path queries, existence checks, directory listing, content
    reads) stays allowed, matching the open() read-mode behaviour.
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # 只检查属性方法调用（obj.method(...)）；裸函数名调用（Path(...)、
        # open(...) 等构造函数/内置）由各自的形状检测负责——否则
        # ``Path('/root/.hermes').exists()`` 的构造参数会被误伤为敏感引用。
        if isinstance(func, ast.Attribute):
            method = func.attr
        else:
            continue

        if method in _EXEC_CODE_READONLY_QUERY_NAMES:
            continue
        # open()/Path() 写形状已在 _execute_code_has_sensitive_write 单独
        # 处理；这里跳过避免重复判定（其只读形态合法，不升级）。
        # 2026-08-28 Blocker 2：io.open/codecs.open 同属 open 形状。
        resolved = _resolve_call_target(
            func, imports, star_modules, raw_aliases, _EXEC_CODE_DANGEROUS_CALLS
        )
        if resolved in _EXEC_CODE_OPEN_SHAPES or resolved == ("pathlib", "open"):
            continue
        # pathlib 变异方法（unlink/touch/chmod/rename/replace/...）的目标
        # 可能是 receiver（sensitive_write 处理）或参数（这里处理）——
        # 不能套用内存容器方法的 NO_IO 白名单：``Path(x).replace(
        # "/root/.ssh/authorized_keys")`` 的 replace 与 str.replace 同名，
        # 但前者是文件覆盖，必须检查参数目标。
        if resolved is not None and resolved[0] == "pathlib" \
                and resolved[1] in _EXEC_CODE_PATHLIB_MUTATORS:
            pass  # 落入下方参数检查
        elif resolved is None and method in _EXEC_CODE_NO_IO_METHODS:
            # 内存容器/字符串方法，敏感字符串参数不构成文件 I/O
            # （``paths.append("/etc/passwd")`` 只是往列表加字符串）。
            # 仅在调用目标**无法解析**时适用——``os.remove`` /
            # ``shutil.copy`` 等已知目标的 method 名（remove/copy）与
            # list.remove/list.copy 撞名，但语义是文件操作，必须继续
            # 参数检查（2026-08-25 回归：曾误跳过 shutil.copy 检测）。
            continue

        # 检查所有位置参数 + 关键字参数是否静态解析为敏感路径
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            target = _resolve_expr_path(arg, raw_aliases, imports)
            if target and _write_target_is_sensitive(target):
                return target
    return None


# =========================================================================
# Layer 1 — Capability Whitelist: Safe Imports Classification
# =========================================================================
# Known-safe stdlib modules whose presence alone does not indicate danger.
# Scripts importing ONLY from these modules (and containing no dangerous
# call patterns) are classified as pure-data / computation — they pass
# through without triggering the approval prompt in CLI sessions.
# (from PR #65592 commit 4, 66e423e4)

_EXEC_CODE_SAFE_IMPORTS = frozenset({
    # Data formats
    "json", "csv", "base64", "binascii", "codecs",
    # Text processing
    "re", "string", "textwrap", "difflib", "unicodedata",
    # Numeric / math
    "math", "statistics", "fractions", "decimal", "numbers",
    "random",
    # Collections / data structures
    "collections", "itertools", "functools", "operator",
    "heapq", "bisect", "array", "struct",
    # Filesystem (read-only / temp)
    "pathlib", "tempfile", "glob", "fnmatch", "fileinput",
    # Date / time
    "datetime", "calendar", "time",
    # Hashing
    "hashlib", "hmac",
    # Type system / introspection
    "typing", "dataclasses", "enum", "inspect", "types",
    # Output / formatting
    "pprint", "textwrap",
    # Debugging / logging (read-only use)
    "traceback", "warnings", "logging",
    # Markup (safe parsing)
    "html",
})

# Modules whose import signals potential danger (process/file/network
# capability). Used by _classify_exec_code_imports for diagnostics.

_EXEC_CODE_DANGEROUS_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "ctypes",
    "socket", "signal", "multiprocessing", "threading",
    "http", "urllib", "ftplib", "smtplib", "poplib", "imaplib",
    "telnetlib", "asyncio",
})


def _classify_exec_code_imports(code: str) -> tuple[list[str], list[str], list[str]]:
    """Classify imports in *code* as (safe, dangerous, unknown).

    Returns three lists of top-level module name strings.  Used by
    Layer 1 (whitelist) to determine whether a script that has no
    dangerous call patterns should still trigger the guard because
    it imports dangerous or unrecognised modules.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], [], ["(syntax error)"]

    safe, dangerous, unknown = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _EXEC_CODE_DANGEROUS_IMPORTS:
                    dangerous.append(top)
                elif top in _EXEC_CODE_SAFE_IMPORTS:
                    safe.append(top)
                else:
                    unknown.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _EXEC_CODE_DANGEROUS_IMPORTS:
                    dangerous.append(top)
                elif top in _EXEC_CODE_SAFE_IMPORTS:
                    safe.append(top)
                else:
                    unknown.append(top)

    return safe, dangerous, unknown


# =========================================================================
# Package Acquisition Detection — execute_code side (#97657 BLOCKER 2)
# =========================================================================
# #97657 (dandckr-ops) introduces the owner-gated package-acquisition
# invariant for *terminal strings*: acquiring packages from registries is
# a supply-chain trust boundary requiring the owner's exact one-operation
# approval; YOLO / approvals.mode=off / Smart Approval / reusable scopes /
# isolated-container skips cannot bypass it. andrexibiza's review of
# #97657 assigned the *execute_code* side to this PR (#65592): a script
# can reach the same package managers via subprocess/os.system process-
# launch calls without passing through terminal approval, so the same
# invariant must hold here — BEFORE the container/YOLO/off short-circuits
# in check_execute_code_guard (isolated backends included).
#
# The word vocabulary mirrors tools/approval.py's _package_argv_is_acquisition
# (#97657) plus the alias spellings named in its review (apk add, npm add,
# uv run --with). Implemented independently — #97657 is unmerged and this
# PR must not import its branch code — so the two PRs compose regardless
# of which approval owner lands first.
#
# Detection is *confirmatory*, not conservative: it fires only when the
# process-launch argument statically resolves to an acquisition argv
# (literal string / literal list / foldable variable). A statically
# unresolvable argument is left to the runtime boundary — blocking every
# unparseable subprocess call would turn ordinary scripts into security
# terminals (npm run build is not acquisition).

_PACKAGE_EXEC_WRAPPERS = frozenset({
    "command", "builtin", "exec", "nohup", "setsid", "time", "nice",
    "timeout", "stdbuf", "sudo", "env", "xargs", "docker", "podman",
    "nerdctl", "cmd", "wsl",
})

# 环境赋值前缀（2026-08-31 举一反三）：``env PATH=/x pip install x`` /
# ``FOO=bar pip install y`` 的 VAR=val 曾中断 wrapper 剥离 → exe 误判为
# 赋值串 → 包获取漏检（yolo/off 下放行）。argv 已 lower()，名字全小写匹配。
_ENV_ASSIGN_RE = re.compile(r"^[a-z_][a-z0-9_]*=")


def _package_words_are_acquisition(words: list[str]) -> str | None:
    """判定 argv 词表是否为包获取命令；返回包管理器名，否则 None。

    与 #97657 ``_package_argv_is_acquisition`` 语义对齐 + review 补别名
    （apk add / npm add / uv run --with）。wrapper（sudo/env/docker/...）
    前缀剥除后检查内部命令；help 标志不构成获取。
    """
    if not words:
        return None
    argv = [w.strip("\"'").lower() for w in words]
    # 容器运行时子命令（2026-08-31 举一反三）：``docker run <img> pip
    # install x`` / ``docker exec <c> pip install x`` 的 run/exec 曾把
    # docker 当普通 wrapper 剥离 → exe 误判为 "run"/"exec" → 包获取漏检
    # （yolo/off 下放行）。必须在剥离**之前**判断（docker 本身是 wrapper
    # 名单成员）。剥离子命令 + flags + 一个目标参数（镜像名/容器名）后
    # 递归判定剩余词表（容器内装包同属获取动作，owner-gate 不区分宿主/
    # 容器边界；``docker pull`` 拉镜像是镜像而非包，不判定）。
    if argv and argv[0] in ("docker", "podman", "nerdctl"):
        if len(argv) >= 2 and argv[1] in ("run", "exec", "create", "start"):
            rest = argv[2:]
            while rest and rest[0].startswith("-"):
                rest = rest[1:]
            if rest:
                rest = rest[1:]  # 镜像名/容器名
            if rest:
                return _package_words_are_acquisition(rest)
        return None
    idx = 0
    # wrapper 前缀（sudo/env/...）+ 环境赋值（VAR=val）剥离。赋值
    # 剥离 2026-08-31 举一反三：``env PATH=/x pip install x`` 的 PATH=/x
    # 曾中断剥离（exe 误判为 "path=/x"）→ 包获取漏检。
    while idx < len(argv) and (
        argv[idx] in _PACKAGE_EXEC_WRAPPERS
        or _ENV_ASSIGN_RE.match(argv[idx])
    ):
        idx += 1
    if idx >= len(argv):
        return None
    exe = re.split(r"[\\/]", argv[idx])[-1].removesuffix(".exe")
    args = argv[idx + 1:]
    if "--help" in args or "-h" in args:
        return None

    def has(action: str) -> bool:
        return action in args

    if re.fullmatch(r"(?:python(?:\d+(?:\.\d+)*)?|py)", exe):
        if any(args[i] == "-m" and args[i + 1] == "pip"
               and "install" in args[i + 2:]
               for i in range(len(args) - 1)):
            return "pip"
        return None
    if re.fullmatch(r"pip(?:\d+(?:\.\d+)*)?", exe):
        return "pip" if has("install") else None
    if exe == "pipx":
        return "pipx" if any(a in args for a in {"install", "run", "runpip"}) else None
    if exe == "uv":
        if "pip" in args and "install" in args[args.index("pip") + 1:]:
            return "uv"
        if any(a in args for a in {"add", "sync"}):
            return "uv"
        if any(args[i] == "tool" and args[i + 1] in {"install", "run"}
               for i in range(len(args) - 1)):
            return "uv"
        # review 补：uv run --with <pkg> 把包装进调用环境（临时安装）
        if "run" in args and "--with" in args:
            return "uv"
        return None
    if exe == "uvx":
        return "uvx"
    if exe == "npm":
        # review 补：npm add 是 npm install 的文档化别名
        if any(a in args for a in {"install", "i", "ci", "exec", "add"}):
            return "npm"
        return None
    if exe == "npx":
        return "npx"
    if exe == "pnpm":
        return "pnpm" if any(a in args for a in {"add", "install", "i", "dlx"}) else None
    if exe == "yarn":
        return "yarn" if any(a in args for a in {"add", "install", "dlx"}) else None
    if exe == "bun":
        return "bun" if any(a in args for a in {"add", "install", "i", "x"}) else None
    if exe == "deno":
        if any(a in args for a in {"install", "add"}):
            return "deno"
        if "run" in args and any(f in args for f in {"--allow-all", "-a"}):
            return "deno"
        return None
    if exe in {"cargo", "gem", "go", "winget", "choco", "scoop", "brew"}:
        return exe if has("install") else None
    if exe in {"apk", "apt", "apt-get", "dnf", "yum", "zypper"}:
        # review 补：apk add 是 Alpine 的包安装操作（apt 系仍以 install 为主，
        # 但 add 一并覆盖无成本）
        return exe if (has("install") or has("add")) else None
    if exe == "pacman":
        return "pacman" if any(a.startswith("-s") and a != "-ss" for a in args) else None
    if exe in {"conda", "mamba", "micromamba"}:
        return exe if any(a in args for a in {"install", "create", "update"}) else None
    if exe == "dotnet":
        return "dotnet" if any(args[i] == "tool" and args[i + 1] == "install"
                               for i in range(len(args) - 1)) else None
    if exe == "poetry":
        return "poetry" if any(a in args for a in {"add", "install", "update"}) else None
    if exe == "composer":
        return "composer" if any(a in args for a in {"require", "install", "update"}) else None
    if exe == "bundle":
        return "bundle" if any(a in args for a in {"install", "update"}) else None
    return None


def _arg_to_argv(expr, raw_aliases, imports) -> list[str] | None:
    """把进程启动调用的参数表达式静态解析为 argv 词表。

    覆盖：字符串字面量（shlex 分词）、bytes 字面量、list/tuple 字面量
    （元素均为字符串字面量）、简单变量别名（递归）、可折叠字符串
    （拼接 / decode / f-string / expanduser，复用 _resolve_expr_path）。
    静态不可解析返回 None —— 包获取是**确认式**检测，不可判定形状
    留给运行时边界，避免误拦正常脚本。
    """
    if expr is None:
        return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        try:
            return shlex.split(expr.value)
        except ValueError:
            return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, bytes):
        try:
            return shlex.split(expr.value.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
    if isinstance(expr, (ast.List, ast.Tuple)):
        words = []
        for elt in expr.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                words.append(elt.value)
            else:
                return None
        return words
    if isinstance(expr, ast.Name) and expr.id in raw_aliases:
        for cand in raw_aliases[expr.id]:
            resolved = _arg_to_argv(cand, raw_aliases, imports)
            if resolved is not None:
                return resolved
        return None
    folded = _resolve_expr_path(expr, raw_aliases, imports)
    if isinstance(folded, str):
        try:
            return shlex.split(folded)
        except ValueError:
            return None
    return None


def _call_arg_expr(node: ast.Call, position: int, keyword: str | None):
    """按位置/关键字取调用参数表达式（位置优先，keyword 兜底）。"""
    if len(node.args) > position:
        return node.args[position]
    if keyword:
        for kw in node.keywords:
            if kw.arg == keyword:
                return kw.value
    return None


def _extract_exec_spawn_argv(m, a, node, raw_aliases, imports):
    """签名感知提取 exec*/spawn* 家族的 (path, argv) 词表（P1 修复）。

    os/posix 各家族的真实签名槽位：
      execv/execve/execvp/execvpe:   path@0, argv@1, (env@2 for *e)
      spawnv/spawnve/spawnvp/spawnvpe: mode@0, path@1, argv@2, (env@3)
      posix.spawn/spawnp:            path@0, argv@1, env@2
      execl/execlp:                  path@0, argv = args[1:]
      execle/execlpe:                path@0, argv = args[1:-1] (尾 env)
      spawnl/spawnlp:                mode@0, path@1, argv = args[2:]
      spawnle/spawnlpe:              mode@0, path@1, argv = args[2:-1]

    返回 (path_words, argv_words)：path_words 是真实可执行文件词表
    （静态不可解析时 None，调用方降级用 argv[0] 判定）；argv_words
    是该家族的 argv 词表（任一元素静态不可解析时返回 None）。
    """
    is_exec = a.startswith("exec")
    is_posix_spawn = a in ("posix_spawn", "posix_spawnp") or (
        m == "posix" and a in ("spawn", "spawnp"))
    is_spawn = a.startswith("spawn") or is_posix_spawn
    if not (is_exec or is_spawn):
        return None, None
    path_idx = 0 if (is_exec or is_posix_spawn) else 1  # spawn* 有 mode 前缀
    path_expr = _call_arg_expr(node, path_idx, None)
    path_words = _arg_to_argv(path_expr, raw_aliases, imports) if path_expr is not None else None
    is_l = a.startswith("execl") or a.startswith("spawnl")
    if is_l:
        has_trailing_env = a.endswith("e")  # execle/execlpe/spawnle/spawnlpe
        argv_exprs = node.args[path_idx + 1:-1 if has_trailing_env else None]
    else:
        # spawnv*(mode,path,argv) argv@2；execv*(path,argv)/posix.spawn(path,argv,env) argv@1
        argv_idx = 2 if (a.startswith("spawn") and not is_posix_spawn) else 1
        argv_expr = _call_arg_expr(node, argv_idx, None)
        argv_exprs = [argv_expr] if argv_expr is not None else []
    words = []
    for e in argv_exprs:
        part = _arg_to_argv(e, raw_aliases, imports)
        if part is None:
            return path_words, None
        words.extend(part)
    return path_words, (words or None)


# fail-closed 标记（P0-3）：command-exec 调用存在但 argv 静态不可解析，
# 无法判定是否包获取。返回此标记 → 调用方要求 owner 审批，不放行。
_PACKAGE_UNRESOLVABLE = "?unresolvable-command-exec?"


def _execute_code_has_package_acquisition(code: str) -> str | None:
    """返回脚本中静态可确认的包获取调用的包管理器名，否则 None。

    只检查命令执行家族的调用（subprocess.* / os.system / os.popen /
    os.spawn* / os.exec* / os.posix_spawn* / pty.spawn /
    asyncio.create_subprocess_*），按调用形状提取 argv 后走
    ``_package_words_are_acquisition`` 词表判定。

    与 #97657 的 terminal 侧同一不变量：包获取必须 owner 精确单操作
    批准，容器/YOLO/approvals-off 均不可绕过（guard 在短路前调用本函数）。

    返回语义（2026-08-31 P0-3 fail-closed）：
      - 包管理器名     → 静态确认的包获取，调用方 owner-gate 拦截
      - _PACKAGE_UNRESOLVABLE → command-exec 调用存在但 argv 静态不可
        解析，无法排除包获取 → 调用方同样要求 owner 审批（不放行）
      - None           → 无命令执行调用，或全部可解析且非包获取
    """
    imports, star_modules, raw_aliases = _collect_exec_code_bindings(code)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        resolved = _resolve_call_target(
            node.func, imports, star_modules, raw_aliases,
            _EXEC_CODE_DANGEROUS_CALLS,
        )
        if resolved is None:
            continue
        m, a = resolved
        # 只检查命令执行能力（精确表或家族规则）
        is_command_exec = (
            _EXEC_CODE_DANGEROUS_CALLS.get((m, a)) == "command-exec"
            or _match_command_exec_family(m, a) is not None
        )
        if not is_command_exec:
            continue
        words = None
        if m == "subprocess":
            # run/call/Popen/check_output/check_call: args 位置 0 或 args=
            words = _arg_to_argv(
                _call_arg_expr(node, 0, "args"), raw_aliases, imports)
        elif m == "os" and a in ("system", "popen"):
            words = _arg_to_argv(_call_arg_expr(node, 0, None), raw_aliases, imports)
        elif m == "pty" and a == "spawn":
            words = _arg_to_argv(_call_arg_expr(node, 0, None), raw_aliases, imports)
        elif m == "asyncio" and a == "create_subprocess_shell":
            words = _arg_to_argv(_call_arg_expr(node, 0, None), raw_aliases, imports)
        elif m == "asyncio" and a == "create_subprocess_exec":
            # 全部位置参数即 argv（程序名 + 参数）
            words = []
            ok = True
            for arg in node.args:
                part = _arg_to_argv(arg, raw_aliases, imports)
                if part is None:
                    ok = False
                    break
                words.extend(part)
            if not ok:
                words = None
        elif (m in ("os", "posix")
              and (a.startswith("spawn") or a.startswith("exec")
                   or a in ("posix_spawn", "posix_spawnp"))):
            # 签名感知：*e 家族最后参数是 env 不是 argv；path 槽位
            # 各族不同（P1 修复，此前 node.args[-1] 一刀切漏检
            # execve/spawnve/posix_spawn）。path 参数是权威执行文件，
            # 解析成功则用它覆盖 argv[0] 判定（防 argv[0] 伪造）。
            path_words, words = _extract_exec_spawn_argv(
                m, a, node, raw_aliases, imports)
            if words and path_words:
                words = [path_words[0]] + words
        if words is None:
            # fail-closed（P0-3，对齐 #98138 bounded 设计）：command-exec
            # 调用存在但 argv 静态不可解析 → 无法判定是否包获取。不放行、
            # 不跳过——交 owner 审批（yolo/off/容器路径在调用本函数之前
            # 已被短路，这里返回的不可判定标记同样不可被它们绕过）。
            return _PACKAGE_UNRESOLVABLE
        if not words:
            continue
        pkg = _package_words_are_acquisition(words)
        if pkg is not None:
            return pkg
    return None
