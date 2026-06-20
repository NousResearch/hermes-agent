#!/usr/bin/env python3
"""主爱马仕本地资源注册表(组织大脑的一部分)—— 协同地基 Phase 1a。

本地优先:子(子爱马仕)把自己的资源(知识库 / 工作流 / agent 能力)**经云中转**上报,
主爱马仕拉取后把 authoritative 副本**存在本地这张表**;后续授权(grant)、鉴权过滤都基于本地这份。
云端 kari-cloud 只做 账号/计费/发 key + store-and-forward 中转,**不长存资源内容**。

表:
    resource(node_uid, kind, resource_id, name, meta, updated_ts)
      node_uid     = 资源所属的(子)账号 uid(= 上报者)
      kind         = 'knowledge' | 'workflow' | 'agent'
      resource_id  = 资源在该节点内的稳定 id
      name         = 展示名
      meta         = JSON 文本(摘要/类型/数量等附加信息)
      PRIMARY KEY (node_uid, kind, resource_id)

注:工作流 kind 只收「ChatInput + ChatOutput」对话流(硬约定,见 easyhermes-copilot-authored-flows-roles)。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

KINDS = ("knowledge", "workflow", "agent")


def _db_path() -> Path:
    override = os.getenv("KARI_RESOURCES_DB")
    if override:
        path = Path(override)
    else:
        from hermes_constants import get_hermes_home  # noqa: PLC0415 — 与 kie_billing 一致的本地路径来源

        path = get_hermes_home() / "kari_resources.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(_db_path()), timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute(
        "CREATE TABLE IF NOT EXISTS resource ("
        "node_uid TEXT NOT NULL, kind TEXT NOT NULL, resource_id TEXT NOT NULL, "
        "name TEXT, meta TEXT, updated_ts REAL NOT NULL, "
        "PRIMARY KEY (node_uid, kind, resource_id))"
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_resource_kind ON resource (kind, node_uid)")
    # 授权策略(Phase 2a):角色被授权可用「某节点上的某资源」。表名避开 SQL 保留字 GRANT。
    c.execute(
        "CREATE TABLE IF NOT EXISTS grant_policy ("
        "role TEXT NOT NULL, node_uid TEXT NOT NULL, kind TEXT NOT NULL, resource_id TEXT NOT NULL, "
        "created_ts REAL NOT NULL, "
        "PRIMARY KEY (role, node_uid, kind, resource_id))"
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_grant_role ON grant_policy (role)")
    # 按具体下级账号微调的直授(slice B):在角色授权之外,把某资源直接授权给某个账号。
    # 与 grant_policy 并存(role 一张、account 一张),解析时取并集;PK 各自独立避免迁移角色表。
    c.execute(
        "CREATE TABLE IF NOT EXISTS grant_user ("
        "target_user_id TEXT NOT NULL, node_uid TEXT NOT NULL, kind TEXT NOT NULL, resource_id TEXT NOT NULL, "
        "created_ts REAL NOT NULL, "
        "PRIMARY KEY (target_user_id, node_uid, kind, resource_id))"
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_grant_user ON grant_user (target_user_id)")
    return c


def _row(r) -> dict:
    try:
        meta = json.loads(r[4] or "{}")
    except (ValueError, TypeError):
        # 单条坏 meta 不该崩掉整张表(list_resources 是授权过滤的数据源)。
        meta = {}
    return {
        "node_uid": r[0],
        "kind": r[1],
        "resource_id": r[2],
        "name": r[3],
        "meta": meta,
        "updated_ts": r[5],
    }


def upsert_resource(node_uid: str, kind: str, resource_id: str, name: str = "", meta: dict | None = None) -> bool:
    """登记/更新一条资源(主拉取子上报后写入)。kind 必须是 KINDS 之一。"""
    node_uid = str(node_uid or "").strip()
    kind = str(kind or "").strip()
    resource_id = str(resource_id or "").strip()
    if not (node_uid and kind in KINDS and resource_id):
        return False
    with closing(_conn()) as c:
        c.execute(
            "INSERT INTO resource (node_uid, kind, resource_id, name, meta, updated_ts) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(node_uid, kind, resource_id) DO UPDATE SET "
            "name=excluded.name, meta=excluded.meta, updated_ts=excluded.updated_ts",
            (node_uid, kind, resource_id, (name or "").strip(), json.dumps(meta or {}, ensure_ascii=False), time.time()),
        )
        c.commit()
    return True


def replace_node_resources(node_uid: str, kind: str, items: list[dict] | None) -> int:
    """用一批资源**整体替换**某节点某 kind 的注册(子重新上报时调用,保证删掉已不存在的)。
    items = [{resource_id, name?, meta?}]。

    **items 为 None → 不动注册表,返回 -1**(区分"拉取失败"和"该节点确实清空了"——
    上报管道拿不到数据时必须传 None 而不是 [],否则会误删主端已有副本)。
    items = [] → 合法清空该节点该 kind。返回写入条数。"""
    if items is None:
        return -1
    node_uid = str(node_uid or "").strip()
    kind = str(kind or "").strip()
    if not (node_uid and kind in KINDS):
        return 0
    now = time.time()
    rows = []
    for it in items:
        rid = str((it or {}).get("resource_id") or "").strip()
        if rid:
            rows.append(
                (
                    node_uid,
                    kind,
                    rid,
                    str(it.get("name") or "").strip(),
                    json.dumps(it.get("meta") or {}, ensure_ascii=False),
                    now,
                )
            )
    with closing(_conn()) as c:
        c.execute("DELETE FROM resource WHERE node_uid=? AND kind=?", (node_uid, kind))
        if rows:
            c.executemany(
                "INSERT INTO resource (node_uid, kind, resource_id, name, meta, updated_ts) VALUES (?,?,?,?,?,?)",
                rows,
            )
        c.commit()
    return len(rows)


def list_resources(kind: str | None = None, node_uid: str | None = None) -> list[dict]:
    """列出已注册资源,可按 kind / node 过滤(主端配置授权、运行时鉴权都从这查)。"""
    where: list[str] = []
    args: list[str] = []
    if kind:
        where.append("kind=?")
        args.append(str(kind))
    if node_uid:
        where.append("node_uid=?")
        args.append(str(node_uid))
    sql = "SELECT node_uid, kind, resource_id, name, meta, updated_ts FROM resource"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY kind, node_uid, name"
    with closing(_conn()) as c:
        return [_row(r) for r in c.execute(sql, args).fetchall()]


def remove_node(node_uid: str, kind: str | None = None) -> int:
    """删除某节点(某 kind)的所有资源(子下线/退出时清)。返回删除条数。"""
    with closing(_conn()) as c:
        if kind:
            cur = c.execute("DELETE FROM resource WHERE node_uid=? AND kind=?", (str(node_uid), str(kind)))
        else:
            cur = c.execute("DELETE FROM resource WHERE node_uid=?", (str(node_uid),))
        c.commit()
        return cur.rowcount


# ---------------------------------------------------------------------------
# 授权(grant)—— 主本地组织大脑的「角色 → 资源」策略,协同地基 Phase 2a。
#
# 核心:一条 grant = (role, node_uid, kind, resource_id) —— 「角色被授权可用 *某节点上的某资源*」。
# 资源由 (node_uid, kind, resource_id) 唯一标识(resource_id 只在节点内唯一,故必须带 node_uid)。
# role 由云端账号体系定义(account.roles);这里只存主本地的授权策略,运行时鉴权(Phase 3)据此过滤。
# grant 与 resource 解耦:不要求资源此刻在注册表里(子可能离线/未上报),也不随资源删除级联清理
# (留陈旧 grant 无害——永远匹配不上;角色删除时用 remove_role_grants 清)。
# ---------------------------------------------------------------------------
def add_grant(role: str, node_uid: str, kind: str, resource_id: str) -> bool:
    """授权:角色可用某节点上的某资源。已存在则保留原 created_ts(幂等)。非法入参返回 False。"""
    role = str(role or "").strip()
    node_uid = str(node_uid or "").strip()
    kind = str(kind or "").strip()
    resource_id = str(resource_id or "").strip()
    if not (role and node_uid and kind in KINDS and resource_id):
        return False
    with closing(_conn()) as c:
        c.execute(
            "INSERT INTO grant_policy (role, node_uid, kind, resource_id, created_ts) VALUES (?,?,?,?,?) "
            "ON CONFLICT(role, node_uid, kind, resource_id) DO NOTHING",
            (role, node_uid, kind, resource_id, time.time()),
        )
        c.commit()
    return True


def remove_grant(role: str, node_uid: str, kind: str, resource_id: str) -> bool:
    """撤销一条授权。返回是否确实删到一条。"""
    with closing(_conn()) as c:
        cur = c.execute(
            "DELETE FROM grant_policy WHERE role=? AND node_uid=? AND kind=? AND resource_id=?",
            (
                str(role or "").strip(),
                str(node_uid or "").strip(),
                str(kind or "").strip(),
                str(resource_id or "").strip(),
            ),
        )
        c.commit()
        return cur.rowcount > 0


def list_grants(role: str | None = None, node_uid: str | None = None, kind: str | None = None) -> list[dict]:
    """列出授权策略,可按 role / node / kind 过滤(团队角色面板回显、Phase 3 鉴权都从这查)。"""
    where: list[str] = []
    args: list[str] = []
    if role:
        where.append("role=?")
        args.append(str(role))
    if node_uid:
        where.append("node_uid=?")
        args.append(str(node_uid))
    if kind:
        where.append("kind=?")
        args.append(str(kind))
    sql = "SELECT role, node_uid, kind, resource_id, created_ts FROM grant_policy"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY role, node_uid, kind, resource_id"
    with closing(_conn()) as c:
        return [
            {"role": r[0], "node_uid": r[1], "kind": r[2], "resource_id": r[3], "created_ts": r[4]}
            for r in c.execute(sql, args).fetchall()
        ]


def is_granted(role: str, node_uid: str, kind: str, resource_id: str) -> bool:
    """运行时鉴权(Phase 3 的基石):该角色是否被授权可用这条资源。"""
    with closing(_conn()) as c:
        row = c.execute(
            "SELECT 1 FROM grant_policy WHERE role=? AND node_uid=? AND kind=? AND resource_id=? LIMIT 1",
            (
                str(role or "").strip(),
                str(node_uid or "").strip(),
                str(kind or "").strip(),
                str(resource_id or "").strip(),
            ),
        ).fetchone()
        return row is not None


def list_authorized_resources(role: str, kind: str | None = None) -> list[dict]:
    """解析「角色 → 实际可用资源」—— 鉴权过滤的产出(Phase 3a)。

    = grant_policy ⨝ resource:既被授权(grant)**又**当前确实在注册表里的资源,带 name/meta。
    与 list_grants 的区别:**排除失效授权**(授权还在但资源已被删/子下线不再上报的,
    不算"可用")。这是子/agent 该被告知"你能用什么"的权威清单,也是 Phase 3b 中继执行前的过滤源。
    """
    role = str(role or "").strip()
    if not role:
        return []
    sql = (
        "SELECT r.node_uid, r.kind, r.resource_id, r.name, r.meta, r.updated_ts "
        "FROM grant_policy g JOIN resource r "
        "ON r.node_uid=g.node_uid AND r.kind=g.kind AND r.resource_id=g.resource_id "
        "WHERE g.role=?"
    )
    args: list[str] = [role]
    if kind:
        sql += " AND r.kind=?"
        args.append(str(kind))
    sql += " ORDER BY r.kind, r.node_uid, r.name"
    with closing(_conn()) as c:
        return [_row(r) for r in c.execute(sql, args).fetchall()]


def remove_role_grants(role: str) -> int:
    """清空某角色的全部授权(角色被删除时调用)。返回删除条数。"""
    with closing(_conn()) as c:
        cur = c.execute("DELETE FROM grant_policy WHERE role=?", (str(role or "").strip(),))
        c.commit()
        return cur.rowcount


# --------------------------- 按具体下级账号微调授权(grant_user,slice B)---------------------------
def add_user_grant(target_user_id: str, node_uid: str, kind: str, resource_id: str) -> bool:
    """直接授权某个下级账号可用某资源(角色授权之外的追加/微调)。幂等。非法入参返回 False。"""
    target_user_id = str(target_user_id or "").strip()
    node_uid = str(node_uid or "").strip()
    kind = str(kind or "").strip()
    resource_id = str(resource_id or "").strip()
    if not (target_user_id and node_uid and kind in KINDS and resource_id):
        return False
    with closing(_conn()) as c:
        c.execute(
            "INSERT INTO grant_user (target_user_id, node_uid, kind, resource_id, created_ts) VALUES (?,?,?,?,?) "
            "ON CONFLICT(target_user_id, node_uid, kind, resource_id) DO NOTHING",
            (target_user_id, node_uid, kind, resource_id, time.time()),
        )
        c.commit()
    return True


def remove_user_grant(target_user_id: str, node_uid: str, kind: str, resource_id: str) -> bool:
    """撤销一条账号直授。返回是否确实删到一条。"""
    with closing(_conn()) as c:
        cur = c.execute(
            "DELETE FROM grant_user WHERE target_user_id=? AND node_uid=? AND kind=? AND resource_id=?",
            (
                str(target_user_id or "").strip(),
                str(node_uid or "").strip(),
                str(kind or "").strip(),
                str(resource_id or "").strip(),
            ),
        )
        c.commit()
        return cur.rowcount > 0


def list_user_grants(
    target_user_id: str | None = None, node_uid: str | None = None, kind: str | None = None
) -> list[dict]:
    """列出账号直授,可按 user / node / kind 过滤(面板回显、slice C 鉴权用)。返回里键为 user_id。"""
    where: list[str] = []
    args: list[str] = []
    if target_user_id:
        where.append("target_user_id=?")
        args.append(str(target_user_id))
    if node_uid:
        where.append("node_uid=?")
        args.append(str(node_uid))
    if kind:
        where.append("kind=?")
        args.append(str(kind))
    sql = "SELECT target_user_id, node_uid, kind, resource_id, created_ts FROM grant_user"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY target_user_id, node_uid, kind, resource_id"
    with closing(_conn()) as c:
        return [
            {"user_id": r[0], "node_uid": r[1], "kind": r[2], "resource_id": r[3], "created_ts": r[4]}
            for r in c.execute(sql, args).fetchall()
        ]


def is_user_granted(target_user_id: str, node_uid: str, kind: str, resource_id: str) -> bool:
    """该账号是否被**直接**授权可用这条资源(角色授权另算;slice C 解析器取两者并集)。"""
    with closing(_conn()) as c:
        row = c.execute(
            "SELECT 1 FROM grant_user WHERE target_user_id=? AND node_uid=? AND kind=? AND resource_id=? LIMIT 1",
            (
                str(target_user_id or "").strip(),
                str(node_uid or "").strip(),
                str(kind or "").strip(),
                str(resource_id or "").strip(),
            ),
        ).fetchone()
        return row is not None


def remove_user_grants(target_user_id: str) -> int:
    """清空某账号的全部直授(账号被删/退出时调)。返回删除条数。"""
    with closing(_conn()) as c:
        cur = c.execute("DELETE FROM grant_user WHERE target_user_id=?", (str(target_user_id or "").strip(),))
        c.commit()
        return cur.rowcount
