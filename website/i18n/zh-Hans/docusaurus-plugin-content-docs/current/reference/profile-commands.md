---
sidebar_position: 7
title: "Profile 命令参考"
---

# Profile 命令参考

本页涵盖 [Hermes profiles](/user-guide/profiles) 相关命令。

## `hermes profile`

```bash
hermes profile <subcommand>
```

用于管理 profile 的顶级命令。常用子命令：`list`、`use`、`create`、`delete`、`show`、`alias`、`rename`、`export`、`import`、`install`、`update`、`info`。

## `hermes profile list`

列出所有 profile，当前激活项会用 `*` 标记。

## `hermes profile use`

```bash
hermes profile use <name>
```

切换默认 profile。

## `hermes profile create`

```bash
hermes profile create <name> [options]
```

创建 profile，支持 `--clone`、`--clone-all`、`--clone-from`。

## `hermes profile delete`

```bash
hermes profile delete <name> [--yes]
```

删除 profile 及其数据目录（不可恢复）。

## `hermes profile show`

```bash
hermes profile show <name>
```

显示 profile 详情：目录、模型、网关状态、技能数量等。

## `hermes profile export`

```bash
hermes profile export <name> [--output <path>]
```

将 profile 导出为 `tar.gz` 归档。

## `hermes profile import`

```bash
hermes profile import <archive> [--name <name>]
```

从 `tar.gz` 归档导入 profile。

## `hermes profile install`

```bash
hermes profile install <git-url-or-dir>
```

从 Git URL 或本地目录安装 distribution。详见 [Profile Distributions](/user-guide/profile-distributions)。

