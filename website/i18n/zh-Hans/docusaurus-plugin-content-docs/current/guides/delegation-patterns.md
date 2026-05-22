---
sidebar_position: 13
title: "委派与并行工作"
description: "何时用 subagent 委派，如何组织并行研究、代码审查和多文件任务。"
---

# 委派与并行工作

Hermes 可以通过 `delegate_task` 派生子 agent。每个子 agent 有独立会话、独立终端和独立工具集，只返回最终摘要。

完整能力请见 [子智能体委派](/user-guide/features/delegation)。

## 何时适合委派

- 推理密集任务（调研、审查、复杂定位）
- 需要上下文隔离，避免污染主会话
- 多个互不依赖的任务可并行

不适合委派:

- 单一步工具调用
- 需要用户交互（`clarify`）
- 必须跨回合长期运行的任务（应改用 Cron 或后台终端）

## 常用模式

1) 并行调研

```text
并行调研三个主题，每个主题输出 5 条结论和来源链接，最后合并成摘要。
```

2) 独立代码审查

```text
对 src/auth 进行安全审查，关注 SQL 注入、JWT 校验和密码处理。修复问题并运行测试。
```

3) 大型改造分治

将不同目录拆给多个子 agent，避免多人同时改同一文件。

## 实践建议

- 在 `context` 中提供明确文件路径、命令和约束
- 根据任务限制 toolsets，减少副作用
- 子 agent 的“已修复”要二次验证（跑测试/看 diff）

## 相关文档

- [子智能体委派](/user-guide/features/delegation)
- [代码执行](/user-guide/features/code-execution)
- [定时任务（Cron）](/user-guide/features/cron)
