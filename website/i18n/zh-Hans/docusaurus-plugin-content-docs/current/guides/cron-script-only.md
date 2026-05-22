---
sidebar_position: 13
title: "仅脚本 Cron 任务（不使用 LLM）"
description: "让定时任务只运行脚本并投递输出，不触发模型调用。适合告警、巡检、心跳与简单报表。"
---

# 仅脚本 Cron 任务（不使用 LLM）

当你已经明确知道要发送什么消息时，不需要让 agent 推理。你只需要定时运行脚本，并把脚本输出投递到 Telegram、Discord、Slack 或本地文件。

这种模式在 Hermes 中叫做 no-agent（`--no-agent` 或 `no_agent=True`）。

## 典型场景

- 内存、磁盘、GPU 告警
- CI 构建状态推送
- 固定格式日报
- 外部 API 轮询与状态变更通知
- 主机在线心跳

## 通过 CLI 创建

```bash
# 1) 写一个脚本（放在 ~/.hermes/scripts/）
cat > ~/.hermes/scripts/memory-watchdog.sh <<'EOF'
#!/usr/bin/env bash
RAM_PCT=$(free | awk '/^Mem:/ {printf "%d", $3 * 100 / $2}')
if [ "$RAM_PCT" -ge 85 ]; then
  echo "RAM ${RAM_PCT}% on $(hostname)"
fi
EOF
chmod +x ~/.hermes/scripts/memory-watchdog.sh

# 2) 创建仅脚本任务
hermes cron create "every 5m" \
  --name "memory-watchdog" \
  --no-agent \
  --script memory-watchdog.sh \
  --deliver telegram

# 3) 验证
hermes cron list
```

## 输出与投递规则

- 脚本退出码为 0 且 stdout 非空: 按原文投递
- 脚本退出码为 0 且 stdout 为空: 静默，不投递
- 脚本非 0 退出: 发送错误告警
- 脚本超时: 发送错误告警

## 何时不要用 no-agent

如果你需要总结、筛选、归纳、改写，请使用普通 LLM Cron 任务。参考 [自动化 Cron 指南](/guides/automate-with-cron)。

## 相关文档

- [定时任务（Cron）](/user-guide/features/cron)
- [自动化 Cron 指南](/guides/automate-with-cron)
- [Webhook 订阅](/user-guide/messaging/webhooks)
