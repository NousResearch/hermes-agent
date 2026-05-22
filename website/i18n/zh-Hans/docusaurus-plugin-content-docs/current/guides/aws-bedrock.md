---
sidebar_position: 14
title: "AWS Bedrock"
description: "在 Hermes 中使用 Amazon Bedrock（Converse API、IAM 鉴权、跨区域推理）。"
---

# AWS Bedrock

Hermes 原生支持 Amazon Bedrock（Converse API），可直接使用 IAM 凭证和 Bedrock 模型目录。

## 前置条件

- 安装依赖: `pip install hermes-agent[bedrock]`
- 可用 AWS 凭证（实例角色、`AWS_PROFILE`、环境变量等）
- 具备 Bedrock 推理权限

## 快速配置

```bash
hermes model
# 选择 AWS Bedrock
```

也可在配置中指定:

```yaml
model:
  provider: bedrock
  default: us.anthropic.claude-sonnet-4-6

bedrock:
  region: us-east-2
```

## 常见问题

- 报无凭证: 检查 `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`、`AWS_PROFILE` 或实例角色
- 模型不可调用: 优先使用 inference profile（通常含 `us.` 或 `global.` 前缀）
- 频率限制: 触发 `ThrottlingException` 时稍后重试或申请配额

## 相关文档

- [AI 提供商](/integrations/providers)
- [配置](/user-guide/configuration)
