# Secrets

Hermes 可以在进程启动时从外部密钥管理器拉取 API 密钥，而不是将其存储在 `~/.hermes/.env` 中。密钥管理器的引导令牌存放在 `.env` 中；其他所有提供商密钥（OpenAI、Anthropic、OpenRouter 等）可以保留在管理器中并集中轮换。

支持的后端：

- [Bitwarden Secrets Manager](./bitwarden) — 使用 `bws` CLI，懒加载安装，免费套餐可用。

更多后端（Vault、AWS Secrets Manager、1Password CLI）可以轻松接入同一接口——只需在 `agent/secret_sources/` 中添加一个模块并实现一个 CLI 处理器。如有特定需求，欢迎提交请求。