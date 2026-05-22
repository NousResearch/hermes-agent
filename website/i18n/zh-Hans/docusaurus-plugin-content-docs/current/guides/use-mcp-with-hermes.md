---
title: "在 Hermes 中使用 MCP"
sidebar_label: "使用 MCP"
---

# 在 Hermes 中使用 MCP

本页用于说明如何把 Model Context Protocol 服务器接入 Hermes Agent。

要点：

- 准备一个可用的 MCP 服务器，确保它能在本地或远程正常启动。
- 在 Hermes 的配置或工具集里启用对应的 MCP 集成。
- 确认权限、网络与凭证设置正确，避免工具调用失败。

如果你需要，我可以继续把这页补成更完整的操作指南和示例命令。

## WSL2 桥接：在 WSL 里运行 Hermes，控制 Windows Chrome {#wsl2-bridge-hermes-in-wsl-to-windows-chrome}

当你在 WSL2 中运行 Hermes、但希望控制 Windows 侧已登录的 Chrome 时，优先选择 MCP 方案（而不是直接使用浏览器连接命令）。这样更稳定，也更容易复用本机浏览器状态。
