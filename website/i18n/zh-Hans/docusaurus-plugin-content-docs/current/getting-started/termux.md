---
sidebar_position: 3
title: "Android / Termux"
description: "通过签名 APT 仓库安装适用于 aarch64 Android 的 Hermes 预发布软件包"
---

# 在 Android 上使用 Termux

Hermes 的 Termux 软件包适用于 aarch64（arm64-v8a）设备，目前处于预发布测试阶段。
它包含 Python、Node.js、npm、uv、ripgrep、FFmpeg 和核心 Python 依赖。
安装时无需在手机上编译核心依赖或组装 Python 环境。

请使用标准 Termux 应用和前缀 `/data/data/com.termux/files/usr`。
其他架构或更改应用包名的 Termux 版本不受支持。
桌面和服务器的 `install.sh` 不是此平台的安装路径。

## 安装

1. 安装仓库配置工具：

   ```bash
   pkg install curl gnupg
   ```

2. 下载公钥：

   ```bash
   mkdir -p "$PREFIX/etc/apt/keyrings"
   curl -fsSL \
     https://hermes-assets.nousresearch.com/releases/termux/canary/key.asc \
     -o "$PREFIX/etc/apt/keyrings/hermes-agent.asc"
   ```

3. 检查主密钥指纹：

   ```bash
   gpg --show-keys --with-fingerprint "$PREFIX/etc/apt/keyrings/hermes-agent.asc"
   ```

   预期指纹：

   ```text
   C572 B5FD D1A2 9CCF A9A9 12B6 840B 0848 E139 156D
   ```

   如果不一致，请停止。不要禁用签名验证。

4. 添加 canary 仓库：

   ```bash
   printf '%s\n' \
     "deb [signed-by=$PREFIX/etc/apt/keyrings/hermes-agent.asc] https://hermes-assets.nousresearch.com/releases/termux/canary hermes-canary main" \
     > "$PREFIX/etc/apt/sources.list.d/hermes-agent.list"
   ```

5. 安装并配置：

   ```bash
   pkg update
   pkg install hermes-agent
   hermes setup
   hermes --tui
   ```

## 数据、更新和卸载

软件包位于 `$PREFIX/lib/hermes-agent/`。
`$PREFIX/bin/` 中的 `hermes`、`hermes-agent` 和 `hermes-acp` 链接使用包内运行时。
它们不依赖 Termux 的 Python 或 Node.js 软件包。
数据位于 `~/.hermes/`，或所选 `HERMES_HOME`。

```bash
pkg update
pkg upgrade hermes-agent
```

`hermes update` 拒绝修改 APT 所有的代码，改由包管理器更新。
Canary 版本带有 `~canary.TIMESTAMP`，排序低于对应的正式版本。

卸载应用软件包：

```bash
pkg uninstall hermes-agent
```

APT 移除软件包和命令链接，但保留用户配置、会话、技能和记忆。

## Gateway 和限制

使用 `hermes gateway run` 在 Termux 会话中运行 gateway。
当前 APT 路径不提供 systemd、launchd 或 Windows 计划任务。
Android 可能挂起或终止后台进程；唤醒锁不保证持续运行。

该包不包含 `nemo-relay` 导出器，也不包含 Electron 桌面应用。
本地 Chromium、桌面控制、Docker daemon 和音频设备集成不能因核心 CLI 可运行就视为可用。
第三方插件仍可能需要 Android 不支持的依赖。

故障排查时运行 `hermes doctor`，并提供 `hermes --version` 和完整错误。
缺少包内核心库或 TUI 输出属于软件包问题，不应要求用户重新编译。
