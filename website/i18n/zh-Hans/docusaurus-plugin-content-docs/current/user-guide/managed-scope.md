---
sidebar_position: 3
title: "托管作用域"
description: "管理员固定的、用户不可修改的配置和密钥，通过系统级托管目录实现"
---

# 托管作用域（Managed Scope）

**托管作用域**允许管理员推送一份配置和密钥的基线，标准（非 root）用户**无法覆盖**。它面向需要统一管控的团队/组织部署，例如，IT 部门需要在一台机器的所有用户上固定模型 Provider、共享 API 基础 URL 或 `security.redact_secrets: true`。

当存在托管作用域时，它指定的值优先于用户的 `~/.hermes/config.yaml`、`~/.hermes/.env`，甚至 shell 环境 —— 仅限于它固定的具体键。其他一切仍然完全由用户控制。

:::note 与包管理器锁定安装的区别
包管理器管理的安装（声明式分发/公式）阻止*所有*配置修改，并告诉你使用包管理器。托管作用域是一个独立机制：它按键注入*特定的不可变值*，而不是锁定整个配置。两者独立，可以共存。
:::

## 存放位置

托管作用域从系统级目录读取，默认为 `/etc/hermes`：

```text
/etc/hermes/
├── config.yaml     # managed config layer (wins over ~/.hermes/config.yaml)
└── .env            # managed env layer (wins over ~/.hermes/.env + shell)
```

目录和文件由 `root` 拥有（目录权限 `0755`，文件 `0644`）：所有人可读，仅管理员可写。**该文件系统权限就是执行机制** —— 标准用户可以读取托管文件但无法编辑。

两个文件都是可选的。缺失的托管目录或文件仅表示"没有托管作用域"，配置解析与没有该功能时完全一致。

### 重新定位目录

位置可通过 `HERMES_MANAGED_DIR` 环境变量重新定位（用于容器或非 `/etc` 部署）。这是一个部署/引导路径控制项 —— 与 `HERMES_HOME` 类似 —— 由拥有托管文件的同一管理员设置。它**绝不会**被 Hermes 持久化到任何 `.env` 中。

```bash
# 将托管作用域指向自定义目录（由 IT / 部署设置，非用户设置）
export HERMES_MANAGED_DIR=/opt/org/hermes-policy
```

:::warning
能够设置 `HERMES_MANAGED_DIR` 的用户可以将托管作用域重新指向他们控制的目录，从而规避它。在实际部署中，该变量应由管理员固定（例如烘焙到服务单元/容器镜像中），而非留给用户设置。`hermes doctor` 报告*已解析的*托管目录，因此重定向是可见的。
:::

## 优先级

对于托管层指定的键，优先级顺序为（越高越优先）：

| 层级 | config.yaml | .env |
|---|---|---|
| 1 | `/etc/hermes/config.yaml`（托管） | `/etc/hermes/.env`（托管） |
| 2 | `~/.hermes/config.yaml`（用户） | `~/.hermes/.env`（用户） |
| 3 | 内置默认值 | 已有的 shell 环境 |

合并是**叶节点级别**的：固定 `model.default` 不会冻结其余的 `model.*`。一个托管的 `config.yaml`：

```yaml
model:
  default: org/standard-model
```

会为每个用户强制 `model.default`，同时将 `model.fallback`（及每个其他键）保持在用户控制下。

:::note 优先级说明
对于它固定的键，托管作用域刻意优先于 shell 环境 —— 否则它就不是"托管"的。这是唯一一处反转通常"环境变量覆盖 config.yaml"规则的地方，仅适用于托管层指定的特定键。
:::

## 查看托管内容

```bash
hermes config        # shows a header naming the managed source + the pinned keys
hermes doctor        # reports the resolved managed dir + pinned key counts
```

如果你尝试更改一个托管值，Hermes 会拒绝并指明来源：

```bash
$ hermes config set model.default my/model
Cannot set 'model.default': it is managed by your administrator
(/etc/hermes/config.yaml) and cannot be changed.
```

同样的规则适用于托管密钥 —— `hermes config set` / 安装不会为被托管 `.env` 固定的环境键写入用户值。

## 设置托管作用域（管理员操作）

```bash
sudo mkdir -p /etc/hermes

# 固定一些配置值给这台机器上的每个用户
sudo tee /etc/hermes/config.yaml >/dev/null <<'YAML'
model:
  provider: nous
security:
  redact_secrets: true
YAML

# 可选：固定一个共享的、非敏感的环境值
sudo tee /etc/hermes/.env >/dev/null <<'ENV'
OPENAI_API_BASE=https://inference.example.com/v1
ENV

sudo chmod 0755 /etc/hermes
sudo chmod 0644 /etc/hermes/config.yaml /etc/hermes/.env
```

变更在下次 Hermes 启动时生效（格式错误的托管文件会被记录警告并忽略 —— 它绝不会阻止启动，但管理员应检查 `hermes doctor` 以确认策略正在生效）。

## 安全模型和限制（v1）

- **执行仅靠文件系统权限。** 如果用户对托管目录有写权限（或以 `root` 运行 Hermes），托管作用域只是建议性的。
- **托管 `.env` 是全局可读的**（`0644`），因此任何本地用户都能读取通过它推送的密钥。用于共享的、非敏感的值（组织 API 基础 URL、功能默认值），而非高敏感密钥。
- **Agent 自身的工具不会被托管 *环境* 值硬性阻止。** 托管环境变量在启动时应用，但无法阻止 Agent 在自己的子进程 shell 中设置不同的值。v1 是面向标准用户的管理便利边界，而非不可逃脱的沙箱。

以下内容在 v1 中刻意**不在范围内**，可能在后续版本实现：

- Agent 自身无法逃脱的硬边界。
- macOS 和 Windows 上的原生托管位置（v1 以 Linux/POSIX 优先）。
- 用于分层策略的片段目录（`managed.d/`）。
- 签名/完整性检查的托管文件。
- 远程/设备管理（MDM）分发。
- 更严格的（组级）托管密钥权限。
