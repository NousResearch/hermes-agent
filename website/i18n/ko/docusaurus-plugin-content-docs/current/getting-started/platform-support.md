---
sidebar_position: 2.5
title: "플랫폼 지원"
description: "Hermes Agent가 지원하는 운영 체제, 배포 방법 및 기능입니다."
---

# 플랫폼 지원

Hermes Agent는 다양한 플랫폼과 배포 방법을 지원하지만, 가능한 모든 설치 방법을 지원할 수는 없습니다.

---

## Tier 1

이 플랫폼의 설치와 업데이트가 중단되지 않도록 최우선으로 관리합니다. Tier 1의 문제와 회귀는 첫 번째 우선순위이며 다른 플랫폼보다 먼저 처리합니다.

| OS / 아키텍처 | 설치 방법 | 참고 |
| --- | --- | --- |
| **macOS** (Apple Silicon) | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | |
| [**Windows 10 / 11**](../user-guide/windows-native.md) (x86_64, aarch64) | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.ps1`](./installation.md#windows-native) | 일부 기능은 [사용할 수 없습니다](../user-guide/windows-native.md#feature-matrix). |
| **Linux / [WSL2](../user-guide/windows-wsl-quickstart.md)** (x86_64, aarch64) | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | 최신 Ubuntu와 WSL2에서 테스트합니다. 배포판이 glibc, systemd를 사용하고 Filesystem Hierarchy Standard를 따른다면 대체로 잘 작동합니다. |
| [**Docker Container**](../user-guide/docker.md#quick-start) (x86_64, aarch64) | [`docker pull`](../user-guide/docker.md#quick-start) | Docker 설치에서는 `hermes update`를 지원하지 않습니다. 새 이미지를 실행해 업데이트합니다. |

---

## Tier 2

이 플랫폼은 저장소에 포함된 상태로 최선의 노력에 한해 유지 관리됩니다. 릴리스로 인해 작동이 중단될 수 있으며, 중단되었을 때 신속히 수정한다고 약속할 수 없습니다.

문제를 수정하는 PR은 받지만 Tier 1 플랫폼 문제를 수정하는 작업보다 우선순위가 낮습니다.

| OS / 아키텍처 | 설치 방법 | 참고 |
| --- | --- | --- |
| **Android (Termux)** (aarch64) | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | 일부 기능은 [사용할 수 없습니다](./termux.md#known-limitations-on-phones). |
| **Nix** (MacOS, Linux, NixOS) | [`install.sh`](./nix-setup.md) | node.js 패키징 문제로 자주 중단됩니다. 행운을 빕니다~! &lt;3 |

## 지원하지 않음

이 플랫폼과 배포 방법은 **지원하지 않습니다**. 지원되는 배포 방법이나 플랫폼으로 이전하는 것을 권장합니다. 지금 작동하지 않을 수도 있고, 앞으로 더 자주 중단될 수도 있습니다.

이를 수정하는 PR은 _받지 않으며_, 호환성을 유지하는 코드는 언제든 제거될 수 있습니다.

- AUR을 통한 설치(도움이 된다면 패치를 upstream할 수 있습니다 &lt;3)
- x86(Intel) 프로세서의 macOS
- `pypi`를 통한 설치(예: `uv tool install hermes-agent`, `pip install hermes-agent` 등)
- `brew`를 통한 설치(`brew install hermes-agent`)

지원되지 않는 배포 방법을 사용 중이라면 [설치 가이드](./installation.md)를 읽고 지원되는 방법으로 전환하세요.
