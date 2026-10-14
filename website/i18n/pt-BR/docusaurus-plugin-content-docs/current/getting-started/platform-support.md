---
sidebar_position: 2.5
title: "Suporte a plataformas"
description: "Quais sistemas operacionais, métodos de distribuição e recursos o Hermes Agent oferece."
---

# Suporte a plataformas

O Hermes Agent mantém suporte a várias plataformas e métodos de distribuição, mas não dá para cobrir todo método de instalação possível.

---

## Tier 1

Nos esforçamos para não quebrar instalação e updates nestes. Issues e regressões no Tier 1 são a primeira prioridade e têm precedência sobre outras plataformas.

| SO / Arquitetura                                                             | Métodos de instalação                                                                                                           | Notas                                                                                                                                                     |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **macOS** (Apple Silicon)                                                     | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) |
| [**Windows 10 / 11**](../user-guide/windows-native.md) (x86_64, aarch64)      | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.ps1`](./installation.md#windows-native)                    | Alguns recursos [não estão disponíveis](../user-guide/windows-native.md#feature-matrix).                                                                       |
| **Linux / [WSL2](../user-guide/windows-wsl-quickstart.md)** (x86_64, aarch64) | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux)                                                           | Testamos no Ubuntu mais recente e no WSL2. Se sua distro tem glibc, systemd e segue o Filesystem Hierarchy Standard, é bem provável que funcione bem. |
| [**Container Docker**](../user-guide/docker.md#quick-start) (x86_64, aarch64) | [`docker pull`](../user-guide/docker.md#quick-start)                                                                           | Instalações Docker não suportam `hermes update`. A atualização é feita rodando uma imagem nova.                                                                  |

---

## Tier 2

Essas plataformas são mantidas na árvore só no melhor esforço.
Releases podem quebrá-las, e não prometemos corrigir rápido quando quebrarem.

PRs para corrigir issues nelas serão aceitos, mas com prioridade abaixo das do Tier 1.

| SO / Arquitetura              | Métodos de instalação                                                 | Notas                                                                        |
| ------------------------------ | -------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Android (Termux)** (aarch64) | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | Alguns recursos [não estão disponíveis](./termux.md#known-limitations-on-phones). |
| **Nix** (MacOS, Linux, NixOS)  | [`install.sh`](./nix-setup.md)                                       | Quebra com frequência por causa de packaging do Node.js. Boa sorte~! &lt;3             |

## Sem suporte

Essas plataformas e métodos de distribuição **não** são suportados.
Sugerimos migrar para um método ou plataforma suportados.
Podem estar quebrados agora e podem quebrar mais no futuro.
PRs para corrigi-los _não_ serão aceitos, e qualquer código que mantenha compatibilidade com eles pode ser removido a qualquer momento.

- instalações via AUR (podemos upstreamar patches se ajudar &lt;3)
- macOS em processadores x86 (Intel)
- instalações via `pypi` (ex.: `uv tool install hermes-agent`, `pip install hermse-agent`, etc.)
- instalações via `brew` (`brew install hermes-agent`)

Se você está usando um método de distribuição sem suporte, leia o [guia de instalação](./installation.md) para saber como migrar para um suportado.
