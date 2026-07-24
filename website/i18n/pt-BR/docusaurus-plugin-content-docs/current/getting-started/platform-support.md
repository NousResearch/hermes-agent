---
sidebar_position: 2.5
title: "Suporte a Plataformas"
description: "Quais sistemas operacionais, métodos de distribuição e recursos o Hermes Agent suporta."
---

# Suporte a Plataformas

O Hermes Agent mantém suporte para muitas plataformas e métodos de distribuição, mas não podemos suportar todos os métodos de instalação possíveis.

---

## Tier 1

Nós nos esforçamos para nunca quebrar instalações e atualizações para estas plataformas. Problemas e regressões no Tier 1 são nossa primeira prioridade e têm precedência sobre outras plataformas.

| SO / Arquitetura                                                                       | Métodos de instalação                                                                                           | Observações                                                                                                                                                              |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **macOS** (Apple Silicon)                                                              | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) |                                                                                                                                                                          |
| [**Windows 10 / 11**](../user-guide/windows-native.md) (x86_64, aarch64)               | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.ps1`](./installation.md#windows-native)     | Alguns recursos [não estão disponíveis](../user-guide/windows-native.md#feature-matrix).                                                                                 |
| **Linux / [WSL2](../user-guide/windows-wsl-quickstart.md)** (x86_64, aarch64)          | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux)                                            | Testamos nas versões mais recentes do Ubuntu e WSL2. Se sua distribuição tem glibc, systemd e segue o Filesystem Hierarchy Standard, provavelmente funcionará bem.        |
| [**Container Docker**](../user-guide/docker.md#quick-start) (x86_64, aarch64)          | [`docker pull`](../user-guide/docker.md#quick-start)                                                            | Instalações Docker não suportam `hermes update`. A atualização é feita executando uma nova imagem.                                                                       |

---

## Tier 2

Estas plataformas são mantidas no repositório apenas como melhor esforço.
Lançamentos podem quebrá-las, e não podemos prometer que as corrigiremos prontamente quando isso acontecer.

PRs serão aceitos para corrigir problemas com elas, mas terão precedência menor do que a correção de problemas com plataformas Tier 1.

| SO / Arquitetura                | Métodos de instalação                                       | Observações                                                                             |
| ------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Android (Termux)** (aarch64)  | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | Alguns recursos [não estão disponíveis](./termux.md#known-limitations-on-phones).       |
| **Nix** (MacOS, Linux, NixOS)   | [`install.sh`](./nix-setup.md)                              | Quebra com frequência devido a problemas de empacotamento do Node.js. Boa sorte~! &lt;3 |

## Não Suportado

Estas plataformas e métodos de distribuição **não** são suportados.
Sugerimos que você migre para um método de distribuição ou plataforma suportada.
Eles podem estar quebrados agora, podem quebrar mais no futuro.
PRs para corrigi-los **não** serão aceitos, e qualquer código que mantenha compatibilidade com eles pode ser removido a qualquer momento.

- Instalações via AUR (podemos upstreamar patches se ajudar &lt;3)
- macOS em processadores x86 (Intel)
- Instalações via `pypi` (ex.: `uv tool install hermes-agent`, `pip install hermes-agent`, etc.)
- Instalações via `brew` (`brew install hermes-agent`)

Se você está usando um método de distribuição não suportado, leia o [guia de instalação](./installation.md) para aprender como migrar para um suportado.
