---
sidebar_position: 2.5
title: "Suporte de Plataforma"
description: "Quais sistemas operacionais, métodos de distribuição e recursos o Hermes Agent suporta."
---

# Suporte de Plataforma

O Hermes Agent mantém suporte para muitas plataformas e métodos de distribuição, mas não podemos suportar todos os métodos de instalação possíveis.

---

## Nível 1

Buscamos nunca quebrar as instalações e atualizações destas plataformas. Problemas e regressões no Nível 1 são nossa primeira prioridade e têm precedência sobre outras plataformas.

| SO / Arquitetura                                                             | Métodos de instalação                                                                                                           | Notas                                                                                                                                                     |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **macOS** (Apple Silicon)                                                     | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | |
| [**Windows 10 / 11**](../user-guide/windows-native.md) (x86_64, aarch64)      | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.ps1`](./installation.md#windows-native)                    | Alguns recursos [não estão disponíveis](../user-guide/windows-native.md#feature-matrix).                                                                  |
| **Linux / [WSL2](../user-guide/windows-wsl-quickstart.md)** (x86_64, aarch64) | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux)                                                           | Testamos no Ubuntu e WSL2 mais recentes. Se sua distro tem glibc, systemd e segue o Filesystem Hierarchy Standard, é provável que funcione bem.             |
| [**Container Docker**](../user-guide/docker.md#quick-start) (x86_64, aarch64) | [`docker pull`](../user-guide/docker.md#quick-start)                                                                           | Instalações via Docker não suportam `hermes update`. A atualização é feita rodando uma nova imagem.                                                        |

---

## Nível 2

Estas plataformas são mantidas no repositório apenas por melhor esforço.
Lançamentos podem quebrá-las, e não podemos prometer que as corrigiremos prontamente quando isso acontecer.

PRs serão aceitos para corrigir problemas nelas, mas terão precedência abaixo da correção de problemas nas plataformas do Nível 1.

| SO / Arquitetura              | Métodos de instalação                                                 | Notas                                                                        |
| ----------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Android (Termux)** (aarch64) | [`install.sh`](./installation.md#linux--macos--wsl2--android-termux) | Alguns recursos [não estão disponíveis](./termux.md#known-limitations-on-phones). |
| **Nix** (MacOS, Linux, NixOS) | [`install.sh`](./nix-setup.md)                                       | Quebra frequentemente por problemas de empacotamento do node.js. Boa sorte~! &lt;3       |

## Sem suporte

Estas plataformas e métodos de distribuição **não** são suportados.
Sugerimos que você migre para um método ou plataforma de distribuição suportada.
Elas podem estar quebradas agora, e podem quebrar mais no futuro.
PRs para corrigi-las **não** serão aceitos, e qualquer código que mantenha compatibilidade com elas pode ser removido a qualquer momento.

- instalações via AUR (podemos aceitar patches upstream se ajudar &lt;3)
- macOS em processadores x86 (Intel)
- instalações via `pypi` (por exemplo, `uv tool install hermes-agent`, `pip install hermes-agent`, etc.)
- instalações via `brew` (`brew install hermes-agent`)

Se você estiver usando um método de distribuição sem suporte, leia o [guia de instalação](./installation.md) para aprender como mudar para um suportado.
