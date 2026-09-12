---
sidebar_position: 2.5
title: "Plattformsstöd"
description: "Operativsystem, installationsmetoder och funktioner som Hermes Agent stöder."
---

# Plattformsstöd {#platform-support}

Hermes Agent stöder flera plattformar och distributionssätt, men inte alla tänkbara installationsmetoder.

## Nivå 1 {#tier-1}

Vi strävar efter att installationer och uppdateringar alltid ska fungera på dessa plattformar. Fel och regressioner på nivå 1 har högsta prioritet och åtgärdas före problem på andra plattformar.

| Operativsystem och arkitektur | Installationsmetoder | Anmärkningar |
| --- | --- | --- |
| **macOS** (Apple Silicon) | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.sh`](/getting-started/installation#linux--macos--wsl2--android-termux) | |
| [**Windows 10 / 11**](/user-guide/windows-native) (x86_64, aarch64) | [Hermes Desktop](https://hermes-agent.nousresearch.com/), [`install.ps1`](/getting-started/installation#windows-native) | Vissa funktioner är [inte tillgängliga](/user-guide/windows-native#feature-matrix). |
| **Linux / [WSL2](/user-guide/windows-wsl-quickstart)** (x86_64, aarch64) | [`install.sh`](/getting-started/installation#linux--macos--wsl2--android-termux) | Vi testar på senaste Ubuntu och WSL2. En distribution med glibc och systemd som följer Filesystem Hierarchy Standard fungerar sannolikt bra. |
| [**Docker-container**](/user-guide/docker#quick-start) (x86_64, aarch64) | [`docker pull`](/user-guide/docker#quick-start) | Docker-installationer stöder inte `hermes update`. Uppdatera genom att köra en ny avbildning. |

## Nivå 2 {#tier-2}

Dessa plattformar underhålls i projektet efter förmåga. Nya utgåvor kan göra att de slutar fungera, och vi kan inte lova snabba rättningar.

PR:er som rättar problem på dessa plattformar tas emot, men prioriteras efter rättningar för nivå 1.

| Operativsystem och arkitektur | Installationsmetoder | Anmärkningar |
| --- | --- | --- |
| **Android (Termux)** (aarch64) | [`install.sh`](/getting-started/installation#linux--macos--wsl2--android-termux) | Vissa funktioner är [inte tillgängliga](/getting-started/termux#known-limitations-on-phones). |
| **Nix** (macOS, Linux, NixOS) | [`install.sh`](/getting-started/nix-setup) | Problem med paketeringen av Node.js orsakar ofta fel. Stödet ges efter förmåga. |

## Utan stöd {#unsupported}

Följande plattformar och distributionssätt stöds **inte**. Vi rekommenderar att du byter till en plattform eller installationsmetod som stöds. De kan vara trasiga redan nu eller sluta fungera framöver. PR:er med rättningar för dem accepteras inte, och kompatibilitetskod kan tas bort när som helst.

- Installation via AUR. Vi kan dock skicka rättningar till upstream om det hjälper.
- macOS på x86-processorer från Intel.
- Installation via `pypi`, exempelvis `uv tool install hermes-agent` eller `pip install hermes-agent`.
- Installation via `brew` (`brew install hermes-agent`).

Om du använder en distributionsmetod utan stöd visar [Installationsguiden](/getting-started/installation) hur du byter till en metod som stöds.
