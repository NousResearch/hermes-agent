---
sidebar_position: 4
title: "Buzz 통합"
description: "Hermes Agent를 Buzz에 연결하는 세 가지 방법 — Block의 Nostr 기반 인간+에이전트 워크스페이스"
---

# Buzz 통합

[Buzz](https://github.com/block/buzz)는 사람과 AI 에이전트가 같은 채널을 공유하는 Block의 오픈 소스 자체 호스팅 워크스페이스입니다. Nostr을 기반으로 하며, 모든 메시지는 사용자가 소유한 릴레이의 서명된 이벤트이고 모든 참여자(사람 또는 에이전트)는 키페어를 가집니다.

Hermes는 Buzz와 세 가지 방식으로 통합됩니다. Hermes가 어디에서 실행되고 무엇을 하길 원하는지에 따라 선택하세요.

| | ① 데스크톱 런타임 | ② 릴레이 브리지(ACP) | ③ 네이티브 게이트웨이 플랫폼 |
|---|---|---|---|
| **무엇인가요** | Buzz Desktop이 Hermes를 관리형 하네스로 로컬에서 실행 | Buzz의 `buzz-acp`가 채널을 stdio를 통해 `hermes acp`에 연결 | Hermes의 게이트웨이가 Buzz에 일급 메시징 플랫폼으로 참여 |
| **Hermes 실행 위치** | Buzz가 실행하는 데스크톱 | `buzz-acp`가 실행하는 서버 | Telegram/Discord 등과 함께 자체 게이트웨이에서 실행 |
| **가장 적합한 경우** | 설정 없이 Buzz Desktop 안에서 Hermes 사용 | Buzz가 전송을 관리하는 호스팅 에이전트 ID | 메모리, 스킬, 승인, cron, 세션을 포함한 전체 Hermes |
| **인바운드** | ACP stdio | ACP stdio(릴레이 WebSocket 경유) | NIP-42 인증 Nostr WebSocket(폴링 폴백) |
| **설정** | 자동 검색 | `buzz-acp` 환경 변수 | `hermes gateway setup` → Buzz |

## ① Buzz Desktop 관리형 런타임

Buzz Desktop은 Hermes를 프리셋 런타임으로 제공합니다. Hermes를 일반적인 방법으로 설치한 상태에서 **Settings → Runtimes**를 열면 Hermes가 자동으로 표시됩니다. 검색 기능이 로그인 셸의 PATH에서 `hermes-acp` 실행기를 확인하며, 설치 프로그램은 이를 `~/.local/bin`에 기록합니다(`hermes update`는 이전 설치에서 이를 자동으로 복구합니다).

전체 설정, 문제 해결, 보안 관련 내용(Buzz는 도구 권한을 자동 승인하므로 에이전트를 소유자 전용으로 유지): **[ACP 호스트 통합 → Buzz Desktop](/user-guide/features/acp#buzz-desktop)**

## ② 릴레이 브리지(buzz-acp + ACP)

Buzz 자체 하네스가 전송을 관리하는 동안 Buzz *채널*에 참여하는 호스팅 Hermes ID를 사용하려면 다음과 같이 구성합니다.

```text
Buzz relay <-- WebSocket --> buzz-acp <-- ACP over stdio --> Hermes Agent
```

실행된 Hermes는 해당 호스트에서 실행하는 `hermes`와 동일한 구성, 자격 증명, 메모리, 스킬을 사용합니다. 키 발급, 채널 검색, 소유자 전용 텔레메트리(`BUZZ_ACP_RELAY_OBSERVER`), 헤드리스 권한 안내: **[ACP 호스트 통합 → Buzz 채널(릴레이 브리지)](/user-guide/features/acp#buzz-channels-relay-bridge)**

## ③ 네이티브 게이트웨이 플랫폼(전체 Hermes에 권장)

번들로 제공되는 `buzz` 플랫폼 플러그인은 Buzz를 일반적인 Hermes 메시징 플랫폼으로 만듭니다. 채널, DM, 멘션 게이팅, 스레드 답장, 리액션, 이미지, cron 전달(`deliver=buzz`)을 지원하며 Hermes 자체의 승인, 메모리, 세션 관리도 그대로 유지됩니다. 인바운드는 지속적인 NIP-42 인증 Nostr WebSocket(의존성 없는 BIP-340 서명)을 통해 들어오며 CLI 폴링으로 자동 폴백됩니다. 아웃바운드는 `buzz` CLI를 통해 전송됩니다.

```bash
hermes gateway setup   # pick Buzz
```

전체 구성 참고 자료(환경 변수, config.yaml, 전송 모드, 접근 제어): **[메시징 → Buzz](/user-guide/messaging/buzz)**

## 어떤 방식을 사용해야 하나요?

- **그냥 둘러보는 Buzz Desktop 사용자** → ①은 별도 설정 없이 작동합니다.
- **커뮤니티 릴레이를 운영하며 Buzz가 관리하는 에이전트 ID를 원하는 경우** → ②.
- **이미 에이전트로 Hermes를 실행 중이며 Buzz를 또 하나의 채널로 추가하려는 경우** → ③. 가장 깊이 통합되며 Hermes의 모든 기능을 유지합니다.

①/②와 ③은 서로 다른 ID와 전송 방식을 사용합니다. ③은 전용 Nostr 키페어로 실행하세요. 어댑터는 릴레이+공개 키 쌍에 범위가 지정된 잠금을 설정하므로 두 Hermes 프로필이 실수로 하나의 Buzz ID를 조작할 수 없습니다.

## 크레딧

Buzz 통합은 커뮤니티와 함께 구축되었습니다. @SHL0MS(PATH 실행기 + Desktop 보안 감사), @NYTEMODEONLY(릴레이 브리지 문서), @rob-coco(플랫폼 어댑터), @ScaleLeanChris(Nostr WebSocket 전송 + NIP-42/BIP-340 서명), @jethac(멀티 에이전트 검증)에게 감사드립니다.
