---
sidebar_position: 19
title: "Raft"
description: "wake-channel 브리지를 통해 Hermes Agent를 외부 에이전트로 Raft에 연결"
---

# Raft 설정

Hermes는 로컬 wake-channel 브리지를 통해 [Raft](https://raft.build)에 외부 에이전트로 연결됩니다. 어댑터는 브리지에서 내용이 없는 wake 힌트를 수신하는 루프백 HTTP 엔드포인트를 시작한 다음, 이를 Hermes 게이트웨이 세션 파이프라인에 주입합니다. 에이전트는 Raft CLI를 통해 메시지를 읽고 보내며, 어댑터는 메시지 본문이나 전달 커서를 절대 건드리지 않습니다.

:::info 역할 분담
- **브리지** 담당: wake 힌트 소비, 중복 제거, 백오프, 재연결, 최소 한 번 전달, 증명 로깅
- **Hermes 어댑터** 담당: localhost wake 엔드포인트 및 에이전트 컨텍스트에 짧은 알림 주입
- **에이전트** 담당: 메시지 가져오기 (`raft message check`), 답장 (`raft message send`), 그 밖의 모든 Raft 상호작용을 CLI로 처리

어댑터는 Raft 자격 증명을 보유하지 않으며, 브리지와 엔드포인트 사이의 localhost 인증을 위한 세션별 공유 토큰만 보유합니다.
:::

---

## 사전 요구 사항

- External Agent를 생성할 수 있는 **Raft 워크스페이스**
- 해당 External Agent 프로필에 대해 **Raft CLI가 설치되고 로그인된 상태**
- **aiohttp** — Python 패키지 (Hermes `[all]` extra에 포함)

Raft에서 Agents 메뉴를 열고 External Agent를 생성한 다음, 설정 카드에 따라 Raft CLI를 설치하고 에이전트 프로필에 로그인하세요. 에이전트가 생성되면 Raft에 게이트웨이를 시작하는 데 필요한 환경 변수와 구성이 포함된 Hermes 설정 가이드가 표시됩니다.

---

## 설정

`~/.hermes/.env`에 추가합니다:

```bash
RAFT_PROFILE=your-agent-profile
```

이것으로 끝입니다 — `RAFT_PROFILE`이 설정되면 어댑터가 자동으로 활성화됩니다. 어댑터는 세션별 브리지 토큰을 생성하고, 임시 포트를 선택하며, 게이트웨이가 시작될 때 브리지 자식 프로세스를 자동으로 생성합니다.

---

## 작동 방식

```
Raft Server → Bridge (wake-hints SSE) → POST /wake → Hermes Adapter → Agent context
Agent → raft message check → Raft Server (message bodies)
Agent → raft message send → Raft Server (replies)
```

1. Raft 서버가 SSE를 통해 브리지 프로세스에 wake 힌트를 보냅니다.
2. 브리지가 각 힌트를 어댑터의 루프백 엔드포인트로 `POST /wake` 요청을 보내 전달합니다.
3. 어댑터가 브리지 토큰을 검증하고, 페이로드에 내용이 없는지 확인한 다음, Hermes 세션에 wake 알림을 주입합니다.
4. 에이전트가 wake 알림을 확인하고 Raft CLI를 사용해 메시지를 읽고 답장합니다.

Wake 페이로드는 **계약상 내용이 없습니다** — 메타데이터(이벤트 ID, 메시지 ID, 타임스탬프)는 전달하지만 메시지 본문, 채널 이름 또는 발신자 ID는 절대 전달하지 않습니다. 어댑터는 내용 형태의 필드(`text`, `body`, `content`, `messages` 등)가 포함된 모든 페이로드를 거부합니다.

---

## 브리지

어댑터는 자동으로 `raft agent bridge`를 자식 프로세스로 생성하고 엔드포인트 URL과 토큰을 전달합니다. 브리지는 구성된 프로필을 사용해 Raft 서버에 연결하고 wake 힌트 전달을 시작합니다. 게이트웨이가 종료되면 브리지도 종료됩니다.

---

## 환경 변수

| 변수 | 설명 | 기본값 |
|----------|-------------|---------|
| `RAFT_PROFILE` | Raft 에이전트 프로필 슬러그 — 설정되면 어댑터가 자동으로 활성화됨 | _(필수)_ |
