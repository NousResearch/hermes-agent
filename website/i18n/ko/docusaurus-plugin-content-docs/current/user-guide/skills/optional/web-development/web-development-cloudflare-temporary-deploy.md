---
title: "Cloudflare 임시 배포 — wrangler --temporary로 계정 없이 Worker를 배포하고 공개하기"
sidebar_label: "Cloudflare 임시 배포"
description: "wrangler --temporary로 계정 없이 Worker를 배포하고 공개하기"
---

{/* 이 페이지는 스킬의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Cloudflare 임시 배포

wrangler --temporary를 사용해 계정 없이 Worker를 공개 배포합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/web-development/cloudflare-temporary-deploy`로 설치 |
| 경로 | `optional-skills/web-development/cloudflare-temporary-deploy` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `cloudflare`, `workers`, `wrangler`, `deploy`, `temporary`, `agent`, `serverless`, `web-development` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보게 되는 내용입니다.
:::

# Cloudflare 임시 배포 스킬

`wrangler deploy --temporary`를 사용해 계정 설정 없이 Cloudflare Worker를 공개 `workers.dev` URL로 배포합니다. Cloudflare는 일회용 계정을 생성하고 배포한 뒤 60분 동안 유효한 클레임 URL을 출력하며, 클레임되지 않은 계정은 자동으로 삭제됩니다. 이를 통해 에이전트는 OAuth, 가입, 토큰 복사 및 붙여넣기 없이도 작성 → 배포 → 검증의 짧은 루프를 실행할 수 있습니다.

이 스킬은 프로덕션 배포(`wrangler login` + 영구 계정 사용)나 아래의 임시 계정 제한을 벗어나는 Worker 이외의 Cloudflare 제품을 다루지 않습니다.

## 사용 시점

다음과 같은 경우 이 스킬을 로드하세요:

- **에이전트가 작성한 코드를 먼저 Cloudflare 계정을 만들지 않고 공개 URL로 배포**하려는 경우 — "이걸 배포하고 링크를 줘"
- **브라우저 OAuth 단계가 중단 요인이 되는 백그라운드/자율 세션**에서 반복 작업을 수행하려는 경우
- **일회용으로 클레임할 수 있는 대상을 사용해 Workers를 빠르게 프로토타이핑하거나 평가**하려는 경우
- **자체 검증 배포 루프**를 구축하려는 경우 — 배포하고, 공개 URL에 `curl`을 실행하고, 출력이 코드와 일치하는지 확인한 뒤 재배포

## 사용하지 않을 시점

- **프로덕션 또는 CI/CD** → 영구 계정(`wrangler login` 또는 `CLOUDFLARE_API_TOKEN`)을 사용하세요. 자격 증명이 있으면 `--temporary`는 오류를 반환합니다.
- **Wrangler가 이미 인증된 경우** → `--temporary`는 의도적으로 오류를 반환합니다. 사용자가 일회용 배포를 명시적으로 원하는 경우에만 먼저 `wrangler logout`을 실행하세요.
- **장기 호스팅** → 임시 배포는 클레임하지 않으면 60분 후 삭제됩니다.

## 사전 요구 사항

- **Wrangler 4.102.0 이상.** `--temporary`가 도입된 버전입니다. 이전 버전에는 이 옵션이 없습니다. `npx wrangler@latest --version`으로 확인하세요.
- **Node 18+ / npm**(또는 `npx`, `yarn`, `pnpm`). 전역 설치는 필요하지 않습니다 — `npx wrangler@latest`가 동작합니다.
- **Cloudflare 자격 증명이 없어야 합니다.** `--temporary`는 Wrangler가 인증되지 않은 경우에만 동작합니다. OAuth 로그인, `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_API_KEY` 환경 변수, `~/.wrangler` / `~/.config/.wrangler`에 캐시된 OAuth가 없어야 합니다. `terminal` 도구의 환경을 그대로 사용하고 해당 변수를 설정하지 마세요.
- `cloudflare.com` 및 `workers.dev`로 네트워크 이그레스가 가능해야 합니다.
- `--temporary`를 사용하면 Cloudflare의 서비스 약관 및 개인정보 처리방침에 동의하게 됩니다.

## 실행 방법

모든 단계에 `terminal` 도구를 사용하세요. 이전 버전의 전역 wrangler를 실수로 실행하지 않도록 항상 버전을 고정하세요(`wrangler@latest` 또는 `wrangler@4.102.0` 이상).

1. **최소 Worker 스캐폴딩** (프로젝트가 이미 있으면 건너뜁니다). Worker에는 `wrangler.toml`(또는 `wrangler.jsonc`)과 진입 스크립트가 필요합니다. 최소 TypeScript 예시 — `write_file`로 다음 파일을 작성하세요:

   `wrangler.jsonc`:
   ```jsonc
   {
     "name": "hello-agent",
     "main": "src/index.ts",
     "compatibility_date": "2025-01-01"
   }
   ```

   `src/index.ts`:
   ```typescript
   export default {
     async fetch(): Promise<Response> {
       return new Response("hello cloudflare");
     },
   };
   ```

2. **`--temporary`로 배포**: 프로젝트 디렉터리에서 실행하세요:
   ```
   npx wrangler@latest deploy --temporary
   ```
   작업 증명 확인으로 인해 짧은 자동 지연이 발생합니다. 성공하면 Wrangler는 `Account: <name> (created)`(또는 `(reused)`) 줄, `Claim URL`, 그리고 실제 `https://<worker>.<account>.workers.dev` URL을 출력합니다.

3. **출력에서 URL 파싱**: 눈으로 확인하는 대신 다음 헬퍼를 실행해 안정적으로 추출하세요:
   ```
   npx wrangler@latest deploy --temporary 2>&1 | python3 scripts/parse_deploy_output.py
   ```
   (이 스킬의 절대 경로로 `scripts/parse_deploy_output.py`를 확인하세요.) JSON이 출력됩니다: `{"live_url", "claim_url", "account", "account_state", "expires_minutes", "deployed"}`.

4. **배포가 실제로 공개 상태인지 검증** — 배포 로그만 신뢰하지 마세요. 공개 URL에 `curl`을 실행하고 본문이 코드의 반환값과 일치하는지 확인하세요:
   ```
   curl -sS <live_url>
   ```

5. **반복**. 코드를 편집하고 동일한 `npx wrangler@latest deploy --temporary`로 재배포하세요. 60분 이내에는 Wrangler가 캐시된 임시 계정을 재사용하므로(`Account: <name> (reused)`) URL이 유지됩니다. 다시 `curl`을 실행해 변경 사항을 확인하세요.

6. **클레임 URL을 사용자에게 전달**. 60분 이내에 URL을 열어 배포와 리소스를 유지해야 하며, 클레임하지 않으면 모두 자동 삭제된다고 안내하세요. 클레임 URL은 계정 소유권을 부여하므로 비밀로 취급하세요.

## 빠른 참조

| 단계 | 명령 |
|---|---|
| 버전 확인(4.102.0+ 필요) | `npx wrangler@latest --version` |
| 배포(계정 없음) | `npx wrangler@latest deploy --temporary` |
| 배포 + URL 파싱 | `npx wrangler@latest deploy --temporary 2>&1 \| python3 scripts/parse_deploy_output.py` |
| 라이브 상태 확인 | `curl -sS <live_url>` |
| 캐시된 임시 계정 초기화 | `npx wrangler@latest logout` |

### 임시 계정 제품 제한

| 제품 | 임시 계정의 제한 |
|---|---|
| Workers | `workers.dev`로 배포 |
| Static Assets | 최대 1,000개 파일, 파일당 5 MiB |
| KV | 허용 |
| D1 | 데이터베이스 1개, DB당 100 MB / 전체 100 MB |
| Durable Objects | 허용 |
| Hyperdrive | 구성 2개, 연결 10개 |
| Queues | 최대 10개 |
| SSL/TLS certs | 허용 |

## 주의 사항

- **`--temporary`는 `wrangler deploy --help`에 표시되지 않으며 전역 플래그도 아닙니다.** 의도적으로 숨겨져 동적으로 표시됩니다. 인증되지 않은 `wrangler deploy`가 실패하면 Wrangler가 "rerun with `--temporary`"를 출력합니다. `--help`에 옵션이 없다고 해서 누락된 것으로 판단하지 마세요 — 대신 버전을 확인하세요.
- **오래된 전역 wrangler.** 오래된 전역 설치 `wrangler`(`< 4.102.0`)에는 이 플래그가 없습니다. 버전을 직접 제어할 수 있도록 항상 `npx wrangler@latest`(또는 고정된 `>=4.102.0`)를 호출하세요.
- **인증 정보가 있으면 하드 오류.** `wrangler login`을 실행한 적이 있거나 `CLOUDFLARE_API_TOKEN`/`CLOUDFLARE_API_KEY`가 설정되어 있으면 `--temporary`가 오류를 반환합니다. 이 셸에서 변수를 해제하거나 `wrangler logout`을 실행하세요. 사용자 실계정 자격 증명을 알리지 않고 제거하지 마세요.
- **속도 제한.** 임시 계정을 너무 빠르게 생성하면 실패합니다. 60분 내에는 새 계정을 강제로 만들지 말고 캐시된 계정을 재사용해 재배포하세요. 속도 제한에 걸리면 기다리거나 영구 계정을 사용하세요.
- **60분의 하드 만료이며 연장할 수 없습니다.** 배포가 한 시간 이상 유지되어야 한다면 사용자가 클레임해야 합니다. 이 점을 명확히 알려 주세요.
- **재배포 직후 `curl`이 잠시 이전 본문을 제공할 수 있습니다.** `workers.dev`에는 짧은 엣지 캐시가 있으며, `(reused)` 줄과 새 `Current Version ID`는 `curl`이 몇 초 동안 오래된 콘텐츠를 보여도 배포가 성공했음을 확인해 줍니다. 다시 `curl`을 실행하거나 캐시 무효화 쿼리 문자열을 추가한 뒤 재배포 실패 여부를 판단하세요.
- **클레임 URL을 "그냥 링크"로 공유 트랜스크립트에 기록하지 마세요.** 이는 자격 증명과 동일한 수준의 비밀입니다.

## 검증

- `npx wrangler@latest --version`이 `>= 4.102.0`을 반환합니다.
- `npx wrangler@latest deploy --temporary`가 `workers.dev` 라이브 URL과 `claim-preview?claimToken=` 클레임 URL을 출력합니다.
- `curl -sS <live_url>`이 Worker 코드가 생성하는 정확한 본문을 반환합니다.
- 두 번째 배포에서 `Account: <name> (reused)`가 보고되고 라이브 URL이 변경되지 않습니다.
- 파서 스크립트 자체 테스트가 통과합니다: `python3 scripts/parse_deploy_output.py --selftest`.
