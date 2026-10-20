---
title: "차단된 페이지 복구 — 아카이브 스냅샷과 리더 대체 경로로 차단/유료 벽/WAF 페이지 복구"
sidebar_label: "차단된 페이지 복구"
description: "아카이브 스냅샷과 리더 대체 경로로 차단/유료 벽/WAF 페이지 복구"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 skill의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 수정하세요. */}

# 차단된 페이지 복구

차단/유료 벽/WAF 페이지를 아카이브 스냅샷과 리더 대체 경로로 복구합니다. `web_extract` 또는 브라우저가 403/429/챌린지 페이지, 유료 벽 또는 봇 감지 중간 페이지에 부딪혔을 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함 (기본으로 설치됨) |
| 경로 | `skills/research/blocked-page-recovery` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Research`, `Archives`, `Wayback`, `Paywall`, `WAF`, `Fallback` |
| 관련 스킬 | [`grounded-citations`](/docs/user-guide/skills/bundled/research/research-grounded-citations) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 스킬 정의 전문입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# 차단된 페이지 복구

페이지를 가져올 수 없을 때 — 403/429, Cloudflare의 "잠시만 기다려 주세요...", 유료 벽 또는 봇 감지 중간 페이지 — 포기하거나 같은 URL을 반복하지 마세요. 서드파티 서비스에 페이지 **사본**이 있는 경우가 많습니다. 비용이 가장 적은 경로부터 다음 순서로 시도하세요.

## 단계별 경로

```
1. Wayback Machine  — archive.org "available" API  (snapshot + timestamp)
2. archive.today    — domain rotation: archive.ph → .md → .li → .is
3. Jina Reader      — only if JINA_API_KEY is set  (live server-side render)
4. API-first pivot  — look for /api/, /graphql, .json, or RSS on the same host
5. Real browser     — browser tool as the last, most expensive resort
```

내장 스크립트로 한 번에 실행하세요.

```bash
python3 scripts/recover_page.py "https://example.com/blocked-article" --json
```

스크립트는 각 경로를 순서대로 시도하고, 모든 본문을 검증한 뒤(아래 "가짜 성공" 참조), 출처 정보와 함께 처음으로 실제로 성공한 결과를 출력합니다.

## 출처 관리 (협상 불가)

복구된 모든 사본에는 인용할 때 반드시 보존해야 하는 출처 정보가 있습니다.

| 경로 | 출처 정보 | 인용 방법 |
|-------|-----------|-------------|
| Wayback / archive.today | `snapshot` | 스냅샷 날짜를 포함해 인용하세요: "2026-08-06에 보관된 자료에 따르면". 스냅샷을 실시간 페이지로 제시하지 마세요 — 오래된 자료일 수 있습니다. |
| Jina Reader | `live` | 서버 측에서 실시간 페이지를 다시 렌더링한 결과이므로 일반적으로 인용하세요. |
| 실시간 가져오기 / 브라우저 | `live` | 일반적으로 인용하세요. |

사용자에게 최신 데이터(가격, 이용 가능 여부, 속보)가 필요하다면 스냅샷은 답변이 아니라 맥락입니다. 이를 명시적으로 밝히고 자료의 시점을 함께 표시하세요.

## 수동 경로

### 1. Wayback Machine (출처 정보가 가장 좋으므로 먼저 시도)

```bash
# Discovery: returns closest snapshot URL + timestamp as JSON
curl -sL "https://archive.org/wayback/available?url={URL}"
# Then fetch archived_snapshots.closest.url
```

여러 스냅샷을 열거하거나 삭제된 페이지를 복구하려면 CDX 색인을 사용하세요.

```bash
curl -sL "https://web.archive.org/cdx/search/cdx?url={URL}&output=json&limit=10"
```

CDX는 부하가 걸리면 간헐적으로 503을 반환합니다. 이 경우 `available` API로 전환하고 반복 요청으로 두드리지 마세요.

작동 대상: 공개적으로 크롤링된 모든 URL. 실패 대상: robots로 차단된 사이트, 한 번도 크롤링되지 않은 URL, 스냅샷이 렌더링하지 못하는 JS 전용 SPA.

### 2. archive.today (유료 벽, 삭제된 콘텐츠)

사용자가 제출한 아카이브로, Wayback에 없는 유료 뉴스 기사를 보관하는 경우가 많습니다. 요청을 적극적으로 제한(429)하고 도메인을 순환하므로 다음과 같이 순회하세요.

```bash
for d in archive.ph archive.md archive.li archive.is; do
  curl -sL --max-time 20 "https://$d/newest/{URL}" -o /tmp/page.html \
    -w "%{http_code}" && break
done
```

**상태 코드가 아니라 본문을 검증하세요** — archive.today의 429 페이지는 수 KB의 HTML을 반환합니다. 단순히 크기만 확인하지 말고 대상의 실제 콘텐츠(제목 단어, 예상 문자열)가 있는지 확인하세요.

### 3. Jina Reader (JINA_API_KEY 필요)

`r.jina.ai`는 실시간 페이지를 서버 측의 실제 브라우저에서 다시 렌더링하고 마크다운으로 반환합니다. 익명 접근은 종료되었습니다(401 → Turnstile). 키가 필요합니다.

```bash
curl -s -H "Authorization: Bearer $JINA_API_KEY" "https://r.jina.ai/{URL}"
```

아카이브가 처리하지 못하는 JS SPA를 처리합니다. 환경 변수가 설정되지 않았다면 이 경로 전체를 건너뛰세요.

### 4. API 우선 전환

WAF는 그 뒤의 데이터 엔드포인트보다 HTML 표면을 훨씬 더 적극적으로 보호합니다. 사이트에서 2~3번 차단되면 HTML과 계속 씨름하지 말고 다음을 찾아보세요.

- 페이지 URL의 `/api/...`, `/graphql` 또는 `.json` 변형
- RSS/Atom 피드 (`/feed`, 복구한 사본의 `<link rel="alternate">`)
- 사이트맵 (`/sitemap.xml`) — 차단되지 않을 수 있는 표준 URL을 확인할 수 있습니다.

## 가짜 성공 — 거짓말하는 경로

이 경로들은 그럴듯한 본문과 함께 HTTP 200을 반환하지만 페이지 자체가 아닙니다. 스크립트가 자동으로 거부하지만, 수동으로도 거부하세요.

- **Google Cache는 종료되었습니다** (2024년 중반 이후). `webcache.googleusercontent.com`은 수십 KB와 함께 200을 반환하지만, 캐시가 아니라 JS 리디렉션이 있는 Google 검색 중간 페이지입니다. 절대 사용하지 마세요.
- **AMP 캐시** (`*.cdn.ampproject.org`)는 대부분 원본(차단된) URL로 되돌아가는 약 300바이트의 `<title>Redirecting</title>` 메타 새로고침 스텁을 반환합니다. 이를 성공으로 처리하면 가져오기 루프가 발생합니다.
- **요청 제한 본문**: archive.today의 429 페이지는 수 KB의 HTML입니다. 크기만 보지 말고 대상의 실제 콘텐츠(제목 단어, 예상 문자열)를 확인하세요.

스크립트가 적용하는 감지 휴리스틱: 경로별 바이트 하한보다 짧은 본문, 원본 호스트를 대상으로 하는 메타 새로고침/JS 리디렉션 스텁, 중간 페이지 제목("잠시만 기다려 주세요", "Redirecting", "Google Search", "Attention Required").

## 프록시 릴레이: 사용하지 마세요

일반적인 "웹 프록시" 릴레이는 구조상 중간자입니다. 쿠키나 Authorization 헤더를 그런 릴레이를 통해 보내지 말고, 사용자가 의존할 어떤 용도로도 사용하지 마세요 — 출처 정보를 검증할 수 없습니다. 최소한 사본에 시점을 기록하는 아카이브를 선호하세요.
