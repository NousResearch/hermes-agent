---
title: "Unbroker — 데이터 브로커 사이트에서 내 정보를 자율적으로 삭제"
sidebar_label: "Unbroker"
description: "데이터 브로커 사이트에서 내 정보를 자율적으로 삭제"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Unbroker

데이터 브로커 사이트에서 내 정보를 자율적으로 삭제합니다.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/security/unbroker`로 설치 |
| 경로 | `optional-skills/security/unbroker` |
| 버전 | `1.0.0` |
| 작성자 | SHL0MS (github.com/SHL0MS) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `privacy`, `data-broker`, `opt-out`, `ccpa`, `gdpr`, `security`, `doxxing` |
| 관련 skill | [`google-workspace`](/docs/user-guide/skills/bundled/productivity/productivity-google-workspace), [`agentmail`](/docs/user-guide/skills/optional/email/email-agentmail), [`himalaya`](/docs/user-guide/skills/bundled/email/email-himalaya), [`scrapling`](/docs/user-guide/skills/optional/research/research-scrapling), [`osint-investigation`](/docs/user-guide/skills/optional/research/research-osint-investigation) |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 트리거될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성화되었을 때 에이전트가 보게 되는 지침입니다.
:::

# unbroker

데이터 브로커와 사람 검색 사이트에서 한 사람의 개인정보(이름, 주소, 전화번호, 이메일, 친척)가 어디에 노출되어 있는지 찾은 다음, 가능한 경우 자동으로 삭제하고 사이트에서 CAPTCHA, 정부 발급 신분증, 전화 통화 또는 팩스를 요구하는 경우에만 안내에 따라 사람이 처리하도록 합니다. 여러 사람을 독립적으로 관리합니다. 안티봇 시스템을 우회하지 않으며, 기록된 동의 없이 누구에게도 조치를 취하지 않고, 공개 기록(유권자/재산/법원 기록)이나 해당 사람이 관리하는 계정은 삭제하지 않습니다.

Python CLI(`scripts/pdd.py`)가 결정론적 상태(config, dossier + consent, 브로커 데이터베이스, tier 계획, ledger, 초안, 보고서, **이메일 전송(SMTP), 확인 링크 폴링(IMAP), 자율 작업 큐(`next`)**)를 관리합니다. 에이전트는 기본 도구인 `web_extract`와 `browser_navigate`로 검색 및 웹 양식 작성을 수행하고, 반복 재검색에는 `cronjob`을 사용합니다.

## 자율성 계약

이 skill은 **손을 대지 않고 실행**되도록 설계되었습니다. intake(+ 기록된 동의) 이후 정당한 사람 개입 지점은 정확히 두 곳입니다. (1) intake 대화 자체, (2) 실행 종료 시 한 번에 모은 사람 작업 요약(`$PDD tasks`). 그 사이에는 다음 규칙이 적용됩니다.

- **운영자에게 config 선택을 요구하지 않습니다.** `$PDD setup --auto`가 기능을 감지하고 가장 자율적인 유효 config를 직접 선택합니다.
- `autonomy=full`(기본값)에서는 **개별 제출 전에 절대 멈추지 않습니다.** intake에서 기록된 동의가 T0-T2 옵트아웃에 대한 지속적인 승인이 됩니다. (`autonomy=assisted`는 신중한 운영자를 위해 제출별 확인을 되돌리므로 `next` 출력의 `confirm_first` 플래그를 준수합니다.)
- **사람만 할 수 있는 작업 때문에 실행을 중단하지 않습니다.** 이를 기록하고(`record ... human_task_queued --reason "..."`) 계속 진행합니다. 모든 항목은 마지막 요약에 나타납니다.
- `$PDD next <subject>`를 반복하는 루프로 전체 실행을 수행합니다. 이 명령은 지금 수행할 정확한 순서의 작업(검색, 확인 폴링, 재확인, 부모 우선 옵트아웃, 차단된 항목 재큐잉)과 사람 요약을 반환합니다. 모든 작업을 실행하고 결과를 기록한 다음 `next`를 다시 실행하고, `done_for_now`가 될 때까지 반복합니다. 그런 다음 요약, 보고서, cron을 제시합니다.

자율성이 절대 넘어설 수 없는 한계는 다음과 같습니다. 기록된 동의 없이 행동하지 않기, `disclosure_fields`를 넘어 공개하지 않기, CAPTCHA/안티봇 우회하지 않기, 확인 재검색 후에만 `confirmed_removed`로 표시하기.

## 사용 시점

- "데이터 브로커/사람 검색 사이트에서 내(또는 가족 구성원의) 데이터를 삭제해 줘."
- "옵트아웃해 줘", "Spokeo/Whitepages 등에서 나를 삭제해 줘", "독싱 이후에 정리해 줘."
- "반복적인 개인정보 모니터링을 설정해 줘"(브로커가 사람을 다시 등록함).
- 어떤 브로커가 여전히 누군가를 노출하고 있는지, 그리고 그 이유를 확인할 때.

## 사전 요구 사항

- `python3`(표준 라이브러리만 필요하며, 핵심 엔진에는 추가 패키지가 필요하지 않음).
- **선택적 업그레이드**(이 skill은 이것들 없이도 zero-config로 작동하며, `setup --auto`가 감지한 항목을 모두 켭니다. 셸 env **및 `$HERMES_HOME/.env`**에서 인증 정보를 읽으므로 Hermes가 자체 도구에 이미 로드한 키를 다시 export하지 않아도 됩니다. 각 항목은 사람 작업의 한 종류를 에이전트 작업으로 바꿉니다):
  - **클라우드 브라우저(권장 기본값): `BROWSERBASE_API_KEY`.** 키가 있으면 `setup --auto`가 이를 선택합니다. 실제 주거용 IP 클라우드 브라우저가 **정상 작업의 일부로 소프트/관리형 CAPTCHA(Cloudflare Turnstile, hCaptcha/reCAPTCHA 체크박스)를 통과**하므로, 해당 브로커는 사람 작업(T2)이 아니라 자동화(T1) 상태로 유지됩니다. 이는 CAPTCHA "풀이"가 아닙니다. solver 서비스나 fingerprint spoofing을 사용하지 않으며, 브라우저가 실제로 통과할 수 없는 대화형/행동 기반("hard") 챌린지만 사람 작업으로 전환합니다. 키가 없으면 일반 에이전트 브라우저를 사용하고 소프트 CAPTCHA 브로커는 T2(사람)로 내려갑니다.
  - 인증 정보가 필요하거나 필요하지 않은 두 가지 이메일 자동화 옵션:
    - **브라우저 모드(비밀번호 없음): `setup --email-mode browser`.** 에이전트가 운영자의 **로그인된 웹메일**을 통해 `browser_*` 도구로 옵트아웃/CCPA 이메일을 보내고 확인 링크를 엽니다. 아무것도 저장하지 않습니다. 이를 위해서는 Hermes가 클라우드 브라우저가 **아닌** 운영자 본인의 로그인된 브라우저를 사용해야 합니다. 헤드리스 클라우드 브라우저(Browserbase)에는 웹메일 세션이 없고, 웹메일과 세션에 묶인 브로커 게이트(예: PeopleConnect guided-mode)에서 자체적으로 Cloudflare/DataDome의 제한을 받습니다. CDP를 통해 운영자의 실제 Chrome을 구동하세요 — `chrome --remote-debugging-port=9222 --user-data-dir="$HOME/.hermes/chrome-debug"`(기본 프로필이 아닌, 웹메일에 한 번 로그인한 전용 debug 프로필)를 실행하고 브라우저 도구를 `127.0.0.1:9222`에 연결합니다. **`$PDD cdp`가 이 과정을 대신 시작합니다**(`--check`로 테스트, `--print`로 명령 확인). `references/methods.md` -> "Browser backends: scan vs execute"를 참고하세요. 받은 편지함에 접근할 수 없으면 이메일을 초안으로 전환합니다.
    - **SMTP/IMAP(인증 정보 저장): `EMAIL_ADDRESS` + `EMAIL_PASSWORD`**(주류가 아닌 제공자는 `EMAIL_SMTP_HOST` / `EMAIL_IMAP_HOST`도 필요하며, gmail/outlook/yahoo/icloud/fastmail은 자동으로 추론). CLI가 `send-email`로 보내고 `poll-verification`으로 확인 링크를 읽습니다. `agentmail` skill(브로커별 별칭)도 사용할 수 있습니다.
  - Google Sheets 추적기: `google-workspace` skill.
  - stealth/Cloudflare 보호 페이지용 `scrapling` skill.

## 실행 방법

모든 작업은 `terminal` 도구를 통해 실행합니다. 이 skill의 디렉터리에서:

```bash
PDD="python3 scripts/pdd.py"
```

엔진은 `$PDD_DATA_DIR`(기본값 `$HERMES_HOME/unbroker`) 아래에 데이터를 저장하며, 권한은 `0600`으로 기록됩니다. `execute_code`가 **아닌** `terminal`로 실행하세요. (`execute_code`는 env를 삭제하고 출력을 redacted 처리하므로 dossier를 읽을 수 없게 됩니다.)

## 빠른 참조

| 명령 | 용도 |
|---|---|
| `$PDD setup --auto` | **자율 설정**: 기능을 감지하고 가장 자율적인 유효 config를 선택(질문 없음) |
| `$PDD doctor` | 준비 상태 확인: config, 브로커 수, 활성화/사용 가능한 업그레이드 |
| `$PDD cdp [--check] [--print] [--port N]` | Phase-2 브라우저 + 웹메일을 위해 운영자의 Chrome을 CDP로 시작/감지(전용 debug 프로필; 웹메일 전송과 세션에 묶인 게이트를 통과하는 신뢰할 수 있는 방법) |
| `$PDD intake --full-name "..." [--alias ...] [--email ... --phone ...] [--city --state] [--prior-location "City,ST"] --consent` | 동의한 대상 생성; 별칭 + 여러 이메일/전화번호 + 이전 거주지를 수집하고 `subject_id` 출력 |
| `$PDD next <subject>` | **자율 루프 드라이버**: 지금 수행할 에이전트 작업 순서 + 사람 요약 + `next_wake_at` |
| `$PDD brokers [--priority crucial]` | 사람 검색 브로커 데이터베이스(엄선 + 실시간) 나열 |
| `$PDD refresh-brokers` | 최신 BADBOOL 사람 검색 목록 **및 CA Data Broker Registry** 가져오기(`cache`가 오래되면 `next`가 자동으로 재큐잉) |
| `$PDD registry [--search NAME]` | 주별 registry 범위(CA 약 545개 수집; VT/OR/TX 포털 노출); DROP/이메일 lane이며 검색 대상은 아님 |
| `$PDD drop <subject> [--filed]` | **한 번에 사용하는 법적 수단**: 등록된 모든 브로커에서 삭제하는 CA DROP 요청 하나를 생성; `--filed`로 제출 기록 |
| `$PDD plan <subject> [--priority crucial]` | 브로커별 tier + method + `search_vectors` + 공개할 정확한 필드 |
| `$PDD plan <subject> --batch` | **Reduce view**: ledger 상태를 덧씌우고, 브로커를 다음 작업별로 그룹화(unscanned/found/indirect/blocked/in_progress/done), 소유권 cluster를 축약하고, **found** cluster를 부모 우선으로 정렬 + 맞춤형 `parent_playbook` 출력, `next_actions` 출력 |
| `$PDD fanout <subject> [--priority crucial] [--size 5]` | 브로커를 병렬 `delegate_task` subagent로 일괄 처리(대규모 실행에 자동 적용; 5개 단위 — 8개 이상은 timeout) |
| `$PDD record <subject> <broker> <state> [--found true] [--evidence JSON] [--disclosed F --channel C] [--reason "..."]` | ledger 갱신(검증된 상태 머신); `next_recheck_at` 자동 기록 |
| `$PDD show <subject> <broker>` | 사례의 기록된 상태 + evidence + 공개 log를 다시 읽기(부모가 subagent의 `found`를 재검증할 수 있도록 listing URL을 다시 도출하지 않음) |
| `$PDD send-email <subject> <broker> --listing <url> [--kind ccpa_indirect ...]` | 요청 렌더링 + 기록(수신자는 브로커 자체 주소로 고정). **browser** mode는 웹메일로 보낼 `compose` payload를 반환(비밀번호 없음); **programmatic** mode는 SMTP로 전송 |
| `$PDD verify-link <subject> <broker> --text '<body>'` | **browser mode**: 읽은 웹메일 본문에서 브로커 확인 링크 추출(피싱 방지 점수 부여) |
| `$PDD poll-verification <subject> [--broker <id>]` | **programmatic mode**: IMAP에서 확인 링크 폴링(피싱 방지 점수 부여); `submitted → verification_pending` 자동 진행 |
| `$PDD render-email <subject> <broker> --listing <url>` | 초안만 생성(이메일 mode가 설정되지 않은 경우의 fallback) |
| `$PDD due <subject>` | 재확인 기간이 도래한 사례(정기 재검색 queue) |
| `$PDD tasks <subject>` | 사람 작업 **한 번에 모은 요약**(실행 종료 시 제시) |
| `$PDD status <subject>` | Markdown 상태 보고서 |
| `$PDD report <subject> --sheets` | Google Sheets 추적기용 행 |

## 배치 작업(2단계: 전체 크롤링 후 삭제)

브로커가 두어 개를 넘는 경우 브로커별로 하나씩 처리하지 말고 **map → reduce → act**로 실행합니다.

- **Phase 1 - DISCOVER(읽기 전용, 병렬, 멱등).** 모든 브로커를 먼저 크롤링하고 각각에 대해 판정(`found` / `not_found` / `indirect_exposure` / `blocked`)을 기록합니다. 검색에는 side effect가 없으므로 병렬화와 재시도가 안전합니다. 행동하기 전에 전체 노출 지도를 얻어야 cluster 중복 제거와 우선순위 지정이 가능합니다. **기본값: 부모가 `web_extract` probe를 직접 수행**합니다. 대부분의 사람 검색 사이트는 이름/전화번호/주소 결과를 정적 HTML로 렌더링하므로 `web_extract`가 몇 초 안에 읽을 수 있습니다. JS 전용 사이트 몇 곳에만 `browser_*`로 승격하고, 대규모 namesake/relative 판별처럼 진정으로 추론이 많이 필요한 작업에만 `delegate_task` subagent를 사용합니다. **브로커 대규모 목록을 browser-toolset subagent에 맡겨 크롤링하지 마세요.** 현장에서는 이 작업이 반복적으로 timeout(600초, 매번 약 5~6개 브로커, 요약 없음)되었고, 살아남은 ledger 기록도 부모의 `web_extract`보다 비용이 10배 들었습니다. DataDome/Cloudflare/`antibot`으로 차단된 사이트도 subagent 작업이 아닙니다. `blocked`를 기록하고 stealth/cloud browser(Browserbase) 단계로 재큐잉합니다. Subagent 보고는 자기 보고이므로 부모가 핵심 URL을 다시 가져와 `found`를 신뢰하기 전에 확인합니다(이 검증은 부모가 false positive로 잘못 가정한 실제 listing도 잡아냈습니다).
- **REDUCE - `$PDD plan <subject> --batch`.** 크롤링 결과를 단계 중심 계획으로 축약합니다. 다음 작업별로 그룹화하고, **소유권 cluster를 축약**하며(자식도 제거하는 부모 삭제 하나는 N개가 아닌 ONE action — 예: 하나의 Intelius/PeopleConnect suppression이 Truthfinder/Instant Checkmate/US Search/…를 함께 처리), `next_actions`를 출력합니다. 아직 검색되지 않은 항목이 있으면 `phase`는 `discover`, 아니면 `delete`입니다.
- **Phase 2 - DELETE(순차적, 되돌릴 수 없음).** 축약된 그룹을 **부모 우선**으로 처리합니다. `plan --batch`가 `found` 그룹을 cluster 부모 우선(가장 많은 자식을 가진 순서)으로 정렬하고, 각 부모에 맞는 순서화된 단계를 담은 `parent_playbook`을 출력합니다. 각 부모를 처리하되 이미 포함된 자식은 건너뛰고, 부모가 확인된 뒤 자식을 **각각 재검색**합니다(대개 사라짐). 그런 다음 독립 listing을 처리하고, `indirect_exposure` 사례는 CCPA/GDPR 개인정보 삭제 이메일(`send-email --kind ccpa_indirect`)로 보내며, `blocked`는 stealth-browser 단계로 미룹니다. 옵트아웃에는 CAPTCHA, 이메일 확인 반복, session binding이 적용되므로 **한 번에 하나씩 신중하게** 처리합니다(이것은 fan-out의 반대입니다). 그러나 `autonomy=full`에서는 제출별로 허가를 묻기 위해 멈추지 말고, `assisted`에서는 각각 확인합니다. **일반적으로 브로커가 삭제와 suppression을 모두 제공하면 삭제를 우선**합니다(Spokeo/BeenVerified). 단, record의 `deletion.prefer`를 따르세요. **PeopleConnect는 예외**(`prefer: false`)입니다. 사용자 데이터를 삭제하면 suppression이 사라지고 공개 기록이 다시 등록되므로, suppression 후 유지 관리합니다.
- **맹목적 옵트아웃이 기본값이지 fallback이 아닙니다.** 접근 가능한 삭제 channel이 있는 모든 사이트에 listing을 먼저 확인하지 않았더라도 옵트아웃/삭제를 제출합니다. 대상 자신의 식별자만 브로커의 공식 channel에 공개하므로 최소 공개 원칙을 위반하지 않습니다. 두 가지 결론이 따릅니다. (1) 이메일+DOB+이름이 일치하는 guided flow에서 "no results"라고 나오는 것은 어떤 scrape보다 **강한 `not_found`**입니다. 옵트아웃 flow 자체가 검색 역할도 합니다. (2) form이 자동화를 적대시하는 경우(강한 CAPTCHA, Cloudflare/DataDome, slide-to-verify slider) `blocked`를 기록하기보다 **기본적으로 브로커가 명시한 권리 요청 이메일**(이름+주+연락 이메일만)을 사용합니다. CAPTCHA 정책: 행동/토큰/slider challenge를 절대 무력화하지 않습니다. 대상 자신의 옵트아웃에서 정적인 왜곡 문자 또는 단순 산술 CAPTCHA를 읽는 것은 허용되지만, 정답을 입력한 뒤 사이트가 전체 제출을 거부하면 중단합니다(자동화를 fingerprinting하는 것입니다). 제3자/간접 기록은 예외이며, 그래도 조치 전에 확인합니다. 사이트별 실행 계획과 meta-search no-op skip-list는 `references/site-playbooks.md`에 있고, 전체 정책은 `references/methods.md`에 있습니다.
- **PeopleConnect delete-wipes-suppression(영구 규칙).** PeopleConnect의 *삭제*는 suppression을 지우고 대상이 전체 제휴 cluster에 다시 등록되게 합니다. "Your deletion request for PeopleConnect.us is Complete" 이메일이 나타나면 suppression이 사라진 것이므로 **suppression을 다시 실행하고 재확인**합니다. Control 단계가 "suppressed"로 표시되는지 확인하기 전에는 이 cluster를 완료된 삭제 상태로 두지 않습니다(`references/brokers/intelius.json` 참고).

Subagent 보고는 자기 보고입니다. 부모는 `found`를 기록하기 전과 삭제 전에 핵심 주장(listing URL, 일치 근거)을 재검증합니다.

## 절차(자율 루프)

1. **Setup(한 번, 질문 없음).** `$PDD setup --auto`를 실행합니다. 기능을 감지하고 가장 자율적인 유효 조합 자체를 구성합니다(프로그램 이메일은 `EMAIL_*` 인증 정보가 있을 때, Browserbase는 키가 있을 때, `age` 암호화는 바이너리가 있을 때, `autonomy=full`). 그런 다음 `$PDD doctor`를 실행하고 운영자에게 준비 상태 출력을 **정보 제공용으로 보여주되 질문으로 제시하지 말고** 즉시 진행합니다. 더 많은 자동화를 가능하게 하는 요소(예: 이메일 인증 정보)를 언급하되 기다리지 않습니다.
2. **Intake + consent(사람과 나누는 ONE 대화).** `--consent`(및 `--consent-method`)와 함께 `$PDD intake ...`를 실행합니다. 동의가 없으면 엔진이 계획 수립이나 실행을 거부합니다. 이름/별칭, 현재 + 이전 도시, 이메일, 전화번호를 한 번에 수집하여 질문을 다시 해야 하는 일이 없게 합니다. California 대상인 경우 `references/legal/drop.md`도 읽습니다. 그러면 모든 등록 브로커(약 545개)에서 한 번에 삭제하는 단일 고효율 작업인 `drop_submit` one-shot이 `next`에 나타납니다. 이를 제출하고 `drop <subject> --filed`를 실행합니다. California 이외의 대상은 `registry --search` 후 `send-email`로 지정된 CCPA/GDPR 이메일을 보내 registry를 처리하며, 사람 검색 사이트는 어느 경우든 직접 처리합니다.
3. **Queue 비우기.** 반복합니다:

   ```
   while true:
     q = $PDD next <subject>
     if q.actions is empty: break
     execute EVERY action in order; record each outcome via $PDD record
   ```

   `next`는 순서대로 `refresh_brokers`(오래된 cache), `fanout_scan`/`scan_inline`(Phase 1 크롤링 — 4단계 참고), `poll_verification`(진행 중인 이메일 확인), `verify_removal`(기한이 된 재확인), `optout_web_form`/`optout_email_send`(playbook 단계에 따른 부모 우선 Phase 2), `indirect_email_send`, `stealth_rescan`을 출력합니다. 사람만 할 수 있는 작업은 action으로 나타나지 않고 `q.human_digest`에 누적됩니다. `autonomy=full`에서는 일시 정지 없이 실행하고, `assisted`에서는 `confirm_first`를 준수합니다.
4. **검색(`next`가 지시할 때).** `fanout_scan`의 경우 `$PDD fanout <subject>`를 실행하고, `batch`마다 하나의 `delegate_task` subagent를 병렬로 생성하여 해당 batch의 이미 준비된 `brief`를 전달합니다. 모든 브로커를 직접 순차 검색하지 마세요. `scan_inline`의 경우 소수의 브로커를 직접 검색합니다. 어느 경우든 `references/methods.md`의 단계(`web_extract` → `site:` probe → `browser_navigate` → `scrapling`)를 통해 각 브로커의 모든 `search_vectors` 항목을 처리합니다. 404는 INCONCLUSIVE(`not_found`가 아님)이며, `antibot`이 설정되어 있고 stealth browser를 사용할 수 없으면 `blocked`를 기록합니다. 기록하기 전에 대상과 namesake/relative를 구분하여 확인합니다: `$PDD record <subject> <broker> <found|not_found|indirect_exposure|blocked> --found <bool> --evidence '{"listing_urls":[...]}'`. 부모는 subagent의 핵심 `found` 주장을 신뢰하기 전에 다시 검증합니다.
5. **옵트아웃(`next`가 지시할 때).** 각 브로커 record의 `optout.playbook`에서 가져온 `steps`와 함께 부모 우선으로 정렬된 action이 제공됩니다(실제 field 검증을 거친 cluster 부모인 PeopleConnect, Whitepages, BeenVerified, Spokeo에는 정확하고 live-checked된 recipe가 있습니다). **삭제가 보통 suppression보다 우선**입니다. action에 `prefer_deletion`이 있으면 단순히 listing 숨김 flow만 실행하지 말고 record의 DELETION lane을 완료합니다. 반대로 `prefer_suppression`이 있으면(**PeopleConnect** — 삭제하면 suppression이 사라지고 재등록을 막지 못함) suppression flow를 수행하고 계속 유지 관리합니다. Delete 버튼은 의도적인 data-purge일 때만 사용합니다. method별로:
   - **web_form** → `browser_navigate`/`browser_type`/`browser_click`로 `optout_url`을 구동하고 `disclosure_fields`만 제출하며, 확인 화면을 screenshot으로 저장한 다음 action의 `after` record 명령을 실행합니다. Playbook이 right-to-delete `send-email` 후속 작업으로 끝날 수 있으므로 실행하세요(단순 listing suppression이 아닌 전체 삭제).
   - **email** → `$PDD send-email <subject> <broker> --kind <ccpa|gdpr|generic> --to <addr> --listing <url>`가 한 단계로 요청을 기록하고 공개합니다(수신자는 브로커 record가 선언한 주소로 고정되며, `next`는 거주지에 따라 종류를 선택합니다. 해당하지 않는 사람에게 CCPA/GDPR이라고 주장하지 마세요). **browser** mode에서는 수신자가 고정된 `compose` payload가 반환됩니다. 운영자의 웹메일에서 `compose.to`로 새 메시지를 작성하고 `compose.subject`/`compose.body`를 정확히 사용하여 `browser_*`로 전송합니다(비밀번호 없음). **programmatic** mode에서는 SMTP로 전송합니다. `next`는 사람 확인이 필요한 form(phone-callback/gov-ID)도 브로커의 삭제 이메일이 존재하면 이를 통해 전달합니다 — **rescue lane**(검증된 Whitepages 패턴)입니다. 초안 전용인 경우 `render-email` + 요약 항목으로 fallback합니다.
   - **captcha** → 기본 클라우드 브라우저에서 소프트/관리형 challenge는 자동으로 통과되므로 정상적으로 진행합니다. 통과할 수 없는 hard 대화형/행동 기반 challenge만 `blocked`로 기록합니다(stealth/operator-browser 단계로 재큐잉). solver service는 절대 사용하지 않습니다.
   - **phone_callback / account / gov_id / fax / mail / voice (T3)** *삭제 이메일이 없는 경우* → 절대 에이전트 action으로 처리하지 않습니다. `next`가 이미 요약으로 전달했습니다. 다음을 기록합니다: `$PDD record <subject> <broker> human_task_queued --reason "..."`.
6. **확인(`next`가 지시할 때).** **programmatic** mode에서는 `$PDD poll-verification <subject>`가 IMAP를 통해 도착한 확인 링크를 찾습니다(피싱 방지 점수 부여, 상태 자동 진행). **browser** mode에서는 운영자의 웹메일에서 브로커 확인 이메일을 열고 `$PDD verify-link <subject> <broker> --text '<body>'`를 실행하여 링크에 점수를 매깁니다. 어느 경우든 **같은 브라우저에서** 링크를 열고(여러 브로커가 링크를 여는 브라우저에 확인 session을 묶음), flow를 완료한 다음 `awaiting_processing`을 기록합니다. listing이 사라졌다는 확인 재검색이 이루어진 경우에만 `confirmed_removed`로 표시합니다. 제출 flow 자체의 확인 페이지만 보고 표시해서는 안 됩니다.
7. **마무리(실행마다 한 번).** `next`가 action을 반환하지 않으면 `$PDD tasks <subject>`(비어 있지 않은 경우 통합된 사람 요약)를 제시한 다음 `$PDD status <subject>`를 제시합니다. Sheets 추적기가 켜져 있으면 `google-workspace` skill을 통해 `$PDD report <subject> --sheets` 행을 추가합니다.
8. **다음 깨우기 예약.** `next`는 가장 이른 재확인 기한인 `next_wake_at`을 반환합니다. 해당 대상에 대해 이 skill의 루프를 다시 실행하는 **하나의** `cronjob`을 생성합니다(예: *"&lt;subject_id>에 대해 unbroker loop를 실행: `$PDD next`를 실행하고 모든 action을 수행"*). 처리 window, 확인 poll, 재등장 sweep가 모두 같은 queue를 거치므로 사람의 관심 없이도 사례가 계속 진행됩니다.

## 주의 사항

- **브로커가 이미 보여 주는 정보보다 더 많이 공개하지 마세요.** `disclosure_fields`만 제출합니다. 엔진은 SSN/ID 번호를 자발적으로 제공하지 않으며, 당신도 그래서는 안 됩니다.
- **동의가 없으면 action도 없습니다.** 엔진이 이를 강제합니다. 제3자를 "조사"하기 위해 우회하지 마세요.
- **`send-email`은 멱등적이며 rate limit이 적용됩니다.** 이미 `submitted` 또는 그 이후인 사례는 재전송을 거부합니다(진짜 재전송이 필요할 때만 `--force` 사용). SMTP 전송은 `email_min_interval_seconds`(기본 20초)에 따라 간격을 두고 retry/backoff됩니다. "확실히 보내기 위해" 반복하지 마세요. SMTP handoff 성공은 전달의 증거가 아니며, due-queue 재검색이 실제 확인입니다.
- **Ledger 쓰기는 잠깁니다.** 동시 실행(cron + 수동)은 안전하게 직렬화됩니다. lock timeout이 보이면 다른 실행이 쓰는 중이므로 끝날 때까지 기다리고 `.lock`을 수동으로 삭제하지 마세요.
- **자율성 ≠ 즉흥적 판단.** 완전한 자율성은 단계 사이에 *묻지 않는 것*을 뜻하며 어떤 gate도 느슨하게 만들지 않습니다. flow 중 브로커가 계획된 `disclosure_fields`보다 **더 많은 정보**를 요구하면 해당 사례를 중단하고(`human_task_queued --reason`) 혼자 추가 개인정보를 공개하지 마세요.
- **질문으로 실행을 중단하지 마세요.** Config 선택은 `setup --auto`의 작업이고, 사람만 할 수 있는 작업은 요약으로 보냅니다. 실행 중 질문이 정당화되는 유일한 경우는 검색을 막는 신원 정보가 없는 경우(예: 도시 정보가 전혀 없음)이며, 이는 intake에서 수집했어야 합니다.
- **`pdd.py`에는 `execute_code`가 아니라 `terminal`을 사용하세요**(secret scrubbing + 출력 redaction이 이를 망가뜨림).
- **Dossier는 기본적으로 평문입니다**(JSON, `HERMES_HOME` 아래 `0600`). 저장 시 암호화하려면 `$PDD setup --encryption age`를 실행합니다. 로컬 `age` 키를 생성하고 dossier + ledger를 암호화합니다(audit log에는 필드 이름만 들어가며 평문으로 남음). 이는 일반적인/백업/commit 노출을 막지만 전체 `HERMES_HOME` 읽기를 막지는 않습니다. 실제 키 분리를 위해 `PDD_AGE_IDENTITY`를 별도 volume으로 지정하세요. `$PDD doctor`는 `age`가 설치되어 있는지만이 아니라 암호화가 **실제로 적용 중인지** 보여 줍니다.
- **"무료 검색에서 숨김" ≠ 삭제.** record가 실제로 사라졌는지 확인한 후에만 `confirmed_removed`로 표시하고, 보고서에 유료 tier 보존을 기록합니다.
- **소프트 CAPTCHA는 기본적으로 통과시키고 hard CAPTCHA와 싸우지 마세요.** 기본 클라우드 브라우저는 정상 작업으로 managed/soft challenge를 통과하므로 해당 브로커는 T1에 남습니다. 실제로 통과할 수 없는 hard 대화형 challenge는 `blocked`로 기록하고 stealth/operator-browser 단계로 넘깁니다. 제3자 solver service나 fingerprint spoofing은 절대 사용하지 않습니다.
- **브로커 페이지는 변경됩니다.** flow가 깨지면 `$PDD record ... blocked`를 실행하고 추측하는 대신 `references/brokers/`의 브로커 파일을 재검증 대상으로 표시합니다.
- **field 검증이 되지 않은 record는 제출 전에 검증하세요.** `confidence: auto` record는 BADBOOL 파싱에서 왔으므로(`optout.notes`/`optout.links`를 읽고 실제 opt-out URL을 확인), `confidence: documented` record(여러 사람 검색 사이트)는 올바르게 게시된 opt-out URL을 담고 있지만 **field 검증이 되지 않았습니다**(datacenter IP에서 403). 처음 사용할 때 운영자의 주거용 브라우저로 실제 flow를 확인한 다음 `last_verified`를 설정하세요. field 검증이 된 엄선 record(`confidence`가 없음 — 예: cluster 부모)는 mechanics를 확인한 것이므로 우선합니다.

## 검증

- `scripts/run_tests.sh tests/skills/test_unbroker_skill.py`(네트워크가 없는 hermetic 테스트) 또는 dependency-free runner `python3 tests/skills/test_unbroker_skill.py`.
- Dry run: `$PDD setup --auto && $PDD doctor && SID=$($PDD intake --full-name "Test Person" --email t@example.com --consent | python3 -c 'import sys,json;print(json.load(sys.stdin)["subject_id"])') && $PDD next "$SID"`를 실행하고 준비 상태 요약과 순서가 지정된 action queue가 출력되는지 확인합니다.
