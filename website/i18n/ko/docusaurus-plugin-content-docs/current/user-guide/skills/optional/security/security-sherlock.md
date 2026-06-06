---
title: "Sherlock — 400개 이상의 소셜 네트워크에서 OSINT 사용자 이름 검색"
sidebar_label: "Sherlock"
description: "400개 이상의 소셜 네트워크에서 OSINT 사용자 이름 검색"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sherlock

400개 이상의 소셜 네트워크에서 OSINT 사용자 이름 검색을 수행합니다. 특정 사용자 이름으로 소셜 미디어 계정을 추적할 수 있습니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/security/sherlock`으로 설치 |
| 경로 | `optional-skills/security/sherlock` |
| 버전 | `1.0.0` |
| 작성자 | unmodeled-tyler |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `osint`, `security`, `username`, `social-media`, `reconnaissance` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Sherlock OSINT 사용자 이름 검색

[Sherlock 프로젝트](https://github.com/sherlock-project/sherlock)를 사용하여 400개 이상의 소셜 네트워크에서 사용자 이름을 기반으로 소셜 미디어 계정을 찾아냅니다.

## 언제 사용하나요

- 사용자가 특정 사용자 이름과 관련된 계정을 찾아달라고 요청할 때
- 여러 플랫폼에서의 사용자 이름 사용 가능 여부를 확인하고 싶을 때
- 사용자가 OSINT(공개 출처 정보) 수집이나 정찰 조사를 수행할 때
- 사용자가 "이 사용자 이름은 어디에 등록되어 있나요?" 등과 같이 질문할 때

## 요구 사항

- Sherlock CLI가 설치되어 있어야 합니다: `pipx install sherlock-project` 또는 `pip install sherlock-project`
- 또는: Docker를 사용할 수 있어야 합니다 (`docker run -it --rm sherlock/sherlock`)
- 소셜 플랫폼에 쿼리를 보내기 위한 네트워크 접근 권한

## 절차

### 1. Sherlock 설치 여부 확인

**다른 작업을 하기 전에** sherlock을 사용할 수 있는지부터 검증하세요:

```bash
sherlock --version
```

명령어가 실패한다면:
- 설치를 제안합니다: `pipx install sherlock-project` (권장) 또는 `pip install sherlock-project`
- **여러 설치 방법을 시도하지 마십시오** — 한 가지를 골라 진행하세요.
- 설치가 실패하면 사용자에게 상황을 알리고 중단하세요.

### 2. 사용자 이름 추출

**사용자가 명확하게 명시한 경우 메시지에서 직접 사용자 이름을 추출하세요.**

명확한 질문이므로 다시 물어볼(clarify) **필요가 없는** 예시:
- "nasa 계정 찾아줘" → 사용자 이름은 `nasa`
- "johndoe123 검색해줘" → 사용자 이름은 `johndoe123`
- "소셜 미디어에 alice가 있는지 확인해줘" → 사용자 이름은 `alice`
- "사용자 bob을 소셜 네트워크에서 조회해줘" → 사용자 이름은 `bob`

**다음과 같은 경우에만 clarify(명확화 질문)를 사용하세요:**
- 여러 개의 사용자 이름이 언급되었을 때 ("alice나 bob을 검색해줘")
- 표현이 모호할 때 (특정하지 않고 "내 사용자 이름을 검색해줘"라고 할 때)
- 사용자 이름이 아예 언급되지 않았을 때 ("OSINT 검색을 실행해줘")

추출할 때, **사용자가 제시한 정확한** 사용자 이름을 가져오세요 — 대소문자, 숫자, 밑줄 등을 모두 유지합니다.

### 3. 명령어 구성

**기본 명령어** (사용자가 특별히 다르게 요청하지 않는 한 이것을 사용하세요):
```bash
sherlock --print-found --no-color "<username>" --timeout 90
```

**선택적 플래그** (사용자가 명시적으로 요청한 경우에만 추가):
- `--nsfw` — 성인(NSFW) 사이트를 포함합니다. (사용자가 요청한 경우에만)
- `--tor` — Tor 네트워크를 통해 라우팅합니다. (사용자가 익명성을 요구할 경우에만)

**명확화 질문(clarify)을 통해 옵션에 대해 묻지 마세요** — 그냥 기본 검색을 실행하십시오. 사용자가 필요하다면 특정 옵션을 직접 요구할 것입니다.

### 4. 검색 실행

`terminal` 도구를 통해 실행합니다. 네트워크 상태와 검사 대상 사이트 수에 따라 대개 30~120초 정도 소요됩니다.

**terminal 도구 호출 예시:**
```json
{
  "command": "sherlock --print-found --no-color \"target_username\"",
  "timeout": 180
}
```

### 5. 결과 파싱 및 표시

Sherlock은 단순한 형식으로 찾은 계정을 출력합니다. 출력을 파싱하여 다음과 같이 제시하십시오:

1. **요약 줄:** "'Y' 사용자 이름에 대해 X개의 계정을 찾았습니다."
2. **분류된 링크 목록:** 도움이 된다면 플랫폼 유형별로(소셜, 전문직, 포럼 등) 분류하십시오.
3. **출력 파일 위치:** Sherlock은 기본적으로 결과를 `<username>.txt` 파일에 저장합니다.

**출력 파싱 예시:**
```
[+] Instagram: https://instagram.com/username
[+] Twitter: https://twitter.com/username
[+] GitHub: https://github.com/username
```

가능한 클릭할 수 있는 링크 형태로 결과를 제공하십시오.

## 주의 사항 (Pitfalls)

### 검색 결과가 없을 때
Sherlock이 어떠한 계정도 찾지 못했다면 보통은 그것이 맞습니다 — 해당 사용자 이름이 검사 대상 플랫폼에 등록되어 있지 않은 것일 수 있습니다. 다음과 같이 제안하세요:
- 철자나 변형된 형태 확인
- `?` 와일드카드를 사용해 비슷한 이름 검색 시도: `sherlock "user?name"`
- 사용자가 개인정보 보호 설정을 해두었거나 계정을 삭제했을 가능성 안내

### 타임아웃 문제
일부 사이트는 느리거나 자동화된 요청을 차단할 수 있습니다. `--timeout 120`을 사용해 대기 시간을 늘리거나, `--site`로 범위를 제한하십시오.

### Tor 구성
`--tor`를 사용하려면 Tor 데몬이 실행 중이어야 합니다. 사용자가 익명성을 원하지만 Tor를 사용할 수 없다면 다음과 같이 제안하세요:
- Tor 서비스 설치
- 다른 프록시와 함께 `--proxy` 사용

### 허위 양성 (False Positives)
응답 구조상의 이유로 일부 사이트는 늘 검색에 "성공(found)"했다고 반환할 수 있습니다. 수동으로 확인해 예상치 못한 결과와 교차 검증하십시오.

### 속도 제한 (Rate Limiting)
공격적인 탐색은 속도 제한(rate limit)을 유발할 수 있습니다. 대량의 사용자 이름 검색 시 호출 사이에 지연(delay) 시간을 추가하거나, 캐시된 데이터를 사용하는 `--local` 옵션을 사용하십시오.

## 설치

### pipx (권장)
```bash
pipx install sherlock-project
```

### pip
```bash
pip install sherlock-project
```

### Docker
```bash
docker pull sherlock/sherlock
docker run -it --rm sherlock/sherlock <username>
```

### 리눅스 패키지
Debian 13+, Ubuntu 22.10+, Homebrew, Kali, BlackArch 등에서 사용할 수 있습니다.

## 윤리적 사용 가이드

이 도구는 합법적인 OSINT 및 연구 목적으로만 사용되어야 합니다. 사용자에게 다음 사항을 상기시키십시오:
- 오직 본인이 소유하거나 조사할 권한을 부여받은 사용자 이름만 검색할 것.
- 각 플랫폼의 서비스 약관을 준수할 것.
- 괴롭힘, 스토킹, 불법적인 활동에 이 도구를 사용하지 말 것.
- 결과를 공유하기 전 개인 정보 침해 소지가 있는지 고려할 것.

## 검증 (Verification)

sherlock을 실행한 후, 다음 사항을 확인하십시오:
1. 결과에 URL이 포함된 발견 사이트들이 나열되어 있는지.
2. 파일 출력을 사용했다면 기본 결과물인 `<username>.txt` 파일이 생성되었는지.
3. `--print-found` 옵션을 사용했다면, 오직 일치하는 항목인 `[+]` 줄만 출력되었는지.

## 대화 예시

**사용자:** "'johndoe123' 사용자 이름이 소셜 미디어에 있는지 확인해줄래?"

**에이전트 절차:**
1. `sherlock --version` 확인 (설치 여부 검증)
2. 사용자 이름이 제공되었으므로 바로 진행
3. 실행: `sherlock --print-found --no-color "johndoe123" --timeout 90`
4. 결과를 분석하고 링크 제공

**응답 형식:**
> 'johndoe123' 사용자 이름에 대해 12개의 계정을 찾았습니다:
>
> • https://twitter.com/johndoe123
> • https://github.com/johndoe123
> • https://instagram.com/johndoe123
> • [... 추가 링크들]
>
> 결과는 다음 위치에 저장되었습니다: johndoe123.txt

---

**사용자:** "성인 사이트를 포함해서 사용자 이름 'alice'를 검색해줘."

**에이전트 절차:**
1. sherlock 설치 여부 확인
2. 사용자 이름 및 성인(NSFW) 검색 플래그 모두 제공됨
3. 실행: `sherlock --print-found --no-color --nsfw "alice" --timeout 90`
4. 결과를 분석하여 제공
