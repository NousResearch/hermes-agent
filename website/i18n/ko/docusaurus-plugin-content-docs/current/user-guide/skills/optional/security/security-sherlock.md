---
title: "Sherlock — 400개 이상의 플랫폼에서 사용자 이름 계정 찾기"
sidebar_label: "Sherlock"
description: "400개 이상의 플랫폼에서 사용자 이름 계정 찾기"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Sherlock

400개 이상의 플랫폼에서 사용자 이름으로 계정을 찾습니다.

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
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보는 내용입니다.
:::

# Sherlock OSINT 사용자 이름 검색

[Sherlock Project](https://github.com/sherlock-project/sherlock)를 사용하여 400개 이상의 소셜 네트워크에서 사용자 이름으로 소셜 미디어 계정을 찾아냅니다.

## 사용 시점

- 사용자가 사용자 이름과 연결된 계정을 찾아 달라고 요청할 때
- 사용자가 플랫폼별 사용자 이름 사용 가능 여부를 확인하려 할 때
- OSINT 또는 정찰 조사를 수행할 때
- 사용자가 "이 사용자 이름은 어디에 등록되어 있나요?"와 비슷한 질문을 할 때

## 요구 사항

- Sherlock CLI 설치: `pipx install sherlock-project` 또는 `pip install sherlock-project`
- 또는 Docker 사용 가능: `docker run -it --rm sherlock/sherlock`
- 소셜 플랫폼을 조회할 네트워크 액세스

## 절차

### 1. Sherlock 설치 여부 확인

**다른 작업을 하기 전에** sherlock을 사용할 수 있는지 확인합니다:

```bash
sherlock --version
```

명령이 실패하면:
- 설치를 제안합니다: `pipx install sherlock-project`(권장) 또는 `pip install sherlock-project`
- **여러 설치 방법을 시도하지 마세요** — 하나를 선택하고 진행합니다.
- 설치에 실패하면 사용자에게 알리고 중지합니다.

### 2. 사용자 이름 추출

**사용자 메시지에 사용자 이름이 명확하게 적혀 있다면 직접 추출합니다.**

명확화 질문을 사용하면 **안 되는** 예:
- "nasa의 계정을 찾아줘" → 사용자 이름은 `nasa`
- "johndoe123을 검색해줘" → 사용자 이름은 `johndoe123`
- "alice가 소셜 미디어에 있는지 확인해줘" → 사용자 이름은 `alice`
- "소셜 네트워크에서 사용자 bob을 찾아줘" → 사용자 이름은 `bob`

**다음 경우에만 명확화 질문을 사용합니다:**
- 여러 후보 사용자 이름이 언급된 경우("alice 또는 bob 검색")
- 모호한 표현("내 사용자 이름을 검색해줘"라고 했지만 지정하지 않은 경우)
- 사용자 이름이 전혀 언급되지 않은 경우("OSINT 검색을 해줘")

추출할 때는 명시된 **정확한** 사용자 이름을 사용합니다 — 대소문자, 숫자, 밑줄 등을 그대로 보존합니다.

### 3. 명령 구성

**기본 명령**(사용자가 특별히 다르게 요청하지 않는 한 사용):
```bash
sherlock --print-found --no-color "<username>" --timeout 90
```

**선택적 플래그**(사용자가 명시적으로 요청한 경우에만 추가):
- `--nsfw` — NSFW 사이트 포함(사용자가 요청한 경우에만)
- `--tor` — Tor를 통해 라우팅(사용자가 익명성을 요청한 경우에만)

**옵션에 대해 명확화 질문을 하지 마세요** — 기본 검색을 실행하면 됩니다. 필요한 경우 사용자가 특정 옵션을 요청할 수 있습니다.

### 4. 검색 실행

`terminal` 도구로 실행합니다. 네트워크 상태와 사이트 수에 따라 명령은 보통 30-120초가 걸립니다.

**예시 terminal 호출:**
```json
{
  "command": "sherlock --print-found --no-color \"target_username\"",
  "timeout": 180
}
```

### 5. 결과 분석 및 제시

Sherlock은 찾은 계정을 간단한 형식으로 출력합니다. 출력 결과를 분석하여 다음을 제시합니다:

1. **요약 줄:** "사용자 이름 'Y'에서 X개 계정을 찾았습니다"
2. **분류된 링크:** 필요한 경우 플랫폼 유형별로 그룹화(소셜, 전문 네트워크, 포럼 등)
3. **출력 파일 위치:** Sherlock은 기본적으로 결과를 `<username>.txt`에 저장합니다.

**출력 분석 예:**
```
[+] Instagram: https://instagram.com/username
[+] Twitter: https://twitter.com/username
[+] GitHub: https://github.com/username
```

가능하면 결과를 클릭 가능한 링크로 제시합니다.

## 주의 사항

### 결과를 찾지 못한 경우
Sherlock이 계정을 찾지 못했다면 이는 올바른 결과인 경우가 많습니다 — 해당 사용자 이름이 확인한 플랫폼에 등록되지 않았을 수 있습니다. 다음을 제안하세요:
- 철자/변형 확인
- `?` 와일드카드로 유사한 사용자 이름 시도: `sherlock "user?name"`
- 사용자가 개인정보 설정을 적용했거나 계정을 삭제했을 가능성

### 시간 초과 문제
일부 사이트는 느리거나 자동 요청을 차단합니다. `--timeout 120`으로 대기 시간을 늘리거나 `--site`로 범위를 제한하세요.

### Tor 구성
`--tor`를 사용하려면 Tor 데몬이 실행 중이어야 합니다. 사용자가 익명성을 원하지만 Tor를 사용할 수 없다면 다음을 제안하세요:
- Tor 서비스 설치
- 대체 프록시와 함께 `--proxy` 사용

### 오탐
일부 사이트는 응답 구조 때문에 항상 "found"를 반환합니다. 예상치 못한 결과는 직접 확인하여 교차 검증하세요.

### 속도 제한
공격적인 검색은 속도 제한을 유발할 수 있습니다. 사용자 이름을 일괄 검색할 때는 호출 사이에 지연을 추가하거나 캐시된 데이터와 함께 `--local`을 사용하세요.

## 설치

### pipx(권장)
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

### Linux 패키지
Debian 13+, Ubuntu 22.10+, Homebrew, Kali, BlackArch에서 사용할 수 있습니다.

## 윤리적 사용

이 도구는 합법적인 OSINT 및 연구 목적만을 위한 것입니다. 사용자에게 다음을 상기시키세요:
- 자신이 소유하거나 조사 허가를 받은 사용자 이름만 검색할 것
- 플랫폼 서비스 약관을 존중할 것
- 괴롭힘, 스토킹 또는 불법 활동에 사용하지 말 것
- 결과를 공유하기 전에 개인정보 보호 영향을 고려할 것

## 검증

sherlock을 실행한 뒤 다음을 확인합니다:
1. 출력에 URL과 함께 찾은 사이트가 나열되는지
2. 파일 출력을 사용했다면 `<username>.txt` 파일이 생성되었는지(기본 출력)
3. `--print-found`를 사용했다면 출력에 일치 항목에 대한 `[+]` 줄만 포함되는지

## 상호 작용 예시

**사용자:** "사용자 이름 'johndoe123'이 소셜 미디어에 있는지 확인해 줄래?"

**에이전트 절차:**
1. `sherlock --version` 확인(설치 여부 확인)
2. 사용자 이름이 제공되었으므로 바로 진행
3. 실행: `sherlock --print-found --no-color "johndoe123" --timeout 90`
4. 결과를 분석하고 링크를 제시

**응답 형식:**
> 사용자 이름 'johndoe123'에서 12개 계정을 찾았습니다:
>
> • https://twitter.com/johndoe123
> • https://github.com/johndoe123
> • https://instagram.com/johndoe123
> • [... 추가 링크]
>
> 결과 저장 위치: johndoe123.txt

---

**사용자:** "NSFW 사이트를 포함하여 사용자 이름 'alice'를 검색해 줘"

**에이전트 절차:**
1. sherlock 설치 여부 확인
2. 사용자 이름과 NSFW 플래그가 모두 제공됨
3. 실행: `sherlock --print-found --no-color --nsfw "alice" --timeout 90`
4. 결과를 제시

