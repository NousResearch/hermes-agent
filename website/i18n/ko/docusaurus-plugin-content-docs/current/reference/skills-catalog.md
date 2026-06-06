---
sidebar_position: 15
title: 번들 스킬 카탈로그 (Bundled Skills)
description: Hermes와 함께 제공되는 기본 스킬에 대한 참고 자료입니다.
---

# 번들 스킬 (Bundled Skills)

Hermes는 일반적인 워크플로우를 즉시 처리할 수 있도록 설계된 미리 설치된 스킬들("번들 스킬")과 함께 제공됩니다. 이러한 스킬들은 `~/.hermes/skills/`에 위치하며 활성 프로필에 자동으로 시드(seed)됩니다.

## 🛠️ 핵심 유틸리티 (Core Utilities)

### Project Manager (`project-manager`)
코드 기반, 문서 작성 및 일반적인 프로젝트 구성을 위한 프로젝트의 부트스트랩 및 관리를 처리합니다.
- **기능:** 저장소(Repository) 뼈대 설정, README 파일 생성, `package.json`/`requirements.txt` 관리 및 일반적인 라이선스 발급.

### File Organizer (`file-organizer`)
작업 공간을 깨끗하고 체계적으로 유지하도록 돕습니다.
- **기능:** 확장자나 날짜별로 혼란스러운 디렉터리 재구성, 중복 파일 찾기, 파일 일괄 이름 변경 및 디렉터리 트리 요약.

### Note Taker (`note-taker`)
기본 메모 작성 및 정보 관리를 위한 스킬입니다.
- **기능:** Markdown 메모 작성/수정, 일일 작업 로그 관리, 해야 할 일 목록(To-Do list) 추적 및 다양한 메모에서 핵심 내용 추출.

## 🌐 리서치 및 웹 (Research & Web)

### Web Researcher (`web-researcher`)
주제를 검색하고 엮어내는 심층 인터넷 리서치 스킬입니다.
- **기능:** 특정 주제에 대한 심층 요약 수행, 다양한 정보 출처에 대한 편향성/신뢰성 교차 검증, 참조 문헌 목록 수집 및 리서치 요약 보고서 작성.
- **필수 사항:** `web_search` 도구 세트가 활성화되어 있고 구성되어 있어야 합니다.

### URL Summarizer (`url-summarizer`)
긴 웹 기사나 문서를 빠르고 간결하게 요약합니다.
- **기능:** 기사 파싱, 핵심 요점(bullet points) 추출, TL;DR 생성 및 특정 질문에 답하기 위한 긴 페이지 검색.

## 🧑‍💻 개발자 경험 (Developer Experience)

### Git Assistant (`git-assistant`)
`git_tools` 위에 구축되어 워크플로우를 간소화합니다.
- **기능:** 커밋 메시지 초안 작성, 복잡한 diff 설명, rebase 충돌 해결 지원 및 분기(branch) 정리.

### Code Reviewer (`code-reviewer`)
수정 또는 변경된 코드에 대한 가벼운 정적 분석 및 피드백을 제공합니다.
- **기능:** 풀 리퀘스트(PR)를 위한 diff 분석, 성능 병목 현상 제안, 보안 취약점 찾기 및 스타일 가이드 준수(PEP8, ESLint 패턴 등) 확인.

### Terminal Wizard (`terminal-wizard`)
셸의 마스터로, 사용자가 복잡한 터미널 명령을 작성하거나 스크립트를 디버깅하도록 돕습니다.
- **기능:** 알 수 없는 오류 메시지 설명, 파이프(pipe)된 명령어 체인 작성(`sed`/`awk`/`grep` 등), 시스템 리소스 사용량 진단 및 셸 별칭(Alias) 작성.

---

## 번들 스킬 관리 (Managing Bundled Skills)

번들 스킬은 코어 설치본(Core Installation)과 함께 업데이트됩니다.

**번들 스킬 비활성화:**
채팅 세션 내에서 특정 스킬을 비활성화하려면:
```text
/skills off code-reviewer
```
또는 CLI를 통해:
```bash
hermes skills config
```

**수정 및 재설정:**
번들 스킬(`~/.hermes/skills/`에 있는 `SKILL.md`)의 프롬프트나 동작을 자유롭게 편집할 수 있습니다. 편집하면 해당 스킬이 `user_modified`로 표시되어 향후 Hermes 업데이트 시 수정된 파일이 덮어쓰이지 않도록 보호됩니다.

공식 버전으로 재설정하려면:
```bash
hermes skills reset code-reviewer --restore --yes
```

**자동 채우기 중단 (Opting Out):**
향후 업데이트에서 프로필에 새로운 번들 스킬이 계속 추가되는 것을 막고 싶다면:
```bash
hermes skills opt-out
```
