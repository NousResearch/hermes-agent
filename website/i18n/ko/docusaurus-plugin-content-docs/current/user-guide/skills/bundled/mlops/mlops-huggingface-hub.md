---
title: "Huggingface Hub — HuggingFace hf CLI: 모델, 데이터셋 검색/다운로드/업로드"
sidebar_label: "Huggingface Hub"
description: "HuggingFace hf CLI: 모델, 데이터셋 검색/다운로드/업로드"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Huggingface Hub

HuggingFace hf CLI로 모델과 데이터셋을 검색/다운로드/업로드합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | Bundled (기본 설치) |
| 경로 | `skills/mlops/huggingface-hub` |
| 버전 | `1.0.1` |
| 작성자 | Hugging Face |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Hugging Face CLI (`hf`) 참조 가이드

`hf` 명령은 Hugging Face Hub와 상호작용하기 위한 최신 명령줄 인터페이스로, 저장소, 모델, 데이터셋, Spaces를 관리하는 도구를 제공합니다.

> **중요:** `hf` 명령은 이제 사용 중단된 `huggingface-cli` 명령을 대체합니다.

## 빠른 시작
*   **설치:** `curl -LsSf https://hf.co/cli/install.sh | bash -s`
*   **도움말:** `hf --help`를 사용해 모든 기능과 실제 예시를 확인합니다.
*   **인증:** `HF_TOKEN` 환경 변수 또는 `--token` 플래그를 사용하는 것을 권장합니다.

---

## 핵심 명령

### 일반 작업
*   `hf download REPO_ID`: Hub에서 파일을 다운로드합니다.
*   `hf upload REPO_ID`: 파일/폴더를 업로드합니다(단일 커밋에 권장되며, 대규모 디렉터리의 재개 가능한 업로드도 처리합니다).
*   `hf upload-large-folder REPO_ID LOCAL_PATH`: **[사용 중단 예정]** — 대신 `hf upload`를 사용합니다.
*   `hf sync`: 로컬 디렉터리와 버킷 간 파일을 동기화합니다.
*   `hf env` / `hf version`: 환경 및 버전 세부 정보를 확인합니다.

### 인증(`hf auth`)
*   `login` / `logout`: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)의 토큰을 사용해 세션을 관리합니다.
*   `list` / `switch`: 저장된 여러 액세스 토큰을 관리하고 전환합니다.
*   `whoami`: 현재 로그인한 계정을 식별합니다.

### 저장소 관리(`hf repos`)
*   `create` / `delete`: 저장소를 생성하거나 영구적으로 삭제합니다.
*   `duplicate`: 모델, 데이터셋 또는 Space를 새 ID로 복제합니다.
*   `move`: 네임스페이스 간 저장소를 이전합니다.
*   `branch` / `tag`: Git과 유사한 참조를 관리합니다.
*   `delete-files`: 패턴을 사용해 특정 파일을 제거합니다.

---

## 전문화된 Hub 상호작용

### 데이터셋 및 모델
*   **데이터셋:** `hf datasets list`, `info`, `parquet`(parquet URL 목록)을 사용합니다.
*   **SQL 쿼리:** `hf datasets sql SQL` — 데이터셋 parquet URL에 대해 DuckDB로 원시 SQL을 실행합니다.
*   **모델:** `hf models list` 및 `info`를 사용합니다.
*   **논문:** `hf papers ls` — 오늘의 논문을 확인합니다.

### 토론 및 풀 리퀘스트(`hf discussions`)
*   Hub 기여의 전체 생명주기를 관리합니다: `list`, `create`, `info`, `comment`, `close`, `reopen`, `rename`.
*   `diff`: PR의 변경 사항을 확인합니다.
*   `merge`: 풀 리퀘스트를 최종 반영합니다.

### 인프라 및 컴퓨팅
*   **엔드포인트:** Inference Endpoint를 배포하고 관리합니다(`deploy`, `pause`, `resume`, `scale-to-zero`, `catalog`).
*   **작업:** HF 인프라에서 컴퓨팅 작업을 실행합니다. 리소스 모니터링을 위한 `hf jobs uv`(인라인 의존성이 포함된 Python 스크립트 실행) 및 `stats`를 포함합니다.
*   **Spaces:** 인터랙티브 앱을 관리합니다. 전체 재시작 없이 Python 파일을 위한 `dev-mode` 및 `hot-reload`를 포함합니다.

### 스토리지 및 자동화
*   **버킷:** S3와 유사한 버킷을 완전히 관리합니다(`create`, `cp`, `mv`, `rm`, `sync`).
*   **캐시:** `list`, `prune`(분리된 리비전 제거), `verify`(체크섬 확인)를 사용해 로컬 스토리지를 관리합니다.
*   **웹훅:** Hub 웹훅을 관리해 워크플로를 자동화합니다(`create`, `watch`, `enable`/`disable`).
*   **컬렉션:** 컬렉션으로 Hub 항목을 구성합니다(`add-item`, `update`, `list`).

---

## 고급 사용법 및 팁

### 전역 플래그
*   `--format json`: 자동화를 위해 기계가 읽을 수 있는 출력을 생성합니다.
*   `-q` / `--quiet`: ID만 출력하도록 제한합니다.

### 확장 기능 및 스킬
*   **확장 기능:** `hf extensions install REPO_ID`를 사용해 GitHub 저장소를 통해 CLI 기능을 확장합니다.
*   **스킬:** `hf skills add`로 AI 어시스턴트 스킬을 관리합니다.
