---
title: "Hugging Face Hub — HuggingFace Hub: 데이터셋 다운로드/업로드 및 모델 가중치 배포"
sidebar_label: "Hugging Face Hub"
description: "HuggingFace Hub: 데이터셋 다운로드/업로드 및 모델 가중치 배포"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Hugging Face Hub

HuggingFace Hub: 데이터셋 다운로드/업로드 및 모델 가중치 배포.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/mlops/huggingface-hub` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `huggingface_hub` |
| 플랫폼 | linux, macos |
| 태그 | `Model Hub`, `Dataset Hub`, `Model Deployment`, `CLI`, `Python API` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Hugging Face Hub

Hugging Face Hub CLI 및 Python API를 통해 모델과 데이터셋을 상호 작용하는 방법에 대한 가이드입니다.

## 포함된 내용

Hugging Face Hub(`huggingface_hub`)는 Hub에서 모델, 데이터셋 및 스페이스(Spaces)를 상호 작용하기 위한 공식 Python 라이브러리이자 CLI입니다. 이 스킬은 이 도구를 사용하여 리소스를 다운로드/업로드하고 허브와 상호 작용하는 방법에 대한 가이드를 제공합니다.

## 빠른 시작

1.  **설치:**
    ```bash
    pip install huggingface_hub
    ```

2.  **로그인:**
    브라우저나 환경 변수(`HUGGING_FACE_HUB_TOKEN`)를 통해 인증합니다. `read` (다운로드 전용) 또는 `write` (업로드 포함) 토큰을 허브에서 생성하세요.
    ```bash
    huggingface-cli login
    # 토큰을 입력하라는 프롬프트가 표시됩니다
    ```

## 핵심 개념

-   **Repository(저장소)**: 모델(`model`), 데이터셋(`dataset`), 또는 스페이스(`space`)의 모음입니다.
-   **Revision(리비전)**: git 브랜치, 태그 또는 커밋 해시와 유사한 개념으로, 특정 버전의 저장소를 나타냅니다.
-   **Snapshot(스냅샷)**: 지정된 리비전에서의 전체 저장소 내용입니다.

## 일반적인 워크플로우

### 워크플로우 1: 모델 및 데이터셋 다운로드

모델과 데이터셋을 로컬로 다운로드합니다. 이 도구는 기본적으로 `~/.cache/huggingface/hub`에 파일 캐싱을 처리합니다.

**Python API (권장):**
특정 파일 또는 전체 저장소를 다운로드할 수 있습니다.
```python
from huggingface_hub import hf_hub_download, snapshot_download

# 특정 모델 파일 (예: config) 다운로드
config_path = hf_hub_download(repo_id="gpt2", filename="config.json")
print(f"Config downloaded to: {config_path}")

# 전체 모델 저장소(snapshot) 다운로드
model_dir = snapshot_download(repo_id="gpt2")
print(f"Model downloaded to: {model_dir}")

# 데이터셋 다운로드 (repo_type 지정)
dataset_dir = snapshot_download(repo_id="wikitext", repo_type="dataset")
```

**CLI:**
빠른 다운로드나 스크립트 작성에 유용합니다.
```bash
# 전체 모델 다운로드
huggingface-cli download gpt2

# 특정 모델 파일 다운로드 및 저장 위치 지정
huggingface-cli download gpt2 config.json --local-dir ./my-gpt2-model

# 데이터셋 다운로드
huggingface-cli download wikitext --repo-type dataset --local-dir ./my-wikitext-data
```

### 워크플로우 2: 모델 및 데이터셋 업로드

학습된 모델이나 커스텀 데이터셋을 공유하거나 저장하기 위해 허브에 업로드합니다.

**Python API:**
```python
from huggingface_hub import HfApi

api = HfApi()

# 저장소 생성 (존재하지 않는 경우)
repo_id = "your-username/my-awesome-model" # 자신의 사용자 이름/조직으로 변경
api.create_repo(repo_id=repo_id, exist_ok=True)

# 단일 파일 업로드
api.upload_file(
    path_or_fileobj="path/to/local/model.bin",
    path_in_repo="model.bin",
    repo_id=repo_id,
)

# 전체 폴더 업로드
api.upload_folder(
    folder_path="path/to/local/model_directory",
    repo_id=repo_id,
    commit_message="Initial model upload"
)
```

**CLI:**
```bash
# 전체 폴더 업로드
huggingface-cli upload your-username/my-awesome-model ./path/to/local/model_directory

# 단일 파일 업로드
huggingface-cli upload your-username/my-awesome-model ./path/to/local/model.bin model.bin
```

### 워크플로우 3: 저장소 관리 및 파일 조작

저장소 내용 조회 및 삭제.

**Python API:**
```python
from huggingface_hub import HfApi

api = HfApi()
repo_id = "your-username/my-awesome-model"

# 저장소 내 파일 목록 보기
files = api.list_repo_files(repo_id)
print(f"Files in repo: {files}")

# 파일 삭제
# api.delete_file(path_in_repo="model.bin", repo_id=repo_id)

# 전체 저장소 삭제 (주의!)
# api.delete_repo(repo_id=repo_id)
```

**CLI:**
```bash
# 로컬 캐시 관리 (캐시된 저장소 보기 및 삭제)
huggingface-cli scan-cache
huggingface-cli delete-cache
```

## 일반적인 문제

**문제:** 권한 거부 오류 (HTTP 401/403)
-   **해결책:** `huggingface-cli login`을 올바른 권한(다운로드의 경우 `read`, 업로드의 경우 `write`)이 있는 유효한 토큰으로 실행했는지 확인하세요. Gated 모델(예: Llama 2)의 경우 허브 웹사이트에서 약관에 동의했는지 확인하세요.

**문제:** 대용량 파일 다운로드 지연/중단
-   **해결책:** 라이브러리가 기본적으로 큰 파일을 재개(resume) 처리합니다. 다운로드 속도가 심각하게 느리다면 네트워크 환경을 확인하거나 `--resume-download` (CLI)을 시도해보세요. 일부 환경에서는 허브 접속이 차단되었을 수 있습니다 (`HF_ENDPOINT` 설정 필요).

**문제:** 로컬 디스크 공간 부족
-   **해결책:** Hugging Face 캐시(기본값 `~/.cache/huggingface/hub`)는 매우 빠르게 용량을 차지할 수 있습니다. `huggingface-cli scan-cache`를 사용하여 공간 사용량을 파악하고, `huggingface-cli delete-cache`를 사용하여 사용하지 않는 모델을 정리하세요.

## 리소스

-   **huggingface_hub 문서:** https://huggingface.co/docs/huggingface_hub/en/index
-   **CLI 가이드:** https://huggingface.co/docs/huggingface_hub/en/guides/cli
-   **모델 허브:** https://huggingface.co/models
-   **데이터셋 허브:** https://huggingface.co/datasets
