---
sidebar_position: 5
title: "Google Gemini"
description: "Hermes Agent에서 Google Gemini 사용하기 (무료 및 유료 티어 모두 지원)"
---

# Google Gemini

Hermes Agent는 **Google Gemini** 모델 제품군을 최고 수준의 프로바이더로 기본 지원합니다. 이 가이드에서는 Gemini의 [무료 티어 (Google AI Studio)](#방법-1-무료-티어-google-ai-studio)를 설정하는 방법과 프로덕션 수준의 [Enterprise/유료 티어 (Google Cloud Vertex AI)](#방법-2-enterprise-티어-google-cloud-vertex-ai)를 설정하는 방법을 다룹니다.

두 연결 방법 모두 완전한 도구 호출(tool calling), 비전(vision) 지원 및 100만 토큰의 컨텍스트 길이를 포함한 Hermes Agent의 핵심 기능을 지원합니다.

---

## 방법 1: 무료 티어 (Google AI Studio)

이것은 개발 및 취미용(개인용) 프로젝트를 위한 가장 쉽고 빠른 방법입니다. 신용카드가 필요 없으며 상당한 무료 사용량 제한(rate limits)을 제공합니다.

### 1단계: API 키 발급
1. [Google AI Studio](https://aistudio.google.com/)로 이동하여 Google 계정으로 로그인합니다.
2. 사이드바에서 **Get API key (API 키 받기)**를 클릭합니다.
3. 새 프로젝트를 생성하고 API 키를 복사합니다.

### 2단계: 환경 변수 설정
`~/.hermes/.env` 파일에 API 키를 추가합니다:

```env
GEMINI_API_KEY=your-api-key-here
```

### 3단계: Hermes 구성
CLI 명령어를 사용하거나 `config.yaml`을 직접 편집하세요.

**명령어 사용:**
```bash
hermes model
# 프로바이더 목록에서 "Google Gemini (AI Studio / Free)"를 선택합니다
# "gemini-3.1-pro-preview" 또는 원하는 모델을 선택합니다
```

**수동 구성 (`~/.hermes/config.yaml`):**
```yaml
model:
  provider: gemini
  default: gemini-3.1-pro-preview
```

---

## 방법 2: Enterprise 티어 (Google Cloud Vertex AI)

프로덕션 배포, 더 높은 사용량 제한, 그리고 데이터 프라이버시/준수(컴플라이언스) 보장이 필요한 경우 이 방법을 사용하세요. 이 방법은 Google Cloud 계정, 결제 수단 및 `gcloud` CLI 도구가 필요합니다.

### 1단계: 필수 구성 요소 및 프로젝트 설정
1. **Google Cloud CLI 설치:** [gcloud 설치 가이드](https://cloud.google.com/sdk/docs/install)를 따르세요.
2. **프로젝트 설정 및 결제 활성화:**
   ```bash
   gcloud init
   ```
   메시지에 따라 새 프로젝트를 생성하거나 기존 프로젝트를 선택하세요. Google Cloud Console에서 해당 프로젝트에 결제가 활성화되어 있는지 확인해야 합니다.
3. **Vertex AI API 활성화:**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

### 2단계: 인증 (Auth)
환경 변수 충돌을 방지하기 위해 Hermes 전용 Google Cloud 서비스 계정을 생성하고 자격 증명 키 파일을 다운로드하는 것이 좋습니다. 또는, 로컬 개발의 경우 애플리케이션 기본 자격 증명(ADC)을 사용할 수 있습니다.

**옵션 A: 애플리케이션 기본 자격 증명 사용 (가장 빠름)**
```bash
gcloud auth application-default login
```
이 명령어는 브라우저를 열어 인증하고 로컬 시스템에 자격 증명을 저장합니다. Hermes는 이를 자동으로 사용합니다.

**옵션 B: 서비스 계정 사용 (프로덕션/서버용 권장)**
1. 서비스 계정 생성:
   ```bash
   gcloud iam service-accounts create hermes-agent --display-name="Hermes Agent"
   ```
2. 역할(권한) 부여:
   ```bash
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:hermes-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```
3. 키 생성 및 다운로드:
   ```bash
   gcloud iam service-accounts keys create ~/.hermes/gcp-key.json \
     --iam-account=hermes-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com
   ```
4. `~/.hermes/.env` 파일에서 환경 변수 설정:
   ```env
   GOOGLE_APPLICATION_CREDENTIALS=/Users/your_username/.hermes/gcp-key.json
   # 또는 Linux의 경우: /home/your_username/.hermes/gcp-key.json
   ```

### 3단계: 환경 변수 설정 (프로젝트 및 위치)
`~/.hermes/.env` 파일에 Google Cloud 프로젝트 ID와 사용할 위치(리전)를 지정합니다:

```env
VERTEX_PROJECT_ID=your-google-cloud-project-id
VERTEX_LOCATION=us-central1 # 또는 사용 가능한 다른 리전(예: us-east1, europe-west4)
```

### 4단계: Hermes 구성
CLI 명령어를 사용하거나 `config.yaml`을 직접 편집하세요.

**명령어 사용:**
```bash
hermes model
# 프로바이더 목록에서 "Google Cloud Vertex AI"를 선택합니다
# 모델을 선택합니다 (Vertex AI 모델 이름 형식: "gemini-3.1-pro-preview")
```

**수동 구성 (`~/.hermes/config.yaml`):**
```yaml
model:
  provider: vertex-ai
  default: gemini-3.1-pro-preview
```

---

## 참고 사항 및 제한 사항
* **컨텍스트 윈도우:** Gemini 모델(특히 Pro 계열)은 최대 100만~200만 토큰의 거대한 컨텍스트 윈도우를 자랑합니다. 긴 문서, 로그, 또는 대규모 코드베이스를 분석할 때 매우 유용합니다. Hermes는 이를 인식하고 프롬프트를 압축하기 전에 최대한 이 윈도우를 활용합니다.
* **시스템 프롬프트 지원:** 두 방법 모두 최신 Gemini 모델의 시스템 프롬프트 기능을 지원하며, 이를 통해 Hermes Agent의 핵심 지침과 도구 정의가 올바르게 전달됩니다.
* **비전(Vision):** Gemini는 이미지를 처리하고 설명할 수 있습니다. Hermes를 통해 이미지를 첨부하면 기본적으로 이 모델들의 멀티모달 기능을 활용합니다.
