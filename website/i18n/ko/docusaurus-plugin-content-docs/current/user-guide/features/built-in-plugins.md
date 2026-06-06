---
sidebar_position: 21
title: "내장 플러그인"
description: "Hermes의 선택적 기능 및 통합을 제공하는 핵심 플러그인 패키지입니다."
---

# 내장 플러그인 (Built-in Plugins)

Hermes는 특정 에코시스템 및 도구와의 통합을 제공하는 몇 가지 내장 플러그인과 함께 제공됩니다. 이러한 플러그인은 핵심 플랫폼과 함께 유지 관리되지만 오버헤드를 줄이기 위해 기본적으로 비활성화되어 있습니다.

대시보드(Plugins 탭) 또는 CLI(`hermes plugins`)를 사용하여 플러그인을 활성화, 비활성화 및 구성할 수 있습니다.

```bash
hermes plugins enable github-integration
hermes plugins configure github-integration
```

---

## Chrome DevTools Plugin (`chrome-devtools-plugin`)

Chrome DevTools Protocol(CDP)을 통해 로컬 Chrome 브라우저의 고급 디버깅 및 제어를 제공합니다.

이 플러그인은 MCP(Model Context Protocol)를 사용하여 로컬 Chrome 브라우저를 디버깅 대상(debugging target)으로 노출시킵니다. 에이전트가 성능을 진단하고, 네트워크 요청을 검사하며, 실행 중인 웹 애플리케이션의 메모리 누수를 프로파일링할 수 있도록 설계되었습니다.

**주요 기능:**
- **실시간 디버깅:** 콘솔 로그 검사, 중단점 설정 및 브라우저에서 직접 JavaScript 평가
- **네트워크 검사:** 실패한 요청 포착, 응답 헤더 분석 및 네트워크 병목 현상 조사
- **성능 프로파일링:** 타임라인 트레이스 기록, 렌더링 성능 분석 및 LCP 문제 해결
- **메모리 분석:** 힙 스냅샷 캡처, 메모리 누수 감지 및 DOM 노드 사용량 추적

이 플러그인은 개발자의 머신에서 직접 프론트엔드 문제를 진단할 때 `browser` 도구 세트보다 우선적으로 사용해야 합니다.

---

## Modern Web Guidance (`modern-web-guidance-plugin`)

최신 웹 개발 표준 및 모범 사례로 에이전트의 지식을 향상시킵니다.

웹 API와 프레임워크는 빠르게 발전하므로 에이전트의 기본 훈련 가중치에 오래된 패턴(obsolete patterns)이 포함되어 있는 경우가 많습니다. 이 플러그인은 최신 브라우저 기능, CSS 발전, 성능 최적화 및 접근성 표준에 대한 최신 컨텍스트를 제공합니다.

**주요 기능:**
- **최신 CSS/UI 패턴:** 앵커 포지셔닝(anchor positioning), 컨테이너 쿼리(container queries), `:has()`, 뷰 전환(view transitions) 및 스크롤 기반 애니메이션
- **성능 지침(Performance Guidance):** Core Web Vitals(CWV), `content-visibility`, fetch priority 및 이미지 최적화
- **최신 브라우저 API:** 로컬 파일 시스템 액세스, WebUSB, WebSockets 및 WebAssembly 통합
- **프레임워크 적용:** React, Vue 및 Angular 환경에서 최신 표준 구현

에이전트는 HTML, CSS 또는 클라이언트 측 JavaScript 관련 작업을 수행할 때 이 플러그인을 먼저 참고해야 합니다.

---

## Android CLI Plugin (`android-cli-plugin`)

명령줄 도구를 사용하여 Android 개발 및 기기 관리 워크플로우를 오케스트레이션(orchestrate)합니다.

이 플러그인은 `adb`(Android Debug Bridge), `sdkmanager` 및 기타 Android 툴체인 유틸리티에 대한 인터페이스를 제공합니다. 에이전트가 에뮬레이터 또는 물리적 기기와 상호 작용하여 Android 앱 테스트, 배포 및 디버깅을 자동화할 수 있습니다.

**주요 기능:**
- **기기 관리:** 에뮬레이터 또는 연결된 기기 시작, 중지 및 상태 확인
- **앱 라이프사이클:** 기기에 APK 설치, 제거 및 시작
- **디버깅:** Logcat 출력 수집, 크래시 로그 분석 및 화면 상의 활동 검사
- **UI 자동화:** 텍스트 입력, 탭 및 스와이프를 위한 입력 이벤트 전송
- **SDK 관리:** 필수 Android SDK 패키지, 빌드 도구 및 시스템 이미지 설치

이 플러그인은 에이전트가 로컬 또는 CI 환경에서 Android 애플리케이션을 빌드하거나 테스트해야 할 때 매우 중요합니다.

---

## Firebase (`firebase`)

에이전트가 Firebase 프로젝트를 배포, 구성 및 관리할 수 있도록 포괄적인 Firebase 통합을 제공합니다.

이 플러그인은 Firebase CLI 및 관리 도구를 래핑하여 에이전트가 인증부터 서버리스 백엔드, 데이터베이스 설계에 이르기까지 복잡한 Firebase 아키텍처를 스캐폴딩(scaffold)하고 유지 관리할 수 있도록 합니다.

**주요 기능:**
- **프로젝트 구성:** Firebase 환경 초기화 및 `google-services.json` / `GoogleService-Info.plist` 구성
- **인증 설정:** Firebase Auth 규칙 구성 및 로그인 플로우 구현 지침
- **Firestore 관리:** 보안 규칙 설계 및 감사, 색인 구성 및 쿼리 최적화
- **데이터베이스 백엔드(Data Connect):** Firebase Data Connect(PostgreSQL)를 사용하여 관계형 백엔드 설계 및 배포
- **호스팅 및 배포:** 정적 앱용 Firebase Hosting 및 서버 사이드 렌더링 프레임워크(Next.js/Angular)용 App Hosting 관리
- **추가 서비스:** Crashlytics 및 Remote Config 기능 설정 지원

Firebase를 기반으로 하는 앱(웹, 모바일, 또는 백엔드)을 구축할 때 이 플러그인은 인프라 관리를 위한 에이전트의 주요 도구가 됩니다.
