---
sidebar_position: 18
title: "대시보드 확장"
description: "커스텀 플러그인을 사용하여 React/Vite 웹 대시보드에 탭, 화면 및 컴포넌트를 추가합니다."
---

# 대시보드 확장 (Extending the Dashboard)

기본 Hermes 웹 대시보드는 플러그인 관리, 자격 증명 설정, 세션 로그 및 프로필 편집과 같은 핵심 에이전트 관리에 중점을 둡니다.

플러그인, 스킬 또는 통합 패키지를 개발하는 경우, 에이전트 사용자 지정 데이터(Custom data), 시각화 및 컨트롤을 렌더링하기 위해 대시보드에 고유한 탭과 화면을 주입하고 싶을 수 있습니다.

Hermes는 **프론트엔드 플러그인(Frontend Plugins)**을 지원합니다. 모든 Python 플러그인은 로컬 React 컴포넌트를 등록할 수 있으며, 이 컴포넌트는 메인 대시보드 앱에 원활하게 번들되어 렌더링됩니다.

## 1. 컴포넌트 마운트 지점 (Component Mount Points)

메인 앱은 확장을 위해 몇 가지 특정 영역을 엽니다. 가장 일반적인 것은 다음과 같습니다:

- `root.tabs` — "플러그인(Plugins)" 옆 상단 탐색 표시줄의 새 탭
- `agent.metrics` — 활성 에이전트의 현재 실행 통계에 패널 삽입
- `settings.panel` — 환경 설정(Settings) 대화 상자의 커스텀 영역

이름이 지정된 익스포트(exports)를 사용하여 컴포넌트를 선언합니다.

## 2. 프론트엔드 코드 작성

플러그인 폴더 안에 `ui/` 폴더를 만들고 진입점(`index.tsx`)을 추가합니다.

```tsx
// my-plugin/ui/index.tsx
import React, { useState, useEffect } from 'react';
// 주입된 컴포넌트는 모든 메인 앱 스타일, 테마 및 tailwind 클래스에 액세스할 수 있습니다.
import { Card, Button, Metric } from '@hermes/ui-kit';

export function RootTab() {
  const [data, setData] = useState(null);

  useEffect(() => {
    // 상대 API 호출은 Hermes의 백엔드로 안전하게 라우팅됩니다.
    // (Python 플러그인이 FastAPI 라우트를 등록했다고 가정)
    fetch('/api/plugins/my-plugin/stats')
      .then(r => r.json())
      .then(setData);
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">My Custom Dashboard</h1>
      <Card>
        <Metric label="Items Processed" value={data?.processed || 0} />
        <Button onClick={() => fetch('/api/plugins/my-plugin/trigger', { method: 'POST' })}>
          Trigger Work
        </Button>
      </Card>
    </div>
  );
}

// Hermes가 이것을 'root.tabs' 익스텐션으로 마운트해야 한다는 것을 알 수 있도록
// default export에 마운트 구성을 선언합니다.
export default {
  mounts: {
    'root.tabs': {
      component: RootTab,
      label: 'My Plugin',
      icon: 'PuzzlePieceIcon',
      path: '/my-plugin',
    }
  }
};
```

*참고: TypeScript 및 JSX는 지원됩니다. 빌드 도구 체인이 이를 자동으로 컴파일합니다.*

## 3. 플러그인 매니페스트 (Plugin Manifest)

다음으로 플러그인의 루트(Python 파일 옆)에 `manifest.json`을 생성하여 Hermes에게 프론트엔드를 찾을 위치를 알려줍니다:

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "frontend": {
    "entry": "ui/index.tsx"
  }
}
```

## 4. 백엔드 API (선택 사항)

종종 UI는 표시할 데이터가 필요합니다. 플러그인의 Python `__init__.py`에서 대시보드가 호출할 수 있는 커스텀 FastAPI 라우트를 노출하세요:

```python
from hermes_agent.plugin import hookimpl

@hookimpl
def register_routes(app):
    @app.get("/api/plugins/my-plugin/stats")
    async def get_stats():
        return {"processed": 42}

    @app.post("/api/plugins/my-plugin/trigger")
    async def trigger_work():
        # 에이전트 작업 또는 내부 로직 수행
        return {"status": "started"}
```

## 5. 빌드 및 배포

Hermes 대시보드는 Vite를 사용합니다. 에이전트가 시작되고 설치된 플러그인에 `frontend.entry`가 매니페스트에 선언되어 있는 것을 감지하면 동적으로 주입하고 프론트엔드를 다시 빌드합니다.

개발 중 UI 변경 사항을 실시간으로 확인하려면 개발 환경에서 대시보드를 실행하세요:

```bash
# 터미널 1
hermes server

# 터미널 2 (Hermes 저장소 내부에서 실행)
cd website
npm run dev
```

이제 `ui/index.tsx`를 저장하면 변경 사항이 대시보드에 즉시 반영(HMR)됩니다.

## 제한 사항 (Limitations)

- **종속성**: 플러그인은 추가 `package.json` 종속성을 선언할 수 없습니다. 컴포넌트는 메인 대시보드의 기존 React 종속성(Tailwind, Framer Motion, Lucide 등)을 재사용해야 합니다.
- **라우팅**: 주입된 탭은 플러그인 ID를 접두사로 사용하는 네임스페이스 하위에 위치합니다. `root.tabs`의 경우 위의 `path: '/my-plugin'`은 `/plugins/my-plugin`에서 렌더링됩니다.
