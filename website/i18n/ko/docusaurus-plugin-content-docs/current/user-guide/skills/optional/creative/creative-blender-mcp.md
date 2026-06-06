---
title: "Blender Mcp — blender-mcp 애드온에 대한 소켓 연결을 통해 Hermes에서 직접 Blender 제어"
sidebar_label: "Blender Mcp"
description: "blender-mcp 애드온에 대한 소켓 연결을 통해 Hermes에서 직접 Blender 제어"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Blender Mcp

blender-mcp 애드온에 대한 소켓 연결을 통해 Hermes에서 직접 Blender를 제어합니다. 3D 객체, 재질, 애니메이션을 만들고 임의의 Blender Python(bpy) 코드를 실행합니다. 사용자가 Blender에서 무언가를 생성하거나 수정하려고 할 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/blender-mcp`를 사용하여 설치 |
| 경로 | `optional-skills/creative/blender-mcp` |
| 버전 | `1.0.0` |
| 작성자 | alireza78a |
| 플랫폼 | linux, macos, windows |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Blender MCP

TCP 포트 9876의 소켓을 통해 Hermes에서 실행 중인 Blender 인스턴스를 제어합니다.

## 설정 (일회성)

### 1. Blender 애드온 설치

    curl -sL https://raw.githubusercontent.com/ahujasid/blender-mcp/main/addon.py -o ~/Desktop/blender_mcp_addon.py

Blender에서:
    Edit > Preferences > Add-ons > Install > blender_mcp_addon.py 선택
    "Interface: Blender MCP" 활성화

### 2. Blender에서 소켓 서버 시작

Blender 뷰포트에서 N을 눌러 사이드바를 엽니다.
"BlenderMCP" 탭을 찾아서 "Start Server"를 클릭합니다.

### 3. 연결 확인

    nc -z -w2 localhost 9876 && echo "OPEN" || echo "CLOSED"

## 프로토콜

TCP 상의 일반 UTF-8 JSON -- 길이 접두사가 없습니다.

송신:     &#123;"type": "&lt;command>", "params": &#123;&lt;kwargs>&#125;&#125;
수신:     &#123;"status": "success", "result": &lt;value>&#125;
          &#123;"status": "error",   "message": "&lt;reason>"&#125;

## 사용 가능한 명령어

| type                    | params            | description                     |
|-------------------------|-------------------|---------------------------------|
| execute_code            | code (str)        | 임의의 bpy Python 코드 실행       |
| get_scene_info          | (없음)            | 씬의 모든 객체 나열               |
| get_object_info         | object_name (str) | 특정 객체에 대한 세부 정보        |
| get_viewport_screenshot | (없음)            | 현재 뷰포트의 스크린샷            |

## Python 헬퍼

execute_code 도구 호출 내에서 다음을 사용하세요:

    import socket, json

    def blender_exec(code: str, host="localhost", port=9876, timeout=15):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.settimeout(timeout)
        payload = json.dumps(&#123;"type": "execute_code", "params": &#123;"code": code&#125;&#125;)
        s.sendall(payload.encode("utf-8"))
        buf = b""
        while True:
            try:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
                try:
                    json.loads(buf.decode("utf-8"))
                    break
                except json.JSONDecodeError:
                    continue
            except socket.timeout:
                break
        s.close()
        return json.loads(buf.decode("utf-8"))

## 일반적인 bpy 패턴

### 씬 지우기
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

### 메쉬 객체 추가
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))
    bpy.ops.mesh.primitive_cube_add(size=2, location=(3, 0, 0))
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2, location=(-3, 0, 0))

### 재질 생성 및 할당
    mat = bpy.data.materials.new(name="MyMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (R, G, B, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.3
    bsdf.inputs["Metallic"].default_value = 0.0
    obj.data.materials.append(mat)

### 키프레임 애니메이션
    obj.location = (0, 0, 0)
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location = (0, 0, 3)
    obj.keyframe_insert(data_path="location", frame=60)

### 파일로 렌더링
    bpy.context.scene.render.filepath = "/tmp/render.png"
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.ops.render.render(write_still=True)

## 주의 사항 (Pitfalls)

- 실행하기 전에 소켓이 열려 있는지 확인해야 합니다 (nc -z localhost 9876)
- 애드온 서버는 각 세션마다 Blender 내에서 시작되어야 합니다 (N-패널 > BlenderMCP > Connect)
- 시간 초과를 피하기 위해 복잡한 씬은 여러 개의 더 작은 execute_code 호출로 나누세요
- 렌더링 출력 경로는 상대 경로가 아닌 절대 경로(/tmp/...)여야 합니다
- shade_smooth()를 사용하려면 객체가 선택되어 있고 객체 모드(object mode)에 있어야 합니다
