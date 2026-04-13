#!/usr/bin/env python3
"""
Epic Manager Tool - 대규모 프로젝트 오케스트레이션

- 하나의 거대한 Epic(목표)을 LLM을 통해 독립적인 의존성 그래프(DAG) 형태의 Task들로 자동 분할합니다.
- 분할된 Task들을 위상 정렬(Topological Sort) 규칙에 따라 순차적으로 개별 `staged_delegate`(Team mode)에게 위임합니다.
- 전체 코드베이스를 들고 다니지 않고, 각 Task의 "완료 상태 요약"만을 다음 Task의 컨텍스트로 전달하여 컨텍스트 윈도우 폭발을 방지합니다.
"""

import json
import logging
import subprocess
from typing import Any, Dict, List, Optional
from tools.delegate_tool import delegate_task
from tools.staged_delegate_tool import staged_delegate

logger = logging.getLogger(__name__)


class CycleError(Exception):
    """Task 의존성에 순환(Cycle)이 발생했을 때 던지는 예외"""
    pass


class EpicManager:
    def __init__(self, verbose: bool = True, parent_agent=None):
        self.verbose = verbose
        self.parent_agent = parent_agent

    def log(self, message: str, level: str = "INFO"):
        """레벨 기반 콘솔 출력 로깅"""
        if self.verbose:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] [EpicManager:{level}] {message}", flush=True)

    def _generate_task_graph(self, goal: str) -> List[Dict]:
        """
        Claude 등 메인 에이전트를 호출하여 목표를 티켓 단위의 JSON 그래프로 분할합니다.
        """
        prompt = f"""
        You are a Master Project Manager and Systems Architect.
        Your task is to break down the following massive project Epic into a set of highly modular, independent development tasks.
        
        EPIC GOAL:
        {goal}
        
        RULES:
        1. Break the epic into realistic, granular tasks (e.g., 3 to 15 tasks).
        2. Define dependencies clearly. If Task B requires Task A to be completed first, Task A's ID MUST be in Task B's depends_on list.
        3. Start with environment/scaffolding tasks, move to core systems, and finish with integrations or UI.
        4. Keep task descriptions highly actionable so that an independent developer agent can write the code.
        5. MANDATORY QUALITY GATE: Provide a "verification_command" that runs the automated tests (e.g., "pytest test_models.py"). If writing tests is truly impossible or unnecessary for this specific task, provide an empty string or basic shell command like "echo 'No test required'".
        6. Return ONLY a pure JSON array of objects. Do NOT include markdown tags like ```json or any other text.
        
        JSON SCHEMA:
        [
          {{
            "id": "t1",
            "title": "Project Initialization",
            "description": "Initialize standard boilerplate, git ignore, and core config...",
            "depends_on": [],
            "verification_command": "echo 'No test required'"
          }},
          {{
            "id": "t2",
            "title": "Database Models",
            "description": "Create SQLAlchemy models for User and Order. Write tests in test_models.py",
            "depends_on": ["t1"],
            "verification_command": "pytest test_models.py -v"
          }}
        ]
        """
        
        self.log("🚀 LLM을 호출하여 대규모 목표(Epic)를 기능별 Task 객체로 분할합니다...")
        
        try:
            # Hermes 내부 하위 에이전트 위임 도구를 사용하여 프롬프트 전송
            result_json = delegate_task(goal=prompt, parent_agent=self.parent_agent)
            print(f"DEBUG: delegate_task returned: {result_json}")
            parsed = json.loads(result_json)
            
            if isinstance(parsed, dict) and "results" in parsed:
                output = parsed["results"][0].get("output", "").strip()
                
                # Markdown 코드 블록 제거
                if output.startswith("```json"):
                    output = output[7:]
                elif output.startswith("```"):
                    output = output[3:]
                if output.endswith("```"):
                    output = output[:-3]
                    
                output = output.strip()
                tasks = json.loads(output)
                
                if isinstance(tasks, list):
                    return tasks
                else:
                    self.log("LLM 반환값이 JSON 리스트 형태가 아닙니다.", "ERROR")
                    return []
            else:
                self.log("delegate_task의 응답 형식을 파싱할 수 없습니다.", "ERROR")
                return []
                
        except json.JSONDecodeError as e:
            self.log(f"LLM이 유효한 JSON을 반환하지 않았습니다: {str(e)}", "ERROR")
            return []
        except Exception as e:
            self.log(f"Task Graph 생성 중 예외 발생: {str(e)}", "ERROR")
            return []

    def save_plan(self, tasks: List[Dict], filepath: str) -> bool:
        """분할된 작업을 JSON 파일로 저장합니다."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(tasks, f, indent=2, ensure_ascii=False)
            self.log(f"청사진이 {filepath} 에 저장되었습니다.")
            return True
        except Exception as e:
            self.log(f"계획 저장 실패: {e}", "ERROR")
            return False

    def load_plan(self, filepath: str) -> List[Dict]:
        """저장된 JSON 파일에서 작업을 읽어옵니다."""
        import os
        if not os.path.exists(filepath):
            self.log(f"파일을 찾을 수 없습니다: {filepath}", "ERROR")
            return []
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
            self.log(f"{filepath} 에서 {len(tasks)}개의 티켓을 로드했습니다.")
            return tasks
        except Exception as e:
            self.log(f"계획 로드 실패: {e}", "ERROR")
            return []

    def _topological_sort(self, tasks: List[Dict]) -> List[Dict]:
        """
        방향성 비순환 그래프(DAG) 형태의 Task 리스트를 위상 정렬하여 
        실행 순서(1D 리스트)로 반환합니다.
        """
        # ID Validation
        task_map = {t["id"]: t for t in tasks}
        in_degree = {t["id"]: 0 for t in tasks}
        adj_list = {t["id"]: [] for t in tasks}

        for t in tasks:
            for dep in t.get("depends_on", []):
                if dep not in task_map:
                    # Ignore invalid/missing dependencies
                    # In a strict environment, we might raise an error here.
                    continue
                adj_list[dep].append(t["id"])
                in_degree[t["id"]] += 1

        queue = [t_id for t_id in in_degree if in_degree[t_id] == 0]
        sorted_tasks = []

        while queue:
            current = queue.pop(0)
            sorted_tasks.append(task_map[current])

            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(sorted_tasks) != len(tasks):
            raise CycleError("Task 목록 내에 순환 참조(의존성 교착)가 존재합니다. 위상 정렬 실패.")

        return sorted_tasks

    def _generate_kanban_text(self, tasks: List[Dict]) -> str:
        """텔레그램 출력용 칸반 보드 마크다운 텍스트 생성"""
        if not tasks:
            return "📌 등록된 티켓이 없습니다."
            
        kanban = ["📋 **프로젝트 대기 보드 (Kanban)**", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        
        for t in tasks:
            deps = t.get("depends_on", [])
            dep_text = f"의존성: {', '.join(deps)}" if deps else "의존성: 없음"
            kanban.append(f"🎫 **[{t['id']}] {t.get('title', 'No Title')}** ({dep_text})")
            kanban.append(f"  - 📝 설명: {t.get('description', 'No explanation provided.')}")
            
            v_cmd = t.get('verification_command')
            if v_cmd:
                kanban.append(f"  - 🛡️ 검증 커맨드: `{v_cmd}`")
                
            kanban.append("") # padding
            
        kanban.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        kanban.append("💡 [안내] 텔레그램에서 'T2 티켓 삭제하고 OOO 추가해줘' 처럼 피드백하거나, 로컬에서 JSON 수정 후 '계획 승인'을 지시해주세요.")
        
        return "\n".join(kanban)

    def _generate_workspace_tree(self, root_dir: str, max_depth: int = 4) -> str:
        """가벼운 폴더/파일 계층 구조 문자열(Workspace Map)을 생성합니다."""
        import os
        
        ignore_dirs = {'.git', 'node_modules', '__pycache__', 'venv', '.env', '.venv', '.hermes', 'build', 'dist'}
        ignore_files = {'.DS_Store'}
        
        tree_lines = []
        root_dir = os.path.abspath(root_dir)
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Exclude ignored directories in-place
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
            
            # Calculate depth relative to root_dir
            rel_path = os.path.relpath(dirpath, root_dir)
            if rel_path == '.':
                depth = 0
            else:
                depth = rel_path.count(os.sep) + 1
                
            if depth > max_depth:
                dirnames.clear() # Stop descending further
                continue
                
            indent = '  ' * depth
            folder_name = os.path.basename(dirpath) if depth > 0 else os.path.basename(root_dir)
            tree_lines.append(f"{indent}📂 {folder_name}/")
            
            # Process files if depth allows
            if depth <= max_depth:
                sub_indent = '  ' * (depth + 1)
                for f in sorted(filenames):
                    if f not in ignore_files:
                        tree_lines.append(f"{sub_indent}📄 {f}")
                        
        if not tree_lines:
            return "Workspace is empty."
            
        tree_str = "\n".join(tree_lines)
        # 1000줄이 넘어가는 거대한 모노레포를 대비한 컷오프
        lines = tree_str.split("\n")
        if len(lines) > 200:
            return "\n".join(lines[:200]) + "\n... (생략됨: 트리가 너무 큽니다)"
        return tree_str

    def _generate_storyboard_elements(self, goal: str, tasks: List[Dict]) -> Dict[str, str]:
        """한 번의 보조 LLM 호출로 Mermaid, HTML, 이미지 프롬프트를 모두 추출합니다."""
        self.log("🎨 스토리보드 시안(Mermaid, HTML, Image Concept) 생성을 시작합니다...")
        try:
            from agent.auxiliary_client import get_text_auxiliary_client, auxiliary_max_tokens_param
            client, model = get_text_auxiliary_client(task="epic_storyboard")
            if not client or not model:
                return {}

            prompt = f"""
            You are an Epic Storyboard Design Agent. Generate a conceptual storyboard for the following project.
            Goal: {goal}
            Tasks: {json.dumps([{{ "id": t["id"], "title": t.get("title"), "deps": t.get("depends_on") }} for t in tasks], ensure_ascii=False)}
            
            Respond EXACTLY with a pure JSON object containing:
            1. "mermaid_chart": A valid Mermaid.js flowchart (graph TD) illustrating the project architecture or logical flow.
            2. "html_prototype": A single-page HTML String using TailwindCSS CDN representing a mockup UI for this project. If purely backend, draw a dashboard.
            3. "image_generation_prompt": A 1-sentence English prompt for an AI Image Generator to create a premium Dribbble-style UI mockup of this project.

            Do NOT wrap in ```json markers. Output only valid JSON.
            """
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **auxiliary_max_tokens_param(8000),
                temperature=0.4,
            )
            
            output = (response.choices[0].message.content or "").strip()
            if output.startswith("```json"): output = output[7:]
            elif output.startswith("```"): output = output[3:]
            if output.endswith("```"): output = output[:-3]
            
            return json.loads(output.strip())
        except Exception as e:
            self.log(f"스토리보드 생성 중 에러 발생: {e}", "ERROR")
            return {}

    def _execute_graph(self, goal: str, tasks: List[Dict]) -> Dict[str, Any]:
        """
        정렬된 Task들을 순회하며 순차적으로 staged_delegate 도구에 할당합니다.
        """
        import os
        try:
            sorted_tasks = self._topological_sort(tasks)
        except CycleError as e:
            self.log(str(e), "ERROR")
            return {"success": False, "error": str(e), "mode": "epic"}
        except Exception as e:
            self.log(f"정렬 중 에러 발생: {e}", "ERROR")
            return {"success": False, "error": str(e), "mode": "epic"}

        self.log(f"✅ 총 {len(sorted_tasks)}개의 Task가 순차 실행 대기열에 등록되었습니다.")
        
        global_context_summaries = []
        history = []
        workspace_root = os.getcwd()

        for idx, task in enumerate(sorted_tasks):
            self.log(f"\n───────────────────────────────────────────────────")
            self.log(f"🎟  진행 중인 티켓 [{idx+1}/{len(sorted_tasks)}]: {task.get('title', 'Untitled')}")
            self.log(f"───────────────────────────────────────────────────")
            
            # 매 턴마다 현재 작업장 맵을 실시간으로 다시 그립니다.
            current_tree = self._generate_workspace_tree(workspace_root)
            
            # PM 에이전트의 전체 에픽 흐름과 이전 작업 결과(성공/메모 등)만 요약으로 전달
            context_dict = {
                "epic_goal": goal,
                "system_context": "당신은 대규모 프로젝트의 일부분(Task)을 할당받은 개발자 에이전트입니다. "
                                  "Task description에 요구된 작업만 집중해서 완료하세요. "
                                  "단, 코드 작성을 শুরু하기 전 아래 제공된 `workspace_tree`를 반드시 확인하세요. "
                                  "만약 다른 에이전트가 만들어놓은 파일(의존성 변수, 함수 등)을 참조해야 한다면, 툴을 사용해 해당 파일의 내용을 먼저 읽어보고 작업하시길 바랍니다.\n"
                                  "MANDATORY TDD QUALITY GATE: 당신의 Task JSON에는 `verification_command`가 명시될 수 있습니다. 만일 명령어가 존재한다면, 코드를 짠 뒤 "
                                  "반드시 그 커맨드가 성공(종료 코드 0)을 반환할 수 있도록 관련 단위 테스트 코드까지 모조리 작성하십시오.",
                "previously_completed_tasks": "\n".join(global_context_summaries),
                "workspace_tree": current_tree
            }
            
            v_cmd = task.get("verification_command", "").strip()
            v_cmd_instructions = f"\n\nREQUIRED VERIFICATION COMMAND: {v_cmd}\n(You MUST implement the code and tests so that this command passes with 0 errors.)" if v_cmd else ""
            
            task_instruction = f"Title: {task.get('title', 'No Title')}\n\nTask Detail:\n{task.get('description', 'No Description')}{v_cmd_instructions}"
            
            # TEAM 모드로 `staged_delegate` 핵심 실행 엔진 호출 (Plan → Exec → Verify → Fix)
            # 순차 실행이므로 한 놈이 끝날 때까지 대기
            result_str = staged_delegate(goal=task_instruction, mode="team", context=context_dict, parent_agent=self.parent_agent)
            
            try:
                result = json.loads(result_str)
                if result.get("success"):
                    # Quality Gate Physical Check
                    if v_cmd:
                        import subprocess
                        self.log(f"🛡️ Quality Gate(물리 검증) 실행 중: {v_cmd}")
                        try:
                            proc = subprocess.run(v_cmd, shell=True, capture_output=True, text=True, cwd=workspace_root)
                            if proc.returncode != 0:
                                self.log(f"❌ Quality Gate 실패. (명령어: {v_cmd})\n[STDOUT] {proc.stdout}\n[STDERR] {proc.stderr}", "ERROR")
                                err_msg = f"에이전트는 성공이라 주장했으나, 실제 터미널 검증({v_cmd})이 실패했습니다 (Return code: {proc.returncode})."
                                global_context_summaries.append(f"❌ [{task['id']}] {task.get('title')} - FAILED: {err_msg}")
                                history.append({
                                    "id": task["id"], 
                                    "title": task.get("title"),
                                    "status": "failed", 
                                    "error": err_msg
                                })
                                break
                            else:
                                self.log(f"✅ Quality Gate 통과. (종료 코드 0)")
                        except Exception as sub_e:
                            self.log(f"❌ Quality Gate 실행 자체 실패: {sub_e}", "ERROR")
                            history.append({"id": task["id"], "title": task.get("title"), "status": "failed", "error": f"Gate Error: {sub_e}"})
                            break
                    
                    self.log(f"✅ 해당 Ticket 완전 성공: {task.get('title')}")
                    global_context_summaries.append(f"☑️ [{task['id']}] {task.get('title')} - SUCCESSfully implemented.")
                    history.append({
                        "id": task["id"],
                        "title": task.get("title"),
                        "status": "success",
                        "iterations": result.get("iterations", 0)
                    })
                else:
                    err_msg = result.get('error', 'Unknown error during team delegation.')
                    self.log(f"❌ Ticket 실패: {task.get('title')} (사유: {err_msg})", "ERROR")
                    global_context_summaries.append(f"❌ [{task['id']}] {task.get('title')} - FAILED: {err_msg}")
                    history.append({
                        "id": task["id"], 
                        "title": task.get("title"),
                        "status": "failed", 
                        "error": err_msg
                    })
                    
                    # 옵션: 실패 시 전체 Epic 중단
                    self.log("🚨 의존성 파이프라인 안전을 위해 연쇄 작업을 중단합니다.", "WARN")
                    break
                    
            except Exception as e:
                self.log(f"Ticket 수행 결과 파싱 에러: {str(e)}", "ERROR")
                history.append({"id": task["id"], "title": task.get("title"), "status": "failed", "error": str(e)})
                break
                
        # 모든 Task 성공 여부 확인
        all_success = (len(history) == len(tasks)) and all(h["status"] == "success" for h in history)
        
        self.log(f"\n🏁 Epic 오케스트레이션 종료 (Success: {all_success})")
        return {
            "success": all_success,
            "mode": "epic",
            "total_tasks": len(tasks),
            "completed_tasks": len([h for h in history if h["status"] == "success"]),
            "history": history
        }


def epic_delegate(goal: str, mode: str = "plan_only", parent_agent=None) -> str:
    """
    Epic Delegate Tool - Main Entry Point
    
    Args:
        goal: 거대한 대규모 프로젝트 목표/명세
        mode: "plan_only" (설계도 작성 후 일시정지) 또는 "execute" (설계도 기반 실행)
    
    Returns:
        JSON 결과 문자열
    """
    import os
    manager = EpicManager(verbose=True, parent_agent=parent_agent)
    plan_file = os.path.join(os.getcwd(), "epic_plan.json")
    
    if mode == "plan_only":
        manager.log("모드: plan_only (티켓 설계도 추출을 시작합니다)")
        tasks = manager._generate_task_graph(goal)
        if not tasks:
            return json.dumps({"success": False, "mode": "epic", "error": "LLM failed to generate a valid task DAG graph."}, ensure_ascii=False)
        
        saved = manager.save_plan(tasks, plan_file)
        if saved:
            kanban_display = manager._generate_kanban_text(tasks)
            msg = f"설계도가 {plan_file}에 생성되었습니다.\n\n{kanban_display}"
            
            # --- Visual Storyboarding Expansion ---
            board_elements = manager._generate_storyboard_elements(goal, tasks)
            if board_elements:
                msg += "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n🌟 **Visual Storyboard Blueprint** 🌟\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                
                # 1. Mermaid Chart
                m_chart = board_elements.get("mermaid_chart")
                if m_chart:
                    msg += f"\n\n### 🗺️ System Architecture\n```mermaid\n{m_chart}\n```\n"
                
                # 2. HTML Prototype Dump
                html_code = board_elements.get("html_prototype")
                if html_code:
                    wireframe_file = os.path.join(os.getcwd(), "epic_wireframe_prototype.html")
                    try:
                        with open(wireframe_file, "w", encoding="utf-8") as f:
                            f.write(html_code)
                        msg += f"\n### 🖥️ Interactive Sandbox (Prototype)\n> 📄 와이어프레임 파일이 생성되었습니다: `{wireframe_file}`\n> 파일을 로컬 브라우저에서 열어보실 수 있습니다.\n"
                    except Exception as e:
                        manager.log(f"HTML 쓰기 에러: {e}", "ERROR")

                # 3. AI Generated Visual Mockup (if image gen supported)
                img_prompt = board_elements.get("image_generation_prompt")
                if img_prompt:
                    from tools.image_generation_tool import check_image_generation_requirements, image_generate_tool
                    if check_image_generation_requirements():
                        manager.log(f"🖼️ 이미지 시안 생성 중...: {img_prompt}")
                        try:
                            # 10~20초 소요 됨을 감안
                            img_result_str = image_generate_tool(
                                prompt=img_prompt,
                                aspect_ratio="landscape",
                                output_format="png"
                            )
                            img_res = json.loads(img_result_str)
                            if img_res.get("success") and img_res.get("image"):
                                msg += f"\n### 🎨 Conceptual UI Mockup\n![Project Concept Design]({img_res['image']})\n"
                        except Exception as e:
                            manager.log(f"이미지 목업 렌더링 에러: {e}", "ERROR")

            return json.dumps({
                "success": True, 
                "mode": "plan_only",
                "message": msg
            }, ensure_ascii=False)
        else:
            return json.dumps({"success": False, "mode": "epic", "error": "Failed to save epic_plan.json"}, ensure_ascii=False)
            
    elif mode == "execute":
        manager.log("모드: execute (청사진 기반 순차 코딩을 시작합니다)")
        tasks = manager.load_plan(plan_file)
        if not tasks:
            return json.dumps({"success": False, "mode": "epic", "error": f"{plan_file} 에서 작업을 불러오지 못했습니다. plan_only 모드를 먼저 실행하세요."}, ensure_ascii=False)
            
        result = manager._execute_graph(goal, tasks)
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    else:
        return json.dumps({"success": False, "mode": "epic", "error": f"Unknown mode: {mode}"}, ensure_ascii=False)


# Register with Hermes tool registry
try:
    from tools.registry import registry

    def _check_epic_delegate() -> bool:
        return True

    registry.register(
        name="epic_delegate",
        toolset="delegation",
        schema={
            "name": "epic_delegate",
            "description": "대규모 소프트웨어 프로젝트 매핑(Epic Manager) 시스템. 큰 목표를 DAG 형태의 기능 티켓으로 분할(plan_only)한 뒤, 승인 후 순차 실행(execute)시켜 프로덕트를 완성합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "진행할 프로젝트의 목표입니다. execute 모드일지라도 전체 컨텍스트용으로 제공해야 합니다."
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["plan_only", "execute"],
                        "description": "plan_only는 JSON 티켓 파일(epic_plan.json)만 바탕화면에 생성하고 대기합니다. execute는 승인 후 실행합니다."
                    }
                },
                "required": ["goal"]
            }
        },
        handler=lambda args, **kw: epic_delegate(
            goal=args.get("goal"),
            mode=args.get("mode", "plan_only"),
            parent_agent=kw.get("parent_agent")
        ),
        check_fn=_check_epic_delegate,
        description="Epic Level Large-Scale Orchestration Strategy using Task DAGs",
        emoji="👑",
    )
except ImportError:
    pass
