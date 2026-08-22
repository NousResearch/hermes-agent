---
sidebar_position: 13
title: "Delegation & งานแบบขนาน"
description: "เมื่อไหร่และใช้ subagent delegation อย่างไร — pattern สำหรับการวิจัยแบบขนาน การรีวิวโค้ด และงานหลายไฟล์"
---

# Delegation & งานแบบขนาน

Hermes spawn child agents ที่แยกขาดจากกันออกมาทำงานแบบขนานได้ subagent แต่ละตัวมี conversation, terminal session และ toolset เป็นของตัวเอง สิ่งที่ส่งกลับมามีเพียง final summary — intermediate tool calls จะไม่เข้าสู่ context window ของคุณเลย

สำหรับข้อมูลอ้างอิงฟีเจอร์ฉบับเต็ม ดูที่ [Subagent Delegation](/user-guide/features/delegation)

---

## เมื่อไหร่ที่ควร Delegate

**งานที่เหมาะกับ delegation:**
- Subtask ที่ต้องใช้การให้เหตุผลหนัก (debugging, code review, research synthesis)
- Task ที่จะทำให้ context ของคุณล้นด้วยข้อมูลระหว่างทาง
- Workstream ที่เป็นอิสระต่อกันและรันขนานกันได้ (วิจัย A และ B พร้อมกัน)
- Task แบบ fresh-context ที่คุณอยากให้ agent เข้าหางานโดยไร้อคติ

**ใช้อย่างอื่นแทน:**
- Single tool call → ใช้ tool นั้นตรงๆ ไปเลย
- งานหลายขั้นแบบกลไกที่มี logic คั่นระหว่างขั้น → `execute_code`
- Task ที่ต้องโต้ตอบกับ user → subagent ใช้ `clarify` ไม่ได้
- แก้ไฟล์เร็วๆ → ทำเองตรงๆ
- งานรันยาวที่ต้องรอดจากการปิด session หรือ process restart → `cronjob` หรือ `terminal(background=True, notify_on_complete=True)` Delegation ระดับ top-level เป็น asynchronous แต่ยังผูกกับ process เดิม

---

## Pattern: วิจัยแบบขนาน

วิจัยสามหัวข้อพร้อมกันแล้วรับ structured summary กลับมา:

```
Research these three topics in parallel:
1. Current state of WebAssembly outside the browser
2. RISC-V server chip adoption in 2025
3. Practical quantum computing applications

Focus on recent developments and key players.
```

เบื้องหลัง Hermes ใช้:

```python
delegate_task(tasks=[
    {
        "goal": "Research WebAssembly outside the browser in 2025",
        "context": "Focus on: runtimes (Wasmtime, Wasmer), cloud/edge use cases, WASI progress"
    },
    {
        "goal": "Research RISC-V server chip adoption",
        "context": "Focus on: server chips shipping, cloud providers adopting, software ecosystem"
    },
    {
        "goal": "Research practical quantum computing applications",
        "context": "Focus on: error correction breakthroughs, real-world use cases, key companies"
    }
])
```

ทั้งสามส่วนรันพร้อมกัน subagent แต่ละตัวค้นเว็บอย่างอิสระแล้วส่ง summary กลับมา จากนั้น parent agent จะ synthesize ทั้งหมดเป็น briefing เดียวที่ต่อเนื่องกัน

---

## Pattern: รีวิวโค้ด

มอบ security review ให้ subagent แบบ fresh-context ที่เข้าถึงโค้ดโดยไม่มีความเข้าใจล่วงหน้า:

```
Review the authentication module at src/auth/ for security issues.
Check for SQL injection, JWT validation problems, password handling,
and session management. Fix anything you find and run the tests.
```

จุดสำคัญคือฟิลด์ `context` — ต้องบรรจุทุกอย่างที่ subagent ต้องการ:

```python
delegate_task(
    goal="Review src/auth/ for security issues and fix any found",
    context="""Project at /home/user/webapp. Python 3.11, Flask, PyJWT, bcrypt.
    Auth files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py
    Test command: pytest tests/auth/ -v
    Focus on: SQL injection, JWT validation, password hashing, session management.
    Fix issues found and verify tests pass."""
)
```

:::warning The Context Problem
Subagents ไม่รู้**อะไรเลย**เกี่ยวกับ conversation ของคุณ พวกมันเริ่มจากศูนย์ทั้งหมด ถ้าคุณ delegate งานว่า "แก้ bug ที่เรากำลังคุยกัน" subagent จะไม่รู้เลยว่าคุณหมายถึง bug ตัวไหน ส่ง file path, error message, project structure และ constraint แบบชัดเจนทุกครั้ง
:::

---

## Pattern: เปรียบเทียบทางเลือก

ประเมินหลายแนวทางสำหรับปัญหาเดียวกันแบบขนาน แล้วค่อยเลือกที่ดีที่สุด:

```
I need to add full-text search to our Django app. Evaluate three approaches
in parallel:
1. PostgreSQL tsvector (built-in)
2. Elasticsearch via django-elasticsearch-dsl
3. Meilisearch via meilisearch-python

For each: setup complexity, query capabilities, resource requirements,
and maintenance overhead. Compare them and recommend one.
```

Subagent แต่ละตัววิจัยหนึ่งทางเลือกอย่างอิสระ เพราะแยกขาดจากกัน จึงไม่มี cross-contamination — การประเมินแต่ละครั้งยืนอยู่บนคุณค่าของตัวเอง Parent agent ได้ summary ทั้งสามมาแล้วทำการเปรียบเทียบ

---

## Pattern: Refactor หลายไฟล์

แบ่งงาน refactor ขนาดใหญ่ให้ subagents ทำงานขนานกัน โดยแต่ละตัวรับผิดชอบคนละส่วนของ codebase:

```python
delegate_task(tasks=[
    {
        "goal": "Refactor all API endpoint handlers to use the new response format",
        "context": """Project at /home/user/api-server.
        Files: src/handlers/users.py, src/handlers/auth.py, src/handlers/billing.py
        Old format: return {"data": result, "status": "ok"}
        New format: return APIResponse(data=result, status=200).to_dict()
        Import: from src.responses import APIResponse
        Run tests after: pytest tests/handlers/ -v"""
    },
    {
        "goal": "Update all client SDK methods to handle the new response format",
        "context": """Project at /home/user/api-server.
        Files: sdk/python/client.py, sdk/python/models.py
        Old parsing: result = response.json()["data"]
        New parsing: result = response.json()["data"] (same key, but add status code checking)
        Also update sdk/python/tests/test_client.py"""
    },
    {
        "goal": "Update API documentation to reflect the new response format",
        "context": """Project at /home/user/api-server.
        Docs at: docs/api/. Format: Markdown with code examples.
        Update all response examples from old format to new format.
        Add a 'Response Format' section to docs/api/overview.md explaining the schema."""
    }
])
```

:::tip
Subagent แต่ละตัวมี terminal session ของตัวเอง ทำงานใน project directory เดียวกันได้โดยไม่เหยียบเท้ากัน — ตราบใดที่แก้ไฟล์คนละไฟล์ ถ้ามีโอกาสที่ subagent สองตัวจะแตะไฟล์เดียวกัน ให้จัดการไฟล์นั้นด้วยตัวเองหลังงานขนานจบ
:::

---

## Pattern: เก็บข้อมูลก่อนแล้ววิเคราะห์

ใช้ `execute_code` สำหรับการเก็บข้อมูลแบบกลไก แล้วค่อย delegate งานวิเคราะห์ที่ต้องใช้การให้เหตุผลหนัก:

```python
# Step 1: Mechanical gathering (execute_code is better here — no reasoning needed)
execute_code("""
from hermes_tools import web_search, web_extract

results = []
for query in ["AI funding Q1 2026", "AI startup acquisitions 2026", "AI IPOs 2026"]:
    r = web_search(query, limit=5)
    for item in r["data"]["web"]:
        results.append({"title": item["title"], "url": item["url"], "desc": item["description"]})

# Extract full content from top 5 most relevant
urls = [r["url"] for r in results[:5]]
content = web_extract(urls)

# Save for the analysis step
import json
with open("/tmp/ai-funding-data.json", "w") as f:
    json.dump({"search_results": results, "extracted": content["results"]}, f)
print(f"Collected {len(results)} results, extracted {len(content['results'])} pages")
""")

# Step 2: Reasoning-heavy analysis (delegation is better here)
delegate_task(
    goal="Analyze AI funding data and write a market report",
    context="""Raw data at /tmp/ai-funding-data.json contains search results and
    extracted web pages about AI funding, acquisitions, and IPOs in Q1 2026.
    Write a structured market report: key deals, trends, notable players,
    and outlook. Focus on deals over $100M."""
)
```

นี่มักเป็น pattern ที่คุ้มค่าที่สุด: `execute_code` จัดการ tool call ลำดับ 10+ ครั้งได้แบบประหยัด แล้ว subagent จะทำงานให้เหตุผลราคาแพงเพียงครั้งเดียวด้วย context ที่สะอาด

---

## การสืบทอดการเข้าถึง Tool

Subagent สืบทอด toolset ที่ parent เปิดใช้งานอยู่ `delegate_task` ไม่รับพารามิเตอร์ `toolsets` ที่ model มองเห็น งานที่ถูก delegate จึงให้ capability ที่ parent ไม่มีแก่ตัวเองไม่ได้ ถ้างานที่ delegate ต้องใช้ web, terminal, file หรือ access อื่น ให้ตั้งค่า tools ของ parent ก่อนเริ่มบทสนทนา Hermes ยังคง strip tools ที่ห้ามสำหรับ child เช่น `clarify`, `memory` และ `send_message` ออก ส่วน children เก็บ `execute_code` ไว้สำหรับ programmatic tool calling

---

## ข้อจำกัด

- **ค่าเริ่มต้น 3 parallel task**: batch เริ่มต้นที่ subagent ขนานพร้อมกัน 3 ตัว (ปรับได้ผ่าน `delegation.max_concurrent_children` ใน config.yaml ไม่มีเพดานจำกัดสูงสุด มีเพียงค่าต่ำสุดเท่ากับ 1)
- **Nested delegation เป็น opt-in**: leaf subagent (ค่าเริ่มต้น) เรียก `delegate_task`, `clarify`, `memory` หรือ `execute_code` ไม่ได้ Orchestrator subagent (`role="orchestrator"`) ยังคงมี `delegate_task` ไว้ delegate งานต่อได้ แต่ต่อเมื่อเพิ่ม `delegation.max_spawn_depth` ให้มากกว่าค่าเริ่มต้น 1 (ต่ำสุด 1 ไม่มีเพดาน) อีกสาม tool ยังถูกบล็อกอยู่ ปิดทั้งระบบผ่าน `delegation.orchestrator_enabled: false`

### ปรับ Concurrency และ Depth

| Config | ค่าเริ่มต้น | ช่วงค่า | ผล |
|--------|---------|-------|--------|
| `max_concurrent_children` | 3 | >=1 | ขนาด batch ขนานต่อการเรียก `delegate_task` หนึ่งครั้ง |
| `max_spawn_depth` | 1 | >=1 | จำนวนชั้น delegation ที่ spawn ต่อได้อีก |

ตัวอย่าง: รัน worker ขนาน 30 ตัวพร้อม nested subagent:

```yaml
delegation:
  max_concurrent_children: 30
  max_spawn_depth: 2
```

- **Terminal แยกกัน** — subagent แต่ละตัวมี terminal session ของตัวเอง พร้อม working directory และ state แยกจากกัน
- **ไม่มีประวัติบทสนทนา** — subagent เห็นเฉพาะ `goal` และ `context` ที่ parent agent ส่งผ่านมาตอนเรียก `delegate_task`
- **ค่าเริ่มต้น 50 iteration** — ตั้ง `max_iterations` ให้ต่ำลงสำหรับ task ง่ายๆ เพื่อประหยัดค่าใช้จ่าย
- **ไม่ทนทาน (not durable)** — top-level delegation รันเบื้องหลังและส่งผลลัพธ์กลับมาทีหลัง แต่ยังผูกอยู่กับ session และ process ของ Hermes ที่เป็นเจ้าของ การปิด session, `/stop`, `/new` หรือ process restart อาจยกเลิกหรือทำให้งานที่กำลังรันค้างคา งานที่ต้องรอดข้ามขอบเขตเหล่านี้ ให้ใช้ `cronjob` หรือ `terminal(background=True, notify_on_complete=True)`

---

## เคล็ดลับ

**ระบุ goal ให้ชัด** "Fix the bug" กว้างเกินไป "Fix the TypeError in api/handlers.py line 47 where process_request() receives None from parse_body()" ให้ข้อมูลเพียงพอแก่ subagent ที่จะลงมือได้เลย

**ใส่ file path มาด้วย** Subagents ไม่รู้ project structure ของคุณ ใส่ absolute path ของไฟล์ที่เกี่ยวข้อง, project root และ test command ทุกครั้ง

**ใช้ delegation เพื่อแยก context** บางครั้งคุณอยากได้มุมมองสดใหม่ การ delegate บังคับให้คุณอธิบายปัญหาให้ชัด และ subagent จะเข้าหาปัญหาโดยไม่ติดสมมติฐานที่สะสมมาจาก conversation ของคุณ

**ตรวจผลลัพธ์** Summary ของ subagent ก็เป็นแค่ summary ถ้า subagent บอกว่า "fixed the bug and tests pass" ให้ยืนยันด้วยการรัน tests เอง หรืออ่าน diff

---

*สำหรับข้อมูลอ้างอิง delegation ฉบับสมบูรณ์ — พารามิเตอร์ทั้งหมด, ACP integration และการตั้งค่าขั้นสูง — ดูที่ [Subagent Delegation](/user-guide/features/delegation).*
