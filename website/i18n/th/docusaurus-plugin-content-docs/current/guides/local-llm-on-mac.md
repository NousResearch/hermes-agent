---
sidebar_position: 2
title: "รัน Local LLM บน Mac"
description: "ตั้งค่า local LLM server ที่ใช้ API แบบเดียวกับ OpenAI บน macOS ด้วย llama.cpp หรือ MLX ครอบคลุมการเลือกโมเดล การปรับหน่วยความจำ และผล benchmark จริงบน Apple Silicon"
---

# การรัน Local LLM บน Mac

คู่มือนี้พาคุณไปทีละขั้นเรื่องการรัน local LLM server บน macOS ที่ใช้ API แบบเดียวกับ OpenAI คุณจะได้ความเป็นส่วนตัวเต็มๆ ไม่มีค่า API และประสิทธิภาพที่ดีเกินคาดบน Apple Silicon

เราครอบคลุม backend สองตัว:

| Backend | ติดตั้ง | เก่งเรื่อง | Format |
|---------|---------|---------|--------|
| **llama.cpp** | `brew install llama.cpp` | Time-to-first-token เร็วที่สุด, quantized KV cache สำหรับเครื่องหน่วยความจำน้อย | GGUF |
| **omlx** | [omlx.ai](https://omlx.ai) | Generate token เร็วที่สุด, optimize Metal แบบ native | MLX (safetensors) |

ทั้งสองตัวเปิด endpoint `/v1/chat/completions` ที่ใช้ API แบบเดียวกับ OpenAI Hermes ใช้ตัวไหนก็ได้ — แค่ชี้ไปที่ `http://localhost:8080` หรือ `http://localhost:8000`

:::info Apple Silicon only
คู่มือนี้เขียนสำหรับ Mac ที่ใช้ Apple Silicon (M1 ขึ้นไป) Mac ที่ใช้ Intel รัน llama.cpp ได้แต่ไม่มี GPU acceleration — เตรียมรับประสิทธิภาพที่ช้าลงอย่างเห็นได้ชัด
:::

---

## เลือกโมเดล

สำหรับเริ่มต้น เราแนะนำ **Qwen3.5-9B** — เป็น model ฝั่ง reasoning ที่แกร่ง และลง unified memory ขนาด 8GB+ ได้อย่างสบายด้วย quantization

| Variant | ขนาดบนดิสก์ | RAM ที่ต้องใช้ (context 128K) | Backend |
|---------|-------------|---------------------------|---------|
| Qwen3.5-9B-Q4_K_M (GGUF) | 5.3 GB | ~10 GB พร้อม quantized KV cache | llama.cpp |
| Qwen3.5-9B-mlx-lm-mxfp4 (MLX) | ~5 GB | ~12 GB | omlx |

**กฎหน่วยความจำคร่าวๆ:** ขนาดโมเดล + KV cache โมเดล 9B Q4 ตัวเป็น ~5 GB ส่วน KV cache ที่ context 128K ด้วย quantization Q4 เพิ่มอีก ~4-5 GB ถ้าใช้ default (f16) จะพุ่งไป ~16 GB flag quantized KV cache ของ llama.cpp คือเคล็ดลับสำคัญสำหรับเครื่องที่หน่วยความจำจำกัด

โมเดลใหญ่กว่านั้น (27B, 35B) ต้องใช้ unified memory 32 GB+ โมเดล 9B จึงเป็นจุด sweet spot ของเครื่อง 8-16 GB

---

## Option A: llama.cpp

llama.cpp เป็น local LLM runtime ที่พกพาง่ายที่สุด บน macOS มันใช้ Metal ทำ GPU acceleration ได้ทันทีตั้งแต่แกะกล่อง

### ติดตั้ง

```bash
brew install llama.cpp
```

คำสั่งนี้ทำให้คุณมีคำสั่ง `llama-server` ใช้ทั้งระบบ

### ดาวน์โหลดโมเดล

คุณต้องมีโมเดล format GGUF แหล่งที่ง่ายที่สุดคือ Hugging Face ผ่าน `huggingface-cli`:

```bash
brew install huggingface-cli
```

แล้วดาวน์โหลด:

```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models
```

:::tip Gated models
บางโมเดลบน Hugging Face ต้อง authenticate ก่อน ถ้าเจอ error 401 หรือ 404 ให้รัน `huggingface-cli login` ก่อน
:::

### เริ่ม server

```bash
llama-server -m ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 \
  -c 131072 \
  -np 1 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --host 0.0.0.0
```

ความหมายของแต่ละ flag:

| Flag | Purpose |
|------|---------|
| `-ngl 99` | Offload layer ทั้งหมดขึ้น GPU (Metal) ใช้เลขสูงๆ เพื่อให้แน่ใจว่าไม่มีอะไรค้างบน CPU |
| `-c 131072` | ขนาด context window (128K tokens) ลดค่านี้ถ้าหน่วยความจำไม่พอ |
| `-np 1` | จำนวน parallel slot ใช้ 1 สำหรับ single user — slot เยอะจะแบ่งงบหน่วยความจำของคุณ |
| `-fa on` | Flash attention ลดการใช้หน่วยความจำและเร่ง inference บน context ยาว |
| `--cache-type-k q4_0` | Quantize key cache เหลือ 4-bit **นี่คือตัวช่วยประหยัดหน่วยความจำตัวใหญ่** |
| `--cache-type-v q4_0` | Quantize value cache เหลือ 4-bit ใช้คู่กับตัวบน ลดหน่วยความจำ KV cache ลง ~75% เทียบกับ f16 |
| `--host 0.0.0.0` | ฟังบนทุก interface ใช้ `127.0.0.1` ถ้าไม่ต้องการเข้าถึงจาก network |

server พร้อมใช้งานเมื่อคุณเห็น:

```
main: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
```

### เพิ่มประสิทธิภาพหน่วยความจำสำหรับระบบที่จำกัด

flag `--cache-type-k q4_0 --cache-type-v q4_0` เป็นการ optimize ที่สำคัญที่สุดสำหรับเครื่องหน่วยความจำจำกัด ผลที่ context 128K เป็นดังนี้:

| ชนิด KV cache | หน่วยความจำ KV cache (128K ctx, โมเดล 9B) |
|---------------|--------------------------------------|
| f16 (default) | ~16 GB |
| q8_0 | ~8 GB |
| **q4_0** | **~4 GB** |

Mac 8 GB ให้ใช้ KV cache `q4_0` และเลือกโมเดลที่เล็กพอจะเหลือที่ให้ context ขั้นต่ำ 64K ของ Hermes ด้วย บน 16 GB คุณใช้ context 128K ได้อย่างสบาย บน 32 GB+ คุณรันโมเดลใหญ่ขึ้นหรือหลาย parallel slot ได้

ถ้าหน่วยความจำยังไม่พอ ให้ลด context ลงแต่คงไว้ไม่ต่ำกว่าค่าขั้นต่ำ 64K ของ Hermes มิฉะนั้นเปลี่ยนไปใช้โมเดลเล็กลงหรือ quantization เบาลง (Q3_K_M แทน Q4_K_M)

### ทดสอบ

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### หาชื่อโมเดล

ถ้าลืมชื่อโมเดล ให้ query ที่ models endpoint:

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

---

## Option B: MLX ผ่าน omlx

[omlx](https://omlx.ai) เป็นแอปแนว macOS โดยเฉพาะ ที่จัดการและ serve โมเดล MLX MLX คือ machine learning framework ของ Apple เอง ที่ optimize มาเพื่อสถาปัตยกรรม unified memory ของ Apple Silicon โดยเฉพาะ

### ติดตั้ง

ดาวน์โหลดและติดตั้งจาก [omlx.ai](https://omlx.ai) มีทั้ง GUI สำหรับจัดการโมเดลและ server ในตัว

### ดาวน์โหลดโมเดล

ใช้แอป omlx เรียกดูและดาวน์โหลดโมเดล ค้นหา `Qwen3.5-9B-mlx-lm-mxfp4` แล้วดาวน์โหลด โมเดลถูกเก็บไว้ในเครื่อง (ปกติอยู่ที่ `~/.omlx/models/`)

### เริ่ม server

omlx serve โมเดลที่ `http://127.0.0.1:8000` เป็นค่าเริ่มต้น เริ่ม serving จาก UI ของแอป หรือใช้ CLI ถ้ามี

### ทดสอบ

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-mlx-lm-mxfp4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### ดูรายการโมเดลที่พร้อมใช้

omlx serve หลายโมเดลพร้อมกันได้:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'
```

---

## Benchmark: llama.cpp vs MLX

ทดสอบ backend ทั้งสองบนเครื่องเดียวกัน (Apple M5 Max, 128 GB unified memory) ด้วยโมเดลเดียวกัน (Qwen3.5-9B) ที่ระดับ quantization เทียบเคียงกันได้ (Q4_K_M สำหรับ GGUF, mxfp4 สำหรับ MLX) ใช้ prompt หลากหลาย 5 แบบ แบบละ 3 รอบ ทดสอบ backend ไล่ทีละตัวเพื่อเลี่ยงการแย่ง resource

### ผลลัพธ์

| Metric | llama.cpp (Q4_K_M) | MLX (mxfp4) | Winner |
|--------|-------------------|-------------|--------|
| **TTFT (avg)** | **67 ms** | 289 ms | llama.cpp (4.3x faster) |
| **TTFT (p50)** | **66 ms** | 286 ms | llama.cpp (4.3x faster) |
| **Generation (avg)** | 70 tok/s | **96 tok/s** | MLX (37% faster) |
| **Generation (p50)** | 70 tok/s | **96 tok/s** | MLX (37% faster) |
| **Total time (512 tokens)** | 7.3s | **5.5s** | MLX (25% faster) |

### ตัวเลขเหล่านี้แปลว่าอะไร

- **llama.cpp** เก่งเรื่อง prompt processing — pipeline flash attention + quantized KV cache ทำให้ได้ token แรกภายใน ~66ms ถ้าคุณสร้าง interactive application ที่ responsiveness ที่ผู้ใช้รู้สึกสำคัญมาก (chatbot, autocomplete) นี่คือ advantage ที่จับต้องได้

- **MLX** generate token เร็วกว่า ~37% เมื่อเข้าที่แล้ว สำหรับ batch workload การสร้างข้อความยาว หรืองานที่เวลารวมสำคัญกว่า latency เริ่มต้น MLX จะจบก่อน

- Backend ทั้งสอง**คงเส้นคงวามาก** — variance ระหว่างรอบน้อยจนแทบไม่มี เชื่อตัวเลขเหล่านี้ได้

### ควรเลือกตัวไหน?

| Use case | Recommendation |
|----------|---------------|
| Interactive chat, tools ที่ต้อง latency ต่ำ | llama.cpp |
| สร้างข้อความยาว, ประมวลผลจำนวนมาก | MLX (omlx) |
| หน่วยความจำจำกัด (8-16 GB) | llama.cpp (quantized KV cache ไม่มีตัวไหนเทียบ) |
| Serve หลายโมเดลพร้อมกัน | omlx (รองรับหลายโมเดลในตัว) |
| ความเข้ากันได้สูงสุด (ใช้ Linux ได้ด้วย) | llama.cpp |

---

## เชื่อมต่อกับ Hermes

เมื่อ local server ของคุณรันอยู่:

```bash
hermes model
```

เลือก **Custom endpoint** แล้วทำตามขั้นตอน มันจะถาม base URL และชื่อโมเดล — ใช้ค่าจาก backend ที่คุณตั้งไว้ข้างบน

---

## Timeouts

Hermes ตรวจจับ local endpoint (localhost, LAN IP) อัตโนมัติ แล้วผ่อน streaming timeout ให้คลายลง ส่วนใหญ่ไม่ต้องตั้งค่าอะไรเพิ่ม

ถ้าคุณยังเจอ timeout error (เช่น context ใหญ่มากบน hardware ที่ช้า) คุณ override streaming read timeout ได้:

```bash
# In your .env — raise from the 120s default to 30 minutes
HERMES_STREAM_READ_TIMEOUT=1800
```

| Timeout | Default | Local auto-adjustment | Env var override |
|---------|---------|----------------------|------------------|
| Stream read (socket-level) | 120s | เพิ่มเป็น 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| Stale stream detection | 180s | ปิดทั้งหมด | `HERMES_STREAM_STALE_TIMEOUT` |
| API call (non-streaming) | 1800s | ไม่ต้องแก้ | `HERMES_API_TIMEOUT` |

Stream read timeout คือตัวที่ก่อปัญหาได้มากที่สุด — เป็น deadline ระดับ socket สำหรับการรับ chunk ข้อมูลถัดไป ระหว่าง prefill บน context ใหญ่ โมเดล local อาจเงียบไม่มี output เป็นนาทีๆ ขณะประมวลผล prompt อยู่ ซึ่ง auto-detection จัดการเรื่องนี้ให้อย่างโปร่งใส

:::tip Turn แรกที่เงียบมักเป็น prefill ไม่ใช่ hang
Hermes ส่ง system prompt และ tool schemas ไปทุก call ดังนั้นบน hardware ที่ช้า turn แรกอาจเงียบเป็นนาทีๆ ขณะโมเดลประมวลผล prompt นั้นก่อนจะเริ่ม generate อะไรออกมา นั่นคือ prefill ทำงานอยู่ ไม่ใช่ session ค้าง ดู [Slow first response (prefill)](./local-ollama-setup.md#slow-first-response-prefill) ในคู่มือ Ollama สำหรับวิธีบรรเทา เช่น การ keep model loaded และการ trim fixed prompt ด้วย `hermes prompt-size`
:::
