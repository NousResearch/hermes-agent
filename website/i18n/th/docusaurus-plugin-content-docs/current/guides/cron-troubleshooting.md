---
sidebar_position: 12
title: "การแก้ปัญหา Cron"
description: "วินิจฉัยและแก้ไขปัญหา cron ที่พบบ่อยของ Hermes — job ไม่ถูกเรียกใช้ การส่งมอบล้มเหลว การโหลด skill ผิดพลาด และปัญหาด้านประสิทธิภาพ"
---

# การแก้ปัญหา Cron

เมื่อ cron job ไม่ทำงานตามที่คาดไว้ ให้ไล่ตรวจสอบตามรายการต่อไปนี้ ปัญหาส่วนใหญ่มักจัดอยู่ในสี่หมวดหมู่เดียวกัน ได้แก่ timing, delivery, permissions หรือการโหลด skill

---

## Job ไม่ถูกเรียกใช้

### ตรวจสอบที่ 1: ยืนยันว่า job มีอยู่และเปิดใช้งานอยู่

```bash
hermes cron list
```

ให้มองหา job นั้นและยืนยันว่าสถานะเป็น `[active]` (ไม่ใช่ `[paused]` หรือ `[completed]`) ถ้าแสดง `[completed]` แสดงว่าจำนวนครั้งที่ทำซ้ำอาจหมดไปแล้ว — ให้แก้ไข job เพื่อรีเซ็ต

### ตรวจสอบที่ 2: ยืนยันว่าตารางเวลาถูกต้อง

ตารางเวลาที่รูปแบบผิดพลาดจะถูกตั้งเป็นแบบครั้งเดียว (one-shot) โดยไม่มีการแจ้งเตือน หรือถูกปฏิเสธทั้งหมด ทดสอบ expression ของคุณ:

| Expression ของคุณ | ควรประเมินเป็น |
|----------------|-------------------|
| `0 9 * * *` | ทุกวัน เวลา 9:00 น. |
| `0 9 * * 1` | ทุกวันจันทร์ เวลา 9:00 น. |
| `every 2h` | ทุก 2 ชั่วโมงนับจากตอนนี้ |
| `30m` | 30 นาทีนับจากตอนนี้ |
| `2025-06-01T09:00:00` | 1 มิถุนายน 2025 เวลา 9:00 น. UTC |

ถ้า job ทำงานครั้งเดียวแล้วหายไปจากรายการ นั่นเพราะมันเป็นตารางเวลาแบบครั้งเดียว (`30m`, `1d` หรือ ISO timestamp) — ซึ่งเป็นพฤติกรรมที่คาดหวัง

### ตรวจสอบที่ 3: gateway กำลังทำงานอยู่หรือไม่?

Cron job ถูกเรียกใช้โดย background ticker thread ของ gateway ซึ่ง tick ทุก 60 วินาที การเปิด CLI chat session ธรรมดาจะ**ไม่**เรียกใช้ cron job โดยอัตโนมัติ

ถ้าคุณคาดหวังให้ job ทำงานเองโดยอัตโนมัติ คุณต้องมี gateway ที่กำลังทำงานอยู่ (`hermes gateway` สำหรับรันใน foreground หรือ `hermes gateway start` สำหรับ service ที่ติดตั้งไว้) สำหรับการดีบักแบบครั้งเดียว คุณสามารถสั่ง tick เองได้ด้วย `hermes cron tick`

### ตรวจสอบที่ 4: ตรวจสอบนาฬิกาและ timezone ของระบบ

Job ใช้ timezone ท้องถิ่น ถ้านาฬิกาของเครื่องคุณคลาดเคลื่อน หรือตั้งอยู่ใน timezone อื่นที่ไม่ใช่ที่คาดไว้ job จะทำงานผิดเวลา ตรวจสอบด้วย:

```bash
date
hermes cron list   # Compare next_run times with local time
```

---

## การส่งมอบล้มเหลว

### ตรวจสอบที่ 1: ยืนยันว่า deliver target ถูกต้อง

Delivery target คำนึงถึงตัวพิมพ์ใหญ่-เล็ก และต้องตั้งค่า platform ที่ถูกต้องเอาไว้ การตั้งค่า target ผิดพลาดจะทำให้ response ถูกทิ้งไปเงียบๆ

| Target | ต้องมี |
|--------|----------|
| `telegram` | `TELEGRAM_BOT_TOKEN` ใน `~/.hermes/.env` |
| `discord` | `DISCORD_BOT_TOKEN` ใน `~/.hermes/.env` |
| `slack` | `SLACK_BOT_TOKEN` ใน `~/.hermes/.env` |
| `whatsapp` | ตั้งค่า WhatsApp gateway แล้ว |
| `signal` | ตั้งค่า Signal gateway แล้ว |
| `matrix` | ตั้งค่า Matrix homeserver แล้ว |
| `email` | ตั้งค่า SMTP ใน `config.yaml` |
| `sms` | ตั้งค่า SMS provider แล้ว |
| `local` | มีสิทธิ์เขียนไปยัง `~/.hermes/cron/output/` |
| `origin` | ส่งมอบกลับไปยัง chat ที่มีการสร้าง job นั้น |

Platform อื่นๆ ที่รองรับ ได้แก่ `mattermost`, `homeassistant`, `dingtalk`, `feishu`, `wecom`, `weixin`, `bluebubbles`, `qqbot` และ `webhook` คุณยังสามารถระบุ chat เฉพาะแห่งด้วย syntax `platform:chat_id` (เช่น `telegram:-1001234567890`)

ถ้าการส่งมอบล้มเหลว job ยังคงทำงานอยู่ — แต่จะไม่ส่งผลลัพธ์ไปที่ใดเลย ให้ดูฟิลด์ `last_error` ที่อัปเดตใน `hermes cron list` (ถ้ามี)

### ตรวจสอบที่ 2: ตรวจสอบการใช้ `[SILENT]`

ถ้า cron job ของคุณไม่มี output เลย การส่งมอบจะถูกระงับ ถ้า response ของ agent มี quiet marker ของ cron คือ `[SILENT]` อยู่ด้วย การส่งมอบก็จะถูกระงับเช่นกัน พฤติกรรมนี้ตั้งใจไว้สำหรับ monitoring job — แต่ตรวจสอบให้แน่ใจว่า prompt ของคุณไม่ได้ระงับทุกอย่างโดยไม่ตั้งใจ

ใช้ prompt ลักษณะเช่น "respond with only [SILENT] if nothing changed." และหลีกเลี่ยงการสั่งให้ agent ใส่ `[SILENT]` ไว้ในคำอธิบายยาวๆ เพราะ cron ถือว่า marker นี้เป็นสัญญาณระงับการส่งมอบ

### ตรวจสอบที่ 3: สิทธิ์ token ของแพลตฟอร์ม

Bot ของแต่ละ messaging platform ต้องการสิทธิ์เฉพาะอย่างเพื่อรับข้อความ ถ้าการส่งมอบล้มเหลวเงียบๆ:

- **Telegram**: Bot ต้องเป็นแอดมินใน group/channel เป้าหมาย
- **Discord**: Bot ต้องมีสิทธิ์ส่งข้อความใน channel เป้าหมาย
- **Slack**: Bot ต้องถูกเพิ่มเข้า workspace และมี scope `chat:write`

### ตรวจสอบที่ 4: การห่อ response

โดยค่าเริ่มต้น response ของ cron จะถูกห่อด้วย header และ footer (`cron.wrap_response: true` ใน `config.yaml`) platform หรือ integration บางตัวอาจรองรับสิ่งนี้ได้ไม่ดี วิธีปิด:

```yaml
cron:
  wrap_response: false
```

---

## การโหลด skill ล้มเหลว

### ตรวจสอบที่ 1: ยืนยันว่าติดตั้ง skills แล้ว

```bash
hermes skills list
```

Skill ต้องติดตั้งก่อนจึงจะแนบกับ cron job ได้ ถ้าขาด skill ใด ให้ติดตั้งก่อนด้วย `hermes skills install <skill-name>` หรือผ่าน `/skills` ใน CLI

### ตรวจสอบที่ 2: เทียบชื่อ skill กับชื่อ folder ของ skill

ชื่อ skill คำนึงถึงตัวพิมพ์ใหญ่-เล็ก และต้องตรงกับชื่อ folder ของ skill ที่ติดตั้งไว้ ถ้า job ของคุณระบุ `ai-funding-report` แต่ folder ของ skill เป็น `ai-funding-daily-report` ให้ยืนยันชื่อที่ถูกต้องจาก `hermes skills list`

### ตรวจสอบที่ 3: Skills ที่ต้องใช้ interactive tools

Cron job ทำงานโดยที่ toolset `cronjob`, `messaging` และ `clarify` ถูกปิดใช้งาน ซึ่งป้องกันการสร้าง cron แบบวนซ้ำ การส่งข้อความโดยตรง (การส่งมอบเป็นหน้าที่ของ scheduler) และ prompt แบบโต้ตอบ ถ้า skill พึ่งพา toolset เหล่านี้ มันจะใช้ไม่ได้ในบริบทของ cron

ให้ดูเอกสารของ skill เพื่อยืนยันว่ามันทำงานในโหมด non-interactive (headless)

### ตรวจสอบที่ 4: ลำดับของหลาย skill

เมื่อใช้หลาย skill skills จะถูกโหลดตามลำดับ ถ้า Skill A พึ่งพา context จาก Skill B ให้แน่ใจว่า B โหลดก่อน:

```bash
/cron add "0 9 * * *" "..." --skill context-skill --skill target-skill
```

ในตัวอย่างนี้ `context-skill` จะถูกโหลดก่อน `target-skill`

---

## ข้อผิดพลาดและความล้มเหลวของ job

### ตรวจสอบที่ 1: ดู output ล่าสุดของ job

ถ้า job ทำงานแล้วล้มเหลว คุณอาจเห็น error context ได้จาก:

1. Chat ที่ job ส่งมอบไป (ถ้าการส่งมอบสำเร็จ)
2. `~/.hermes/logs/agent.log` สำหรับข้อความจาก scheduler (หรือ `errors.log` สำหรับ warning)
3. metadata `last_run` ของ job ผ่าน `hermes cron list`

### ตรวจสอบที่ 2: รูปแบบข้อผิดพลาดที่พบบ่อย

**"No such file or directory" ของ script**
Path `script` ต้องเป็น absolute path (หรือ relative กับ Hermes config directory) ตรวจสอบด้วย:
```bash
ls ~/.hermes/scripts/your-script.py   # Must exist
hermes cron edit <job_id> --script ~/.hermes/scripts/your-script.py
```

**"Skill not found" ตอนรัน job**
Skill ต้องติดตั้งอยู่บนเครื่องที่รัน scheduler ถ้าคุณย้ายไปเครื่องอื่น skills จะไม่ sync อัตโนมัติ — ติดตั้งใหม่ด้วย `hermes skills install <skill-name>`

**Job ทำงานแต่ไม่ส่งมอบอะไรเลย**
สาเหตุที่เป็นไปได้คือปัญหา delivery target (ดูหัวข้อ การส่งมอบล้มเหลว ด้านบน), ไม่มี output หรือ response มี quiet marker ของ cron คือ `[SILENT]`

**Job ค้างหรือหมดเวลา**
Scheduler ใช้ timeout แบบอิงความไม่มีกิจกรรม (ค่าเริ่มต้น 600s, ปรับได้ผ่าน env var `HERMES_CRON_TIMEOUT`, `0` คือไม่จำกัด) Agent ทำงานต่อได้นานเท่าที่ยังเรียก tools อยู่ — ตัวจับเวลาจะทำงานหลังจากไม่มีกิจกรรมติดต่อกันนานเท่านั้น Job ที่รันนานควรใช้ script จัดการการรวบรวมข้อมูล แล้วส่งมอบเฉพาะผลลัพธ์สุดท้าย

### ตรวจสอบที่ 3: การแย่งกันของ lock

Scheduler ใช้ file-based locking ป้องกันการ tick ทับซ้อนกัน ถ้ามี gateway instance รันอยู่สองตัว (หรือ CLI session ขัดแย้งกับ gateway) job อาจถูกหน่วงหรือถูกข้ามไป

สั่ง kill gateway process ที่ซ้ำซ้อน:
```bash
ps aux | grep hermes
# Kill duplicate processes, keep only one
```

### ตรวจสอบที่ 4: สิทธิ์ของไฟล์ jobs.json

Job ถูกเก็บไว้ใน `~/.hermes/cron/jobs.json` ถ้า user ของคุณอ่าน/เขียนไฟล์นี้ไม่ได้ scheduler จะล้มเหลวอย่างเงียบๆ:

```bash
ls -la ~/.hermes/cron/jobs.json
chmod 600 ~/.hermes/cron/jobs.json   # Your user should own it
```

---

## ปัญหาด้านประสิทธิภาพ

### Job เริ่มทำงานช้า

Cron job แต่ละตัวสร้าง AIAgent session ใหม่ ซึ่งอาจรวมการยืนยันตัวตนกับ provider และการโหลด model สำหรับตารางเวลาที่ไวต่อเวลา ให้เผื่อ buffer time (เช่น `0 8 * * *` แทน `0 9 * * *`)

### Job ทับซ้อนกันมากเกินไป

Scheduler รัน job ตามลำดับภายในแต่ละ tick ถ้ามีหลาย job ถึงกำหนดพร้อมกัน มันจะทำงานไล่ทีละตัว พิจารณาเว้นระยะตารางเวลา (เช่น `0 9 * * *` กับ `5 9 * * *` แทนที่จะทั้งคู่เป็น `0 9 * * *`) เพื่อเลี่ยงความล่าช้า

### Output ของ script ขนาดใหญ่

Script ที่ dump output เป็นเมกะไบต์จะทำให้ agent ช้าลงและอาจชน token limit ให้กรอง/สรุปที่ระดับ script — emit เฉพาะสิ่งที่ agent ต้องใช้ในการให้เหตุผล

---

## คำสั่งสำหรับวินิจฉัย

```bash
hermes cron list                    # Show all jobs, states, next_run times
hermes cron run <job_id>            # Schedule for next tick (for testing)
hermes cron edit <job_id>           # Fix configuration issues
hermes logs                         # View recent Hermes logs
hermes skills list                  # Verify installed skills
```

---

## ขอความช่วยเหลือเพิ่มเติม

ถ้าคุณไล่ตรวจสอบตามคู่มือนี้จนครบแล้วปัญหายังอยู่:

1. รัน job ด้วย `hermes cron run <job_id>` (จะทำงานใน gateway tick ถัดไป) แล้วสังเกต error ใน chat output
2. ดู `~/.hermes/logs/agent.log` สำหรับข้อความจาก scheduler และ `~/.hermes/logs/errors.log` สำหรับ warning
3. เปิด issue ที่ [github.com/NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) พร้อมข้อมูล:
   - The job ID and schedule
   - The delivery target
   - What you expected vs. what happened
   - Relevant error messages from the logs

---

*สำหรับข้อมูลอ้างอิง cron ฉบับสมบูรณ์ ดูที่ [Automate Anything with Cron](/guides/automate-with-cron) และ [Scheduled Tasks (Cron)](/user-guide/features/cron).*
