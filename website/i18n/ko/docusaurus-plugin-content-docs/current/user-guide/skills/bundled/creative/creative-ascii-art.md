---
title: "Ascii Art — ASCII 아트: pyfiglet, cowsay, boxes, 이미지-to-ASCII"
sidebar_label: "Ascii Art"
description: "ASCII 아트: pyfiglet, cowsay, boxes, 이미지-to-ASCII"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ascii Art

ASCII 아트: pyfiglet, cowsay, boxes, 이미지-to-ASCII.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공(기본 설치됨) |
| 경로 | `skills/creative/ascii-art` |
| 버전 | `4.0.0` |
| 작성자 | 0xbyt4, Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `ASCII`, `Art`, `Banners`, `Creative`, `Unicode`, `Text-Art`, `pyfiglet`, `figlet`, `cowsay`, `boxes` |
| 관련 스킬 | [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 전체 SKILL.md 참고

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보는 내용입니다.
:::

# ASCII Art 스킬

ASCII 아트에 필요한 여러 도구입니다. 모든 도구는 로컬 CLI 프로그램 또는 무료 REST API이며 API 키가 필요하지 않습니다.

## 도구 1: 텍스트 배너(pyfiglet — 로컬)

텍스트를 큰 ASCII 아트 배너로 렌더링합니다. 571개의 내장 글꼴을 제공합니다.

### 설정

```bash
pip install pyfiglet --break-system-packages -q
```

### 사용법

```bash
python3 -m pyfiglet "YOUR TEXT" -f slant
python3 -m pyfiglet "TEXT" -f doom -w 80    # Set width
python3 -m pyfiglet --list_fonts             # List all 571 fonts
```

### 권장 글꼴

| 스타일 | 글꼴 | 적합한 용도 |
|-------|------|----------|
| 깔끔하고 현대적 | `slant` | 프로젝트 이름, 헤더 |
| 굵고 블록형 | `doom` | 제목, 로고 |
| 크고 읽기 쉬움 | `big` | 배너 |
| 클래식 배너 | `banner3` | 넓은 화면 |
| 컴팩트 | `small` | 부제목 |
| 사이버펑크 | `cyberlarge` | 기술 테마 |
| 3D 효과 | `3-d` | 시작 화면 |
| 고딕 | `gothic` | 극적인 텍스트 |

### 팁

- 2~3개의 글꼴을 미리 보고 사용자가 가장 좋아하는 글꼴을 선택하게 하세요.
- 짧은 텍스트(1~8자)는 `doom` 또는 `block`처럼 세부적인 글꼴에서 가장 잘 표현됩니다.
- 긴 텍스트에는 `small` 또는 `mini`처럼 컴팩트한 글꼴이 더 적합합니다.

## 도구 2: 텍스트 배너(asciified API — 원격, 설치 불필요)

텍스트를 ASCII 아트로 변환하는 무료 REST API입니다. 250개 이상의 FIGlet 글꼴을 제공합니다. 결과를 파싱할 필요 없이 일반 텍스트로 직접 반환합니다. pyfiglet이 설치되지 않았거나 빠른 대안이 필요할 때 사용하세요.

### 사용법(터미널 curl 사용)

```bash
# Basic text banner (default font)
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello+World"

# With a specific font
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Slant"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Doom"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Star+Wars"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=3-D"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Banner3"

# List all available fonts (returns JSON array)
curl -s "https://asciified.thelicato.io/api/v2/fonts"
```

### 팁

- text 매개변수에서 공백은 `+`로 URL 인코딩하세요.
- 응답은 일반 텍스트 ASCII 아트이며 JSON으로 감싸지지 않아 바로 표시할 수 있습니다.
- 글꼴 이름은 대소문자를 구분하므로 fonts 엔드포인트에서 정확한 이름을 확인하세요.
- curl이 있는 모든 터미널에서 작동하며 Python이나 pip가 필요하지 않습니다.

## 도구 3: Cowsay(메시지 아트)

텍스트를 ASCII 문자와 함께 말풍선으로 감싸는 클래식 도구입니다.

### 설정

```bash
sudo apt install cowsay -y    # Debian/Ubuntu
# brew install cowsay         # macOS
```

### 사용법

```bash
cowsay "Hello World"
cowsay -f tux "Linux rules"       # Tux the penguin
cowsay -f dragon "Rawr!"          # Dragon
cowsay -f stegosaurus "Roar!"     # Stegosaurus
cowthink "Hmm..."                  # Thought bubble
cowsay -l                          # List all characters
```

### 사용 가능한 문자(50개 이상)

`beavis.zen`, `bong`, `bunny`, `cheese`, `daemon`, `default`, `dragon`,
`dragon-and-cow`, `elephant`, `eyes`, `flaming-skull`, `ghostbusters`,
`hellokitty`, `kiss`, `kitty`, `koala`, `luke-koala`, `mech-and-cow`,
`meow`, `moofasa`, `moose`, `ren`, `sheep`, `skeleton`, `small`,
`stegosaurus`, `stimpy`, `supermilker`, `surgery`, `three-eyes`,
`turkey`, `turtle`, `tux`, `udder`, `vader`, `vader-koala`, `www`

### 눈/혀 수정자

```bash
cowsay -b "Borg"       # =_= eyes
cowsay -d "Dead"       # x_x eyes
cowsay -g "Greedy"     # $_$ eyes
cowsay -p "Paranoid"   # @_@ eyes
cowsay -s "Stoned"     # *_* eyes
cowsay -w "Wired"      # O_O eyes
cowsay -e "OO" "Msg"   # Custom eyes
cowsay -T "U " "Msg"   # Custom tongue
```

## 도구 4: Boxes(장식 테두리)

어떤 텍스트든 장식용 ASCII 아트 테두리나 프레임으로 감쌉니다. 70개 이상의 내장 디자인을 제공합니다.

### 설정

```bash
sudo apt install boxes -y    # Debian/Ubuntu
# brew install boxes         # macOS
```

### 사용법

```bash
echo "Hello World" | boxes                    # Default box
echo "Hello World" | boxes -d stone           # Stone border
echo "Hello World" | boxes -d parchment       # Parchment scroll
echo "Hello World" | boxes -d cat             # Cat border
echo "Hello World" | boxes -d dog             # Dog border
echo "Hello World" | boxes -d unicornsay      # Unicorn
echo "Hello World" | boxes -d diamonds        # Diamond pattern
echo "Hello World" | boxes -d c-cmt           # C-style comment
echo "Hello World" | boxes -d html-cmt        # HTML comment
echo "Hello World" | boxes -a c               # Center text
boxes -l                                       # List all 70+ designs
```

### pyfiglet 또는 asciified와 함께 사용

```bash
python3 -m pyfiglet "HERMES" -f slant | boxes -d stone
# Or without pyfiglet installed:
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=HERMES&font=Slant" | boxes -d stone
```

## 도구 5: TOIlet(색상 텍스트 아트)

ANSI 색상 효과와 시각 필터를 제공하는 pyfiglet과 유사한 도구입니다. 터미널을 화려하게 꾸미는 데 적합합니다.

### 설정

```bash
sudo apt install toilet toilet-fonts -y    # Debian/Ubuntu
# brew install toilet                      # macOS
```

### 사용법

```bash
toilet "Hello World"                    # Basic text art
toilet -f bigmono12 "Hello"            # Specific font
toilet --gay "Rainbow!"                 # Rainbow coloring
toilet --metal "Metal!"                 # Metallic effect
toilet -F border "Bordered"             # Add border
toilet -F border --gay "Fancy!"         # Combined effects
toilet -f pagga "Block"                 # Block-style font (unique to toilet)
toilet -F list                          # List available filters
```

### 필터

`crop`, `gay`(무지개), `metal`, `flip`, `flop`, `180`, `left`, `right`, `border`

**참고**: toilet은 색상용 ANSI 이스케이프 코드를 출력하므로 터미널에서는 작동하지만 모든 환경(예: 일반 텍스트 파일, 일부 채팅 플랫폼)에서 올바르게 렌더링되지 않을 수 있습니다.

## 도구 6: 이미지를 ASCII 아트로 변환

이미지(PNG, JPEG, GIF, WEBP)를 ASCII 아트로 변환합니다.

### 옵션 A: ascii-image-converter(권장, 최신)

```bash
# Install
sudo snap install ascii-image-converter
# OR: go install github.com/TheZoraiz/ascii-image-converter@latest
```

```bash
ascii-image-converter image.png                  # Basic
ascii-image-converter image.png -C               # Color output
ascii-image-converter image.png -d 60,30         # Set dimensions
ascii-image-converter image.png -b               # Braille characters
ascii-image-converter image.png -n               # Negative/inverted
ascii-image-converter https://url/image.jpg      # Direct URL
ascii-image-converter image.png --save-txt out   # Save as text
```

### 옵션 B: jp2a(경량, JPEG 전용)

```bash
sudo apt install jp2a -y
jp2a --width=80 image.jpg
jp2a --colors image.jpg              # Colorized
```

## 도구 7: 미리 만들어진 ASCII 아트 검색

웹에서 엄선된 ASCII 아트를 검색합니다. `terminal`과 `curl`을 사용하세요.

### 출처 A: ascii.co.uk(미리 만들어진 아트에 권장)

주제별로 정리된 클래식 ASCII 아트 모음입니다. 아트는 HTML `<pre>` 태그 안에 있습니다. 페이지를 curl로 가져온 다음 작은 Python 스니펫으로 아트를 추출하세요.

**URL 패턴:** `https://ascii.co.uk/art/{subject}`

**1단계 — 페이지 가져오기:**

```bash
curl -s 'https://ascii.co.uk/art/cat' -o /tmp/ascii_art.html
```

**2단계 — pre 태그에서 아트 추출:**

```python
import re, html
with open('/tmp/ascii_art.html') as f:
    text = f.read()
arts = re.findall(r'<pre[^>]*>(.*?)</pre>', text, re.DOTALL)
for art in arts:
    clean = re.sub(r'<[^>]+>', '', art)
    clean = html.unescape(clean).strip()
    if len(clean) > 30:
        print(clean)
        print('\n---\n')
```

**사용 가능한 주제**(URL 경로로 사용):
- 동물: `cat`, `dog`, `horse`, `bird`, `fish`, `dragon`, `snake`, `rabbit`, `elephant`, `dolphin`, `butterfly`, `owl`, `wolf`, `bear`, `penguin`, `turtle`
- 사물: `car`, `ship`, `airplane`, `rocket`, `guitar`, `computer`, `coffee`, `beer`, `cake`, `house`, `castle`, `sword`, `crown`, `key`
- 자연: `tree`, `flower`, `sun`, `moon`, `star`, `mountain`, `ocean`, `rainbow`
- 캐릭터: `skull`, `robot`, `angel`, `wizard`, `pirate`, `ninja`, `alien`
- 기념일: `christmas`, `halloween`, `valentine`

**팁:**
- 작가 서명/이니셜을 보존하세요. 이는 중요한 예의입니다.
- 페이지마다 여러 작품이 있으므로 사용자에게 가장 적합한 작품을 고르세요.
- JavaScript가 필요하지 않아 curl로 안정적으로 작동합니다.

### 출처 B: GitHub Octocat API(재미있는 이스터 에그)

현명한 문구와 함께 무작위 GitHub Octocat을 반환합니다. 인증이 필요하지 않습니다.

```bash
curl -s https://api.github.com/octocat
```

## 도구 8: 재미있는 ASCII 유틸리티(curl 사용)

이 무료 서비스들은 ASCII 아트를 직접 반환하므로 재미있는 추가 요소에 적합합니다.

### ASCII 아트로 QR 코드 만들기

```bash
curl -s "qrenco.de/Hello+World"
curl -s "qrenco.de/https://example.com"
```

### ASCII 아트로 날씨 보기

```bash
curl -s "wttr.in/London"          # Full weather report with ASCII graphics
curl -s "wttr.in/Moon"            # Moon phase in ASCII art
curl -s "v2.wttr.in/London"       # Detailed version
```

## 도구 9: LLM으로 사용자 지정 아트 생성(대체 수단)

위 도구로 필요한 결과를 얻을 수 없을 때 다음 유니코드 문자를 사용해 ASCII 아트를 직접 생성하세요.

### 문자 팔레트

**상자 그리기:** `╔ ╗ ╚ ╝ ║ ═ ╠ ╣ ╦ ╩ ╬ ┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ ╭ ╮ ╰ ╯`

**블록 요소:** `░ ▒ ▓ █ ▄ ▀ ▌ ▐ ▖ ▗ ▘ ▝ ▚ ▞`

**기하 도형 및 기호:** `◆ ◇ ◈ ● ○ ◉ ■ □ ▲ △ ▼ ▽ ★ ☆ ✦ ✧ ◀ ▶ ◁ ▷ ⬡ ⬢ ⌂`

### 규칙

- 최대 너비: 줄당 60자(터미널 안전)
- 최대 높이: 배너는 15줄, 장면은 25줄
- 고정폭 글꼴 전용: 고정폭 글꼴에서 올바르게 렌더링되어야 합니다.

## 결정 흐름

1. **배너로 표시할 텍스트** → 설치되어 있으면 pyfiglet, 그렇지 않으면 curl을 통한 asciified API
2. **메시지를 재미있는 캐릭터 아트로 감싸기** → cowsay
3. **장식 테두리/프레임 추가** → boxes(pyfiglet/asciified와 조합 가능)
4. **특정 대상(고양이, 로켓, 용)의 아트** → ascii.co.uk를 curl로 가져온 뒤 파싱
5. **이미지를 ASCII로 변환** → ascii-image-converter 또는 jp2a
6. **QR 코드** → qrenco.de를 curl로 사용
7. **날씨/달 아트** → wttr.in을 curl로 사용
8. **사용자 지정/창의적인 결과** → 유니코드 팔레트를 사용한 LLM 생성
9. **필요한 도구가 설치되지 않음** → 설치하거나 다음 옵션으로 대체
