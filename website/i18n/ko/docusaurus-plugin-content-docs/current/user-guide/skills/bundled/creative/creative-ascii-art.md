---
title: "Ascii Art — ASCII art: pyfiglet, cowsay, boxes, image-to-ascii"
sidebar_label: "Ascii Art"
description: "ASCII art: pyfiglet, cowsay, boxes, image-to-ascii"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Ascii Art

ASCII 아트: pyfiglet, cowsay, boxes, image-to-ascii.

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/creative/ascii-art` |
| Version | `4.0.0` |
| Author | 0xbyt4, Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `ASCII`, `Art`, `Banners`, `Creative`, `Unicode`, `Text-Art`, `pyfiglet`, `figlet`, `cowsay`, `boxes` |
| Related skills | [`excalidraw`](/docs/user-guide/skills/bundled/creative/creative-excalidraw) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# ASCII Art Skill

다양한 ASCII 아트 요구 사항을 충족하는 여러 도구. 모든 도구는 로컬 CLI 프로그램이거나 무료 REST API입니다 — API 키가 필요하지 않습니다.

## 도구 1: 텍스트 배너 (pyfiglet — 로컬)

텍스트를 큰 ASCII 아트 배너로 렌더링합니다. 571개의 내장 글꼴을 제공합니다.

### 설정

```bash
pip install pyfiglet --break-system-packages -q
```

### 사용법

```bash
python3 -m pyfiglet "YOUR TEXT" -f slant
python3 -m pyfiglet "TEXT" -f doom -w 80    # 너비 설정
python3 -m pyfiglet --list_fonts             # 571개 모든 글꼴 목록 보기
```

### 추천 글꼴

| 스타일 | 글꼴 | 적합한 용도 |
|-------|------|----------|
| 깔끔하고 모던함 | `slant` | 프로젝트 이름, 헤더 |
| 굵고 각진 스타일 | `doom` | 제목, 로고 |
| 크고 가독성 높음 | `big` | 배너 |
| 클래식 배너 | `banner3` | 넓은 디스플레이 |
| 컴팩트함 | `small` | 부제목 |
| 사이버펑크 | `cyberlarge` | 기술 관련 테마 |
| 3D 효과 | `3-d` | 스플래시 화면 |
| 고딕 | `gothic` | 극적인 텍스트 |

### 팁

- 2~3개의 글꼴을 미리 보여주고 사용자가 마음에 드는 것을 고르게 하세요.
- 짧은 텍스트(1~8자)는 `doom`이나 `block`과 같이 디테일한 글꼴과 잘 어울립니다.
- 긴 텍스트는 `small`이나 `mini`와 같이 컴팩트한 글꼴과 더 잘 어울립니다.

## 도구 2: 텍스트 배너 (asciified API — 원격, 설치 불필요)

텍스트를 ASCII 아트로 변환하는 무료 REST API입니다. 250개 이상의 FIGlet 글꼴을 지원합니다. 일반 텍스트를 직접 반환하므로 파싱이 필요하지 않습니다. pyfiglet이 설치되어 있지 않거나 빠른 대안이 필요할 때 사용하세요.

### 사용법 (터미널 curl 경유)

```bash
# 기본 텍스트 배너 (기본 글꼴)
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello+World"

# 특정 글꼴 사용
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Slant"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Doom"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Star+Wars"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=3-D"
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=Hello&font=Banner3"

# 사용 가능한 모든 글꼴 목록 보기 (JSON 배열 반환)
curl -s "https://asciified.thelicato.io/api/v2/fonts"
```

### 팁

- 텍스트 매개변수에서 공백을 `+`로 URL 인코딩하세요.
- 응답은 일반 텍스트 ASCII 아트입니다 — JSON 래핑이 없어 즉시 표시할 수 있습니다.
- 글꼴 이름은 대소문자를 구분합니다. 정확한 이름을 얻으려면 fonts 엔드포인트를 사용하세요.
- Python이나 pip 없이 curl이 있는 모든 터미널에서 작동합니다.

## 도구 3: Cowsay (메시지 아트)

ASCII 캐릭터가 말풍선 안에 텍스트를 전달하는 클래식 도구입니다.

### 설정

```bash
sudo apt install cowsay -y    # Debian/Ubuntu
# brew install cowsay         # macOS
```

### 사용법

```bash
cowsay "Hello World"
cowsay -f tux "Linux rules"       # 펭귄 턱스(Tux)
cowsay -f dragon "Rawr!"          # 용
cowsay -f stegosaurus "Roar!"     # 스테고사우루스
cowthink "Hmm..."                  # 생각 말풍선
cowsay -l                          # 모든 캐릭터 목록 보기
```

### 사용 가능한 캐릭터 (50+)

`beavis.zen`, `bong`, `bunny`, `cheese`, `daemon`, `default`, `dragon`,
`dragon-and-cow`, `elephant`, `eyes`, `flaming-skull`, `ghostbusters`,
`hellokitty`, `kiss`, `kitty`, `koala`, `luke-koala`, `mech-and-cow`,
`meow`, `moofasa`, `moose`, `ren`, `sheep`, `skeleton`, `small`,
`stegosaurus`, `stimpy`, `supermilker`, `surgery`, `three-eyes`,
`turkey`, `turtle`, `tux`, `udder`, `vader`, `vader-koala`, `www`

### 눈/혀 수정자

```bash
cowsay -b "Borg"       # =_= 눈
cowsay -d "Dead"       # x_x 눈
cowsay -g "Greedy"     # $_$ 눈
cowsay -p "Paranoid"   # @_@ 눈
cowsay -s "Stoned"     # *_* 눈
cowsay -w "Wired"      # O_O 눈
cowsay -e "OO" "Msg"   # 사용자 지정 눈
cowsay -T "U " "Msg"   # 사용자 지정 혀
```

## 도구 4: Boxes (장식용 테두리)

모든 텍스트 주위에 장식용 ASCII 아트 테두리/프레임을 그립니다. 70개 이상의 내장 디자인을 제공합니다.

### 설정

```bash
sudo apt install boxes -y    # Debian/Ubuntu
# brew install boxes         # macOS
```

### 사용법

```bash
echo "Hello World" | boxes                    # 기본 상자
echo "Hello World" | boxes -d stone           # 돌 테두리
echo "Hello World" | boxes -d parchment       # 양피지 두루마리
echo "Hello World" | boxes -d cat             # 고양이 테두리
echo "Hello World" | boxes -d dog             # 개 테두리
echo "Hello World" | boxes -d unicornsay      # 유니콘
echo "Hello World" | boxes -d diamonds        # 다이아몬드 패턴
echo "Hello World" | boxes -d c-cmt           # C 스타일 주석
echo "Hello World" | boxes -d html-cmt        # HTML 주석
echo "Hello World" | boxes -a c               # 텍스트 중앙 정렬
boxes -l                                       # 70개 이상의 전체 디자인 목록 보기
```

### pyfiglet 또는 asciified와 결합

```bash
python3 -m pyfiglet "HERMES" -f slant | boxes -d stone
# 또는 pyfiglet이 설치되지 않은 경우:
curl -s "https://asciified.thelicato.io/api/v2/ascii?text=HERMES&font=Slant" | boxes -d stone
```

## 도구 5: TOIlet (컬러 텍스트 아트)

pyfiglet과 유사하지만 ANSI 색상 효과와 시각적 필터가 있습니다. 터미널을 화려하게 꾸미는 데 좋습니다.

### 설정

```bash
sudo apt install toilet toilet-fonts -y    # Debian/Ubuntu
# brew install toilet                      # macOS
```

### 사용법

```bash
toilet "Hello World"                    # 기본 텍스트 아트
toilet -f bigmono12 "Hello"            # 특정 글꼴
toilet --gay "Rainbow!"                 # 무지개 색칠
toilet --metal "Metal!"                 # 금속 효과
toilet -F border "Bordered"             # 테두리 추가
toilet -F border --gay "Fancy!"         # 효과 결합
toilet -f pagga "Block"                 # 블록 스타일 글꼴 (toilet 전용)
toilet -F list                          # 사용 가능한 필터 목록 보기
```

### 필터

`crop`, `gay` (무지개), `metal`, `flip`, `flop`, `180`, `left`, `right`, `border`

**참고**: toilet은 색상을 위해 ANSI 이스케이프 코드를 출력합니다 — 터미널에서는 작동하지만 일반 텍스트 파일이나 일부 채팅 플랫폼과 같은 모든 컨텍스트에서 렌더링되지는 않을 수 있습니다.

## 도구 6: 이미지를 ASCII 아트로 변환

이미지(PNG, JPEG, GIF, WEBP)를 ASCII 아트로 변환합니다.

### 옵션 A: ascii-image-converter (권장, 모던)

```bash
# 설치
sudo snap install ascii-image-converter
# 또는: go install github.com/TheZoraiz/ascii-image-converter@latest
```

```bash
ascii-image-converter image.png                  # 기본
ascii-image-converter image.png -C               # 컬러 출력
ascii-image-converter image.png -d 60,30         # 크기 설정
ascii-image-converter image.png -b               # 점자 문자
ascii-image-converter image.png -n               # 반전/네거티브
ascii-image-converter https://url/image.jpg      # 직접 URL 사용
ascii-image-converter image.png --save-txt out   # 텍스트로 저장
```

### 옵션 B: jp2a (경량, JPEG 전용)

```bash
sudo apt install jp2a -y
jp2a --width=80 image.jpg
jp2a --colors image.jpg              # 컬러화
```

## 도구 7: 미리 만들어진 ASCII 아트 검색

웹에서 선별된 ASCII 아트를 검색합니다. `curl`과 함께 `terminal`을 사용하세요.

### 출처 A: ascii.co.uk (미리 만들어진 아트로 권장)

주제별로 정리된 방대한 고전 ASCII 아트 컬렉션입니다. 아트는 HTML `<pre>` 태그 안에 있습니다. curl로 페이지를 가져온 다음 작은 Python 스니펫으로 아트를 추출합니다.

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

**사용 가능한 주제** (URL 경로로 사용):
- 동물: `cat`, `dog`, `horse`, `bird`, `fish`, `dragon`, `snake`, `rabbit`, `elephant`, `dolphin`, `butterfly`, `owl`, `wolf`, `bear`, `penguin`, `turtle`
- 사물: `car`, `ship`, `airplane`, `rocket`, `guitar`, `computer`, `coffee`, `beer`, `cake`, `house`, `castle`, `sword`, `crown`, `key`
- 자연: `tree`, `flower`, `sun`, `moon`, `star`, `mountain`, `ocean`, `rainbow`
- 캐릭터: `skull`, `robot`, `angel`, `wizard`, `pirate`, `ninja`, `alien`
- 기념일: `christmas`, `halloween`, `valentine`

**팁:**
- 아티스트의 서명/이니셜을 유지하세요 — 중요한 에티켓입니다.
- 페이지당 여러 개의 예술 작품이 있습니다 — 사용자를 위해 가장 좋은 것을 선택하세요.
- curl을 통해 안정적으로 작동하며 JavaScript가 필요하지 않습니다.

### 출처 B: GitHub Octocat API (재미있는 이스터에그)

현명한 인용구와 함께 무작위 GitHub Octocat을 반환합니다. 인증이 필요하지 않습니다.

```bash
curl -s https://api.github.com/octocat
```

## 도구 8: 재미있는 ASCII 유틸리티 (curl 경유)

다음 무료 서비스들은 ASCII 아트를 직접 반환합니다 — 재미있는 추가 기능으로 좋습니다.

### ASCII 아트 형태의 QR 코드

```bash
curl -s "qrenco.de/Hello+World"
curl -s "qrenco.de/https://example.com"
```

### ASCII 아트 날씨

```bash
curl -s "wttr.in/London"          # ASCII 그래픽이 포함된 전체 날씨 보고서
curl -s "wttr.in/Moon"            # ASCII 아트로 나타낸 달의 위상
curl -s "v2.wttr.in/London"       # 상세 버전
```

## 도구 9: LLM 생성 사용자 지정 아트 (대체 수단)

위의 도구에서 필요한 것을 찾을 수 없는 경우 다음 유니코드 문자를 사용하여 ASCII 아트를 직접 생성합니다.

### 문자 팔레트

**상자 그리기:** `╔ ╗ ╚ ╝ ║ ═ ╠ ╣ ╦ ╩ ╬ ┌ ┐ └ ┘ │ ─ ├ ┤ ┬ ┴ ┼ ╭ ╮ ╰ ╯`

**블록 요소:** `░ ▒ ▓ █ ▄ ▀ ▌ ▐ ▖ ▗ ▘ ▝ ▚ ▞`

**기하학 & 기호:** `◆ ◇ ◈ ● ○ ◉ ■ □ ▲ △ ▼ ▽ ★ ☆ ✦ ✧ ◀ ▶ ◁ ▷ ⬡ ⬢ ⌂`

### 규칙

- 최대 너비: 한 줄당 60자 (터미널 안전)
- 최대 높이: 배너는 15줄, 씬은 25줄
- 고정폭만 사용: 출력은 고정폭 글꼴에서 올바르게 렌더링되어야 합니다.

## 결정 흐름도

1. **배너 형태의 텍스트** → 설치된 경우 pyfiglet, 그렇지 않으면 curl을 통한 asciified API
2. **재미있는 캐릭터 아트 안에 메시지 넣기** → cowsay
3. **장식용 테두리/프레임 추가** → boxes (pyfiglet/asciified와 결합 가능)
4. **특정 대상의 아트** (고양이, 로켓, 용) → curl + 파싱을 통해 ascii.co.uk 이용
5. **이미지를 ASCII로 변환** → ascii-image-converter 또는 jp2a
6. **QR 코드** → curl을 통해 qrenco.de 이용
7. **날씨/달 아트** → curl을 통해 wttr.in 이용
8. **맞춤형/창의적인 무언가** → 유니코드 팔레트를 사용하여 LLM 생성
9. **설치되지 않은 도구** → 설치하거나 다음 옵션으로 넘어감
