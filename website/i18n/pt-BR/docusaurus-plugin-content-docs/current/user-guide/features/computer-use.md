---
title: Computer Use
sidebar_position: 16
---

# Computer Use

O Hermes Agent pode dirigir seu desktop — clicar, digitar, rolar,
arrastar — em **segundo plano** no **macOS, Windows e Linux**. Seu
cursor não se move, o foco do teclado não muda e seus desktops virtuais / Spaces
não trocam para você. Você e o agente trabalham juntos na
mesma máquina.

Diferente da maioria das integrações de computer-use, isso funciona com **qualquer modelo
capaz de ferramentas** — Claude, GPT, Gemini ou um modelo aberto num endpoint
compatível com OpenAI local. Não há schema nativo Anthropic com que se preocupar.

## Como funciona {#how-it-works}

O toolset built-in `computer_use` é a integração Hermes recomendada. Ele
fala MCP over stdio com
[`cua-driver`](https://github.com/trycua/cua), um driver open-source de computer-use
em segundo plano. Cada plataforma usa a pilha de acessibilidade +
input apropriada por baixo:

| Plataforma | Árvore de acessibilidade | Dispatch de input |
|---|---|---|
| macOS | AX (SPIs SkyLight privados) | `SLPSPostEventRecordTo` — escopo por pid, sem warp do cursor |
| Windows | UIAutomation | `SendInput` + `PostMessage` — sem roubo de foco |
| Linux | AT-SPI (X11 + Wayland) | XTest (X11) / virtual-keyboard (Wayland) |

O resultado é o mesmo em toda plataforma: o agente pode ler a
árvore de acessibilidade de qualquer janela visível E postar eventos sintetizados
sem trazer ao front, trocar desktops virtuais ou mover o
cursor real do SO.

Para o contrato subjacente — *por que* o modo background importa, a
invariante no-foreground, internals de dispatch de clique — veja
**[cua.ai/docs/explanation/the-no-foreground-contract](https://cua.ai/docs/explanation/the-no-foreground-contract)**.

## Habilitar {#enabling}

**Instalações novas já têm o driver.** O instalador Hermes
(`install.sh` / `install.ps1`) pré-instala `cua-driver` (best-effort;
passe `--skip-computer-use` / `-SkipComputerUse` para optar out), então habilitar
Computer Use é só um flip de config:

- **`hermes tools`** → escolha `🖱️  Computer Use` — instala o driver
  automaticamente se ainda estiver faltando.
- **Dashboard / desktop app** → toggle o toolset Computer Use — se o
  driver estiver faltando, o toggle dispara o install em background
  automaticamente (acompanhe o progresso no painel de toolset).

**Fallback manual (instalações antigas, passo do instalador pulado):**

```
hermes computer-use install
```

Isso busca e executa o instalador upstream cua-driver — `install.sh`
no macOS/Linux, `install.ps1` no Windows. Use `hermes computer-use
status` para verificar a instalação.

Já tem cua-driver? O Hermes o reutiliza quando ele suporta o contrato de runtime 0.20.
Durante setup, habilitação de toolset, `hermes update`, e a primeira
chamada `computer_use` de uma sessão, o Hermes checa a versão local e
o manifest. Ele repara uma instalação padrão antiga ou incompleta pelo
instalador upstream (no máximo uma vez por sessão em runtime). Um binário
selecionado com `HERMES_CUA_DRIVER_CMD` permanece
sob seu controle, então o Hermes reporta a incompatibilidade e o deixa
inalterado.

Se você instala Cua Driver primeiro, `cua-driver skills install` instala o
skill pack da Cua em `~/.cua-driver/skills/cua-driver`. Autodetecção Hermes é um
follow-up planejado do cua-driver, então atualmente aponte o Hermes a esse diretório ou
symlink no seu espaço de skills. Você também pode registrar tools MCP Cua brutas como um
servidor MCP custom, mas isso é uma alternativa para usuários que precisam da interface
low-level. O toolset built-in fornece as ações Hermes, configuração,
aprovações e diagnósticos.

Depois de instalar, independente do caminho, conceda os
prereqs da plataforma:

| Plataforma | Pré-requisitos |
|---|---|
| **macOS** | System Settings → Privacy & Security → **Accessibility** + **Screen Recording**. Conceda a identidade nomeada por `hermes computer-use doctor`. Modo standard usa CuaDriver.app; modos bounded e unrestricted usam a identidade host do Hermes. |
| **Windows** | Nenhum na instalação. Se você estiver dirigindo via SSH (não RDP / console), precisa do padrão autostart — veja [cua.ai/docs/how-to-guides/driver/windows-ssh](https://cua.ai/docs/how-to-guides/driver/windows-ssh) para o proxy Session 0 ↔ Session 1+. |
| **Linux** | Servidor de display acessível: `DISPLAY` definido para X11, ou `XDG_SESSION_TYPE=wayland`. Sessões Wayland precisam de ponte XWayland para captura. AT-SPI deve estar ligado (padrão em GNOME/KDE/Xfce). |

Depois inicie uma sessão com o toolset habilitado:

```
hermes -t computer_use chat
```

ou adicione `computer_use` aos toolsets habilitados em `~/.hermes/config.yaml`.

## Modos de permissão e profiles de browser logados {#permission-modes-and-logged-in-browser-profiles}

O Hermes mapeia seu UX de aprovação existente nos modos de runtime imutáveis
do cua-driver. Modo de permissão, aprovação de capability manifest, e o grant de
existing-profile são settings de launch. Não podem mudar depois que o runtime inicia:

| Sessão Hermes | Modo cua-driver | Intervenção humana | `existing_profile` |
|---|---|---|---|
| Aprovações manual ou smart (padrão) | `standard` | Aprovações Hermes normais; Cua para na fronteira protegida | Recusa a menos que `computer_use.grant_existing_profile: true` (opt-in de config one-time) |
| `computer_use.permission_mode: bounded` + manifest revisado | daemon privado `bounded` | Você revisa e aprova o capability manifest uma vez, no launch | Permitido só dentro dos profiles/origins/tools declarados no manifest; todo o resto falha closed |
| `--yolo`, `/yolo`, ou `approvals.mode: off` | daemon privado `unrestricted` | Uma aceitação de risco Hermes explícita; sem prompts Cua em runtime | Recusa a menos que `computer_use.grant_existing_profile: true`; YOLO não substitui este grant |

### Anexar ao seu browser signed-in {#attaching-to-your-signed-in-browser}

O agente pode dirigir uma janela Chrome/Edge que você já tem aberta — incluindo um
profile signed-in — **sem restartar o browser, copiar o profile, ou
tocar suas tabs**. Como acesso DevTools expõe as páginas live,
cookies e storage daquele profile, o cua-driver exige um grant humano explícito que
aprovação de ferramenta comum não pode substituir. Você opta in uma vez, no config.yaml:

```yaml
computer_use:
  grant_existing_profile: true
```

O Hermes então lança o runtime cua-driver com o grant de trusted-launcher
(`--grant existing-profile`), e
`cua_browser_prepare` com um existing profile sucede contra o par exato
`(pid, window_id)` que o agente prova. Deixe `false` (o padrão) e
anexação de existing-profile falha closed; profiles isolados owned pelo driver funcionam
de qualquer forma e são o que o agente prefere.

### Modo bounded para automação repetível {#bounded-mode-for-repeatable-automation}

Para automação de browser recorrente (jobs cron, pesquisa agendada contra um
app autenticado), o modo `bounded` usa um capability manifest que você revisa uma vez:

```yaml
# config.yaml
computer_use:
  permission_mode: bounded
  capability_manifest: ~/.hermes/cua-manifest.yaml
```

O manifest nomeia os apps, kinds de browser profile, origins permitidos, e
tools tipadas que a sessão pode usar (veja a
[referência de modos de permissão cua-driver](https://cua.ai/docs/reference/cua-driver/permission-modes)
para o formato). O Hermes lança um runtime privado com
`--capability-manifest ... --approve-capability-manifest`; qualquer coisa fora
do manifest falha closed dentro do cua-driver. Um manifest faltando ou ilegível
falha ruidosamente no início da sessão em vez de dar downgrade silencioso. YOLO da sessão
ainda sobrescreve bounded para aquela sessão.

Cada transporte MCP possui uma sessão de lifecycle privada dentro do runtime. Um
nome de sessão público é só um label para identidade de cursor e estado
escopado à sessão. Não seleciona, compartilha ou mantém um runtime vivo. Desligar `/yolo`,
resetar ou fechar a sessão Hermes, cancellation cleanup, ou saída do processo
fecha essa sessão de transporte. O Hermes também para runtimes privados que lançou
para acesso bounded, unrestricted ou existing-profile. Uma conversa Hermes
não pode mudar o modo ou grants de outro runtime. No macOS, um
runtime standard com grant existing-profile usa um daemon CuaDriver.app fresh
num socket privado. Modos bounded e unrestricted usam um serviço embedded privado
sob a identidade host do Hermes.

Aprovação `smart` permanece `standard`: uma classificação LLM não pode substituir
um manifest revisado ou um grant de launch-time.

<div class="alert alert--warning">

YOLO/unrestricted mode does not protect against prompt injection or unintended
input. Use it only in a disposable VM or with accounts and data whose full
compromise you accept.

</div>

## `hermes computer-use doctor` — sua primeira parada de triagem {#hermes-computer-use-doctor-your-first-triage-stop}

`hermes computer-use doctor` executa a ferramenta MCP estruturada
`health_report` do cua-driver e imprime uma matriz por check. É a forma
mais rápida de descobrir *por que* uma ação não funciona.

```
$ hermes computer-use doctor
⚠️  cua-driver VERSION on darwin: degraded
  ✅ binary_version: cua-driver VERSION
  ✅ platform_supported: macOS 26.4.1 (arm64)
  ✅ session_active: MCP session is active.
  ❌ bundle_identity: Process has no CFBundleIdentifier.
      → Execute o binário dentro de CuaDriver.app para o TCC atribuir corretamente.
  ✅ tcc_accessibility: Accessibility is granted.
  ✅ tcc_screen_recording: Screen Recording is granted.
  ✅ ax_capability: AX is trusted and reachable.
  ✅ screen_capture_capability: ScreenCaptureKit reachable; 1 display(s) shareable.
```

- **Exit code 0** quando overall é `ok` — tudo conectado.
- **Exit code 1** quando `degraded` ou `failed` — pelo menos um check falhou; a dica em cada falha diz o que corrigir.
- **Exit code 2** quando o binário cua-driver em si não é alcançável.

Flags úteis:

- `--include CHECK` — roda só os checks listados (repita para vários)
- `--skip CHECK` — pula um check (vence sobre `--include`)
- `--json` — emite o payload estruturado bruto, mesma forma da resposta MCP
  `tools/call health_report`

A matriz de checks é consciente de plataforma: `bundle_identity` / `tcc_*` são
`skip` no Windows + Linux porque esses conceitos não se aplicam.
`ax_capability` verifica AX no macOS, UIA no Windows, AT-SPI no Linux —
cada um com a dica diagnóstica certa quando não alcança.

## O cursor do agente e sessões {#the-agent-cursor-and-sessions}

Quando o agente age, você verá um **cursor overlay tingido** deslizar
pela tela até onde cada clique / digitação / scroll cai. O cursor real
do SO nunca se move. O overlay mostra onde o agente está agindo. Cada
execução Hermes declara um **session name** público cua-driver (algo como
`hermes-3a7b9c14d2e8`). O nome rotula identidade de cursor e estado relacionado, então
execuções e subagentes concorrentes ganham cursores distintos. O transporte MCP possui a
sessão de lifecycle privada dentro do runtime; o nome público não.

Ajuste o cursor com flags CLI do `cua-driver` ou a ferramenta MCP runtime
`set_agent_cursor_style` — veja
[cua.ai/docs/how-to-guides/driver/personalize-cursor](https://cua.ai/docs/how-to-guides/driver/personalize-cursor)
para o menu completo (silhueta built-in `arrow` vs `teardrop`, SVG / PNG / ICO
custom via `--cursor-icon`, cores de gradiente runtime, halo
bloom).

## Aprofundando — o skill pack cua-driver {#going-deeper-the-cua-driver-skill-pack}

O Hermes mantém sua wrapper skill (`skills/autonomous-ai-agents/computer-use/SKILL.md`)
focada no workflow e vocabulário de ações `computer_use` do lado Hermes. Para
detalhes de plataforma, semântica de gravação, interação com página de
browser, e outro comportamento Cua profundo, instale o skill pack que a equipe cua-driver
envia e mantém diretamente:

```
cua-driver skills install
```

O comando instala o pack em `~/.cua-driver/skills/cua-driver`. Autodetecção Hermes
é um follow-up planejado do cua-driver, então atualmente aponte o Hermes a
esse diretório ou symlink no seu espaço de skills. A wrapper permanece a
camada de workflow e aponta para a skill instalada da Cua para comportamento do driver. O
pack contém:

| Arquivo | Tópico |
|---|---|
| `SKILL.md` | Núcleo cross-platform (invariante de snapshot, contrato no-foreground, dispatch de clique, mecânica de árvore AX) |
| `MACOS.md` | Específicos macOS: contrato no-foreground, navegação AXMenuBar, dispatch de clique SkyLight, ponte JS Apple Events |
| `WINDOWS.md` | Específicos Windows: árvore UIA, hospedagem UWP / `ApplicationFrameHost`, isolamento Session 0, padrão autostart |
| `LINUX.md` | Específicos Linux: árvore AT-SPI, X11 / Wayland, detecção de emulador de terminal |
| `RECORDING.md` | Semântica de gravação de trajetória + vídeo |
| `WEB_APPS.md` | Dicas de interação com páginas de browser |
| `TESTS.md` | Fluxo de replay por trajetória |

São **deep dives de plataforma, não duplicatas da skill Hermes** —
quando um agente reporta "no Windows, meu clique caiu no elemento
errado," ele lê `WINDOWS.md` pelo contexto UIA / UWP que
explica por quê e o que fazer diferente.

`cua-driver skills status` mostra o que está instalado e em quais harnesses
de agente está linkado. Hoje a lista autodetect cobre Claude
Code, Codex, OpenCode, OpenClaw e Antigravity; **autodetecção Hermes
está planejada como follow-up em `trycua/cua`** — até
lá, execute `cua-driver skills install` uma vez e aponte seu harness ao
diretório resultante `~/.cua-driver/skills/cua-driver` (ou symlink
no seu espaço de skills usual).

## Exemplo rápido {#quick-example}

Prompt do usuário: *"Find my latest email from Stripe and summarise what they want me to do."*

O plano do agente (mesma forma no macOS / Windows / Linux —
o modelo substitui o atalho idiomático e nome do app da plataforma):

1. `computer_use(action="capture", mode="som", app="Mail")` — obtém
   screenshot do app de email com cada item da sidebar, botão da toolbar
   e linha de mensagem numerados.
2. `computer_use(action="click", element=14)` — clica no campo de busca.
3. `computer_use(action="type", text="from:stripe")`
4. `computer_use(action="key", keys="return", capture_after=True)` —
   envia e obtém o novo screenshot.
5. Clique no resultado superior, leia o corpo, resuma.

Durante tudo isso, seu cursor fica onde você deixou e o app de email
nunca vem ao front.

## Recebendo o screenshot de verdade {#receiving-the-actual-screenshot}

Screenshots tirados durante o controle do computador são normalmente
internos — existem para o modelo ver a tela, e o agente responde em texto.
Mas cada captura de imagem também salva uma cópia limitada e compartilhável
no cache de imagens do Hermes e reporta o caminho, então em superfícies que
aceitam anexos (Telegram, Discord, Desktop e outras plataformas do gateway)
você pode simplesmente pedir:

> *"Me manda um screenshot da minha tela."*

e o agente entrega a imagem real como anexo nativo, não só uma descrição.
Na CLI não há canal de anexo, então o agente te dá o caminho do arquivo
salvo.

Só os 20 arquivos de captura mais recentes são mantidos, e screenshots
nunca são enviados automaticamente — só quando você pede um.

## Compatibilidade de providers {#provider-compatibility}

| Provider | Visão? | Funciona? | Notas |
|---|---|---|---|
| Anthropic (Claude Sonnet/Opus 3+) | ✅ | ✅ | Melhor no geral; SOM + coordenadas brutas. |
| OpenRouter (qualquer modelo com visão) | ✅ | ✅ | Mensagens de ferramenta multi-part suportadas. |
| OpenAI (GPT-4+, GPT-5) | ✅ | ✅ | Igual ao acima. |
| Google (Gemini 2+) | ✅ | ✅ | Tool-calling + visão ambos suportados. |
| Local vLLM / LM Studio / Ollama (modelo com visão) | ✅ | ✅ | Se o modelo suporta conteúdo multi-part em ferramentas. |
| Modelos só texto | ❌ | ✅ (degradado) | Use `mode="ax"` para operação só com árvore de acessibilidade. |

Screenshots são enviados inline com resultados de ferramenta como partes `image_url`
estilo OpenAI. Para Anthropic, o adapter converte em blocos `tool_result`
de imagem nativos. O MIME type da imagem vem do campo explícito
`mimeType` do cua-driver (`image/png` ou `image/jpeg`) — sem
sniffing magic-byte no client.

## Segurança {#safety}

O Hermes aplica guardrails em camadas:

- Ações destrutivas (click, type, drag, scroll, key, focus_app)
  exigem aprovação — interativamente via diálogo CLI ou via
  botões de aprovação da plataforma de mensagens.
- Combinações de teclas hard-blocked no nível da ferramenta: esvaziar lixeira, delete forçado,
  bloquear tela, log out, force log out.
- Padrões de type hard-blocked: `curl | bash`, `sudo rm -rf /`, fork
  bombs, etc.
- O system prompt do agente diz explicitamente: não clicar diálogos de permissão,
  não digitar senhas, não seguir instruções embutidas em
  screenshots.

Combine com `approvals.mode: manual` em `~/.hermes/config.yaml` se quiser
cada ação confirmada.

## Eficiência de tokens {#token-efficiency}

Screenshots são caros. O Hermes aplica quatro camadas de otimização:

- **Eviction de screenshot** — o adapter Anthropic mantém só os 3
  screenshots mais recentes no contexto; os mais antigos viram placeholders `[screenshot removed
  to save context]`.
- **Podagem de compressão client-side** — o compressor de contexto detecta
  resultados de ferramenta multimodais e remove partes de imagem dos antigos.
- **Estimativa de tokens consciente de imagem** — cada imagem conta como ~1500
  tokens (taxa flat Anthropic) em vez do comprimento char base64.
- **Context editing server-side (só Anthropic)** — quando ativo, o
  adapter habilita `clear_tool_uses_20250919` via `context_management` para
  a API Anthropic limpar resultados antigos de ferramenta server-side.

Uma sessão de 20 ações num display 1568×900 tipicamente custa ~30K tokens
de contexto de screenshot, não ~600K.

## Limitações {#limitations}

- **Desempenho.** Modo background é mais lento que foreground —
  eventos roteados por acessibilidade levam ~5–20 ms no macOS, ~3–10 ms no
  Windows UIA, ~5–15 ms no Linux AT-SPI vs posting HID direto. Imperceptível
  para cliques em velocidade de agente; perceptível se tentar gravar
  speed-run.
- **Sem entrada de senha por teclado.** `type` tem padrões hard-block em
  payloads de shell de comando; para senhas, use autofill do sistema
  (macOS Keychain / Windows Credential Manager / GNOME Keyring /
  KWallet).
- **Alguns apps não expõem árvore de acessibilidade.** Apps UWP modernos no
  Windows, Electron < 28 no Linux e alguns apps macOS com desenho
  custom (Logic, Final Cut, alguns jogos) têm árvores AX esparsas ou vazias.
  Caia para coordenadas de pixel se a árvore estiver vazia — ou pule a
  tarefa inteira.
- **Windows: janelas elevadas (admin) não podem ser dirigidas por um agente
  normal.** UIPI (User Interface Privilege Isolation) do Windows impõe
  limites de integrity level: um processo Medium-integrity (agente Hermes padrão)
  não pode enumerar a árvore UIA de, nem injetar input de mouse em, uma janela
  de processo High-integrity (Administrator).
  Sintoma: `capture(mode='som')` retorna 0 elementos e `click(...)`
  reporta sucesso sem fazer nada, embora o screenshot
  renderize bem (captura GDI fica abaixo do check de integridade). Eventos de
  teclado contornam parcialmente UIPI, então Tab / Enter ainda navegam um
  diálogo elevado. Restrição do SO, não bug do cua-driver — afeta
  toda pilha de automação Windows. Para dirigir janelas elevadas,
  rode o agente Hermes em High integrity (launch de terminal
  elevado); caso contrário mire janelas não elevadas.
- **Gotchas de deploy por plataforma:**
  - **macOS** usa SPIs SkyLight privados. Apple pode mudá-los em qualquer
    update de SO. O Hermes avisa quando o cua-driver instalado é mais antigo que
    a versão contra a qual foi testado.
  - **Windows** sessões SSH rodam em **Session 0**, que não tem
    desktop interativo. Dirija o Hermes de dentro da sessão RDP / console,
    ou configure a Scheduled Task autostart do cua-driver —
    [windows-ssh](https://cua.ai/docs/how-to-guides/driver/windows-ssh)
    tem a receita.
  - **Linux** exige display server alcançável. Servidores headless
    precisam Xvfb (`Xvfb :99 -screen 0 1920x1080x24`) antes de
    `computer_use` capturar ou injetar eventos. Sessões Wayland puras
    precisam ponte XWayland para captura de tela (o caminho inject Wayland
    do cua-driver trata input independentemente).

Para automação GUI cross-platform sem overhead de desktop (e
sem setup TCC / Session 0 / X11), o toolset `browser` usa um
Chromium headless real e é a resposta certa para tarefas só web.

## Configuração {#configuration}

Modo de permissão e manifest (veja
[Modos de permissão](#permission-modes-and-logged-in-browser-profiles) acima):

```yaml
computer_use:
  permission_mode: standard        # standard (default) | bounded
  capability_manifest: ""          # capability manifest path, required for bounded
  grant_existing_profile: false    # opt-in: attach in standard or unrestricted mode
```

Sobrescreva o caminho do binário driver (testes / CI / builds locais):

```
HERMES_CUA_DRIVER_CMD=/path/to/your/cua-driver
```

Troque o backend inteiro (para testes):

```
HERMES_COMPUTER_USE_BACKEND=noop   # records calls, no side effects
```

### Telemetria {#telemetry}

cua-driver vem com telemetria de uso anônima (PostHog) habilitada por padrão
upstream. **O Hermes desabilita para você** — em toda invocação cua-driver
(backend MCP, `status`, `doctor` e install) o Hermes define
`CUA_DRIVER_RS_TELEMETRY_ENABLED=0` no ambiente do driver.

Para optar de volta (deixar cua-driver usar seu default e enviar telemetria), defina
isto em `config.yaml`:

```yaml
computer_use:
  cua_telemetry: true   # default: false (telemetry off)
```

Quando ligado, `hermes computer-use doctor` reporta `telemetry: enabled`;
quando off (padrão), reporta `telemetry: disabled via
CUA_DRIVER_RS_TELEMETRY_ENABLED`.

## Testar contra build local cua-driver {#testing-against-a-local-cua-driver-build}

Quando você desenvolve o cua-driver — ou quer testar um
fix não publicado — aponte o Hermes a um binário que você buildou do source em vez
do release publicado. O Hermes resolve o driver com
`shutil.which("cua-driver")` e **não impõe
`HERMES_CUA_DRIVER_VERSION`**, então um build local (reportado como
`0.0.0-local-*`) é aceito como está. Duas abordagens:

### Opção A — `install-local` (build + colocar no PATH) {#option-a-install-local-build-put-it-on-path}

Do checkout `trycua/cua`, execute o instalador local upstream. Ele
builda o backend Rust em release mode e coloca `cua-driver` no
mesmo layout de install que o instalador de produção usa, adicionando o bin dir
ao PATH:

```powershell
# Windows (PowerShell), from the cua repo root
./libs/cua-driver/scripts/install-local.ps1 -NoAutoStart
```

```bash
# macOS / Linux, from the cua repo root  (defaults to a debug build without --release)
./libs/cua-driver/scripts/install-local.sh --release
```

- Windows stageia o build sob `%USERPROFILE%\.cua-driver\packages\…`
  e junctiona
  `%LOCALAPPDATA%\Programs\Cua\cua-driver\bin` (adicionado ao User
  PATH) a ele. macOS/Linux symlink `cua-driver` em `~/.local/bin`
  (override com `--bin-dir <path>`).
- `-NoAutoStart` pula registrar o daemon logon `cua-driver-serve`
  — você não precisa dele para testes Hermes (veja notas).

Depois abra um shell novo (para o PATH mudar ser visível) e confirme:

```
cua-driver --version                 # local builds report 0.0.0-local-release
# Windows:      (Get-Command cua-driver).Source
# macOS/Linux:  which cua-driver
```

### Opção B — apontar Hermes direto ao binário buildado (loop mais rápido) {#option-b-point-hermes-straight-at-the-built-binary-fastest-loop}

Pule a cerimônia de install: `cargo build` e defina
`HERMES_CUA_DRIVER_CMD` para o binário resultante. Melhor para loop rápido
edit/build/test.

```bash
cargo build -p cua-driver            # add --release for a release build; run from libs/cua-driver/rust
```

```
# Windows (.env)
HERMES_CUA_DRIVER_CMD=C:\path\to\cua\libs\cua-driver\rust\target\debug\cua-driver.exe
# macOS / Linux (.env)
HERMES_CUA_DRIVER_CMD=/path/to/cua/libs/cua-driver/rust/target/debug/cua-driver
```

### Confirmar que o Hermes usa seu build {#confirm-hermes-is-using-your-build}

- `hermes computer-use status` imprime o caminho do binário resolvido e
  versão.
- `hermes computer-use doctor` confirma que o binário é alcançável e
  exercita o caminho MCP completo end-to-end.
- Num session, `computer_use(action="capture")` exercita o child process
  `cua-driver mcp` spawnado.

### Notas e gotchas {#notes-gotchas}

- **Hermes spawna seu próprio child `cua-driver mcp` over stdio** — *não*
  anexa ao daemon autostart long-running `cua-driver serve`
  ou seu named pipe. Então a Scheduled Task / LaunchAgent é desnecessária
  para testes (`-NoAutoStart` serve). O daemon autostart e o
  worker UIAccess Windows (`cua-driver-uia.exe`) só importam para
  input foreground-safe em alguns apps (ex. WPF); a superfície padrão de ferramenta
  funciona pelo child stdio. Em sessões Windows SSH, o
  padrão autostart É necessário — veja a seção Limitações.
- **Binário locked no Windows.** Um daemon `cua-driver-serve` rodando pode
  segurar `cua-driver.exe` e bloquear overwrite no rebuild.
  `install-local.ps1` renomeia o binário locked automaticamente;
  se você `cargo build` manualmente (Opção B), pare primeiro com `cua-driver autostart disable` (ou `schtasks /End /TN
  cua-driver-serve`).
- **Loop de rebuild.** Depois de editar source cua-driver, re-execute
  `install-local` (rebuild, restage, flip junction `current`)
  para Opção A, ou só re-`cargo build` para Opção B — sem mudança Hermes
  de qualquer forma.
- **Builds locais pulam version check.** O Hermes avisa quando o
  cua-driver instalado é mais antigo que sua baseline testada por SO, mas
  isenta builds dev `0.0.0-local-*` — então seu build local nunca
  dispara esse aviso.

## Solução de problemas {#troubleshooting}

**Primeira ação quando algo estiver errado: execute `hermes computer-use doctor`.**
A matriz estruturada por check diz a você (e qualquer agente ajudando a
debugar) exatamente o que está errado.

Modos de falha específicos que o doctor não pega:

**`computer_use backend unavailable: cua-driver is not installed`** —
Execute `hermes computer-use install` para buscar o binário cua-driver, ou
execute `hermes tools` e habilite o toolset Computer Use.

**Cliques parecem não ter efeito** — Capture e verifique. Um modal que você
não viu pode estar bloqueando input. Feche com `escape` ou o botão close.

**Índices de elemento estão stale** — índices SOM só valem até o
próximo `capture`. Re-capture após qualquer ação que muda estado. O
wrapper carrega `element_token`s opacos para detecção stale — você
verá erro explícito em vez de clique errado.

**"blocked pattern in type text"** — O texto que tentou `type`
corresponde à lista de padrões de shell perigosos. Quebre o comando ou
reconsidere.

**Captures vazios no Linux** — `DISPLAY` não definido, ou você está em Wayland
puro sem ponte XWayland. `hermes computer-use doctor` sinalizará
`ax_capability: fail` com dica `Set DISPLAY (X11)…`.

**Captures vazios no Windows over SSH** — Você está em Session 0 (sessão
de serviços). Dirija de RDP / console diretamente, ou configure o
padrão autostart — veja
[cua.ai/docs/how-to-guides/driver/windows-ssh](https://cua.ai/docs/how-to-guides/driver/windows-ssh).

## Veja também {#see-also}

- **Skill do lado Hermes** — `skills/computer-use/SKILL.md` — ensina o
  vocabulário de ações `computer_use` Hermes; é o que o agente carrega.
- **Skill pack cua-driver** — para deep dives por plataforma
  (contrato no-foreground macOS, Windows UIA + Session 0, Linux AT-SPI
  + X11/Wayland, gravação, páginas browser), execute
  `cua-driver skills install` e leia `MACOS.md` / `WINDOWS.md` /
  `LINUX.md` / `RECORDING.md` / `WEB_APPS.md`. Autodetecção Hermes é um
  follow-up planejado; atualmente aponte o Hermes ao diretório do pack instalado
  ou symlink no seu espaço de skills.
- **cua.ai/docs** — documentação do projeto cua-driver:
  - [What is computer use?](https://cua.ai/docs/explanation/what-is-computer-use) — introdução conceitual
  - [The no-foreground contract](https://cua.ai/docs/explanation/the-no-foreground-contract) — *por que* o modo background importa
  - [Install reference](https://cua.ai/docs/how-to-guides/driver/install) — detalhes de instalação cross-platform
  - [Personalize the agent cursor](https://cua.ai/docs/how-to-guides/driver/personalize-cursor) — formas built-in, assets customizados, overrides em runtime
  - [Drive Windows over SSH](https://cua.ai/docs/how-to-guides/driver/windows-ssh) — o padrão autostart Session 0 → Session 1+
  - [Keep cua-driver running](https://cua.ai/docs/how-to-guides/driver/keep-running) — autostart / ciclo de vida do daemon
  - [Connect your agent](https://cua.ai/docs/how-to-guides/driver/connect-your-agent) — registrar cua-driver com vários harnesses (Hermes entre eles)
- [Automação de browser](./browser.md) para tarefas web cross-platform quando você não precisa dirigir apps nativos.
