---
title: "Guia Windows (Nativo)"
description: "Execute o Hermes Agent nativamente no Windows 10 / 11 — instalação, matriz de recursos, console UTF-8, Git Bash, gateway como Tarefa Agendada, editor, PATH, desinstalação e armadilhas comuns"
sidebar_label: "Windows (Nativo)"
sidebar_position: 3
---

# Guia Windows (Nativo)

O Hermes roda nativamente no Windows 10 e Windows 11 — sem WSL, sem Cygwin, sem Docker. Esta página é o mergulho profundo: o que funciona nativamente, o que é só WSL, o que o instalador realmente faz e os ajustes específicos do Windows que você pode precisar tocar.

Se você só quer instalar, o one-liner na [página inicial](/) ou na [página de Instalação](../getting-started/installation#windows-native-powershell) é tudo o que precisa. Volte aqui quando algo te surpreender.

:::tip Prefere WSL?
Se você prefere um ambiente POSIX de verdade (para o terminal incorporado do dashboard, semântica `fork`, file watchers estilo Linux, etc.), veja o **[Guia Windows (WSL2)](./windows-wsl-quickstart.md)**. Os dois coexistem sem conflito: dados nativos ficam em `%LOCALAPPDATA%\hermes`, dados WSL em `~/.hermes`.
:::

## Instalação rápida

Abra o **PowerShell** (ou Windows Terminal) e execute:

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

Não exige direitos de administrador. O instalador vai para `%LOCALAPPDATA%\hermes\` e adiciona `hermes` ao seu **User PATH** — abra um terminal novo depois que terminar.

**Opções do instalador** (exige a forma scriptblock para passar parâmetros):

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1))) -NoVenv -SkipSetup -Branch main
```

| Parameter | Default | Purpose |
|---|---|---|
| `-Branch` | `main` | Clone a specific branch (useful for testing PRs) |
| `-Commit` | unset | Pin install to a specific commit SHA (overrides `-Branch`) |
| `-Tag` | unset | Pin install to a specific git tag (e.g. `v0.14.0`) |
| `-NoVenv` | off | Skip venv creation (advanced — you manage Python yourself) |
| `-SkipSetup` | off | Skip the post-install `hermes setup` wizard |
| `-HermesHome` | `%LOCALAPPDATA%\hermes` | Override data directory |
| `-InstallDir` | `%LOCALAPPDATA%\hermes\hermes-agent` | Override code location |

O instalador tenta de novo automaticamente git fetches instáveis e remove BOM de qualquer payload `install.ps1` baixado, então um BOM UTF-8 capturado durante trânsito HTTP não quebra mais a forma `[scriptblock]::Create((irm ...))`.

### Instalador Desktop (alternativa)

Um instalador GUI fino também está disponível — útil se você prefere dar duplo clique em um `.exe` a abrir o PowerShell. Baixe o Hermes Desktop, execute o instalador e, na primeira inicialização, a GUI chama `install.ps1` por baixo dos panos para provisionar Python (via `uv`), Node, PortableGit e o restante do bootstrap de dependências descrito abaixo. Depois da primeira execução, o app desktop e o `hermes` CLI instalado pelo PowerShell compartilham a mesma instalação `%LOCALAPPDATA%\hermes\hermes-agent` e o mesmo diretório de dados `%LOCALAPPDATA%\hermes` — alterne livremente entre GUI e CLI.

Use o instalador desktop quando quiser uma experiência de instalação Windows familiar ou estiver entregando o Hermes para alguém que não é desenvolvedor; use o one-liner PowerShell quando já estiver em um terminal.

### Bootstrap de dependências (`dep_ensure`)

Na primeira inicialização (e sob demanda quando uma ferramenta ausente é detectada), o Hermes executa um pequeno bootstrapper Python — `hermes_cli/dep_ensure.py` — que verifica e instala preguiçosamente as dependências não-Python de que precisa. No Windows, as relevantes são:

| Dependency | Why Hermes needs it |
|---|---|
| **PortableGit** | Provides `bash.exe` for the terminal tool and `git` for in-session clones. Provisioned at install time, not by `dep_ensure`. |
| **Node.js 22** | Required for the browser tool (`agent-browser`), the TUI's web bridge, and the WhatsApp bridge. |
| **ffmpeg** | Audio format conversion for TTS / voice messages. |
| **ripgrep** | Fast file search — falls back to `grep` if unavailable. |
| **npm packages** | `agent-browser`, Playwright Chromium, and any per-toolset Node deps are installed once at first browser-tool use. |

Cada dep tem uma verificação estilo `shutil.which(...)`; se um binário estiver ausente e a execução for interativa, `dep_ensure` oferece instalá-lo (delegando a `scripts\install.ps1 -ensure <dep>` para a lógica real de instalação). Execuções não interativas (gateway, cron, lançamentos headless do desktop) pulam o prompt e mostram um erro claro `this feature needs <dep>`.

## O que o instalador realmente faz

De cima a baixo, em ordem:

1. **Bootstrap do `uv`** — o gerenciador Python rápido da Astral. Instalado em `%USERPROFILE%\.local\bin`.
2. **Instala Python 3.11** via `uv`. Não precisa de Python existente.
3. **Instala Node.js 22** (winget se disponível, senão um tarball Node portátil descompactado em `%LOCALAPPDATA%\hermes\node`). Usado pela ferramenta de browser e pela ponte WhatsApp.
4. **Instala Git portátil** — se `git` já estiver no PATH, o instalador o usa; caso contrário, baixa um **PortableGit** reduzido e autocontido (~45 MB, do release oficial `git-for-windows`) para `%LOCALAPPDATA%\hermes\git`. Sem admin, sem registro do instalador Windows, sem interferir com mais nada na máquina.
5. **Clona o repo** para `%LOCALAPPDATA%\hermes\hermes-agent` e cria um virtualenv dentro dele.
6. **`uv pip install` em camadas** — tenta `.[all]` primeiro, faz fallback para conjuntos progressivamente menores (`[messaging,dashboard,ext]` → `[messaging]` → `.`) se uma dep `git+https` falhar com rate limit no GitHub. Evita o modo de falha "um flake te deixa com instalação mínima".
7. **Auto-instala SDKs de mensagens** conforme `.env` — se `TELEGRAM_BOT_TOKEN` / `DISCORD_BOT_TOKEN` / `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` / `WHATSAPP_ENABLED` estiverem presentes, executa `python -m ensurepip --upgrade` e `pip install` direcionados para que o SDK de cada plataforma seja importável de fato.
8. **Define `HERMES_GIT_BASH_PATH`** para o `bash.exe` resolvido, para o Hermes encontrá-lo deterministicamente em shells novos.
9. **Adiciona `%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts` ao User PATH e define `HERMES_HOME=%LOCALAPPDATA%\hermes`** — expõe o comando `hermes` (e aponta para seu dir de dados) depois que você abrir um terminal novo.
10. **Executa `hermes setup`** — o wizard normal de primeira execução (modelo, provedor, toolsets). Pule com `-SkipSetup`.

:::tip Pule a caça a chaves de API no Windows
No Windows, a configuração de chaves de API por ferramenta (Firecrawl, FAL, Browser Use, OpenAI TTS) é a parte de maior atrito para ter um agente útil. Uma assinatura [Nous Portal](/user-guide/features/tool-gateway) cobre o modelo **e** todas essas ferramentas com um login OAuth. Depois que o instalador terminar, execute `hermes setup --portal` para conectar tudo.
:::

## Matriz de recursos {#feature-matrix}

Tudo exceto o painel de terminal incorporado do dashboard roda nativamente no Windows.

| Feature | Native Windows | WSL2 |
|---|---|---|
| CLI (`hermes chat`, `hermes setup`, `hermes gateway`, …) | ✓ | ✓ |
| Interactive TUI (`hermes --tui`) | ✓ | ✓ |
| Messaging gateway (Telegram, Discord, Slack, WhatsApp, 15+ platforms) | ✓ | ✓ |
| Cron scheduler | ✓ | ✓ |
| Browser tool (Chromium via Node) | ✓ | ✓ |
| MCP servers (stdio and HTTP) | ✓ | ✓ |
| Local Ollama / LM Studio / llama-server | ✓ | ✓ (via WSL networking) |
| Web dashboard (sessions, jobs, metrics, config) | ✓ | ✓ |
| Dashboard `/chat` embedded terminal pane | ✗ (needs POSIX PTY) | ✓ |
| Auto-start at login | ✓ (schtasks) | ✓ (systemd) |

A aba `/chat` do dashboard incorpora um terminal real via PTY POSIX (`ptyprocess`). O Windows nativo não tem primitivo equivalente; `pywinpty` / Windows ConPTY funcionariam, mas é implementação separada — trate como trabalho futuro. **O restante do dashboard funciona nativamente** — só aquela aba mostra um banner "use WSL2 for this".

## Como o Hermes executa comandos shell no Windows

A ferramenta terminal do Hermes executa comandos via **Git Bash**, mesma estratégia do Claude Code. Isso contorna a lacuna POSIX-vs-Windows sem reescrever cada ferramenta.

Ordem de resolução para `bash.exe`:

1. Variável de ambiente `HERMES_GIT_BASH_PATH` se definida.
2. `%LOCALAPPDATA%\hermes\git\usr\bin\bash.exe` (PortableGit gerenciado pelo instalador).
3. `%LOCALAPPDATA%\hermes\git\bin\bash.exe` (layout antigo Git-for-Windows).
4. Instalação Git-for-Windows do sistema (`%ProgramFiles%\Git\bin\bash.exe`, etc.).
5. MSYS2, Cygwin ou qualquer `bash.exe` no PATH como último recurso.

O instalador define `HERMES_GIT_BASH_PATH` explicitamente para sessões PowerShell novas não precisarem redescobrir. Substitua se quiser que o Hermes use um bash específico — por exemplo, seu Git Bash do sistema ou um bash hospedado no WSL via symlink.

**Armadilha:** o layout do MinGit difere do instalador completo Git-for-Windows — o bash fica em `usr\bin\bash.exe`, não `bin\bash.exe`. O Hermes verifica ambos. Se você descompactar manualmente um zip MinGit, escolha a variante **não-busybox** (`MinGit-*-64-bit.zip`, não `MinGit-*-busybox*.zip`) — builds busybox trazem `ash` em vez de `bash` e faltam a maioria dos coreutils.

## Console UTF-8 no Windows

O stdio padrão do Python no Windows usa a code page ativa do console (geralmente cp1252 ou cp437). O banner do Hermes, lista de slash commands, feed de ferramentas, painéis Rich e descrições de skills contêm Unicode. Sem intervenção, qualquer um disso quebra com `UnicodeEncodeError: 'charmap' codec can't encode character…`.

A correção está em `hermes_cli/stdio.py::configure_windows_stdio()`, chamada cedo em cada entry point (`cli.py::main`, `hermes_cli/main.py::main`, `gateway/run.py::main`). Ela:

1. Muda a code page do console para CP_UTF8 (65001) via `kernel32.SetConsoleCP` / `SetConsoleOutputCP`.
2. Reconfigura `sys.stdout` / `sys.stderr` / `sys.stdin` para UTF-8 com `errors='replace'`.
3. Define `PYTHONIOENCODING=utf-8` e `PYTHONUTF8=1` (via `setdefault`, para valores explícitos do usuário prevalecerem) para subprocessos Python filhos herdarem UTF-8.
4. Define `EDITOR=notepad` se nem `EDITOR` nem `VISUAL` estiverem definidos (veja a seção Editor abaixo).

Idempotente. No-op fora do Windows.

**Opt out:** `HERMES_DISABLE_WINDOWS_UTF8=1` no ambiente volta ao caminho stdio legado cp1252. Útil para bissectar bug de encoding; improvável ser a configuração certa no uso normal.

## O editor (`Ctrl-X Ctrl-E`, `/edit`)

Antes do #21561, pressionar `Ctrl-X Ctrl-E` ou digitar `/edit` silenciosamente não fazia nada no Windows. O prompt_toolkit tem uma lista fallback hardcoded POSIX-absoluta (`/usr/bin/nano`, `/usr/bin/pico`, `/usr/bin/vi`, …) que nunca resolve no Windows — mesmo com Git for Windows completo instalado.

O shim stdio Windows do Hermes agora define `EDITOR=notepad` como padrão. O Notepad vem em toda instalação Windows e funciona como editor bloqueante — `subprocess.call(["notepad", file])` bloqueia até a janela fechar.

**Substituições do usuário ainda prevalecem** (são verificadas antes do setdefault):

| Editor | PowerShell command |
|---|---|
| VS Code | `$env:EDITOR = "code --wait"` |
| Notepad++ | `$env:EDITOR = "'C:\Program Files\Notepad++\notepad++.exe' -multiInst -nosession"` |
| Neovim | `$env:EDITOR = "nvim"` |
| Helix | `$env:EDITOR = "hx"` |

A flag `--wait` no VS Code é crítica — sem ela o editor retorna imediatamente e o Hermes recebe um buffer vazio.

Defina permanentemente no seu perfil PowerShell:

```powershell
# In $PROFILE
$env:EDITOR = "code --wait"
```

Ou como variável de ambiente User em Configurações do Sistema para todo shell novo pegar.

## `Ctrl+Enter` para nova linha no CLI

O Windows Terminal passa `Ctrl+Enter` como sequência de tecla dedicada. O Hermes associa a "inserir nova linha" para você compor prompts multilinha no CLI sem recorrer a `Esc`-then-`Enter`. Funciona no Windows Terminal, terminal integrado do VS Code e qualquer host de console Windows moderno que honre sequências de escape VT.

Em consoles legados `cmd.exe`, `Ctrl+Enter` colapsa para `Enter` simples — use `Esc Enter`, ou atualize para Windows Terminal (é gratuito e vem por padrão no Windows 11).

## Executando o gateway no login do Windows

`hermes gateway install` no Windows usa **Tarefas Agendadas** com fallback para pasta Startup — sem admin.

### Instalar

```powershell
hermes gateway install
```

O que acontece por baixo:

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN HermesGateway` — registra uma tarefa que roda no seu login com permissões padrão (não elevadas). Sem prompt UAC.
2. Se schtasks estiver bloqueado por group policy, faz fallback escrevendo um atalho `start /min cmd.exe /d /c <wrapper>` em `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`. Mesmo efeito, um pouco mais bruto.
3. Inicia o gateway **desanexado via `pythonw.exe`** — não `python.exe`. `pythonw.exe` não tem console anexado, o que o imuniza contra broadcasts `CTRL_C_EVENT` de processos irmãos (problema real que matava o gateway quando você Ctrl+C'd qualquer coisa no mesmo process group).

Flags usadas ao spawnar: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`.

### Gerenciar

```powershell
hermes gateway status      # Merged view: schtasks + Startup folder + running PID
hermes gateway start       # Starts the scheduled task now
hermes gateway stop        # Graceful SIGTERM equivalent (TerminateProcess via psutil)
hermes gateway restart
hermes gateway uninstall   # Removes schtasks entry, Startup shortcut, pid file
```

`hermes gateway status` é idempotente — chame mil vezes seguidas e nunca mata o gateway acidentalmente. (Antes do PR #21561, matava silenciosamente, via `os.kill(pid, 0)` colidindo com `CTRL_C_EVENT` no nível C — veja "process management internals" abaixo se quiser a história.)

### Por que não um Serviço Windows?

Serviços exigem direitos de admin para instalar e amarram o ciclo de vida do gateway ao boot da máquina, não ao login do usuário. O usuário típico do Hermes quer: logar → gateway disponível, deslogar → gateway some. Tarefas Agendadas fazem exatamente isso sem elevação. Se você realmente quer um serviço, use `nssm` ou `sc create` manualmente — mas provavelmente não precisa.

## Layout de dados

| Path | Contents |
|---|---|
| `%LOCALAPPDATA%\hermes\hermes-agent\` | Git checkout + venv. `venv\Scripts\hermes.exe` is the command added to User PATH. Safe to `Remove-Item -Recurse` and reinstall. |
| `%LOCALAPPDATA%\hermes\git\` | PortableGit (only if the installer provisioned it). |
| `%LOCALAPPDATA%\hermes\node\` | Portable Node.js (only if the installer provisioned it). |
| `%LOCALAPPDATA%\hermes\bin\` | Hermes's managed `uv.exe` (the Python manager it uses for updates). |
| `%LOCALAPPDATA%\hermes\` (root) | Your config, auth, skills, sessions, logs (`config.yaml`, `.env`, `skills\`, `sessions\`, `logs\`, …). **Survives reinstalls.** |

No Windows nativo, o instalador define `HERMES_HOME=%LOCALAPPDATA%\hermes`, então seus dados e a instalação descartável ficam sob a **mesma** raiz `%LOCALAPPDATA%\hermes`: instalação/runtime são os subdiretórios `hermes-agent\`, `git\`, `node\` e `bin\`, enquanto seus arquivos de dados ficam diretamente em `%LOCALAPPDATA%\hermes`. Reinstalar só substitui o checkout `hermes-agent\`, então seus dados sobrevivem — mas como os dois compartilham uma raiz, **não** faça `Remove-Item -Recurse %LOCALAPPDATA%\hermes` se quiser manter seus dados; delete o subdiretório `hermes-agent\` em vez disso. Seu diretório de dados tem forma idêntica a um `~/.hermes` Linux, então você pode espelhá-lo entre máquinas.

**Substituir `HERMES_HOME`:** defina a variável de ambiente para apontar a outro dir de dados (ex.: `%USERPROFILE%\.hermes` para combinar com layout Linux/WSL). Funciona igual ao Linux.

## Ferramenta de browser

A ferramenta de browser usa `agent-browser` (helper Node) para dirigir Chromium. No Windows:

- O instalador coloca `agent-browser` no PATH via npm.
- `shutil.which("agent-browser", path=...)` pega o shim `.cmd` automaticamente — `CreateProcessW` não executa shebang sem extensão, então o Hermes sempre resolve para o wrapper `.CMD`. Não invoque manualmente o script shebang; sempre use o `.cmd`.
- Playwright Chromium é auto-instalado na primeira execução (`npx playwright install chromium`). Se a instalação falhar, `hermes doctor` mostra com dica de correção.

## Executando Hermes no Windows — notas práticas

### PATH após instalação

O instalador adiciona `%LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts` ao seu **User PATH** via `[Environment]::SetEnvironmentVariable`. Terminais existentes não pegam isso — abra uma janela PowerShell nova (ou aba do Windows Terminal) após a instalação. Feche e reabra; não faça `$env:PATH += …` manualmente a menos que saiba o que está fazendo.

Verifique:

```powershell
Get-Command hermes        # should print C:\Users\<you>\AppData\Local\hermes\hermes-agent\venv\Scripts\hermes.exe
hermes --version
```

### Variáveis de ambiente

O Hermes honra tanto `$env:X` (escopo de processo) quanto variáveis User (permanentes, em Propriedades do Sistema → Variáveis de Ambiente). Definir chaves de API em `%LOCALAPPDATA%\hermes\.env` (seu `HERMES_HOME`) é o caminho normal — igual ao Linux:

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

Não coloque segredos em variáveis User a menos que queira explicitamente que todo processo Windows os veja (não é o que você quer).

### Env vars específicas do Windows

Estas afetam só instalações Windows nativas:

| Variable | Effect |
|---|---|
| `HERMES_GIT_BASH_PATH` | Override bash.exe discovery. Point at any bash — full Git-for-Windows, WSL bash via symlink, MSYS2, Cygwin. The installer sets this automatically. |
| `HERMES_DISABLE_WINDOWS_UTF8` | Set to `1` to disable the UTF-8 stdio shim and fall back to the locale code page. Useful for bisecting an encoding bug. |
| `EDITOR` / `VISUAL` | Your editor for `/edit` and `Ctrl-X Ctrl-E`. Hermes defaults to `notepad` if both are unset. |

## Desinstalação

No PowerShell:

```powershell
hermes uninstall
```

Esse é o caminho limpo — remove a entrada schtasks, atalho da pasta Startup, shim `hermes.cmd`, exclui `%LOCALAPPDATA%\hermes\hermes-agent\` e apara o User PATH. Deixa o restante de `%LOCALAPPDATA%\hermes\` intacto (sua config, auth, skills, sessões, logs) caso você esteja reinstalando.

Para apagar tudo:

```powershell
hermes uninstall
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hermes"
# Also remove a legacy CLI/WSL data dir if you ever used one:
Remove-Item -Recurse -Force "$env:USERPROFILE\.hermes"
```

O subcomando CLI `hermes uninstall` também trata o caso em que a entrada schtasks foi registrada com outro nome de tarefa (instalações antigas) — busca pelo caminho de instalação em vez de nome de tarefa hardcoded.

## Internals de gerenciamento de processos

Isso é material de fundo — pule a menos que esteja depurando uma estranheza "está se matando sozinho".

No Linux e macOS, o idiom POSIX `os.kill(pid, 0)` é uma verificação de permissão no-op: "este PID está vivo e posso sinalizá-lo?" No Windows, `os.kill` do Python mapeia `sig=0` para `CTRL_C_EVENT` — colidem no valor inteiro 0 — e roteia via `GenerateConsoleCtrlEvent(0, pid)`, que transmite Ctrl+C para **todo o process group do console** que contém o PID alvo. Isso é [bpo-14484](https://bugs.python.org/issue14484), aberto desde 2012. Não será corrigido porque mudar quebraria scripts que dependem do comportamento atual.

Consequência: qualquer caminho que dizia "verificar se este PID está vivo" via `os.kill(pid, 0)` no Windows estava matando silenciosamente o alvo. O Hermes migrou todo site assim (14 em 11 arquivos) para `gateway.status._pid_exists()`, que usa `psutil.pid_exists()` (que por sua vez usa `OpenProcess + GetExitCodeProcess` no Windows — sem sinais). Se você escreve um plugin ou patch, use `psutil.pid_exists()` diretamente ou `gateway.status._pid_exists()` — nunca `os.kill(pid, 0)`.

`scripts/check-windows-footguns.py` impõe isso no CI: qualquer nova chamada `os.kill(pid, 0)` falha a verificação `Windows footguns (blocking)` a menos que a linha tenha marcador `# windows-footgun: ok — <reason>`.

## Armadilhas comuns

**`hermes: command not found` logo após instalar.**
Abra uma janela PowerShell nova. O instalador adicionou `%LOCALAPPDATA%\hermes\bin` ao User PATH, mas shells existentes precisam reiniciar para pegar. Enquanto isso, execute `& "$env:LOCALAPPDATA\hermes\bin\hermes.cmd"`.

**`WinError 193: %1 is not a valid Win32 application` ao executar uma ferramenta.**
Você acertou uma invocação de script shebang que contornou o shim `.cmd`. O Hermes resolve comandos via `shutil.which(cmd, path=local_bin)` para PATHEXT pegar `.CMD` — se você invoca a ferramenta por caminho hardcoded, mude para a variante `.cmd` (ex.: `npx.cmd`, não `npx`).

**`[scriptblock]::Create(...)` falha com `The assignment expression is not valid`.**
Seu download de `install.ps1` pegou BOM UTF-8. A forma `irm | iex` remove BOMs automaticamente; `[scriptblock]::Create((irm ...))` não. Reexecute com a forma simples `irm | iex`, ou baixe o script manualmente e salve sem BOM via `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))`.

**Gateway não permanece rodando após reiniciar.**
Verifique `hermes gateway status` — mescla entrada schtasks, atalho da pasta Startup (se usado) e PID ao vivo. Se schtasks está registrado mas não rodando, group policy pode estar bloqueando triggers `ONLOGON`. Execute `schtasks /Query /TN HermesGateway /V /FO LIST` para ver o motivo da falha, ou faça fallback para o caminho da pasta Startup desinstalando e reinstalando com `HERMES_GATEWAY_FORCE_STARTUP=1`.

**`/edit` ainda não faz nada depois de definir `$env:EDITOR`.**
Você definiu só no processo atual; feche e reabra o shell, ou defina em escopo User em Propriedades do Sistema → Variáveis de Ambiente. Verifique com `echo $env:EDITOR` em uma janela PowerShell nova.

**Ferramenta de browser inicia mas ferramentas dão timeout.**
Chromium é auto-instalado na primeira execução. Se a instalação falhou (GitHub rate-limited, soluço na CDN Playwright), execute `hermes doctor` — mostrará o Chromium ausente e imprimirá o comando exato `npx playwright install chromium` para corrigir.

**`agent-browser` falha com erro estranho de versão Node.**
O instalador provisiona Node 22 em `%LOCALAPPDATA%\hermes\node`, mas seu PATH pode ter um Node 18 do sistema primeiro. Mova o dir node do Hermes mais cedo no PATH, ou delete a instalação do sistema se não usa Node em outro lugar.

**Caracteres chineses / japoneses / árabes aparecem como `?` no CLI.**
O shim stdio UTF-8 não ativou. Verifique que `HERMES_DISABLE_WINDOWS_UTF8` NÃO está definido (`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`). Se estiver vazio e você ainda vê `?`, o host do console (`cmd.exe` muito antigo) pode não suportar UTF-8 — mude para Windows Terminal.

**Gateway não envia fotos Telegram — "`BadRequest: payload contains invalid characters`".**
Isso não tem relação com Windows, mas às vezes aparece primeiro lá. Geralmente significa que seu caminho de arquivo contém barras invertidas não escapadas em um corpo JSON. Telegram deveria receber caminhos que o Hermes normaliza, não caminhos Windows crus — se você vê isso dentro de um plugin customizado, passe o caminho fornecido pelo Hermes, não `str(Path(...))` de input do usuário.

**Estranheza de encoding "funciona na minha outra máquina" após `git pull`.**
Se você editou config Hermes ou uma skill no Windows com editor não-UTF-8 (Notepad em versões antigas do Windows, alguns IMEs chineses), o arquivo pode ter sido salvo com BOM. O Hermes tolera `utf-8-sig` na maioria das leituras de config, mas BOM dentro de um escalar YAML dobrado (`description: >`) quebra silenciosamente o parsing YAML. Re-salve o arquivo como UTF-8 puro sem BOM.

## Próximos passos

- **[Installation](../getting-started/installation.md)** — a página completa de instalação, incluindo Linux/macOS/WSL2/Termux.
- **[Windows (WSL2) Guide](./windows-wsl-quickstart.md)** — se quiser semântica POSIX ou o painel de terminal do dashboard.
- **[CLI Reference](../reference/cli-commands.md)** — cada subcomando `hermes`.
- **[FAQ](../reference/faq.md)** — perguntas comuns não específicas do Windows.
- **[Messaging Gateway](./messaging/index.md)** — executando Telegram/Discord/Slack no Windows.
