---
title: "Guia Windows (WSL2)"
description: "Execute o Hermes Agent no Windows via WSL2 — setup, acesso ao filesystem entre Windows e Linux, rede e armadilhas comuns"
sidebar_label: "Windows (WSL2)"
sidebar_position: 2
---

# Guia Windows (WSL2)

O Hermes Agent agora suporta **tanto** Windows nativo quanto WSL2. Esta página cobre o caminho WSL2; para a instalação nativa PowerShell, veja o **[Guia Windows (Nativo)](./windows-native.md)** dedicado.

**Quando escolher WSL2 em vez de nativo:**
- Você quer usar o terminal incorporado do dashboard (aba `/chat`) — aquele painel exige PTY POSIX e é só WSL2.
- Você faz desenvolvimento pesado em POSIX e quer que suas sessões Hermes compartilhem o mesmo filesystem / caminhos das suas ferramentas de dev.
- Você já tem um ambiente WSL2 e não quer manter uma segunda instalação.

**Quando nativo basta (ou é melhor):**
- Chat interativo, gateway (Telegram/Discord/etc.), agendador cron, ferramenta de browser, servidores MCP e a maioria dos recursos Hermes rodam nativamente no Windows.
- Você não quer pensar em cruzar o limite WSL↔Windows toda vez que referencia um arquivo ou abre uma URL.

No WSL2 existem efetivamente dois computadores em jogo: seu host Windows e uma VM Linux gerenciada pelo WSL. A maior parte da confusão vem de não ter certeza em qual você está a cada momento.

Este guia cobre as partes dessa divisão que afetam especificamente o Hermes: instalar WSL2, mover arquivos entre Windows e Linux, rede nos dois sentidos e as armadilhas que as pessoas realmente encontram.

:::info 简体中文
Um walkthrough em chinês do caminho mínimo de instalação é mantido nesta mesma página — troque pelo menu **language** (canto superior direito) e selecione **简体中文**.
:::

## Por que WSL2 (vs. Windows nativo)

A instalação nativa Windows roda diretamente no Windows: seu terminal Windows (PowerShell, Windows Terminal, etc.), caminhos do filesystem Windows (`C:\Users\…`) e processos Windows. O Hermes usa Git Bash para executar comandos shell, como o Claude Code e outros agentes fazem no Windows hoje — contorna a lacuna POSIX-vs-Windows sem reescrita completa.

O WSL2 roda um kernel Linux real em uma VM leve, então o Hermes dentro dele é essencialmente idêntico a rodar no Ubuntu. Isso é valioso quando você quer um ambiente POSIX de verdade: `fork`, `/tmp`, sockets UNIX, semântica de sinais, terminais com PTY, shells como `bash`/`zsh` e ferramentas como `rg`, `git`, `ffmpeg` que se comportam como no Linux.

Consequências práticas do WSL2:

- O CLI Hermes, gateway, sessões, memória, skills e runtimes de ferramentas vivem dentro da VM Linux.
- Programas Windows (navegadores, apps nativos, Chrome com seu perfil logado) ficam fora dela.
- Toda vez que você quer que os dois conversem — compartilhar arquivos, abrir URLs, controlar Chrome, alcançar um servidor de modelo local, expor o gateway Hermes ao seu celular — você cruza um limite. Esses limites são o foco deste guia.

## Instalar WSL2

De um **PowerShell Admin** ou Windows Terminal:

```powershell
wsl --install
```

Em uma caixa Windows 10 22H2+ ou Windows 11 nova, isso instala o kernel WSL2, o recurso Virtual Machine Platform e uma distro Ubuntu padrão. Reinicie quando solicitado. Após reiniciar, o Ubuntu abrirá e pedirá username + password Linux — este é um **usuário Linux novo**, sem relação com sua conta Windows.

Verifique que você está de fato no WSL2 (não WSL1 legado):

```powershell
wsl --list --verbose
```

Você deve ver `VERSION  2`. Se uma distro mostrar `VERSION  1`, converta:

```powershell
wsl --set-version Ubuntu 2
wsl --set-default-version 2
```

O Hermes não funciona de forma confiável no WSL1 — o WSL1 traduz syscalls Linux na hora e alguns comportamentos (procfs, sinais, rede) divergem de Linux real.

### Escolha de distro

Ubuntu (LTS) é o que testamos. Debian funciona. Arch e NixOS funcionam para quem os quer, mas o instalador one-liner assume um sistema Debian-derivado com `apt` — veja o [guia de setup Nix](/getting-started/nix-setup) para esse caminho.

### Habilitar systemd (recomendado)

O gateway hermes (e qualquer coisa que você queira manter rodando) é mais fácil de gerenciar com systemd. No WSL moderno, habilite uma vez dentro da sua distro:

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true

[interop]
enabled=true
appendWindowsPath=true

[automount]
options = "metadata,umask=22,fmask=11"
EOF
```

Depois, no PowerShell:

```powershell
wsl --shutdown
```

Reabra seu terminal WSL. `ps -p 1 -o comm=` deve imprimir `systemd`.

A opção de mount `metadata` acima é importante — sem ela, arquivos em `/mnt/c/...` não podem armazenar bits de permissão Linux reais, o que quebra coisas como `chmod +x` em scripts sob caminhos Windows.

### Instalar Hermes dentro do WSL

Com um shell WSL2 aberto:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
hermes
```

O instalador trata WSL2 como Linux comum — nada específico de WSL é necessário. Veja [Installation](/getting-started/installation) para o layout completo.

## Filesystem: cruzando o limite Windows ↔ WSL2

Esta é a parte que mais confunde as pessoas. Existem **dois filesystems**, e onde você coloca seus arquivos importa — por desempenho, correção e o que as ferramentas conseguem ver.

### As duas direções

| Direction | Path inside | Path you use |
|---|---|---|
| Windows disk, seen from WSL | `C:\Users\you\Documents` | `/mnt/c/Users/you/Documents` |
| WSL disk, seen from Windows | `/home/you/code` | `\\wsl$\Ubuntu\home\you\code` (or `\\wsl.localhost\Ubuntu\...` on newer builds) |

Ambos são reais, ambos funcionam, mas **não são o mesmo filesystem** — são ligados por um protocolo de rede 9P por baixo. Isso tem consequências reais de desempenho e semântica.

### Onde colocar Hermes e seus projetos

**Regra prática: mantenha tudo "Linux-ish" dentro do filesystem Linux.**

- Sua instalação Hermes (`~/.hermes/`) — lado Linux. O instalador já faz isso.
- Seus repos git nos quais você trabalha a partir do WSL — lado Linux (`~/code/...`, `~/projects/...`).
- Seus modelos, datasets, venvs — lado Linux.

O que você ganha seguindo esta regra:

- **I/O rápido.** Operações em `/mnt/c/...` passam por 9P e são 10–100× mais lentas que ext4 nativo. `git status` em um repo de 10k arquivos que parece instantâneo em `~/code` pode levar 15+ segundos em `/mnt/c`.
- **Permissões corretas.** Bits de permissão Linux são emulação best-effort em `/mnt/c`. Coisas como `ssh` recusando chave com "bad permissions" ou `chmod +x` falhando silenciosamente são comuns.
- **File watchers confiáveis.** inotify através de 9P é instável — watchers (dev servers, test runners) rotineiramente perdem mudanças em `/mnt/c`.
- **Sem surpresas de case-sensitivity.** Caminhos Windows são case-insensitive por padrão; Linux é case-sensitive. Projetos com `Readme.md` e `README.md` se comportam diferente dependendo do lado.

Coloque coisas em `/mnt/c` só quando **precisa** que o arquivo viva no lado Windows — ex.: quer abrir de um app GUI Windows, ou o DevTools MCP do Chrome Windows precisa que o diretório atual seja um caminho alcançável pelo Windows.

### Movendo arquivos de um lado para o outro

**Do Windows → para o WSL:** o mais fácil é abrir o Explorer e digitar `\\wsl.localhost\Ubuntu` na barra de endereço. Você pode arrastar para `\home\<you>\...`. Ou no PowerShell:

```powershell
wsl cp /mnt/c/Users/you/Downloads/file.pdf ~/incoming/
```

**Do WSL → para o Windows:** copie para `/mnt/c/Users/<you>/...` e aparece imediatamente no Explorer Windows:

```bash
cp ~/reports/output.pdf /mnt/c/Users/you/Desktop/
```

**Abrir um arquivo WSL em um app Windows** (editor GUI, navegador, etc.): use `explorer.exe` ou `wslview`:

```bash
sudo apt install wslu     # once — gives you wslview, wslpath, wslopen, etc.
wslview ~/reports/output.pdf    # opens with the Windows default handler
explorer.exe .                  # opens the current WSL dir in Windows Explorer
```

**Converter caminhos entre os dois universos:**

```bash
wslpath -w ~/code/project        # → \\wsl.localhost\Ubuntu\home\you\code\project
wslpath -u 'C:\Users\you'        # → /mnt/c/Users/you
```

### Finais de linha, BOMs e git

Se você edita arquivos no lado Windows com um editor Windows, podem ganhar finais de linha `CRLF`. Quando `bash` ou Python no lado Linux os lê, scripts shell quebram com `bad interpreter: /bin/bash^M` e Python pode falhar em `.env` com BOM.

A correção é uma config git sensata dentro do WSL (não no Windows):

```bash
git config --global core.autocrlf input
git config --global core.eol lf
```

Para arquivos que já têm CRLF:

```bash
sudo apt install dos2unix
dos2unix path/to/script.sh
```

### "Clonar dentro do WSL ou em `/mnt/c`?"

Clone dentro do WSL. Sempre, a menos que tenha motivo específico. Um fluxo Hermes típico (`hermes chat`, chamadas de ferramenta que `rg`/`ripgrep` o repo, file watchers, gateway em background) será dramaticamente mais rápido e confiável contra `~/code/myrepo` do que `/mnt/c/Users/you/myrepo`.

Uma exceção: **pontes MCP que lançam binários Windows.** Se você usa `chrome-devtools-mcp` via `cmd.exe` (veja [MCP guide: WSL → Windows Chrome](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)), o Windows pode reclamar com aviso `UNC` se o cwd atual do Hermes for `~`. Nesse caso, inicie o Hermes de algum lugar sob `/mnt/c/` para o processo Windows ter um cwd com letra de unidade.

## Rede: WSL ↔ Windows

O WSL2 roda em uma VM leve com sua própria pilha de rede. Isso significa que `localhost` dentro do WSL **não é o mesmo que** `localhost` no Windows — são dois hosts separados do ponto de vista da rede. Você precisa decidir, para cada serviço, em qual direção o tráfego flui e escolher a ponte certa.

Dois casos aparecem constantemente.

### Caso 1 — Hermes no WSL conversa com um serviço no Windows

Mais comum: você roda **Ollama, LM Studio ou llama-server no Windows**, e o Hermes (dentro do WSL) precisa alcançá-lo.

O how-to canônico para isso está no guia de provedores: **[WSL2 Networking for Local Models →](/integrations/providers#wsl2-networking-windows-users)**

Versão curta:

- **Windows 11 22H2+:** ligue o modo mirrored networking (`networkingMode=mirrored` em `%USERPROFILE%\.wslconfig`, depois `wsl --shutdown`). `localhost` passa a funcionar nos dois sentidos.
- **Windows 10 ou builds antigas:** use o IP do host Windows (o default gateway da rede virtual do WSL) e certifique-se de que o servidor no Windows vincula a `0.0.0.0`, não só `127.0.0.1`. O Firewall Windows geralmente também precisa de uma regra para a porta.

Para a tabela completa (endereços de bind Ollama / LM Studio / vLLM / SGLang, one-liners de regra de firewall, helpers de IP dinâmico, workaround Hyper-V firewall), siga o link acima — não duplique aqui.

### Caso 2 — Algo no Windows (ou sua LAN) conversa com Hermes no WSL

Esta é a direção reversa e está menos documentada em outros lugares, mas é o que você precisa para:

- Usar o **web dashboard** Hermes de um navegador Windows.
- Usar o **servidor API compatível com OpenAI** (exposto por `hermes gateway` quando `API_SERVER_ENABLED=true`) de uma ferramenta no lado Windows. Veja a [página do recurso API Server](/user-guide/features/api-server).
- Testar um **gateway de mensagens** (Telegram, Discord, etc.) onde a plataforma pinga uma URL de webhook local — geralmente você usaria `cloudflared`/`ngrok` em vez de port forwarding cru.

#### Subcaso 2a: do próprio host Windows

No **Windows 11 22H2+ com mirrored mode habilitado**, não há nada a fazer. Um processo no WSL que vincula a `0.0.0.0:8080` (ou até `127.0.0.1:8080`) é alcançável de um navegador Windows em `http://localhost:8080`. O WSL publica o bind de volta ao host automaticamente.

No **modo NAT** (Windows 10 / Windows 11 antigo), o "localhost forwarding" padrão no WSL2 geralmente encaminha binds `127.0.0.1` do lado Linux para `localhost` Windows, então um serviço Hermes iniciado com `--host 127.0.0.1` costuma ser alcançável como `http://localhost:PORT` a partir do Windows. Se não for:

- Vincule explicitamente a `0.0.0.0` dentro do WSL.
- Encontre o IP da VM WSL com `ip -4 addr show eth0 | grep inet` e acesse a partir do Windows.

#### Subcaso 2b: de outro dispositivo na sua LAN (celular, tablet, outro PC)

Esta é a dor de verdade. O tráfego flui **dispositivo LAN → host Windows → VM WSL**, e você precisa configurar os dois saltos:

1. **Vincule em todas as interfaces dentro do WSL.** Um processo escutando em `127.0.0.1` nunca será alcançável de fora da VM. Use `0.0.0.0`.

2. **Port-forward Windows → VM WSL.** No mirrored mode isso é automático. No NAT mode você faz manualmente, por porta, em PowerShell Admin:

   ```powershell
   # Grab the WSL VM's current IP (it changes on every WSL restart under NAT)
   $wslIp = (wsl hostname -I).Trim().Split(' ')[0]

   # Forward Windows port 8080 → WSL:8080
   netsh interface portproxy add v4tov4 `
     listenaddress=0.0.0.0 listenport=8080 `
     connectaddress=$wslIp connectport=8080

   # Allow it through Windows Firewall
   New-NetFirewallRule -DisplayName "Hermes WSL 8080" `
     -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
   ```

   Remova depois com `netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080`.

3. **Aponte o dispositivo LAN para `http://<windows-lan-ip>:8080`.**

Como o IP da VM WSL muda a cada reinício no NAT mode, uma regra one-shot sobrevive só até o próximo `wsl --shutdown`. Para algo persistente, use mirrored mode ou coloque o passo port-proxy em um script que roda no login Windows.

Para webhooks de provedores de mensagens na nuvem (Telegram `setWebhook`, eventos Slack, etc.), não lute com port-forwarding — use túneis `cloudflared`. Veja o [guia de webhooks](/user-guide/messaging/webhooks).

## Executando serviços Hermes a longo prazo no Windows

O [Tool Gateway](/user-guide/features/tool-gateway) Hermes e o servidor API são processos de longa duração. No WSL2 você tem algumas opções para mantê-los ativos.

### Atalho na área de trabalho para abrir Hermes rapidamente

Se você só quer um launcher de duplo clique para um shell Hermes interativo, crie
no lado Windows e faça-o entrar no WSL:

1. Clique com o botão direito na área de trabalho Windows e escolha **New -> Shortcut**.
2. Para o destino, use o nome da sua distro (substitua `Ubuntu` se necessário):

   ```text
   wt.exe -w 0 -p "Ubuntu" wsl.exe -d Ubuntu --cd ~ -- bash -ic "hermes"
   ```

3. Dê um nome óbvio como `Hermes`.

Isso abre o Windows Terminal, inicia sua distro WSL, deixa você no home
Linux e lança o Hermes. Se `hermes` ainda não estiver no PATH, abra o WSL
manualmente uma vez e execute `source ~/.bashrc`, ou substitua o comando por
`uv run hermes` dentro do checkout do projeto.

Polimento opcional:

- **Ícone customizado:** abra **Properties -> Change Icon** e aponte para um `.ico`,
  como o favicon Hermes do repo.
- **Launcher fixado:** quando o atalho funcionar, fixe no Start ou Taskbar para
  não precisar procurá-lo de novo.

### Dentro do WSL com systemd (recomendado)

Se você habilitou systemd conforme a seção de setup acima, `hermes gateway` e o servidor API funcionam como em qualquer máquina Linux. Use o wizard de setup do gateway:

```bash
hermes gateway setup
```

Ele oferecerá instalar uma unit systemd de usuário para o gateway subir automaticamente quando o WSL iniciar.

### Fazer o WSL iniciar no login do Windows

A VM WSL só permanece viva enquanto algo a usa. Para manter seu gateway alcançável sem janela de terminal aberta, inicie um processo WSL no login Windows via Agendador de Tarefas:

- **Trigger:** At log on (seu usuário).
- **Action:** Start a program
  - Program: `C:\Windows\System32\wsl.exe`
  - Arguments: `-d Ubuntu --exec /bin/sh -c "sleep infinity"`

Isso mantém a VM viva para o gateway gerenciado por systemd continuar rodando. No Windows 11, os fluxos mais novos `wsl --install --no-launch` + auto-start também funcionam; o truque `sleep infinity` é a versão portável.

## GPU passthrough (modelos locais)

O WSL2 suporta GPUs **NVIDIA** nativamente desde o kernel WSL 5.10.43+ — instale o driver NVIDIA padrão no Windows ( **não** instale driver NVIDIA Linux dentro do WSL), e `nvidia-smi` dentro do WSL verá a GPU. A partir daí, toolkits CUDA, `torch`, `vllm`, `sglang` e `llama-server` compilam contra a GPU real como de costume.

Suporte AMD ROCm e Intel Arc dentro do WSL2 ainda evolui e está fora da matriz de testes do Hermes — pode funcionar com drivers atuais, mas não temos receita para recomendar.

Se você roda um servidor de modelo local **Windows-native** (Ollama for Windows, LM Studio) que já usa sua GPU via drivers Windows, não precisa de GPU passthrough WSL — siga o Caso 1 acima e acesse pela rede a partir do WSL.

## Armadilhas comuns

**"Connection refused" para Ollama / LM Studio hospedado no Windows.**
Veja [WSL2 Networking](/integrations/providers#wsl2-networking-windows-users). Noventa por cento das vezes o servidor está vinculado a `127.0.0.1` e precisa de `0.0.0.0` (Ollama: `OLLAMA_HOST=0.0.0.0`), ou falta uma regra de firewall.

**Lentidão massiva em `git status` / `hermes chat` em um repo.**
Provavelmente você está trabalhando sob `/mnt/c/...`. Mova o repo para `~/code/...` (lado Linux). Ordem de magnitude mais rápido.

**`bad interpreter: /bin/bash^M` em scripts.**
Finais de linha CRLF de um editor Windows. `dos2unix script.sh`, e defina `core.autocrlf input` na config git WSL.

**Aviso "UNC paths are not supported" de binários Windows lançados via MCP.**
O cwd do Hermes está dentro do filesystem Linux, e `cmd.exe` Windows não sabe o que fazer. Inicie o Hermes de `/mnt/c/...` para aquela sessão, ou use um wrapper que faz `cd` para um caminho alcançável pelo Windows antes de invocar o executável Windows.

**Deriva de relógio após sleep/hibernate.**
O relógio WSL2 pode atrasar minutos após o host retomar do sleep, o que quebra qualquer coisa baseada em cert (OAuth, APIs HTTPS). Corrija sob demanda:

```bash
sudo hwclock -s
```

Ou instale `ntpdate` e execute no login.

**DNS para de funcionar após habilitar mirrored mode, ou com VPN conectada.**
Mirrored mode faz proxy das configurações de rede do host para o WSL — se DNS Windows estiver estranho (split-tunnel VPN, resolver corporativo), WSL herda. Workaround: substitua `resolv.conf` manualmente (defina `generateResolvConf=false` em `/etc/wsl.conf`, depois escreva seu `/etc/resolv.conf` com `1.1.1.1` ou o DNS da sua VPN).

**`hermes` not found após executar o instalador.**
O instalador adiciona `~/.local/bin` ao PATH do seu shell via `~/.bashrc`. Você precisa `source ~/.bashrc` (ou abrir um terminal novo) para surtir efeito na sessão atual.

**Windows Defender lento em arquivos WSL.**
Defender escaneia arquivos via ponte 9P quando acessados do Windows, o que amplifica a lentidão de acesso cross-boundary estilo `/mnt/c`. Se você só toca arquivos WSL de dentro do WSL, isso não importa. Se usa ferramentas Windows contra `\\wsl$\...` frequentemente, considere excluir o caminho da distro do scan em tempo real.

**Ficando sem disco.**
WSL2 armazena o disco da VM como VHDX esparso sob `%LOCALAPPDATA%\Packages\...`. Cresce mas não encolhe automaticamente quando você apaga arquivos. Para recuperar espaço: `wsl --shutdown`, depois de PowerShell Admin execute `Optimize-VHD -Path <path-to-ext4.vhdx> -Mode Full` (requer ferramentas Hyper-V) — ou o caminho `diskpart` mais simples documentado nos docs WSL.

## Próximos passos

- **[Installation](/getting-started/installation)** — passos reais de instalação (Linux/WSL2/Termux usam o mesmo instalador).
- **[Integrations → Providers → WSL2 Networking](/integrations/providers#wsl2-networking-windows-users)** — mergulho canônico de rede para servidores de modelo locais.
- **[MCP guide → WSL → Windows Chrome](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)** — controlar seu Chrome Windows logado a partir do Hermes no WSL.
- **[Tool Gateway](/user-guide/features/tool-gateway)** e **[Web Dashboard](/user-guide/features/web-dashboard)** — os serviços de longa duração que você mais often quer expor do WSL para o resto da rede.
