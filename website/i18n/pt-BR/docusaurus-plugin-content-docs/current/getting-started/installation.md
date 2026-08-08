---
sidebar_position: 2
title: "Instalação"
description: "Instale o Hermes Agent no Linux, macOS, WSL2, Windows nativo ou Android via Termux"
---

# Instalação

Coloque o Hermes Agent para rodar em menos de dois minutos!

:::tip Suporte a plataformas
Para a matriz completa de suporte (quais SOs, métodos de distribuição e
recursos condicionados a plataforma são suportados), veja **[Suporte a plataformas](./platform-support.md)**.
:::

## Instalação rápida
### Com o instalador do Hermes Desktop no macOS ou Windows (recomendado)
Para instalar com facilidade o CLI e o app desktop, [baixe o instalador do Hermes Desktop](https://hermes-agent.nousresearch.com/) no site e execute-o.

### Sem o Hermes Desktop:
Para uma instalação só de linha de comando, sem o Hermes Desktop, execute:

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (nativo) {#windows-native}

No PowerShell:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

Se quiser instalar e rodar o Hermes Desktop depois de uma instalação só de CLI, basta executar
```bash
hermes desktop
```

### O que o instalador faz

O instalador cuida de tudo automaticamente — todas as dependências (Python, Node.js, ripgrep, ffmpeg), o clone do repositório, o ambiente virtual, o comando global `hermes` e a configuração do provider de LLM. No fim, você já pode conversar.

#### Layout da instalação

Onde o instalador coloca as coisas depende de você instalar como usuário normal ou como root:

| Instalador                              | Código fica em                  | Binário `hermes`                         | Diretório de dados                       |
| -------------------------------------- | ------------------------------ | --------------------------------------- | ------------------------------------ |
| Por usuário (instalador git)               | `~/.hermes/hermes-agent/`      | `~/.local/bin/hermes` (symlink)         | `~/.hermes/`                         |
| Modo root (`sudo curl … \| sudo bash`) | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes`                 | `/root/.hermes/` (ou `$HERMES_HOME`) |

O **layout FHS** do modo root (`/usr/local/lib/…`, `/usr/local/bin/hermes`) bate com onde outras tools de desenvolvimento system-wide ficam no Linux. É útil em máquinas compartilhadas em que uma instalação do sistema deve servir todos os usuários. Config por usuário (auth, skills, sessões) continua em `~/.hermes/` de cada um ou num `HERMES_HOME` explícito.

### Depois da instalação

Recarregue o shell e comece a conversar:

```bash
source ~/.bashrc   # or: source ~/.zshrc
hermes             # Start chatting!
```

Para reconfigurar opções individuais depois, use os comandos dedicados:

```bash
hermes model          # Choose your LLM provider and model
hermes tools          # Configure which tools are enabled
hermes gateway setup  # Set up messaging platforms
hermes config set     # Set individual config values
hermes config get     # Inspect individual config values
hermes setup          # Or run the full setup wizard to configure everything at once
```

:::tip Caminho mais rápido: Nous Portal
Uma assinatura cobre mais de 300 modelos e o [Tool Gateway](/user-guide/features/tool-gateway) (busca na web, geração de imagem, TTS, browser na nuvem). Sem ficar malabarizando chave por tool:

```bash
hermes setup --portal
```

Isso faz login, define a Nous como provider e liga o Tool Gateway num único comando.
:::

---

## Pré-requisitos

**Instalador:** Fora do Windows, o único pré-requisito é **Git**. No Linux, também garanta que `curl` e `xz-utils` estejam disponíveis (o instalador baixa o Node.js como archive `.tar.xz`). O app desktop ainda precisa de `g++` (ou `build-essential` no Debian/Ubuntu) para compilar módulos nativos. O instalador cuida do resto automaticamente:

- **uv** (gerenciador rápido de pacotes Python)
- **Python 3.11** (via uv, sem sudo)
- **Node.js v22** (para automação de browser e bridge do WhatsApp)
- **ripgrep** (busca rápida em arquivos)
- **ffmpeg** (conversão de áudio para TTS)

:::info
Você **não** precisa instalar Python, Node.js, ripgrep ou ffmpeg na mão. O instalador detecta o que falta e instala por você. Só garanta que o `git` esteja disponível (`git --version`). No Linux, tenha `curl` e `xz-utils` (`sudo apt install curl xz-utils` no Debian/Ubuntu). Para o app desktop, instale também `build-essential` (`sudo apt install build-essential`).
:::

:::tip Usuários Nix
O Nix **não é mais um caminho de instalação oficialmente suportado** (só best-effort). Se você já usa Nix (em NixOS, macOS ou Linux), há um caminho dedicado com flake Nix, módulo declarativo NixOS e modo container opcional. Veja o guia **[Nix & NixOS Setup](./nix-setup.md)**.
:::

---

## Instalação manual / para desenvolvimento

Se quiser clonar o repo e instalar a partir do source — para contribuir, rodar de um branch específico ou ter controle total do ambiente virtual — veja a seção [Development Setup](../developer-guide/contributing.md#development-setup) no guia de contribuição.

---

## Instalações sem sudo / usuário de serviço do sistema

Rodar o Hermes como usuário sem privilégios dedicado (ex.: conta de serviço systemd `hermes`, ou qualquer usuário sem `sudo`) é suportado. A única coisa no caminho de instalação que de fato precisa de root é o passo `--with-deps` do Playwright, que usa `apt` para instalar bibliotecas compartilhadas (`libnss3`, `libxkbcommon`, etc.) usadas pelo Chromium. O instalador detecta se o sudo está disponível e degrada com elegância quando não está — instala o binário do Chromium no cache Playwright do próprio usuário de serviço e imprime o comando exato que um administrador precisa rodar à parte.

**Divisão recomendada (Debian/Ubuntu):**

1. **Uma vez, como usuário admin com sudo**, instale as bibliotecas de sistema que o Chromium precisa:
   ```bash
   sudo npx playwright install-deps chromium
   ```
   (Pode rodar de qualquer lugar — o `npx` busca o Playwright na hora.)

2. **Como o usuário de serviço sem privilégios**, rode o instalador normal. Ele detecta a falta de sudo, pula `--with-deps` e instala o Chromium no cache Playwright local do usuário:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

   Se quiser pular o passo do Playwright por completo — por exemplo porque está rodando headless e não precisa de automação de browser — passe `--skip-browser`:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
   ```

3. **Deixe o `hermes` disponível nos shells do usuário de serviço.** O instalador grava o launcher em `~/.local/bin/hermes`. Contas de serviço do sistema costumam ter um PATH mínimo que não inclui `~/.local/bin`. Ou adicione ao ambiente do usuário, ou faça symlink do launcher para um local do sistema:
   ```bash
   # Option A — add to the service user's profile
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # Option B — symlink system-wide (run as an admin)
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **Verifique:** `hermes doctor` deve rodar limpo. Se aparecer `ModuleNotFoundError: No module named 'dotenv'`, você está invocando o arquivo `hermes` do source do repo (`~/.hermes/hermes-agent/hermes`) com o Python do sistema em vez do launcher do venv (`~/.hermes/hermes-agent/venv/bin/hermes`) — corrija o passo 3.

O mesmo padrão funciona no Arch (o instalador usa pacman com a mesma lógica de detecção de sudo), Fedora/RHEL e openSUSE — nessas distros o `--with-deps` não é suportado de jeito nenhum, então um administrador sempre instala as bibliotecas de sistema à parte. Os comandos `dnf`/`zypper` relevantes são impressos pelo instalador.

---

## Troubleshooting

| Problema | Solução |
|---------|----------|
| `hermes: command not found` | Recarregue o shell (`source ~/.bashrc`) ou confira o PATH |
| `API key not set` | Rode `hermes model` para configurar o provider, ou `hermes config set OPENROUTER_API_KEY your_key` |
| Config sumiu depois de um update | Rode `hermes config check` e depois `hermes config migrate` |

Para mais diagnósticos, rode `hermes doctor` — ele diz exatamente o que está faltando e como corrigir.

## Detecção automática do método de instalação

O Hermes detecta sozinho se foi instalado via `pip`, instalador git, Homebrew ou NixOS, e o `hermes update` imprime o comando de update correspondente. Não há env var para setar — a detecção se baseia no layout da instalação (site-packages do Python, `~/.hermes/hermes-agent/`, prefixo Homebrew ou path do store Nix). O `hermes doctor` também mostra o método detectado no resumo do ambiente.
