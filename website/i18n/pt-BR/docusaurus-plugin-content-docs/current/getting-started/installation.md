---
sidebar_position: 2
title: "Instalação"
description: "Instale o Hermes Agent no Linux, macOS, WSL2, Windows nativo ou Android via Termux"
---

# Instalação

Coloque o Hermes Agent em funcionamento em menos de dois minutos!

:::tip Suporte de Plataforma
Para a matriz completa de suporte de plataforma (quais SOs, métodos de distribuição e recursos por plataforma são suportados), veja **[Suporte de Plataforma](./platform-support.md)**.
:::

## Instalação Rápida
### Com o instalador do Hermes Desktop no macOS ou Windows (recomendado)
Para instalar facilmente os aplicativos de linha de comando e desktop, [baixe o instalador do Hermes Desktop](https://hermes-agent.nousresearch.com/) em nosso site e execute-o.

### Sem o Hermes Desktop:
Para uma instalação apenas de linha de comando sem o Hermes Desktop, execute:

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (nativo)

Execute no PowerShell:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Se quiser instalar e executar o Hermes Desktop após uma instalação apenas de linha de comando, basta executar
```bash
hermes desktop
```

### O que o instalador faz

O instalador cuida de tudo automaticamente — todas as dependências (Python, Node.js, ripgrep, ffmpeg), o clone do repositório, o ambiente virtual, o comando global `hermes` e a configuração do provedor de LLM. Ao final, você está pronto para conversar.

#### Layout de instalação

Onde o instalador coloca as coisas depende se você está instalando como usuário normal ou como root:

| Instalador                              | O código fica em                | binário `hermes`                         | Diretório de dados                       |
| --------------------------------------- | ------------------------------- | ---------------------------------------- | ---------------------------------------- |
| Por usuário (instalador git)            | `~/.hermes/hermes-agent/`       | `~/.local/bin/hermes` (symlink)          | `~/.hermes/`                             |
| Modo root (`sudo curl … \| sudo bash`)  | `/usr/local/lib/hermes-agent/`  | `/usr/local/bin/hermes`                  | `/root/.hermes/` (ou `$HERMES_HOME`)     |

O **layout FHS em modo root** (`/usr/local/lib/…`, `/usr/local/bin/hermes`) combina com onde outras ferramentas de desenvolvimento do sistema caem no Linux. É útil para implantações de máquina compartilhada onde uma instalação de sistema deve servir a todos os usuários. A configuração por usuário (auth, skills, sessões) ainda vive no `~/.hermes/` de cada usuário ou no `HERMES_HOME` explícito.

### Após a instalação

Recarregue seu shell e comece a conversar:

```bash
source ~/.bashrc   # ou: source ~/.zshrc
hermes             # Comece a conversar!
```

Para reconfigurar configurações individuais depois, use os comandos dedicados:

```bash
hermes model          # Escolha seu provedor e modelo de LLM
hermes tools          # Configure quais ferramentas estão ativas
hermes gateway setup  # Configure as plataformas de mensagens
hermes config set     # Defina valores de configuração individuais
hermes config get     # Inspecione valores de configuração individuais
hermes setup          # Ou rode o assistente completo para configurar tudo de uma vez
```

:::tip Caminho mais rápido: Nous Portal
Uma assinatura cobre 300+ modelos mais o [Tool Gateway](/user-guide/features/tool-gateway) (busca web, geração de imagens, TTS, navegador em nuvem). Evite o vai-e-vem de chaves por ferramenta:

```bash
hermes setup --portal
```

Isso faz login, define a Nous como provedor e ativa o Tool Gateway em um único comando.
:::

---

## Pré-requisitos

**Instalador:** Em plataformas não-Windows, o único pré-requisito é **Git**. No Linux, garanta também que `curl` e `xz-utils` estejam disponíveis (o instalador baixa o Node.js como arquivo `.tar.xz`). O aplicativo desktop adicionalmente requer `g++` (ou `build-essential` no Debian/Ubuntu) para compilar módulos nativos. O instalador cuida automaticamente de todo o resto:

- **uv** (gerenciador de pacotes Python rápido)
- **Python 3.11** (via uv, sem precisar de sudo)
- **Node.js v22** (para automação de navegador e ponte do WhatsApp)
- **ripgrep** (busca rápida de arquivos)
- **ffmpeg** (conversão de formato de áudio para TTS)

:::info
Você **não** precisa instalar Python, Node.js, ripgrep ou ffmpeg manualmente. O instalador detecta o que falta e instala por você. Apenas garanta que o `git` esteja disponível (`git --version`). No Linux, garanta que `curl` e `xz-utils` estejam instalados (`sudo apt install curl xz-utils` no Debian/Ubuntu). Para o app desktop, instale também `build-essential` (`sudo apt install build-essential`).
:::

:::tip Usuários de Nix
O Nix **não é mais um caminho de instalação explicitamente suportado** (apenas melhor esforço). Se você já usa Nix (no NixOS, macOS ou Linux), existe um caminho de configuração dedicado com flake Nix, módulo declarativo NixOS e modo de container opcional. Veja o guia **[Nix e NixOS](./nix-setup.md)**.
:::

---

## Instalação Manual / para Desenvolvedores

Se quiser clonar o repositório e instalar a partir do código-fonte — para contribuir, rodar de uma branch específica ou ter controle total sobre o ambiente virtual — veja a seção [Configuração de Desenvolvimento](../developer-guide/contributing.md#development-setup) no guia de Contribuição.

---

## Instalações Sem Sudo / para Usuários de Serviço de Sistema

Rodar o Hermes como um usuário dedicado sem privilégios (por exemplo, uma conta de serviço `hermes` do systemd, ou qualquer usuário sem acesso `sudo`) é suportado. A única coisa no caminho de instalação que realmente precisa de root é o passo `--with-deps` do Playwright, que instala via `apt` as bibliotecas compartilhadas (`libnss3`, `libxkbcommon`, etc.) usadas pelo Chromium. O instalador detecta se o sudo está disponível e degrada graciosamente quando não está — instala o binário do Chromium no cache local do Playwright do usuário de serviço e imprime o comando exato que um administrador precisa executar separadamente.

**Divisão recomendada (Debian/Ubuntu):**

1. **Uma vez, como usuário admin com sudo**, instale as bibliotecas de sistema que o Chromium precisa:
   ```bash
   sudo npx playwright install-deps chromium
   ```
   (Você pode executar de qualquer lugar — o `npx` buscará o Playwright na hora.)

2. **Como o usuário de serviço sem privilégios**, execute o instalador normal. Ele detectará o sudo ausente, pulará o `--with-deps` e instalará o Chromium no cache local do Playwright do usuário:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

   Se quiser pular o passo do Playwright por completo — por exemplo, porque você está rodando sem interface (headless) e não precisa de automação de navegador — use `--skip-browser`:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
   ```

3. **Disponibilize o `hermes` para os shells do usuário de serviço.** O instalador escreve o lançador em `~/.local/bin/hermes`. Contas de serviço de sistema muitas vezes têm um PATH mínimo que não inclui `~/.local/bin`. Ou adicione ao ambiente do usuário, ou crie um symlink para uma localização de sistema:
   ```bash
   # Opção A — adicione ao perfil do usuário de serviço
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # Opção B — symlink em todo o sistema (execute como admin)
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **Verifique:** `hermes doctor` deve rodar sem erros. Se receber `ModuleNotFoundError: No module named 'dotenv'`, você está invocando o arquivo `hermes` da fonte do repositório (`~/.hermes/hermes-agent/hermes`) com o Python do sistema em vez do lançador do venv (`~/.hermes/hermes-agent/venv/bin/hermes`) — corrija o passo 3.

O mesmo padrão funciona no Arch (o instalador usa pacman com a mesma lógica de detecção de sudo), Fedora/RHEL e openSUSE — essas distros não suportam `--with-deps` de forma alguma, então um administrador sempre instala as bibliotecas de sistema separadamente. Os comandos relevantes `dnf`/`zypper` são impressos pelo instalador.

---

## Solução de Problemas

| Problema | Solução |
|----------|---------|
| `hermes: command not found` | Recarregue seu shell (`source ~/.bashrc`) ou verifique o PATH |
| `API key not set` | Rode `hermes model` para configurar seu provedor, ou `hermes config set OPENROUTER_API_KEY sua_chave` |
| Config ausente após atualização | Rode `hermes config check` e depois `hermes config migrate` |

Para mais diagnósticos, rode `hermes doctor` — ele dirá exatamente o que está faltando e como corrigir.

## Auto-detecção do método de instalação

O Hermes auto-detecta se foi instalado via instalador git, Docker ou NixOS, e o `hermes update` imprime o comando de atualização correspondente para aquele caminho. Não há variável de ambiente para definir — a detecção é baseada no layout da instalação (checkout `~/.hermes/hermes-agent/`, stamp de imagem Docker ou caminho do store Nix). O `hermes doctor` também expõe o método detectado em seu resumo de ambiente.
