---
sidebar_position: 2
title: "Instalação"
description: "Instale o Hermes Agent no Linux, macOS, WSL2, Windows nativo ou Android via Termux"
---

# Instalação

Instale o Hermes Agent em menos de dois minutos!

:::tip Suporte a Plataformas
Para a matriz completa de suporte a plataformas (quais SOs, métodos de distribuição e
recursos vinculados a plataformas são suportados), veja **[Suporte a Plataformas](./platform-support.md)**.
:::

## Instalação Rápida
### Com o instalador Hermes Desktop no macOS ou Windows (recomendado)
Para instalar facilmente os aplicativos de linha de comando e desktop, [baixe o instalador do Hermes Desktop](https://hermes-agent.nousresearch.com/) do nosso site e execute-o.

### Sem o Hermes Desktop:
Para uma instalação apenas com linha de comando sem o Hermes Desktop, execute:

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (nativo)

Execute no powershell:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

Se você quiser instalar e executar o Hermes Desktop após uma instalação apenas com linha de comando, simplesmente execute
```bash
hermes desktop
```

### O que o Instalador Faz

O instalador cuida de tudo automaticamente — todas as dependências (Python, Node.js, ripgrep, ffmpeg), o clone do repositório, ambiente virtual, configuração do comando global `hermes` e configuração do provider de LLM. Ao final, você estará pronto para conversar.

#### Estrutura da Instalação

Onde o instalador coloca as coisas depende se você está instalando como usuário normal ou como root:

| Instalador                              | Código fica em                    | Binário `hermes`                              | Diretório de dados                    |
| --------------------------------------- | --------------------------------- | --------------------------------------------- | ------------------------------------- |
| Por-usuário (git installer)             | `~/.hermes/hermes-agent/`         | `~/.local/bin/hermes` (symlink)               | `~/.hermes/`                          |
| Modo root (`sudo curl … | sudo bash`) | `/usr/local/lib/hermes-agent/`   | `/usr/local/bin/hermes`                      | `/root/.hermes/` (ou `$HERMES_HOME`) |

O modo root com **layout FHS** (`/usr/local/lib/…`, `/usr/local/bin/hermes`) segue o padrão onde outras ferramentas de desenvolvedor do sistema são instaladas no Linux. É útil para implantações em máquinas compartilhadas onde uma única instalação do sistema deve atender a todos os usuários. A configuração por usuário (auth, skills, sessões) ainda fica no `~/.hermes/` de cada usuário ou no `HERMES_HOME` explícito.

### Após a Instalação

Recarregue seu shell e comece a conversar:

```bash
source ~/.bashrc   # ou: source ~/.zshrc
hermes             # Comece a conversar!
```

Para reconfigurar configurações individuais depois, use os comandos dedicados:

```bash
hermes model          # Escolha seu provider e modelo LLM
hermes tools          # Configure quais ferramentas estão ativadas
hermes gateway setup  # Configure plataformas de mensagem
hermes config set     # Defina valores de configuração individuais
hermes config get     # Inspecione valores de configuração individuais
hermes setup          # Ou execute o assistente de configuração completo para configurar tudo de uma vez
```

:::tip Caminho mais rápido: Nous Portal
Uma assinatura cobre mais de 300 modelos mais o [Tool Gateway](/user-guide/features/tool-gateway) (pesquisa web, geração de imagem, TTS, navegador na nuvem). Pule a gestão de chaves por ferramenta:

```bash
hermes setup --portal
```

Isso faz login, define Nous como seu provider e ativa o Tool Gateway em um único comando.
:::

---

## Pré-requisitos

**Instalador:** Em plataformas que não sejam Windows, o único pré-requisito é **Git**. No Linux, certifique-se também de que `curl` e `xz-utils` estão disponíveis (o instalador baixa o Node.js como um arquivo `.tar.xz`). O aplicativo desktop requer adicionalmente `g++` (ou `build-essential` no Debian/Ubuntu) para compilar módulos nativos. O instalador cuida automaticamente de todo o resto:

- **uv** (gerenciador de pacotes Python rápido)
- **Python 3.11** (via uv, sem necessidade de sudo)
- **Node.js v22** (para automação de navegador e bridge WhatsApp)
- **ripgrep** (busca rápida em arquivos)
- **ffmpeg** (conversão de formato de áudio para TTS)

:::info
Você **não** precisa instalar Python, Node.js, ripgrep ou ffmpeg manualmente. O instalador detecta o que está faltando e instala para você. Apenas certifique-se de que `git` está disponível (`git --version`). No Linux, garanta que `curl` e `xz-utils` estejam instalados (`sudo apt install curl xz-utils` no Debian/Ubuntu). Para o aplicativo desktop, instale também `build-essential` (`sudo apt install build-essential`).
:::

:::tip Usuários Nix
Nix **não é mais um caminho de instalação explicitamente suportado** (apenas melhor esforço). Se você já usa Nix (no NixOS, macOS ou Linux), há um caminho de configuração dedicado com um flake Nix, módulo NixOS declarativo e modo container opcional. Veja o guia **[Nix e NixOS Setup](./nix-setup.md)**.
:::

---

## Instalação Manual / Desenvolvedor

Se você quiser clonar o repositório e instalar a partir do código fonte — para contribuir, executar a partir de um branch específico, ou ter controle total sobre o ambiente virtual — veja a seção [Development Setup](../developer-guide/contributing.md#development-setup) no guia de Contribuição.

---

## Instalações sem Sudo / Usuário de Serviço do Sistema

Executar o Hermes como um usuário dedicado sem privilégios (ex.: uma conta de serviço systemd `hermes`, ou qualquer usuário sem acesso `sudo`) é suportado. A única coisa no caminho de instalação que realmente precisa de root é o passo `--with-deps` do Playwright, que instala bibliotecas compartilhadas (`libnss3`, `libxkbcommon`, etc.) usadas pelo Chromium via `apt`. O instalador detecta se o sudo está disponível e degrada graciosamente quando não está — ele instalará o binário do Chromium no cache do Playwright do próprio usuário do serviço e imprimirá o comando exato que um administrador precisa executar separadamente.

**Divisão recomendada (Debian/Ubuntu):**

1. **Uma vez, como usuário administrador com sudo**, instale as bibliotecas de sistema que o Chromium precisa:
   ```bash
   sudo npx playwright install-deps chromium
   ```
   (Você pode executar isso de qualquer lugar — `npx` baixará o Playwright na hora.)

2. **Como usuário do serviço sem privilégios**, execute o instalador normal. Ele detectará a falta de sudo, pulará `--with-deps` e instalará o Chromium no cache local do Playwright do usuário:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

   Se você quiser pular a etapa do Playwright completamente — por exemplo, porque está executando headless e não precisa de automação de navegador — passe `--skip-browser`:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
   ```

3. **Disponibilize `hermes` para os shells do usuário do serviço.** O instalador escreve o lançador em `~/.local/bin/hermes`. Contas de serviço do sistema geralmente têm um PATH mínimo que não inclui `~/.local/bin`. Adicione-o ao ambiente do usuário ou crie um symlink do lançador para um local do sistema:
   ```bash
   # Opção A — adicione ao perfil do usuário do serviço
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # Opção B — symlink em todo o sistema (execute como administrador)
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **Verifique:** `hermes doctor` agora deve executar sem problemas. Se você receber `ModuleNotFoundError: No module named 'dotenv'`, está invocando o arquivo `hermes` do código fonte (`~/.hermes/hermes-agent/hermes`) com o Python do sistema em vez do lançador do venv (`~/.hermes/hermes-agent/venv/bin/hermes`) — corrija o passo 3.

O mesmo padrão funciona no Arch (o instalador usa pacman com a mesma lógica de detecção de sudo), Fedora/RHEL e openSUSE — essas distribuições não suportam `--with-deps`, então um administrador sempre instala as bibliotecas de sistema separadamente. Os comandos `dnf`/`zypper` relevantes são impressos pelo instalador.

---

## Solução de Problemas

| Problema                          | Solução                                                            |
|-----------------------------------|--------------------------------------------------------------------|
| `hermes: command not found`       | Recarregue seu shell (`source ~/.bashrc`) ou verifique o PATH      |
| `API key not set`                 | Execute `hermes model` para configurar seu provider, ou `hermes config set OPENROUTER_API_KEY sua_key` |
| Config ausente após atualização   | Execute `hermes config check` e depois `hermes config migrate`     |

Para mais diagnósticos, execute `hermes doctor` — ele dirá exatamente o que está faltando e como corrigir.

## Detecção automática do método de instalação

O Hermes detecta automaticamente se foi instalado via `pip`, git installer, Homebrew ou NixOS, e `hermes update` imprime o comando de atualização correspondente para aquele caminho. Não há variável de ambiente para definir — a detecção é baseada no layout da instalação (Python site-packages, `~/.hermes/hermes-agent/`, prefixo Homebrew ou caminho da Nix store). `hermes doctor` também exibe o método detectado em seu resumo de ambiente.
