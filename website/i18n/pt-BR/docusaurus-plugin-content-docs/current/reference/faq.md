---
sidebar_position: 3
title: "FAQ e Solução de Problemas"
description: "Perguntas frequentes e soluções para problemas comuns com o Hermes Agent"
---

# FAQ e Solução de Problemas

Respostas rápidas e correções para as perguntas e problemas mais comuns.

---

## Perguntas Frequentes {#frequently-asked-questions}

### Quais provedores de LLM funcionam com o Hermes? {#what-llm-providers-work-with-hermes}

O Hermes Agent funciona com qualquer API compatível com OpenAI. Provedores suportados incluem:

- **[OpenRouter](https://openrouter.ai/)** — acesso a centenas de modelos com uma única chave de API (recomendado para flexibilidade)
- **[Nous Portal](/integrations/nous-portal)** — o gateway de assinatura da Nous Research — mais de 300 modelos além de web/imagem/TTS/browser com um único login OAuth (recomendado para novos usuários)
- **OpenAI** — GPT-5.4, GPT-5-codex, GPT-4.1, GPT-4o, etc.
- **Anthropic** — modelos Claude (API direta, OAuth via `hermes auth add anthropic`, OpenRouter, ou qualquer proxy compatível)
- **Google** — modelos Gemini (API direta via provedor `gemini`, OpenRouter, ou proxy compatível)
- **z.ai / ZhipuAI** — modelos GLM
- **Kimi / Moonshot AI** — modelos Kimi
- **MiniMax** — endpoints globais e da China
- **Modelos locais** — via [Ollama](https://ollama.com/), [vLLM](https://docs.vllm.ai/), [llama.cpp](https://github.com/ggerganov/llama.cpp), [SGLang](https://github.com/sgl-project/sglang), ou qualquer servidor compatível com OpenAI

Defina seu provedor com `hermes model` ou editando `~/.hermes/.env`. Veja a referência de [Variáveis de Ambiente](./environment-variables.md) para todas as chaves de provedor.

### Funciona no Windows/Android/Termux/minha plataforma?? {#does-it-work-on-windowsandroidtermuxmy-plataform}
Veja **[Suporte a Plataformas](../getting-started/platform-support.md)** para a matriz completa de disponibilidade por plataforma.

### Rodo o Hermes no WSL2. Qual é a melhor forma de controlar meu Chrome normal do Windows? {#i-run-hermes-in-wsl2-whats-the-best-way-to-control-my-normal-windows-chrome}

Prefira uma bridge MCP em vez de `/browser connect`.

Padrão recomendado:

- execute o Hermes dentro do WSL2
- continue usando seu Chrome normal já logado no Windows
- adicione `chrome-devtools-mcp` como um servidor MCP via `cmd.exe` ou `powershell.exe`
- deixe o Hermes usar as ferramentas de browser MCP resultantes

Isso é mais confiável do que tentar forçar o transporte core de browser do Hermes a se conectar diretamente através da fronteira WSL2/Windows.

Veja:

- [Usar MCP com o Hermes](../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)
- [Automação de Navegador](../user-guide/features/browser.md#wsl2--windows-chrome-prefer-mcp-over-browser-connect)

### Meus dados são enviados para algum lugar? {#is-my-data-sent-anywhere}

As chamadas de API vão **apenas para o provedor de LLM que você configurar** (ex.: OpenRouter, sua instância local do Ollama). O Hermes Agent não coleta telemetria, dados de uso ou análises. Suas conversas, memória e skills são armazenadas localmente em `~/.hermes/`.

### Posso usar offline / com modelos locais? {#can-i-use-it-offline--with-local-models}

Sim. Execute `hermes model`, selecione **Custom endpoint**, e digite a URL do seu servidor:

```bash
hermes model
# Select: Custom endpoint (enter URL manually)
# API base URL: http://localhost:11434/v1
# API key: ollama
# Model name: qwen3.5:27b
# Context length: 64000   ← Hermes minimum; set this to match your server's actual context window
```

Ou configure diretamente no `config.yaml`:

```yaml
model:
  default: qwen3.5:27b
  provider: custom
  base_url: http://localhost:11434/v1
```

O Hermes persiste o endpoint, provedor e URL base no `config.yaml` para sobreviver a reinicializações. Se seu servidor local tiver exatamente um modelo carregado, `/model custom` o detecta automaticamente. Você também pode definir `provider: custom` no config.yaml — é um provedor de primeira classe, não um alias para outra coisa.

Isso funciona com Ollama, vLLM, servidor llama.cpp, SGLang, LocalAI, e outros. Veja o [guia de Configuração](../user-guide/configuration.md) para detalhes.

:::tip Usuários do Ollama
Se você definiu um `num_ctx` customizado no Ollama (ex.: `ollama run --num_ctx 64000`), certifique-se de definir o tamanho de contexto correspondente no Hermes — o `/api/show` do Ollama reporta o contexto *máximo* do modelo, não o `num_ctx` efetivo que você configurou.
:::

:::tip Timeouts com modelos locais
O Hermes detecta automaticamente endpoints locais e relaxa os timeouts de streaming (timeout de leitura elevado de 120s para 1800s, detecção de stream obsoleto desativada). Se você ainda encontrar timeouts em contextos muito grandes, defina `HERMES_STREAM_READ_TIMEOUT=1800` no seu `.env`. Veja o [guia de LLM local](../guides/local-llm-on-mac.md#timeouts) para detalhes.
:::

### Quanto custa? {#how-much-does-it-cost}

O Hermes Agent em si é **gratuito e de código aberto** (licença MIT). Você paga apenas pelo uso da API de LLM do provedor escolhido. Modelos locais são completamente gratuitos para executar.

### Várias pessoas podem usar uma instância? {#can-multiple-people-use-one-instance}

Sim. O [gateway de mensagens](../user-guide/messaging/index.md) permite que vários usuários interajam com a mesma instância do Hermes Agent via Telegram, Discord, Slack, WhatsApp ou Home Assistant. O acesso é controlado por allowlists (IDs de usuário específicos) e pareamento por DM (o primeiro usuário a enviar mensagem reivindica o acesso).

### Qual é a diferença entre memória e skills? {#whats-the-difference-between-memory-and-skills}

- **Memória** armazena **fatos** — coisas que o agente sabe sobre você, seus projetos e preferências. Memórias são recuperadas automaticamente com base na relevância.
- **Skills** armazenam **procedimentos** — instruções passo a passo de como fazer coisas. Skills são lembradas quando o agente encontra uma tarefa semelhante.

Ambas persistem entre sessões. Veja [Memória](../user-guide/features/memory.md) e [Skills](../user-guide/features/skills.md) para detalhes.

### Posso usar em meu próprio projeto Python? {#can-i-use-it-in-my-own-python-project}

Sim. Importe a classe `AIAgent` e use o Hermes programaticamente:

```python
from run_agent import AIAgent

agent = AIAgent(model="anthropic/claude-opus-4.7")
response = agent.chat("Explain quantum computing briefly")
```

Veja o [guia de Biblioteca Python](../user-guide/features/code-execution.md) para uso completo da API.

---

## Solução de Problemas {#troubleshooting}

### Problemas de Instalação {#installation-issues}

#### `hermes: command not found` após a instalação {#hermes-command-not-found-after-installation}

**Causa:** Seu shell não recarregou o PATH atualizado.

**Solução:**
```bash
# Reload your shell profile
source ~/.bashrc    # bash
source ~/.zshrc     # zsh

# Or start a new terminal session
```

Se ainda não funcionar, verifique o local de instalação:
```bash
which hermes
ls ~/.local/bin/hermes
```

:::tip
O instalador adiciona `~/.local/bin` ao seu PATH. Se você usa uma configuração de shell não padrão, adicione `export PATH="$HOME/.local/bin:$PATH"` manualmente.
:::

#### Versão do Python muito antiga {#python-version-too-old}

**Causa:** O Hermes requer Python 3.11 ou mais recente.

**Solução:**
```bash
python3 --version   # Check current version

# Install a newer Python
sudo apt install python3.12   # Ubuntu/Debian
brew install python@3.12      # macOS
```

O instalador trata isso automaticamente — se você ver esse erro durante a instalação manual, atualize o Python primeiro.

#### Comandos de terminal dizem `node: command not found` (ou `nvm`, `pyenv`, `asdf`, …) {#terminal-commands-say-node-command-not-found-or-nvm-pyenv-asdf}

**Causa:** O Hermes constrói um snapshot de ambiente por sessão executando `bash -l` uma vez na inicialização. Um shell de login bash lê `/etc/profile`, `~/.bash_profile` e `~/.profile`, mas **não carrega `~/.bashrc`** — então ferramentas que se instalam ali (`nvm`, `asdf`, `pyenv`, `cargo`, exports customizados de `PATH`) ficam invisíveis para o snapshot. Isso acontece mais comumente quando o Hermes roda sob systemd ou em um shell minimalista onde nada pré-carregou o perfil de shell interativo.

**Solução:** O Hermes carrega automaticamente `~/.bashrc` por padrão. Se isso não for suficiente — ex.: você é um usuário zsh cujo PATH vive em `~/.zshrc`, ou você inicia o `nvm` a partir de um arquivo independente — liste os arquivos extras a carregar em `~/.hermes/config.yaml`:

```yaml
terminal:
  shell_init_files:
    - ~/.zshrc                     # zsh users: pulls zsh-managed PATH into the bash snapshot
    - ~/.nvm/nvm.sh                # direct nvm init (works regardless of shell)
    - /etc/profile.d/cargo.sh      # system-wide rc files
  # When this list is set, the default ~/.bashrc auto-source is NOT added —
  # include it explicitly if you want both:
  #   - ~/.bashrc
  #   - ~/.zshrc
```

Arquivos ausentes são pulados silenciosamente. O carregamento acontece em bash, então arquivos que dependem de sintaxe exclusiva do zsh podem dar erro — se isso for uma preocupação, carregue apenas a parte que define o PATH (ex.: o `nvm.sh` do nvm diretamente) em vez do arquivo rc inteiro.

Para desativar o comportamento de carregamento automático (semântica estrita de shell de login apenas):

```yaml
terminal:
  auto_source_bashrc: false
```

#### `uv: command not found` {#uv-command-not-found}

**Causa:** O gerenciador de pacotes `uv` não está instalado ou não está no PATH.

**Solução:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

#### Erros de permissão negada durante a instalação {#permission-denied-errors-during-install}

**Causa:** Permissões insuficientes para escrever no diretório de instalação.

**Solução:**
```bash
# Don't use sudo with the installer — it installs to ~/.local/bin
# If you previously installed with sudo, clean up:
sudo rm /usr/local/bin/hermes
# Then re-run the standard installer
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

---

### Problemas de Provedor e Modelo {#provider--model-issues}

#### `/model` mostra apenas um provedor / não consigo trocar de provedor {#model-only-shows-one-provider--cant-switch-providers}

**Causa:** `/model` (dentro de uma sessão de chat) só pode alternar entre provedores que você **já configurou**. Se você só configurou o OpenRouter, isso é tudo que `/model` vai mostrar.

**Solução:** Saia da sua sessão e use `hermes model` no seu terminal para adicionar novos provedores:

```bash
# Exit the Hermes chat session first (Ctrl+C or /quit)

# Run the full provider setup wizard
hermes model

# This lets you: add providers, run OAuth, enter API keys, configure endpoints
```

Após adicionar um novo provedor via `hermes model`, inicie uma nova sessão de chat — `/model` agora mostrará todos os seus provedores configurados.

:::tip Referência rápida
| Quero... | Usar |
|-----------|-----|
| Adicionar um novo provedor | `hermes model` (no terminal) |
| Digitar/alterar chaves de API | `hermes model` (no terminal) |
| Trocar de modelo no meio da sessão | `/model <name>` (dentro da sessão) |
| Trocar para um provedor diferente já configurado | `/model provider:model` (dentro da sessão) |
:::

#### Chave de API não funciona {#api-key-not-working}

**Causa:** A chave está ausente, expirada, definida incorretamente, ou é para o provedor errado.

**Solução:**
```bash
# Check your configuration
hermes config show

# Re-configure your provider
hermes model

# Or set directly
hermes config set OPENROUTER_API_KEY sk-or-v1-xxxxxxxxxxxx
```

:::warning
Certifique-se de que a chave corresponde ao provedor. Uma chave da OpenAI não funcionará com o OpenRouter e vice-versa. Verifique `~/.hermes/.env` por entradas conflitantes.
:::

#### Modelo não disponível / modelo não encontrado {#model-not-available--model-not-found}

**Causa:** O identificador do modelo está incorreto ou não está disponível no seu provedor.

**Solução:**
```bash
# List available models for your provider
hermes model

# Set a valid model
hermes config set HERMES_MODEL anthropic/claude-opus-4.7

# Or specify per-session
hermes chat --model openrouter/meta-llama/llama-3.1-70b-instruct
```

#### Limitação de taxa (erros 429) {#rate-limiting-429-errors}

**Causa:** Você excedeu os limites de taxa do seu provedor.

**Solução:** Espere um momento e tente novamente. Para uso sustentado, considere:
- Fazer upgrade do seu plano de provedor
- Trocar para um modelo ou provedor diferente
- Usar `hermes chat --provider <alternative>` para rotear para um backend diferente

#### Tamanho de contexto excedido {#context-length-exceeded}

**Causa:** A conversa cresceu demais para a janela de contexto do modelo, ou o Hermes detectou o tamanho de contexto errado para seu modelo.

**Solução:**
```bash
# Compress the current session
/compress

# Or start a fresh session
hermes chat

# Use a model with a larger context window
hermes chat --model openrouter/google/gemini-3-flash-preview
```

Se isso ocorrer na primeira conversa longa, o Hermes pode ter o tamanho de contexto errado para o seu modelo. Verifique o que ele detectou:

Observe a linha de inicialização da CLI — ela mostra o tamanho de contexto detectado (ex.: `📊 Context limit: 128000 tokens`). Você também pode verificar com `/usage` durante uma sessão.

Para corrigir a detecção de contexto, defina-a explicitamente:

```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  context_length: 131072  # your model's actual context window
```

Ou para endpoints customizados, adicione por modelo:

```yaml
custom_providers:
  - name: "My Server"
    base_url: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
```

Veja [Detecção de Tamanho de Contexto](../integrations/providers.md#context-length-detection) para como a detecção automática funciona e todas as opções de sobrescrita.

---

### Problemas de Terminal {#terminal-issues}

#### Comando bloqueado como perigoso {#command-blocked-as-dangerous}

**Causa:** O Hermes detectou um comando potencialmente destrutivo (ex.: `rm -rf`, `DROP TABLE`). Este é um recurso de segurança.

**Solução:** Quando solicitado, revise o comando e digite `y` para aprová-lo. Você também pode:
- Pedir ao agente para usar uma alternativa mais segura
- Ver a lista completa de padrões perigosos na [documentação de Segurança](../user-guide/security.md)

:::tip
Isso está funcionando como pretendido — o Hermes nunca executa comandos destrutivos silenciosamente. O prompt de aprovação mostra exatamente o que será executado.
:::

#### `sudo` não funciona via gateway de mensagens {#sudo-not-working-via-messaging-gateway}

**Causa:** O gateway de mensagens roda sem um terminal interativo, então o `sudo` não pode solicitar uma senha.

**Solução:**
- Evite `sudo` em mensagens — peça ao agente para encontrar alternativas
- Se você precisar usar `sudo`, configure sudo sem senha para comandos específicos em `/etc/sudoers`
- Ou troque para a interface de terminal para tarefas administrativas: `hermes chat`

#### Backend do Docker não conecta {#docker-backend-not-connecting}

**Causa:** O daemon do Docker não está rodando ou o usuário não tem permissões.

**Solução:**
```bash
# Check Docker is running
docker info

# Add your user to the docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker run hello-world
```

---

### Problemas de Mensagens {#messaging-issues}

#### Bot não responde a mensagens {#bot-not-responding-to-messages}

**Causa:** O bot não está rodando, não está autorizado, ou seu usuário não está na allowlist.

**Solução:**
```bash
# Check if the gateway is running
hermes gateway status

# Start the gateway
hermes gateway start

# Check logs for errors
cat ~/.hermes/logs/gateway.log | tail -50
```

#### Mensagens não estão sendo entregues {#messages-not-delivering}

**Causa:** Problemas de rede, token do bot expirado, ou webhook da plataforma mal configurado.

**Solução:**
- Verifique se o token do seu bot é válido com `hermes gateway setup`
- Verifique os logs do gateway: `cat ~/.hermes/logs/gateway.log | tail -50`
- Para plataformas baseadas em webhook (Slack, WhatsApp), garanta que seu servidor esteja publicamente acessível

#### Confusão de allowlist — quem pode falar com o bot? {#allowlist-confusion--who-can-talk-to-the-bot}

**Causa:** O modo de autorização determina quem tem acesso.

**Solução:**

| Modo | Como funciona |
|------|-------------|
| **Allowlist** | Apenas os IDs de usuário listados na config podem interagir |
| **Pareamento por DM** | O primeiro usuário a enviar mensagem na DM reivindica acesso exclusivo |
| **Aberto** | Qualquer pessoa pode interagir (não recomendado para produção) |

Configure em `~/.hermes/config.yaml` nas configurações do seu gateway. Veja a [documentação de Mensagens](../user-guide/messaging/index.md).

#### O gateway não inicia {#gateway-wont-start}

**Causa:** Dependências ausentes, conflitos de porta, ou tokens mal configurados.

**Solução:**
```bash
# Install core messaging gateway dependencies
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"  # Telegram, Discord, Slack, and shared gateway deps

# Check for port conflicts
lsof -i :8080

# Verify configuration
hermes config show
```

#### WSL: o gateway continua desconectando ou `hermes gateway start` falha {#wsl-gateway-keeps-disconnecting-or-hermes-gateway-start-fails}

**Causa:** O suporte a systemd do WSL é pouco confiável. Muitas instalações WSL2 não têm systemd habilitado, e mesmo quando habilitado, os serviços podem não sobreviver a reinicializações do WSL ou desligamentos por inatividade do Windows.

**Solução:** Use o modo em primeiro plano em vez do serviço systemd:

```bash
# Option 1: Direct foreground (simplest)
hermes gateway run

# Option 2: Persistent via tmux (survives terminal close)
tmux new -s hermes 'hermes gateway run'
# Reattach later: tmux attach -t hermes

# Option 3: Background via nohup
nohup hermes gateway run > ~/.hermes/logs/gateway.log 2>&1 &
```

Se você quiser tentar o systemd mesmo assim, certifique-se de que está habilitado:

1. Abra `/etc/wsl.conf` (crie-o se não existir)
2. Adicione:
   ```ini
   [boot]
   systemd=true
   ```
3. No PowerShell: `wsl --shutdown`
4. Reabra seu terminal WSL
5. Verifique: `systemctl is-system-running` deve dizer "running" ou "degraded"

:::tip Iniciar automaticamente na inicialização do Windows
Para uma inicialização automática confiável, use o Agendador de Tarefas do Windows para iniciar o WSL + o gateway no login:
1. Crie uma tarefa que execute `wsl -d Ubuntu -- bash -lc 'hermes gateway run'`
2. Configure-a para disparar no logon do usuário
:::

#### macOS: Node.js / ffmpeg / outras ferramentas não encontradas pelo gateway {#macos-nodejs--ffmpeg--other-tools-not-found-by-gateway}

**Causa:** Serviços launchd herdam um PATH mínimo (`/usr/bin:/bin:/usr/sbin:/sbin`) que não inclui Homebrew, nvm, cargo, ou outros diretórios de ferramentas instaladas pelo usuário. Isso comumente quebra a bridge do WhatsApp (`node not found`) ou a transcrição de voz (`ffmpeg not found`).

**Solução:** O gateway captura o PATH do seu shell quando você executa `hermes gateway install`. Se você instalou ferramentas depois de configurar o gateway, execute a instalação novamente para capturar o PATH atualizado:

```bash
hermes gateway install    # Re-snapshots your current PATH
hermes gateway start      # Detects the updated plist and reloads
```

Você pode verificar se o plist tem o PATH correto:
```bash
/usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:PATH" \
  ~/Library/LaunchAgents/ai.hermes.gateway.plist
```

---

### Problemas de Desempenho {#performance-issues}

#### Respostas lentas {#slow-responses}

**Causa:** Modelo grande, servidor de API distante, ou system prompt pesado com muitas ferramentas.

**Solução:**
- Tente um modelo mais rápido/menor: `hermes chat --model openrouter/meta-llama/llama-3.1-8b-instruct`
- Reduza os toolsets ativos: `hermes chat -t "terminal"`
- Verifique sua latência de rede até o provedor
- Para modelos locais, garanta que você tenha VRAM de GPU suficiente

#### Alto uso de tokens {#high-token-usage}

**Causa:** Conversas longas, system prompts verbosos, ou muitas chamadas de ferramenta acumulando contexto.

**Solução:**
```bash
# Compress the conversation to reduce tokens
/compress

# Check session token usage
/usage
```

:::tip
Use `/compress` regularmente durante sessões longas. Isso resume o histórico da conversa e reduz significativamente o uso de tokens preservando o contexto.
:::

#### Sessão ficando muito longa {#session-getting-too-long}

**Causa:** Conversas estendidas acumulam mensagens e saídas de ferramentas, se aproximando dos limites de contexto.

**Solução:**
```bash
# Compress current session (preserves key context)
/compress

# Start a new session with a reference to the old one
hermes chat

# Resume a specific session later if needed
hermes chat --continue
```

---

### Problemas de MCP {#mcp-issues}

#### Servidor MCP não conecta {#mcp-server-not-connecting}

**Causa:** Binário do servidor não encontrado, caminho de comando errado, ou runtime ausente.

**Solução:**
```bash
# Ensure MCP dependencies are installed (already included in standard install)
cd ~/.hermes/hermes-agent && uv pip install -e ".[mcp]"

# For npm-based servers, ensure Node.js is available
node --version
npx --version

# Test the server manually
npx -y @modelcontextprotocol/server-filesystem /tmp
```

Verifique a configuração MCP do seu `~/.hermes/config.yaml`:
```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/docs"]
```

#### Ferramentas não aparecem do servidor MCP {#tools-not-showing-up-from-mcp-server}

**Causa:** O servidor iniciou mas a descoberta de ferramentas falhou, as ferramentas foram filtradas pela config, ou o servidor não suporta a capacidade MCP que você esperava.

**Solução:**
- Verifique os logs do gateway/agente por erros de conexão MCP
- Garanta que o servidor responda ao método RPC `tools/list`
- Revise quaisquer configurações `tools.include`, `tools.exclude`, `tools.resources`, `tools.prompts`, ou `enabled` sob esse servidor
- Lembre-se de que ferramentas utilitárias de resource/prompt só são registradas quando a sessão realmente suporta essas capacidades
- Use `/reload-mcp` após alterar a config

```bash
# Verify MCP servers are configured
hermes config show | grep -A 12 mcp_servers

# Restart Hermes or reload MCP after config changes
hermes chat
```

Veja também:
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [Usar MCP com o Hermes](/guides/use-mcp-with-hermes)
- [Referência de Configuração MCP](/reference/mcp-config-reference)

#### Erros de timeout do MCP {#mcp-timeout-errors}

**Causa:** O servidor MCP está demorando muito para responder, ou travou durante a execução.

**Solução:**
- Aumente o timeout na config do seu servidor MCP, se suportado
- Verifique se o processo do servidor MCP ainda está em execução
- Para servidores MCP HTTP remotos, verifique a conectividade de rede

:::warning
Se um servidor MCP travar no meio de uma solicitação, o Hermes reportará um timeout. Verifique os próprios logs do servidor (não apenas os logs do Hermes) para diagnosticar a causa raiz.
:::

---

## Perfis {#profiles}

### Como os perfis diferem de apenas definir HERMES_HOME? {#how-do-profiles-differ-from-just-setting-hermes_home}

Perfis são uma camada gerenciada sobre o `HERMES_HOME`. Você *poderia* definir manualmente `HERMES_HOME=/some/path` antes de cada comando, mas os perfis cuidam de toda a parte técnica para você: criando a estrutura de diretórios, gerando aliases de shell (`hermes-work`), rastreando o perfil ativo em `~/.hermes/active_profile`, e sincronizando atualizações de skills entre todos os perfis automaticamente. Eles também se integram ao autocompletar por tab para que você não precise memorizar caminhos.

### Dois perfis podem compartilhar o mesmo token de bot? {#can-two-profiles-share-the-same-bot-token}

Não. Cada plataforma de mensagens (Telegram, Discord, etc.) requer acesso exclusivo a um token de bot. Se dois perfis tentarem usar o mesmo token simultaneamente, o segundo gateway falhará ao conectar. Crie um bot separado por perfil — para o Telegram, fale com o [@BotFather](https://t.me/BotFather) para criar bots adicionais.

### Os perfis compartilham memória ou sessões? {#do-profiles-share-memory-or-sessions}

Não. Cada perfil tem seu próprio armazenamento de memória, banco de dados de sessões e diretório de skills. Eles são completamente isolados. Se você quiser iniciar um novo perfil com memórias e sessões existentes, use `hermes profile create newname --clone-all` para copiar tudo do perfil atual, ou adicione `--clone-from <profile>` para copiar de um perfil de origem específico.

### O que acontece quando executo `hermes update`? {#what-happens-when-i-run-hermes-update}

`hermes update` baixa o código mais recente e reinstala as dependências **uma vez** (não por perfil). Em seguida, sincroniza automaticamente as skills atualizadas para todos os perfis. Você só precisa executar `hermes update` uma vez — isso cobre todo perfil na máquina.


### Quantos perfis posso executar? {#how-many-profiles-can-i-run}

Não há limite fixo. Cada perfil é apenas um diretório sob `~/.hermes/profiles/`. O limite prático depende do seu espaço em disco e de quantos gateways simultâneos seu sistema pode suportar (cada gateway é um processo Python leve). Executar dezenas de perfis é normal; cada perfil ocioso não usa recursos.

---

## Workflows e Padrões {#workflows--patterns}

### Usando modelos diferentes para tarefas diferentes (workflows multi-modelo) {#using-different-models-for-different-tasks-multi-model-workflows}

**Cenário:** Você usa o GPT-5.4 como seu modelo do dia a dia, mas o Gemini ou o Grok escreve melhor conteúdo para redes sociais. Trocar de modelo manualmente todas as vezes é tedioso.

**Solução: config de delegação.** O Hermes pode rotear subagentes para um modelo diferente automaticamente. Defina isso em `~/.hermes/config.yaml`:

```yaml
delegation:
  model: "google/gemini-3-flash-preview"   # subagents use this model
  provider: "openrouter"                    # provider for subagents
```

Agora, quando você diz ao Hermes "escreva uma thread do Twitter sobre X" e ele cria um subagente `delegate_task`, esse subagente roda no Gemini em vez do seu modelo principal. Sua conversa primária permanece no GPT-5.4.

Você também pode ser explícito no seu prompt: *"Delegue uma tarefa para escrever posts de redes sociais sobre o lançamento do nosso produto. Use seu subagente para a escrita em si."* O agente usará o `delegate_task`, que capta automaticamente a config de delegação.

Para trocas de modelo pontuais sem delegação, use `/model` na CLI:

```bash
/model google/gemini-3-flash-preview    # switch for this session
# ... write your content ...
/model openai/gpt-5.4                   # switch back
```

:::warning
Cada troca de `/model` reinicia o cache de prompt — a chave de cache inclui o modelo, então a primeira mensagem após cada troca relê a conversa inteira ao preço total de entrada. Em sessões longas, prefira delegação (subagentes recebem seu próprio contexto novo) ou uma nova sessão em vez de trocas repetidas de ida e volta.
:::

Veja [Delegação de Subagentes](../user-guide/features/delegation.md) para mais sobre como a delegação funciona.

### Executando vários agentes em um único número do WhatsApp (vinculação por chat) {#running-multiple-agents-on-one-whatsapp-number-per-chat-binding}

**Cenário:** No OpenClaw, você tinha vários agentes independentes vinculados a chats específicos do WhatsApp — um para um grupo de lista de compras da família, outro para seu chat privado. O Hermes pode fazer isso?

**Limitação atual:** Cada perfil do Hermes requer seu próprio número/sessão do WhatsApp. Você não pode vincular vários perfis a chats diferentes no mesmo número do WhatsApp — a bridge do WhatsApp (Baileys) usa uma sessão autenticada por número.

**Soluções alternativas:**

1. **Use um único perfil com troca de personalidade.** Crie arquivos de contexto `AGENTS.md` diferentes ou use o comando `/personality` para mudar o comportamento por chat. O agente vê em qual chat está e pode se adaptar.

2. **Use jobs de cron para tarefas especializadas.** Para um rastreador de lista de compras, configure um job de cron que monitora um chat específico e gerencia a lista — sem necessidade de um agente separado.

3. **Use números separados.** Se você precisar de agentes verdadeiramente independentes, pareie cada perfil com seu próprio número do WhatsApp. Números virtuais de serviços como o Google Voice funcionam para isso.

4. **Use o Telegram ou o Discord em vez disso.** Essas plataformas suportam vinculação por chat de forma mais natural — cada grupo do Telegram ou canal do Discord recebe sua própria sessão, e você pode executar vários tokens de bot (um por perfil) na mesma conta.

Veja [Perfis](../user-guide/profiles.md) e [configuração do WhatsApp](../user-guide/messaging/whatsapp.md) para mais detalhes.

### Controlando o que aparece no Telegram (ocultando logs e raciocínio) {#controlling-what-shows-up-in-telegram-hiding-logs-and-reasoning}

**Cenário:** Você vê logs de execução do gateway, raciocínio do Hermes e detalhes de chamadas de ferramenta no Telegram em vez de apenas a saída final.

**Solução:** A configuração `display.tool_progress` no `config.yaml` controla quanta atividade de ferramenta é mostrada:

```yaml
display:
  tool_progress: "off"   # options: off, new, all, verbose
```

- **`off`** — Apenas a resposta final. Sem chamadas de ferramenta, sem raciocínio, sem logs.
- **`new`** — Mostra novas chamadas de ferramenta conforme acontecem (frases curtas de uma linha).
- **`all`** — Mostra toda a atividade de ferramenta incluindo resultados.
- **`verbose`** — Detalhe completo incluindo argumentos e saídas de ferramentas.

Para plataformas de mensagens, `off` ou `new` é geralmente o que você quer. Após editar o `config.yaml`, reinicie o gateway para que as mudanças tenham efeito.

Você também pode alternar isso por sessão com o comando `/verbose` (se habilitado):

```yaml
display:
  tool_progress_command: true   # enables /verbose in the gateway
```

### Gerenciando skills no Telegram (limite de slash commands) {#managing-skills-on-telegram-slash-command-limit}

**Cenário:** O Telegram tem um limite de 100 slash commands, e suas skills estão passando desse limite. Você quer desativar skills que não precisa no Telegram, mas as configurações do `hermes skills config` não parecem ter efeito.

**Solução:** Use `hermes skills config` para desativar skills por plataforma. Isso escreve no `config.yaml`:

```yaml
skills:
  disabled: []                    # globally disabled skills
  platform_disabled:
    telegram: [skill-a, skill-b]  # disabled only on telegram
```

Depois de alterar isso, **reinicie o gateway** (`hermes gateway restart` ou mate e reinicie). O menu de comandos do bot do Telegram é reconstruído na inicialização.

:::tip
Skills com descrições muito longas são truncadas para 40 caracteres no menu do Telegram para permanecer dentro dos limites de tamanho de payload. Se as skills não estiverem aparecendo, pode ser um problema de tamanho total do payload em vez do limite de contagem de 100 comandos — desativar skills não usadas ajuda com ambos.
:::

### Sessões de thread compartilhadas (vários usuários, uma conversa) {#shared-thread-sessions-multiple-users-one-conversation}

**Cenário:** Você tem uma thread do Telegram ou do Discord onde várias pessoas mencionam o bot. Você quer que todas as menções nessa thread façam parte de uma única conversa compartilhada, não sessões separadas por usuário.

**Comportamento atual:** O Hermes cria sessões identificadas por ID de usuário na maioria das plataformas, então cada pessoa recebe seu próprio contexto de conversa. Isso é intencional, por privacidade e isolamento de contexto.

**Soluções alternativas:**

1. **Use o Slack.** As sessões do Slack são identificadas por thread, não por usuário. Vários usuários na mesma thread compartilham uma conversa — exatamente o comportamento que você está descrevendo. Este é o encaixe mais natural.

2. **Use um chat de grupo com um único usuário.** Se uma pessoa for o "operador" designado que retransmite perguntas, a sessão permanece unificada. Outros podem acompanhar a leitura.

3. **Use um canal do Discord.** As sessões do Discord são identificadas por canal, então todos os usuários no mesmo canal compartilham o contexto. Use um canal dedicado para a conversa compartilhada.

### Exportando o Hermes para outra máquina {#exporting-hermes-to-another-machine}

**Cenário:** Você acumulou skills, jobs de cron e memórias em uma máquina e quer mover tudo para uma nova máquina Linux dedicada.

**Solução:**

1. Instale o Hermes Agent na nova máquina:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

2. Na **máquina de origem**, crie um backup completo:
   ```bash
   hermes backup
   ```
   Isso cria um zip do seu diretório `~/.hermes/` inteiro — config, chaves de API, memórias, skills, sessões e perfis — salvo no seu diretório home como `~/hermes-backup-<timestamp>.zip`.

3. Copie o zip para a nova máquina e importe-o:
   ```bash
   # On the source machine
   scp ~/hermes-backup-<timestamp>.zip newmachine:~/

   # On the new machine
   hermes import ~/hermes-backup-<timestamp>.zip
   ```

4. Na nova máquina, execute `hermes setup` para verificar se as chaves de API e a config de provedor estão funcionando.

### Movendo um único perfil para outra máquina {#moving-a-single-profile-to-another-machine}

**Cenário:** Você quer mover ou compartilhar um perfil específico — não sua instalação completa.

```bash
# On the source machine
hermes profile export work ./work-backup.tar.gz

# Copy the file to the target machine, then:
hermes profile import ./work-backup.tar.gz work
```

O perfil importado terá toda a config, memórias, sessões e skills da exportação. Você pode precisar atualizar caminhos ou reautenticar com provedores se a nova máquina tiver uma configuração diferente.

### `hermes backup` vs `hermes profile export` {#hermes-backup-vs-hermes-profile-export}

| Recurso | `hermes backup` | `hermes profile export` |
| :--- | :--- | :--- |
| **Caso de uso** | **Migração completa de máquina** | **Portar/compartilhar um perfil específico** |
| **Escopo** | Global (diretório `~/.hermes` inteiro) | Local (diretório de um único perfil) |
| **Inclui** | Todos os perfis, config global, chaves de API, sessões | Perfil único: SOUL.md, memórias, sessões, skills |
| **Credenciais** | **Incluídas** (`.env` e `auth.json`) | **Excluídas** (removidas para compartilhamento seguro) |
| **Formato** | `.zip` | `.tar.gz` |

**Alternativa manual (rsync):** Se você preferir copiar arquivos diretamente, exclua o repositório de código:
```bash
rsync -av --exclude='hermes-agent' ~/.hermes/ newmachine:~/.hermes/
```

:::tip
`hermes backup` produz um snapshot consistente mesmo enquanto o Hermes está ativamente em execução. O arquivo restaurado exclui arquivos de runtime locais da máquina como `gateway.pid` e `cron.pid`.
:::

### Permissão negada ao recarregar o shell após a instalação {#permission-denied-when-reloading-shell-after-install}

**Cenário:** Após executar o instalador do Hermes, `source ~/.zshrc` retorna um erro de permissão negada.

**Causa:** Isso geralmente acontece quando `~/.zshrc` (ou `~/.bashrc`) tem permissões de arquivo incorretas, ou quando o instalador não conseguiu escrever nele corretamente. Não é um problema específico do Hermes — é um problema de permissões de configuração de shell.

**Solução:**
```bash
# Check permissions
ls -la ~/.zshrc

# Fix if needed (should be -rw-r--r-- or 644)
chmod 644 ~/.zshrc

# Then reload
source ~/.zshrc

# Or just open a new terminal window — it picks up PATH changes automatically
```

Se o instalador adicionou a linha do PATH mas as permissões estão erradas, você pode adicioná-la manualmente:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
```

### Erro 400 na primeira execução do agente {#error-400-on-first-agent-run}

**Cenário:** A configuração é concluída sem problemas, mas a primeira tentativa de chat falha com HTTP 400.

**Causa:** Geralmente uma incompatibilidade de nome de modelo — o modelo configurado não existe no seu provedor, ou a chave de API não tem acesso a ele.

**Solução:**
```bash
# Check what model and provider are configured
hermes config show | head -20

# Re-run model selection
hermes model

# Or test with a known-good model
hermes chat -q "hello" --model anthropic/claude-opus-4.7
```

Se estiver usando o OpenRouter, certifique-se de que sua chave de API tem créditos. Um 400 do OpenRouter geralmente significa que o modelo requer um plano pago ou o ID do modelo tem um erro de digitação.

---

## Ainda com problemas? {#still-stuck}

Se seu problema não está coberto aqui:

1. **Busque nos issues existentes:** [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
2. **Pergunte à comunidade:** [Discord da Nous Research](https://discord.gg/nousresearch)
3. **Registre um bug:** Inclua seu SO, versão do Python (`python3 --version`), versão do Hermes (`hermes --version`), e a mensagem de erro completa
