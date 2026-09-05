---
sidebar_position: 1
title: "Dicas e Boas Práticas"
description: "Conselhos práticos para aproveitar ao máximo o Hermes Agent — dicas de prompt, atalhos de CLI, arquivos de contexto, memória, otimização de custos e segurança"
---

# Dicas e Boas Práticas {#tips--best-practices}

Uma coleção de dicas práticas e rápidas que tornam você imediatamente mais eficiente com o Hermes Agent. Cada seção aborda um aspecto diferente — percorra os títulos e vá direto ao que for relevante.

:::tip Não sabe qual modelo escolher?
Execute `hermes setup --portal` — você ganha acesso a mais de 300 modelos, incluindo Claude, GPT-5 e Gemini, sob uma única assinatura. Veja [Nous Portal](/integrations/nous-portal).
:::

---

## Obtendo os Melhores Resultados {#getting-the-best-results}

### Seja Específico Sobre o que Você Quer {#be-specific-about-what-you-want}

Prompts vagos produzem resultados vagos. Em vez de "corrija o código", diga "corrija o `TypeError` em `api/handlers.py` na linha 47 — a função `process_request()` recebe `None` de `parse_body()`." Quanto mais contexto você fornecer, menos iterações você precisará.

### Forneça Contexto de Antemão {#provide-context-up-front}

Comece sua solicitação já com os detalhes relevantes: caminhos de arquivo, mensagens de erro, comportamento esperado. Uma mensagem bem elaborada vale mais do que três rodadas de esclarecimento. Cole os tracebacks de erro diretamente — o agente consegue interpretá-los.

### Use Arquivos de Contexto para Instruções Recorrentes {#use-context-files-for-recurring-instructions}

Se você se pega repetindo as mesmas instruções ("use tabs, não espaços", "usamos pytest", "a API está em `/api/v2`"), coloque-as em um arquivo `AGENTS.md`. O agente o lê automaticamente em toda sessão — esforço zero após a configuração inicial.

### Deixe o Agente Usar Suas Ferramentas {#let-the-agent-use-its-tools}

Não tente guiar cada passo manualmente. Diga "encontre e corrija o teste que está falhando" em vez de "abra `tests/test_foo.py`, olhe a linha 42, depois...". O agente tem busca de arquivos, acesso a terminal e execução de código — deixe-o explorar e iterar.

### Use Skills para Fluxos de Trabalho Complexos {#use-skills-for-complex-workflows}

Antes de escrever um prompt longo explicando como fazer algo, verifique se já existe uma skill para isso. Digite `/skills` para navegar pelas skills disponíveis, ou simplesmente invoque uma diretamente, como `/axolotl` ou `/github-pr-workflow`.

## Dicas de Power User da CLI {#cli-power-user-tips}

### Entrada Multilinha {#multi-line-input}

Pressione **Alt+Enter**, **Ctrl+J** ou **Shift+Enter** para inserir uma nova linha sem enviar. `Shift+Enter` só funciona quando o terminal o envia como uma tecla distinta (Kitty / foot / WezTerm / Ghostty por padrão; iTerm2 / Alacritty / terminal do VS Code depois que o protocolo de teclado Kitty é habilitado). As outras duas opções funcionam em qualquer terminal.

### Detecção de Colagem {#paste-detection}

A CLI detecta automaticamente colagens de múltiplas linhas. Basta colar um bloco de código ou traceback de erro diretamente — ele não enviará cada linha como uma mensagem separada. A colagem é armazenada em buffer e enviada como uma única mensagem.

### Interromper e Redirecionar {#interrupt-and-redirect}

Pressione **Ctrl+C** uma vez para interromper o agente no meio da resposta. Você pode então digitar uma nova mensagem para redirecioná-lo. Pressione Ctrl+C duas vezes em até 2 segundos para forçar a saída. Isso é inestimável quando o agente começa a seguir o caminho errado.

### Retomar Sessões com `-c` {#resume-sessions-with--c}

Esqueceu algo da sua última sessão? Execute `hermes -c` para retomar exatamente de onde você parou, com o histórico completo da conversa restaurado. Você também pode retomar pelo título: `hermes -r "my research project"`.

### Colar Imagem da Área de Transferência {#clipboard-image-paste}

Pressione **Ctrl+V** para colar uma imagem da sua área de transferência diretamente no chat. O agente usa visão computacional para analisar capturas de tela, diagramas, pop-ups de erro ou mockups de UI — sem precisar salvar em um arquivo antes.

### Autocompletar de Comandos com Barra {#slash-command-autocomplete}

Digite `/` e pressione **Tab** para ver todos os comandos disponíveis. Isso inclui comandos integrados (`/compress`, `/model`, `/title`) e todas as skills instaladas. Você não precisa memorizar nada — o autocompletar com Tab cuida disso.

:::tip
Use `/verbose` para alternar entre os modos de exibição de saída de ferramentas: **off → new → all → verbose**. O modo "all" é ótimo para acompanhar o que o agente está fazendo; "off" é o mais limpo para perguntas e respostas simples.
:::

## Arquivos de Contexto {#context-files}

### AGENTS.md: O Cérebro do Seu Projeto {#agentsmd-your-projects-brain}

Crie um `AGENTS.md` na raiz do seu projeto com decisões de arquitetura, convenções de código e instruções específicas do projeto. Isso é injetado automaticamente em toda sessão, então o agente sempre conhece as regras do seu projeto.

```markdown
# Project Context
- This is a FastAPI backend with SQLAlchemy ORM
- Always use async/await for database operations
- Tests go in tests/ and use pytest-asyncio
- Never commit .env files
```

### SOUL.md: Personalize a Personalidade {#soulmd-customize-personality}

Quer que o Hermes tenha uma voz padrão estável? Edite `~/.hermes/SOUL.md` (ou `$HERMES_HOME/SOUL.md` se você usar um diretório home personalizado do Hermes). O Hermes agora gera automaticamente um SOUL inicial e usa esse arquivo global como a fonte de personalidade de toda a instância.

Para um passo a passo completo, veja [Use SOUL.md com o Hermes](/guides/use-soul-with-hermes).

```markdown
# Soul
You are a senior backend engineer. Be terse and direct.
Skip explanations unless asked. Prefer one-liners over verbose solutions.
Always consider error handling and edge cases.
```

Use `SOUL.md` para personalidade duradoura. Use `AGENTS.md` para instruções específicas do projeto.

### Compatibilidade com .cursorrules {#cursorrules-compatibility}

Já tem um arquivo `.cursorrules` ou `.cursor/rules/*.mdc`? O Hermes também os lê. Não é necessário duplicar suas convenções de código — eles são carregados automaticamente a partir do diretório de trabalho.

### Descoberta {#discovery}

O Hermes carrega o `AGENTS.md` de nível superior do diretório de trabalho atual no início da sessão. Arquivos `AGENTS.md` em subdiretórios são descobertos de forma lazy durante as chamadas de ferramentas (via `subdirectory_hints.py`) e injetados nos resultados das ferramentas — eles não são carregados antecipadamente no prompt do sistema.

:::tip
Mantenha os arquivos de contexto focados e concisos. Cada caractere conta contra seu orçamento de tokens, já que eles são injetados em toda mensagem.
:::

## Memória e Skills {#memory--skills}

### Memória vs. Skills: O que Vai Onde {#memory-vs-skills-what-goes-where}

**Memória** é para fatos: seu ambiente, preferências, localizações de projetos e coisas que o agente aprendeu sobre você. **Skills** são para procedimentos: fluxos de trabalho de múltiplas etapas, instruções específicas de ferramentas e receitas reutilizáveis. Use memória para "o quê", skills para "como".

### Quando Criar Skills {#when-to-create-skills}

Se você encontrar uma tarefa que leva 5 ou mais passos e você fará isso novamente, peça ao agente para criar uma skill para ela. Diga "salve o que você acabou de fazer como uma skill chamada `deploy-staging`." Da próxima vez, basta digitar `/deploy-staging` e o agente carrega o procedimento completo.

### Gerenciando a Capacidade da Memória {#managing-memory-capacity}

A memória é intencionalmente limitada (~2.200 caracteres para o MEMORY.md, ~1.375 caracteres para o USER.md). Quando ela se enche, o agente consolida as entradas. Você pode ajudar dizendo "limpe sua memória" ou "substitua a nota antiga sobre Python 3.9 — agora estamos na 3.12."

### Deixe o Agente Lembrar {#let-the-agent-remember}

Depois de uma sessão produtiva, diga "lembre-se disso para a próxima vez" e o agente salvará os principais pontos. Você também pode ser específico: "salve na memória que nosso CI usa GitHub Actions com o workflow `deploy.yml`."

:::warning
A memória é um snapshot congelado — mudanças feitas durante uma sessão não aparecem no prompt do sistema até que a próxima sessão comece. O agente grava no disco imediatamente, mas o cache do prompt não é invalidado no meio da sessão.
:::

## Desempenho e Custo {#performance--cost}

### Não Quebre o Cache do Prompt {#dont-break-the-prompt-cache}

A maioria dos provedores de LLM faz cache do prefixo da conversa (prompt do sistema + histórico). Se você mantiver seu prompt do sistema estável (mesmos arquivos de contexto, mesma memória), as mensagens subsequentes em uma sessão recebem **acertos de cache** que são significativamente mais baratos. O cache é vinculado ao modelo e à conta — então uma troca explícita de `/model`, um [fallback automático de provedor](../user-guide/features/fallback-providers.md) ou uma [rotação de pool de credenciais](../user-guide/features/credential-pools.md) forçam a próxima resposta a reler toda a conversa a preço total de entrada. Trocas ocasionais não são um problema; trocas frequentes em uma sessão longa multiplicam seu custo.

### Use /compress Antes de Atingir os Limites {#use-compress-before-hitting-limits}

Sessões longas acumulam tokens. Quando você notar que as respostas estão ficando mais lentas ou truncadas, execute `/compress`. Isso resume o histórico da conversa, preservando o contexto principal enquanto reduz drasticamente a contagem de tokens. Use `/usage` para verificar sua situação atual.

### Delegue para Trabalho Paralelo {#delegate-for-parallel-work}

Precisa pesquisar três tópicos ao mesmo tempo? Peça ao agente para usar `delegate_task` com subtarefas paralelas. Cada subagente é executado de forma independente com seu próprio contexto, e apenas os resumos finais retornam — reduzindo drasticamente o uso de tokens da sua conversa principal.

### Use execute_code para Operações em Lote {#use-execute_code-for-batch-operations}

Em vez de executar comandos de terminal um a um, peça ao agente para escrever um script que faça tudo de uma vez. "Escreva um script Python para renomear todos os arquivos `.jpeg` para `.jpg` e execute-o" é mais barato e rápido do que renomear arquivos individualmente.

### Escolha o Modelo Certo {#choose-the-right-model}

Use `/model` para trocar de modelo no meio da sessão. Use um modelo de ponta (Claude Sonnet/Opus, GPT-4o) para raciocínio complexo e decisões de arquitetura. Troque para um modelo mais rápido para tarefas simples, como formatação, renomeação ou geração de boilerplate. Lembre-se de que cada troca reinicia o cache do prompt (veja acima), então em sessões longas é geralmente mais barato começar uma sessão nova no outro modelo do que ficar alternando entre eles.

:::tip
Execute `/usage` periodicamente para ver seu consumo de tokens. Execute `/insights` para uma visão mais ampla dos padrões de uso nos últimos 30 dias.
:::

## Dicas de Mensageria {#messaging-tips}

### Defina um Canal Principal {#set-a-home-channel}

Use `/sethome` no seu chat preferido do Telegram ou Discord para designá-lo como o canal principal. Os resultados de tarefas cron e saídas de tarefas agendadas são entregues aqui. Sem isso, o agente não tem para onde enviar mensagens proativas.

### Use /title para Organizar Sessões {#use-title-to-organize-sessions}

Nomeie suas sessões com `/title auth-refactor` ou `/title research-llm-quantization`. Sessões nomeadas são fáceis de encontrar com `hermes sessions list` e retomar com `hermes -r "auth-refactor"`. Sessões sem nome se acumulam e ficam impossíveis de distinguir.

### Pareamento por DM para Acesso da Equipe {#dm-pairing-for-team-access}

Em vez de coletar manualmente IDs de usuário para listas de permissão, habilite o pareamento por DM. Quando um colega de equipe envia uma DM para o bot, ele recebe um código de pareamento único. Você o aprova com `hermes pairing approve telegram XKGH5N7P` — simples e seguro.

### Modos de Exibição de Progresso de Ferramentas {#tool-progress-display-modes}

Use `/verbose` para controlar quanta atividade de ferramentas você vê. Em plataformas de mensageria, menos geralmente é mais — mantenha em "new" para ver apenas novas chamadas de ferramentas. Na CLI, "all" oferece uma visão satisfatória em tempo real de tudo o que o agente faz.

:::tip
Por padrão, sessões de mensageria nunca reiniciam automaticamente — o contexto permanece até você usar `/reset` ou a compressão ser acionada. Se você quiser que as sessões reiniciem automaticamente (após tempo ocioso ou diariamente em um horário fixo), opte por isso através da seção `session_reset` em `~/.hermes/config.yaml`.
:::

## Segurança {#security}

### Use Docker para Código Não Confiável {#use-docker-for-untrusted-code}

Ao trabalhar com repositórios não confiáveis ou executar código desconhecido, use Docker ou Daytona como seu backend de terminal. Defina `TERMINAL_ENV=docker` no seu `.env`. Comandos destrutivos dentro de um container não conseguem prejudicar seu sistema host.

```bash
# In your .env:
TERMINAL_ENV=docker
TERMINAL_DOCKER_IMAGE=hermes-sandbox:latest
```

### Evite Armadilhas de Codificação no Windows {#avoid-windows-encoding-pitfalls}

No Windows, algumas codificações padrão (como `cp125x`) não conseguem representar todos os caracteres Unicode, o que pode causar `UnicodeEncodeError` ao gravar arquivos em testes ou scripts.

- Prefira abrir arquivos com codificação UTF-8 explícita:

```python
with open("results.txt", "w", encoding="utf-8") as f:
    f.write("✓ All good\n")
```

- No PowerShell, você também pode alternar a sessão atual para UTF-8 para a saída do console e de comandos nativos:

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
```

Isso mantém o PowerShell e os processos filhos em UTF-8 e ajuda a evitar falhas exclusivas do Windows.

### Revise Antes de Escolher "Sempre" {#review-before-choosing-always}

Quando o agente aciona uma aprovação de comando perigoso (`rm -rf`, `DROP TABLE`, etc.), você recebe quatro opções: **uma vez**, **sessão**, **sempre** e **negar**. Pense com cuidado antes de escolher "sempre" — isso permite permanentemente aquele padrão. Comece com "sessão" até se sentir confortável.

### A Aprovação de Comandos é Sua Rede de Segurança {#command-approval-is-your-safety-net}

O Hermes verifica todo comando contra uma lista selecionada de padrões perigosos antes da execução. Isso inclui exclusões recursivas, drops de SQL, redirecionamento de curl para shell e mais. Não desabilite isso em produção — existe por boas razões.

:::warning
Ao executar em um backend de container (Docker, Singularity, Modal, Daytona), as verificações de comandos perigosos são **ignoradas** porque o container é a fronteira de segurança. Certifique-se de que suas imagens de container estejam devidamente protegidas.
:::

### Use Listas de Permissão para Bots de Mensageria {#use-allowlists-for-messaging-bots}

Nunca defina `GATEWAY_ALLOW_ALL_USERS=true` em um bot com acesso a terminal. Sempre use listas de permissão específicas por plataforma (`TELEGRAM_ALLOWED_USERS`, `DISCORD_ALLOWED_USERS`) ou pareamento por DM para controlar quem pode interagir com seu agente.

```bash
# Recommended: explicit allowlists per platform
TELEGRAM_ALLOWED_USERS=123456789,987654321
DISCORD_ALLOWED_USERS=123456789012345678

# Or use cross-platform allowlist
GATEWAY_ALLOWED_USERS=123456789,987654321
```

---

*Tem uma dica que deveria estar nesta página? Abra uma issue ou PR — contribuições da comunidade são bem-vindas.*
