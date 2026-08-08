---
sidebar_position: 3
title: 'Trilha de aprendizado'
description: 'Escolha sua trilha na documentação do Hermes Agent conforme seu nível de experiência e objetivos.'
---

# Trilha de aprendizado

O Hermes Agent faz muita coisa — assistente CLI, bot no Telegram/Discord, automação de tarefas, treino RL e mais. Esta página ajuda a descobrir por onde começar e o que ler, conforme seu nível e o que você quer realizar.

:::tip Comece aqui
Se ainda não instalou o Hermes Agent, comece pelo [guia de Instalação](/getting-started/installation) e depois passe pelo [Início rápido](/getting-started/quickstart). Tudo abaixo assume que você já tem uma instalação funcionando.
:::

:::tip Setup de provider pela primeira vez
Quem está começando quase sempre quer `hermes setup --portal` — um OAuth cobre um modelo e as quatro tools do Tool Gateway (busca/imagem/TTS/browser). Veja [Nous Portal](/integrations/nous-portal).
:::

## Como usar esta página

- **Já sabe o seu nível?** Vá à [tabela por nível de experiência](#by-experience-level) e siga a ordem de leitura da sua faixa.
- **Tem um objetivo específico?** Pule para [Por caso de uso](#by-use-case) e ache o cenário que bate.
- **Só explorando?** Veja a tabela [Recursos principais](#key-features-at-a-glance) para um panorama rápido do que o Hermes Agent faz.

## Por nível de experiência {#by-experience-level}

| Nível | Objetivo | Leitura recomendada | Estimativa de tempo |
|---|---|---|---|
| **Iniciante** | Colocar para rodar, ter conversas básicas, usar tools embutidas | [Instalação](/getting-started/installation) → [Início rápido](/getting-started/quickstart) → [Uso do CLI](/user-guide/cli) → [Configuração](/user-guide/configuration) | ~1 hora |
| **Intermediário** | Montar bots de messaging, usar recursos avançados como memória, cron e skills | [Sessões](/user-guide/sessions) → [Messaging](/user-guide/messaging) → [Tools](/user-guide/features/tools) → [Skills](/user-guide/features/skills) → [Memória](/user-guide/features/memory) → [Cron](/user-guide/features/cron) | ~2–3 horas |
| **Avançado** | Criar tools custom, skills, treinar modelos com RL, contribuir no projeto | [Arquitetura](/developer-guide/architecture) → [Adding Tools](/developer-guide/adding-tools) → [Creating Skills](/developer-guide/creating-skills) → [Contributing](/developer-guide/contributing) | ~4–6 horas |

## Por caso de uso {#by-use-case}

Escolha o cenário que bate com o que você quer fazer. Cada um aponta para a docs relevante na ordem em que você deve ler.

### "Quero um assistente de coding no CLI"

Use o Hermes Agent como assistente interativo no terminal para escrever, revisar e rodar código.

1. [Instalação](/getting-started/installation)
2. [Início rápido](/getting-started/quickstart)
3. [Uso do CLI](/user-guide/cli)
4. [Code Execution](/user-guide/features/code-execution)
5. [Context Files](/user-guide/features/context-files)
6. [Tips & Tricks](/guides/tips)

:::tip
Passe arquivos direto na conversa com context files. O Hermes Agent consegue ler, editar e rodar código nos seus projetos.
:::

### "Quero um bot no Telegram/Discord"

Faça o deploy do Hermes Agent como bot na sua plataforma de messaging favorita.

1. [Instalação](/getting-started/installation)
2. [Configuração](/user-guide/configuration)
3. [Messaging Overview](/user-guide/messaging)
4. [Telegram Setup](/user-guide/messaging/telegram)
5. [Discord Setup](/user-guide/messaging/discord)
6. [Modo de voz](/user-guide/features/voice-mode)
7. [Usar modo de voz com o Hermes](/guides/use-voice-mode-with-hermes)
8. [Segurança](/user-guide/security)

Para exemplos de projeto completos, veja:
- [Daily Briefing Bot](/guides/daily-briefing-bot)
- [Team Telegram Assistant](/guides/team-telegram-assistant)

### "Quero automatizar tarefas"

Agende tarefas recorrentes, rode jobs em batch ou encadeie ações do agente.

1. [Início rápido](/getting-started/quickstart)
2. [Cron Scheduling](/user-guide/features/cron)
3. [Batch Processing](/user-guide/features/batch-processing)
4. [Delegation](/user-guide/features/delegation)
5. [Hooks](/user-guide/features/hooks)

:::tip
Jobs de cron deixam o Hermes Agent rodar tarefas sob agendamento — resumos diários, checagens periódicas, relatórios automáticos — sem você estar presente.
:::

### "Quero criar tools/skills custom"

Estenda o Hermes Agent com suas próprias tools e pacotes de skill reutilizáveis.

1. [Plugins](/user-guide/features/plugins)
2. [Build a Hermes Plugin](/developer-guide/plugins)
3. [Tools Overview](/user-guide/features/tools)
4. [Skills Overview](/user-guide/features/skills)
5. [MCP (Model Context Protocol)](/user-guide/features/mcp)
6. [Architecture](/developer-guide/architecture)
7. [Adding Tools](/developer-guide/adding-tools)
8. [Creating Skills](/developer-guide/creating-skills)

:::tip
Para a maioria das tools custom, comece por plugins. A página [Adding Tools](/developer-guide/adding-tools)
é para desenvolvimento do core embutido do Hermes, não o caminho usual de tool custom do usuário.
:::

### "Quero treinar modelos"

Use reinforcement learning para fine-tune do comportamento do modelo com o pipeline de treino RL do Hermes Agent (powered by [Atropos](https://github.com/NousResearch/atropos)).

1. [Início rápido](/getting-started/quickstart)
2. [Configuração](/user-guide/configuration)
3. [Atropos RL Environments](https://github.com/NousResearch/atropos) (externo)
4. [Provider Routing](/user-guide/features/provider-routing)
5. [Architecture](/developer-guide/architecture)

:::tip
Treino RL funciona melhor quando você já entende o básico de como o Hermes Agent lida com conversas e tool calls. Passe pela trilha Iniciante primeiro se estiver começando.
:::

### "Quero usar como biblioteca Python"

Integre o Hermes Agent nas suas próprias aplicações Python de forma programática.

1. [Instalação](/getting-started/installation)
2. [Início rápido](/getting-started/quickstart)
3. [Python Library Guide](/guides/python-library)
4. [Architecture](/developer-guide/architecture)
5. [Tools](/user-guide/features/tools)
6. [Sessions](/user-guide/sessions)

## Recursos principais em um olhar {#key-features-at-a-glance}

Não tem certeza do que está disponível? Aqui vai um diretório rápido dos recursos principais:

| Recurso | O que faz | Link |
|---|---|---|
| **Tools** | Tools embutidas que o agente pode chamar (file I/O, busca, shell, etc.) | [Tools](/user-guide/features/tools) |
| **Skills** | Pacotes de plugin instaláveis que acrescentam novas capacidades | [Skills](/user-guide/features/skills) |
| **Memory** | Memória persistente entre sessões | [Memory](/user-guide/features/memory) |
| **Context Files** | Alimenta arquivos e diretórios nas conversas | [Context Files](/user-guide/features/context-files) |
| **MCP** | Conecta a servidores de tools externos via Model Context Protocol | [MCP](/user-guide/features/mcp) |
| **Cron** | Agenda tarefas recorrentes do agente | [Cron](/user-guide/features/cron) |
| **Delegation** | Spawna subagentes para trabalho em paralelo | [Delegation](/user-guide/features/delegation) |
| **Code Execution** | Roda scripts Python que chamam tools do Hermes de forma programática | [Code Execution](/user-guide/features/code-execution) |
| **Browser** | Navegação e scraping na web | [Browser](/user-guide/features/browser) |
| **Hooks** | Callbacks e middleware orientados a eventos | [Hooks](/user-guide/features/hooks) |
| **Batch Processing** | Processa vários inputs em lote | [Batch Processing](/user-guide/features/batch-processing) |
| **Provider Routing** | Roteia requests por vários providers de LLM | [Provider Routing](/user-guide/features/provider-routing) |

## O que ler a seguir

Com base em onde você está agora:

- **Acabou de instalar?** → Vá ao [Início rápido](/getting-started/quickstart) para rodar a primeira conversa.
- **Terminou o Início rápido?** → Leia [Uso do CLI](/user-guide/cli) e [Configuração](/user-guide/configuration) para personalizar o setup.
- **Confortável com o básico?** → Explore [Tools](/user-guide/features/tools), [Skills](/user-guide/features/skills) e [Memory](/user-guide/features/memory) para liberar o poder completo do agente.
- **Montando para um time?** → Leia [Segurança](/user-guide/security) e [Sessões](/user-guide/sessions) para entender controle de acesso e gestão de conversas.
- **Pronto para construir?** → Entre no [Developer Guide](/developer-guide/architecture) para entender os internals e começar a contribuir.
- **Quer exemplos práticos?** → Veja a seção [Guides](/guides/tips) para projetos do mundo real e dicas.

:::tip
Você não precisa ler tudo. Escolha a trilha que bate com o seu objetivo, siga os links na ordem e você fica produtivo rápido. Sempre pode voltar a esta página para achar o próximo passo.
:::
