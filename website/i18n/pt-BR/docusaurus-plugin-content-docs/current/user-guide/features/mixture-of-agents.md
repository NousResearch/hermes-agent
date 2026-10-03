---
sidebar_position: 7
title: "Mixture of Agents"
description: "Crie presets nomeados de MoA que aparecem como modelos selecionáveis sob o provedor Mixture of Agents"
---

# Mixture of Agents

Mixture of Agents é um provedor de modelo virtual. Cada preset nomeado de MoA aparece como um modelo selecionável sob o provedor `moa`.

Quando você seleciona um preset de MoA, o agregador do preset é o modelo em atuação. É o modelo que escreve a resposta do assistente e emite chamadas de ferramenta. Os modelos de referência rodam primeiro e fornecem análise para o agregador usar.

Use o MoA quando uma tarefa difícil se beneficia de múltiplas perspectivas de modelo, mas ainda precisa do loop de agente normal do Hermes: chamadas de ferramenta, iterações de acompanhamento, interrupções, persistência de transcrição e o mesmo contexto de sessão de qualquer outra mensagem.

## Selecione um preset de MoA como seu modelo {#select-a-moa-preset-as-your-model}

Você pode selecionar um preset através das superfícies normais do seletor de modelo:

```bash
/model default --provider moa
/model review --provider moa
```

Presets de MoA são selecionáveis em **toda superfície do Hermes**, porque o MoA é um provedor normal no sistema de modelos:

- **CLI / gateway / TUI `/model`** — `/model <preset> --provider moa`, ou `/model --provider moa` para o preset padrão. Um `/model <preset>` simples também funciona quando o nome corresponde exatamente a um preset configurado.
- **`hermes model`** e o **seletor de modelo do Dashboard** — aparece uma linha de provedor `Mixture of Agents` com os nomes dos seus presets como seus modelos.
- **App de desktop (GUI)** — o menu suspenso de modelo mostra uma seção `MoA presets`; selecionar um (`MoA: <preset>`) muda o modelo ativo para aquele preset. O painel de configurações do Desktop também cria e edita presets.

Presets configurados, portanto, aparecem em qualquer lugar onde você escolheria qualquer outro modelo.

## Atalho de comando de barra {#slash-command-shortcut}

`/moa` é um açúcar sintático de conveniência de uso único. Ele executa um único prompt através do preset **padrão** de MoA, e depois restaura o modelo em que você estava:

```bash
/moa design and implement a migration plan for this flaky test cluster
```

O Hermes muda temporariamente para o preset padrão de MoA por aquele único turno, envia o prompt e restaura seu modelo anterior depois. O argumento inteiro é o prompt — `/moa` não interpreta mais isso como um nome de preset.

```bash
/moa
```

`/moa` sozinho (sem prompt) apenas imprime as instruções de uso.

Para **mudar** para um preset de MoA pelo resto da sessão, selecione-o no seletor de modelo — presets de MoA aparecem sob um provedor `Mixture of Agents` em toda superfície de seleção de modelo (veja acima). `/moa` é deliberadamente não uma troca de modelo, então um prompt normal nunca pode mudar acidentalmente seu modelo.

## Como funciona no loop do agente {#how-it-works-in-the-agent-loop}

A cada chamada do modelo principal quando o provedor `moa` está selecionado, o Hermes:

1. resolve o preset selecionado pelo nome;
2. executa os modelos de referência configurados sem esquemas de ferramenta (eles recebem apenas o texto usuário/assistente da conversa — não o prompt de sistema do Hermes nem a transcrição de chamadas de ferramenta — para que as chamadas de referência permaneçam baratas e evitem rejeições de provedores rigorosos);
3. anexa as saídas de referência como contexto privado para o agregador;
4. chama o agregador configurado com o esquema de ferramenta normal do Hermes;
5. trata a resposta do agregador como a resposta real do modelo;
6. se o agregador chama ferramentas, o Hermes executa essas ferramentas normalmente;
7. na próxima iteração do modelo, o mesmo processo de MoA roda novamente sobre a conversa atualizada, incluindo os resultados das ferramentas.

Como o MoA é selecionado através do sistema de modelos normal, ele se compõe automaticamente com `/goal`, sessões de gateway, sessões de TUI e o chat do Desktop.

## Configure presets {#configure-presets}

Você pode configurar presets nomeados de MoA a partir de:

- Dashboard → Models → Model Settings → Mixture of Agents
- App Desktop → Settings → Model → Mixture of Agents
- `hermes moa configure [name]`
- `config.yaml`

A configuração armazena pares explícitos de provedor/modelo, então você pode misturar provedores e usar múltiplos modelos do mesmo provedor:

```yaml
moa:
  default_preset: default
  presets:
    default:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
        - provider: openrouter
          model: deepseek/deepseek-v4-pro
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      # Optional: pin sampling temperatures. When omitted (the default),
      # temperature is NOT sent and each model uses its provider default —
      # the same behavior as a single-model Hermes agent.
      # reference_temperature: 0.6
      # aggregator_temperature: 0.4
      max_tokens: 4096
      enabled: true
```

Preset padrão:

- referência: `openai-codex:gpt-5.5`
- referência: `openrouter:deepseek/deepseek-v4-pro`
- agregador / modelo em atuação: `openrouter:anthropic/claude-opus-4.8`

### Ajustando a velocidade dos conselheiros com `reference_max_tokens` {#tuning-advisor-speed-with-reference_max_tokens}

A cada turno, o MoA executa os modelos de referência (conselheiros) em
paralelo e depois o agregador atua. A geração dos conselheiros é a latência
dominante por turno — o tempo de parede do turno se correlaciona fortemente
com quantos tokens os conselheiros emitem, porque o turno espera o
conselheiro mais lento terminar de escrever. Por padrão os conselheiros não
têm limite (`reference_max_tokens` não definido), então eles podem escrever
conselhos longos, do tamanho de um ensaio.

Defina `reference_max_tokens` em um preset para limitar a saída dos
conselheiros e dar conselhos concisos em vez disso. O agregador só precisa
da essência do julgamento de cada conselheiro, então um limite (por exemplo,
`600`) reduz mensuravelmente o tempo de parede por turno com pouco impacto
na qualidade. Ele limita **apenas os conselheiros** — a saída do agregador
em atuação (a resposta visível ao usuário) nunca é limitada.

```yaml
moa:
  presets:
    fast:
      reference_models:
        - provider: openrouter
          model: anthropic/claude-opus-4.8
        - provider: openrouter
          model: openai/gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
      reference_max_tokens: 600   # concise advice → faster turns
```

Deixe sem definir (ou `0`/em branco) para manter o comportamento anterior
sem limite.

### Cadência dos conselheiros com `fanout` {#advisor-cadence-with-fanout}

Por padrão os conselheiros rodam **uma vez por turno de usuário**
(`fanout: user_turn`) — eles sintetizam conselhos em nível de plano na
primeira mensagem do turno, e depois o agregador em atuação percorre o
restante do loop de ferramentas sozinho. Essa é a cadência mais barata — o
custo dos conselheiros não se multiplica com o número de chamadas de
ferramenta em um turno. Duas cadências alternativas trocam custo por
atualidade do conselho:

- `fanout: per_iteration` — os conselheiros rodam novamente em **cada
  iteração de ferramenta**, então seu conselho sempre acompanha os
  resultados mais recentes das ferramentas — ao custo de multiplicar a
  latência e o gasto dos conselheiros pelo número de chamadas de ferramenta
  em um turno.
- `fanout: every_n:3` — o meio-termo: os conselheiros rodam na
  **primeira** iteração de cada turno de usuário e depois a cada **3ª**
  iteração de ferramenta (qualquer `N >= 2` funciona). Iterações
  intermediárias reutilizam a orientação em cache da última execução dos
  conselheiros, então o agregador ainda recebe conselho a cada passo — ele
  só é atualizado a cada N passos em vez de a cada passo. O contador reinicia
  a cada nova mensagem de usuário, então cada turno começa com conselhos
  frescos. A forma de mapeamento `fanout: {mode: every_n, n: 3}` também é
  aceita e normalizada para a forma de string.

```yaml
moa:
  presets:
    fresh:
      reference_models:
        - provider: openrouter
          model: anthropic/claude-opus-4.8
      aggregator:
        provider: openrouter
        model: openai/gpt-5.5
      fanout: per_iteration   # advisors refresh on every tool iteration
```

Valores desconhecidos ou malformados recaem para `user_turn`.

:::note Mudança de padrão
Antes de julho de 2026 a cadência padrão era `per_iteration`. O padrão agora
é `user_turn` — a cadência mais barata e de menor impacto — até que
benchmarks por modo justifiquem um padrão mais custoso. Presets que querem
aconselhamento a cada passo de volta definem `fanout: per_iteration`
explicitamente.
:::

### Filtro de privacidade para saídas dos conselheiros {#privacy-filter-for-advisor-outputs}

As saídas dos conselheiros podem ecoar dados sensíveis da conversa —
e-mails, números de telefone formatados, chaves de API, JWTs — nos blocos de
referência exibidos na UI, nos traces de MoA salvos, e no prompt do
agregador. `moa.privacy_filter` (desativado por padrão) oculta essas
superfícies:

```yaml
moa:
  privacy_filter: display   # or: full
```

- `display` — oculta **apenas as superfícies visíveis ao usuário**: os
  blocos de referência rotulados renderizados na UI e os registros escritos
  por `save_traces`. O agregador ainda recebe o texto bruto dos
  conselheiros, então a qualidade da resposta não é afetada.
- `full` — adicionalmente oculta o texto dos conselheiros injetado no
  prompt do agregador (e a entrada de síntese do `/moa` de uso único).

Formatos de credencial (prefixos de chave de API, JWTs, chaves privadas,
strings de conexão de banco de dados) são mascarados pelo redator central de
segredos do Hermes; o filtro do MoA adiciona ocultação de e-mail e de
números de telefone claramente formatados por cima disso. Os padrões são
deliberadamente conservadores para conselhos no estilo de revisão de código:
sequências de dígitos soltas, números de linha, timestamps, SHAs do git e
endereços IP nunca são tocados — só formatos de telefone delimitados como
`(555) 123-4567` ou `555-123-4567` correspondem.

### Esforço de raciocínio por slot {#per-slot-reasoning-effort}

Os slots de referência e agregador também podem definir
`reasoning_effort`. Use isso quando você quiser que o mesmo modelo contribua
em profundidades diferentes, ou quando o agregador deve pensar mais do que
as referências consultivas. Valores válidos correspondem aos controles
normais de raciocínio do Hermes: `none`, `minimal`, `low`, `medium`, `high`,
`xhigh`, `max` e `ultra`.

```yaml
moa:
  presets:
    deep_review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.6-sol
          reasoning_effort: low
        - provider: openai-codex
          model: gpt-5.6-sol
          reasoning_effort: xhigh
        - provider: xai-oauth
          model: grok-4.5
      aggregator:
        provider: openai-codex
        model: gpt-5.6-sol
        reasoning_effort: high
```

Omita `reasoning_effort` para usar o padrão do provedor/Hermes para aquele
slot.

## Gerenciamento de presets pelo terminal {#terminal-preset-management}

```bash
hermes moa list
hermes moa configure              # update the default preset
hermes moa configure review       # create or update a named preset
hermes moa delete review
```

## Benchmarks {#benchmarks}

No HermesBench, um preset de MoA de dois modelos — `claude-opus-4.8` agregando sobre uma referência `gpt-5.5` — supera qualquer um dos dois modelos rodando sozinho:

| Modelo | Pontuação no HermesBench |
|---|---|
| **Agregador Opus (opus-4.8 + referência gpt-5.5) — MoA** | **0.8202** |
| `anthropic/claude-opus-4.8` | 0.7607 |
| `openai/gpt-5.5` | 0.7412 |

A configuração de MoA supera seu componente mais forte (opus-4.8) em ~6 pontos, confirmando que agregar uma segunda perspectiva eleva a qualidade em tarefas difíceis, em vez de apenas fazer a média das duas.

## Cache de prompt {#prompt-caching}

O MoA é construído de forma que o **cache de prompt da conversa principal nunca seja quebrado**. Selecionar um preset de MoA é uma seleção de modelo normal: não altera contexto passado, não troca toolsets, nem reconstrói o prompt de sistema no meio da conversa. Seu histórico de conversa, prompt de sistema e esquema de ferramenta permanecem byte a byte estáveis, então o prefixo em cache do qual todo outro modelo depende é preservado exatamente como seria para um modelo simples. Mudar para ou de um preset de MoA custa a mesma invalidação de cache que qualquer outra troca de `/model` — nada mais.

Ambos os tipos de chamada interna fazem cache normalmente:

- **Os modelos de referência** recebem uma visão da conversa recortada e determinística (prompt de sistema e transcrição de ferramentas removidos — veja o loop acima). Como essa visão é uma função estável do histórico estável, o prefixo de prompt de um modelo de referência se repete entre iterações e faz cache normalmente. As referências são chamadas consultivas curtas, sem ferramentas.
- **O agregador** é o modelo em atuação. As saídas de referência são anexadas ao *final* do último turno de usuário como orientação privada. Como esse texto fica na cauda — abaixo de todo o prefixo estável (prompt de sistema + histórico anterior) — ele não invalida nenhum prefixo em cache: o agregador tem um acerto de cache em tudo acima da injeção, e só a cauda recém-anexada é nova. É exatamente assim que todo turno normal se comporta, onde cada nova mensagem de usuário também é tokens de cauda sem cache.

Então o MoA não sacrifica o cache de prompt em nenhum dos dois tipos de chamada. Seu único custo real são as chamadas de referência extras por iteração — você paga por múltiplas perspectivas de modelo, não por caches quebrados. O prefixo de conversa de longa duração compartilhado com o resto do Hermes permanece totalmente intacto.

## Notas {#notes}

- O MoA não está mais listado em `hermes tools`; não há toolset `moa` para ativar.
- Definir `enabled: false` em um preset desativa o fan-out de referência para aquele preset: o agregador atua sozinho, exatamente como se você o tivesse selecionado como um modelo simples. Esse é o interruptor de desativação por preset exposto nas configurações do dashboard e do desktop.
- O agregador de um preset não pode ser outro preset de MoA. Árvores recursivas de MoA são bloqueadas intencionalmente.
- Falhas de credencial em um modelo de referência não abortam o turno. O Hermes inclui a falha no contexto de referência e continua com quaisquer modelos que responderam.
- O MoA aumenta a contagem de chamadas de modelo. Uma única iteração de modelo pode envolver múltiplas chamadas de referência mais a chamada do agregador.
