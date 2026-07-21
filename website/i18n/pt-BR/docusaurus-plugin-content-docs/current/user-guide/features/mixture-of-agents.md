---
sidebar_position: 7
title: "Mixture of Agents"
description: "Crie presets MoA nomeados que aparecem como modelos selecionáveis sob o provider Mixture of Agents"
---

# Mixture of Agents

Mixture of Agents é um provider de modelo virtual. Cada preset MoA nomeado aparece como um modelo selecionável sob o provider `moa`.

Quando você seleciona um preset MoA, o agregador do preset é o modelo em atuação. É ele que escreve a resposta do assistente e emite tool calls. Modelos de referência rodam primeiro e fornecem análise para o agregador usar.

Use MoA quando uma tarefa difícil se beneficia de múltiplas perspectivas de modelo, mas ainda precisa do loop normal de agente do Hermes: tool calls, iterações de follow-up, interrupts, persistência de transcript e o mesmo contexto de sessão de qualquer outra mensagem.

## Selecionar um preset MoA como seu modelo {#select-a-moa-preset-as-your-model}

Você pode selecionar um preset pelas superfícies normais de escolha de modelo:

```bash
/model default --provider moa
/model review --provider moa
```

Presets MoA são selecionáveis em **toda superfície do Hermes**, porque MoA é um provider normal no sistema de modelos:

- **CLI / gateway / TUI `/model`** — `/model <preset> --provider moa`, ou `/model --provider moa` para o preset padrão. Um `/model <preset>` simples também funciona quando o nome corresponde exatamente a um preset configurado.
- **`hermes model`** e o **seletor de modelo do Dashboard** — uma linha de provider `Mixture of Agents` aparece com os nomes dos seus presets como modelos.
- **App Desktop GUI** — o dropdown de modelo mostra uma seção `MoA presets`; selecionar um (`MoA: <preset>`) troca o modelo ativo para esse preset. O painel de configurações do Desktop também cria e edita presets.

Presets configurados portanto aparecem onde você escolheria qualquer outro modelo.

## Atalho com slash command {#slash-command-shortcut}

`/moa` é açúcar de conveniência one-shot. Ele roda um único prompt pelo preset MoA **padrão**, depois restaura qualquer modelo em que você estava:

```bash
/moa design and implement a migration plan for this flaky test cluster
```

O Hermes troca temporariamente para o preset MoA padrão naquele turno, envia o prompt, depois restaura seu modelo anterior. O argumento inteiro é o prompt — `/moa` não o interpreta mais como nome de preset.

```bash
/moa
```

`/moa` sem argumentos (bare) só imprime o uso.

Para **trocar** para um preset MoA pelo resto da sessão, selecione-o no seletor de modelo — presets MoA aparecem sob um provider `Mixture of Agents` em toda superfície de seleção de modelo (veja acima). `/moa` deliberadamente não é uma troca de modelo, então um prompt normal nunca muda seu modelo acidentalmente.

## Como funciona no loop do agente {#how-it-works-in-the-agent-loop}

Para cada chamada principal de modelo quando o provider `moa` está selecionado, o Hermes:

1. resolve o preset selecionado pelo nome;
2. executa os modelos de referência configurados sem schemas de ferramenta (eles recebem só o texto user/assistant da conversa — não o system prompt Hermes nem o transcript de tool-call — para as chamadas de referência ficarem baratas e evitarem rejeições de providers strict);
3. anexa as saídas de referência como contexto privado para o agregador;
4. chama o agregador configurado com o schema normal de ferramentas Hermes;
5. trata a resposta do agregador como a resposta real do modelo;
6. se o agregador chama ferramentas, o Hermes as executa normalmente;
7. na próxima iteração de modelo, o mesmo processo MoA roda de novo sobre a conversa atualizada, incluindo resultados de ferramentas.

Como MoA é selecionado pelo sistema normal de modelos, compõe automaticamente com `/goal`, sessões de gateway, sessões TUI e chat Desktop.

## Configurar presets {#configure-presets}

Você pode configurar presets MoA nomeados em:

- Dashboard → Models → Model Settings → Mixture of Agents
- App Desktop → Settings → Model → Mixture of Agents
- `hermes moa configure [name]`
- `config.yaml`

O config armazena pares explícitos provider/model, então você pode misturar providers e usar vários modelos do mesmo provider:

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

### Ajustar velocidade dos advisors com `reference_max_tokens` {#tuning-advisor-speed-with-reference_max_tokens}

A cada turno, MoA executa os modelos de referência (advisors) em paralelo e depois o
agregador age. A geração dos advisors é a latência dominante por turno — o tempo
de parede do turno correlaciona fortemente com quantos tokens os advisors emitem, porque
o turno espera o advisor mais lento terminar de escrever. Por padrão os advisors
estão **sem teto** (`reference_max_tokens` unset), então podem escrever conselhos longos,
estilo ensaio.

Defina `reference_max_tokens` em um preset para limitar a saída dos advisors e dar conselhos
concisos. O agregador só precisa da essência do julgamento de cada advisor,
então um teto (ex.: `600`) corta mensuravelmente o tempo de parede por turno com pouco
impacto na qualidade. Limita **só advisors** — a saída do agregador em atuação (a
resposta visível ao usuário) nunca é limitada.

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

Deixe unset (ou `0`/blank) para manter o comportamento anterior sem teto.

### Esforço de reasoning por slot {#per-slot-reasoning-effort}

Slots de referência e agregador também podem definir `reasoning_effort`. Use quando
quiser o mesmo modelo contribuindo em profundidades diferentes, ou quando o
agregador deve pensar mais que as referências consultivas. Valores válidos correspondem
aos controles normais de reasoning do Hermes: `none`, `minimal`, `low`, `medium`, `high`,
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

Omita `reasoning_effort` para usar o padrão provider/Hermes daquele slot.

## Gerenciamento de presets no terminal {#terminal-preset-management}

```bash
hermes moa list
hermes moa configure              # atualiza o preset padrão
hermes moa configure review       # create or update a named preset
hermes moa delete review
```

## Benchmarks {#benchmarks}

No HermesBench, um preset MoA de dois modelos — `claude-opus-4.8` agregando sobre referência `gpt-5.5` — supera qualquer um dos modelos rodando sozinho:

| Modelo | Pontuação HermesBench |
|---|---|
| **Opus aggregator (opus-4.8 + gpt-5.5 reference) — MoA** | **0.8202** |
| `anthropic/claude-opus-4.8` | 0.7607 |
| `openai/gpt-5.5` | 0.7412 |

A configuração MoA supera seu componente mais forte (opus-4.8) por ~6 pontos, confirmando que agregar uma segunda perspectiva eleva a qualidade em tarefas difíceis em vez de só fazer a média dos dois.

## Prompt caching {#prompt-caching}

MoA é construído para que o **cache de prompt da conversa principal nunca seja quebrado**. Selecionar um preset MoA é uma seleção normal de modelo: não muta contexto passado, não troca toolsets nem reconstrói o system prompt no meio da conversa. Seu histórico, system prompt e schema de ferramentas permanecem byte-estáveis, então o prefixo em cache em que todo outro modelo confia é preservado exatamente como seria para um modelo simples. Trocar para ou de um preset MoA custa a mesma invalidação de cache de qualquer outra troca `/model` — nada a mais.

Ambos os tipos de chamada interna fazem cache normalmente:

- **Modelos de referência** recebem uma visão recortada e determinística da conversa (system prompt e transcript de ferramentas removidos — veja o loop acima). Como essa visão é função estável do histórico estável, o prefixo de prompt de um modelo de referência se repete entre iterações e faz cache normalmente. Referências são chamadas consultivas curtas sem ferramentas.
- **O agregador** é o modelo em atuação. As saídas de referência são anexadas ao *final* do último turno user como orientação privada. Como esse texto fica na cauda — abaixo de todo o prefixo estável (system prompt + histórico anterior) — não invalida nenhum prefixo em cache: o agregador obtém cache hit em tudo acima da injeção, e só a cauda recém-anexada é nova. É exatamente como todo turno normal se comporta, onde cada nova mensagem user também são tokens de cauda não cacheados.

Então MoA não sacrifica prompt caching em nenhum tipo de chamada. Seu custo real são as chamadas extras de referência por iteração — você paga por múltiplas perspectivas de modelo, não por caches quebrados. O prefixo de conversa de longa duração compartilhado com o resto do Hermes permanece totalmente intacto.

## Notas {#notes}

- MoA não aparece mais em `hermes tools`; não há toolset `moa` para habilitar.
- Definir `enabled: false` em um preset desabilita o fan-out de referência daquele preset: o agregador age sozinho, exatamente como se você o selecionasse como modelo simples. Esse é o interruptor off por preset exposto no dashboard e nas configurações desktop.
- O agregador de um preset não pode ser outro preset MoA. Árvores MoA recursivas são bloqueadas intencionalmente.
- Falhas de credencial em um modelo de referência não abortam o turno. O Hermes inclui a falha no contexto de referência e continua com os modelos que retornaram.
- MoA aumenta a contagem de chamadas de modelo. Uma única iteração de modelo pode envolver várias chamadas de referência mais a chamada do agregador.
