---
sidebar_position: 27
title: "Solução de problemas: \"Meu agente parece mais burro\""
description: "Um checklist de diagnóstico para quando o Hermes parece menos capaz do que antes ou esquece coisas no meio da sessão — trocas de modelo, pressão de contexto, detecção de contexto errada e o snapshot congelado de memória"
---

# Solução de problemas: "Meu agente parece mais burro" {#troubleshooting-my-agent-feels-dumber}

Às vezes o Hermes parece menos afiado do que ontem, ou esquece algo que você disse vinte minutos atrás. Quase nunca é mistério — geralmente há uma causa específica e verificável. Percorra este checklist em ordem: os passos estão ordenados pela frequência com que cada um acaba sendo a resposta.

## 1. Verifique qual modelo a sessão está usando de fato {#1-check-which-model-the-session-is-actually-using}

**Sintoma:** Respostas mais superficiais, qualidade de código caiu, raciocínio parece estranho — de forma generalizada.

**Verificação:** Execute `/model` sem argumentos para mostrar o modelo atual, ou `/status` para ver modelo, provedor e perfil da sessão em uma única visão.

**O que significa:** Trocar de modelo é uma mudança de capacidade, e é fácil acabar em um modelo diferente do que você imagina:

- Uma troca simples `/model <name>` é **somente da sessão** por padrão (a menos que `model.persist_switch_by_default: true` esteja definido), então o modelo em uso pode não corresponder ao que está em `config.yaml`.
- Alterar o modelo principal na página Models do dashboard aplica-se apenas a **novas sessões** — um chat já aberto continua rodando com o modelo com que começou.
- Se você trocou para um modelo mais rápido para uma tarefa simples (um padrão que [Dicas e boas práticas](/guides/tips#choose-the-right-model) recomenda), lembre de voltar para trabalho de raciocínio complexo.

Se o modelo estiver errado, `/model <name>` corrige para esta sessão; adicione `--global` para persistir a mudança em `config.yaml`. Note que uma troca no meio da sessão reinicia o cache de prompt, então o próximo turno relê a conversa pelo preço completo de input — em uma sessão longa pode ser mais barato começar do zero no modelo certo.

## 2. Verifique o uso de contexto {#2-check-context-usage}

**Sintoma:** A sessão começou forte, mas as respostas estão ficando mais lentas, truncadas ou perdendo detalhes anteriores.

**Verificação:** Execute `/usage` para ver uso de tokens e estado da janela de contexto, ou `/context` para um breakdown visual do que ocupa a janela (system prompt, definições de ferramentas, skills, memória, conversa) versus espaço livre.

**O que significa:** Conversas longas acumulam mensagens e saídas de ferramentas, aproximando-se dos limites de contexto. Quando notar degradação em uma sessão longa:

```bash
# Compress the conversation (summarizes history, preserves key context)
/compress

# Or start a fresh session
/new
```

`/compress` resume o histórico da conversa, reduzindo drasticamente a contagem de tokens enquanto preserva o contexto-chave. `/compress here [N]` mantém as N trocas mais recentes verbatim e resume o resto, e um tópico de foco (`/compress focus <topic>`) estreita o que um resumo completo preserva.

:::tip
Use `/compress` regularmente durante sessões longas em vez de esperar problemas, e `/usage` periodicamente para ver onde você está.
:::

## 3. Verifique o context length detectado {#3-verify-the-detected-context-length}

**Sintoma:** Problemas de contexto aparecem surpreendentemente cedo — a primeira conversa longa já atinge limites, ou a compressão dispara bem antes do que a janela anunciada do modelo deveria permitir.

**Verificação:** Olhe a linha de startup do CLI — ela mostra o context length detectado (ex.: `📊 Context limit: 128000 tokens`). Você também pode verificar com `/usage` durante uma sessão.

**O que significa:** O Hermes pode ter detectado automaticamente o context length errado para seu modelo. Defina explicitamente:

```yaml
# In ~/.hermes/config.yaml
model:
  default: your-model-name
  context_length: 131072  # your model's actual context window
```

Ou para endpoints customizados, por modelo na entrada do provedor:

```yaml
providers:
  my-server:
    api: "http://localhost:11434/v1"
    models:
      qwen3.5:27b:
        context_length: 64000
```

Usuários de Ollama: se você definir um `num_ctx` customizado, defina o context length correspondente no Hermes — o `/api/show` do Ollama reporta o contexto *máximo* do modelo, não o `num_ctx` efetivo que você configurou. Em um gateway em execução, edições em `model.context_length` ou qualquer chave `compression.*` entram em vigor na próxima mensagem — sem reiniciar.

Veja [Detecção de context length](/integrations/providers#context-length-detection) para como a detecção automática funciona e todas as opções de override.

## 4. "Eu disse algo e ele esqueceu" — o snapshot congelado de memória {#4-i-told-it-something-and-it-forgot--the-frozen-memory-snapshot}

**Sintoma:** Você pediu ao Hermes para lembrar algo durante esta sessão, ele confirmou o save, mas depois na *mesma* sessão parece não saber.

**Verificação:** Nada está quebrado — verifique o timing. Memória salva no meio da sessão é gravada no disco imediatamente, mas o system prompt não refletirá isso até a próxima sessão.

**O que significa:** Isso é comportamento documentado e intencional. A memória é injetada no system prompt como um **snapshot congelado no início da sessão**, e essa injeção nunca muda no meio da sessão — preserva o prefix cache do LLM por desempenho. Quando o agente adiciona ou remove entradas de memória durante uma sessão, as mudanças persistem no disco na hora, mas aparecem no system prompt apenas quando a próxima sessão começa. Respostas de ferramentas sempre mostram o estado ao vivo, então o save em si é confirmado e real.

:::info
Snapshot congelado na prática: "remember X" durante uma sessão significa que X estará garantido **na próxima** sessão. Na sessão atual, o fato ainda existe no histórico da conversa — o agente só esquece se aquela parte da conversa foi comprimida depois (veja o passo 7).
:::

Veja [Memória persistente](/user-guide/features/memory#how-memory-appears-in-the-system-prompt) para a mecânica completa.

## 5. Memória é limitada e curada — não uma transcrição {#5-memory-is-bounded-and-curated--not-a-transcript}

**Sintoma:** O Hermes não lembra um detalhe de uma sessão da semana passada, embora vocês tenham discutido extensamente.

**Verificação:** Capacidade e conteúdo da memória. O cabeçalho de memória do system prompt mostra uso (ex.: `[67% — 1,474/2,200 chars]`), e `hermes journey list` mostra cada entrada de memória salva e skill.

**O que significa:** Memória persistente é intencionalmente limitada — 2.200 chars (~800 tokens) para MEMORY.md e 1.375 chars (~500 tokens) para USER.md. Ela guarda fatos-chave curados, não transcrições de conversa. Coisas que valem salvar são preferências, fatos de ambiente, convenções e correções; detalhe bruto de discussão não fica lá por design.

Para recall de "discutimos X na semana passada?", o agente tem um mecanismo separado: `session_search` consulta todas as sessões passadas (armazenadas em SQLite com busca full-text) e encontra coisas discutidas semanas atrás mesmo quando não estão na memória ativa. Basta pedir — "search our past sessions for the deploy discussion."

Você também pode ajudar diretamente: diga "remember this for next time" após uma sessão produtiva, ou "clean up your memory" quando estiver perto da capacidade para o agente consolidar entradas. Veja [Dicas de memória e skills](/guides/tips#memory--skills) e [Gerenciamento de capacidade](/user-guide/features/memory#capacity-management).

## 6. Verifique se skills e ferramentas estão carregadas {#6-check-that-skills-and-tools-are-loaded}

**Sintoma:** O Hermes costumava lidar com um fluxo específico com expertise e agora aborda de forma ingênua, ou diz que não consegue fazer algo que fez antes.

**Verificação:**

- `/skills` — navegue pelas skills instaladas (uma skill da qual o agente dependia pode ter sido removida).
- `/reload-skills` — reescaneie `~/.hermes/skills/` para skills recém-instaladas ou removidas.
- `/tools list` — veja ferramentas disponíveis; uma ferramenta desabilitada antes com `/tools disable` permanece fora do toolset do agente na sessão.
- `/context all` — listagem de custo por skill e por toolset, que também funciona como inventário do que está realmente carregado.

**O que significa:** Skills são o conhecimento procedural do agente — fluxos multi-etapa e instruções específicas de ferramentas. Se uma skill estiver ausente ou um toolset foi reduzido (ex.: sessão iniciada com `hermes chat -t "terminal"` para aliviar o prompt), o agente genuinamente tem menos com que trabalhar nessa sessão. Reative ferramentas com `/tools enable`, ou invoque a skill explicitamente pelo nome (`/github-pr-workflow`) para confirmar que carrega.

## 7. Efeitos colaterais da compressão {#7-compression-side-effects}

**Sintoma:** Após uma sessão longa (ou logo após `/compress`), o Hermes lembra o panorama geral, mas perdeu detalhes finos do início da conversa.

**Verificação:** Se a compressão disparou — `/usage` e `/context` mostram estatísticas de compressão e estado de contexto em plataformas de mensageria, e `/compress` manual sempre reporta seu resultado.

**O que significa:** Compressão substitui histórico de conversa mais antigo por um resumo — esse é o ponto, e necessariamente troca detalhe por folga. Conheça o formato:

- Mensagens recentes são protegidas: por padrão as últimas 20 mensagens ficam sem compressão (`protect_last_n`) e a troca inicial fica fixada (`protect_first_n: 3`) para o objetivo original permanecer visível.
- Compactação é não destrutiva: com o padrão `compression.in_place: true`, a sessão mantém um id durável e turnos pré-compactação são arquivados soft — ainda pesquisáveis via `session_search` e recuperáveis, não deletados.
- Com `in_place: false` (comportamento legado), cada compactação rotaciona para uma **nova sessão ligada à antiga** — uma sessão titulada vira `"my project" → "my project #2" → "my project #3"`. Se você retomar por título, `hermes -c "my project"` escolhe automaticamente a variante mais recente.
- Um tópico de foco estreita o que um resumo completo preserva: `/compress focus auth-refactor` mantém detalhe daquele thread à custa do resto.

Se um detalhe comprimido importa, peça ao agente para buscá-lo (`session_search` alcança os turnos arquivados) ou cole novamente os fatos-chave na conversa.

Veja [Compressão de contexto](/user-guide/configuration#context-compression) para a referência completa de configurações e [Linha de linhagem automática na compressão](/user-guide/sessions#auto-lineage-on-compression) para como sessões tituladas encadeiam.

---

## Referência rápida {#quick-reference}

| Sintoma | Primeiro comando | Causa provável |
|---------|--------------|--------------|
| Tudo parece menos capaz | `/model` | Sessão está em um modelo diferente do que você imagina |
| Sessão longa degradando | `/usage` | Pressão de contexto — comprima ou comece do zero |
| Limites atingidos surpreendentemente cedo | Linha de startup do CLI / `/usage` | Context length detectado automaticamente errado |
| Esqueceu o que eu disse nesta sessão | — (por design) | Snapshot congelado de memória — aparece na próxima sessão |
| Esqueceu discussão da semana passada | peça para `session_search` | Memória é limitada, apenas fatos curados |
| Perdeu uma capacidade específica | `/skills`, `/tools list` | Skill ou toolset não carregado nesta sessão |
| Perdeu detalhe antigo após sessão longa | `/usage`, `/context` | Compressão resumiu histórico mais antigo |
