---
title: "Glossário de símbolos da CLI"
description: "O que cada símbolo na UI de terminal do Hermes significa — marcadores de transcript, badges da status bar, glyphs de overlay e prompts de aprovação."
---

# Glossário de símbolos da CLI {#cli-symbols-glossary}

As interfaces de terminal do Hermes falam uma linguagem visual compacta: pontos, chevrons, spinners braille e glyphs de status. Esta página é o decoder ring. Cobre a [TUI](../user-guide/tui.md) (onde a maior parte disso renderiza) e nota as peças compartilhadas com a [Classic CLI](../user-guide/cli.md).

:::note Skins podem reestilizar alguns destes
Glyphs marcados *themeable* abaixo são os defaults da marca — uma [skin](../user-guide/features/skins.md) pode sobrescrevê-los (por exemplo `tool_prefix` e o símbolo do prompt). Todo o resto é fixo no renderer.
:::

## Símbolos do transcript {#transcript-symbols}

O que você vê no fluxo da conversa enquanto o agente trabalha.

| Símbolo | Significado |
|--------|---------|
| `❯` | Prompt de input — onde você digita. *Themeable* (skins definem o próprio símbolo de prompt). |
| `●` | Uma tool call. O bullet precede o nome da tool e seus argumentos. |
| `┊` | Rail de atividade de tool mostrado com as linhas de tool. *Themeable* (`tool_prefix`). |
| `✓` / `✗` | Resultado da tool: sucesso / falha. Anexado ao fim de uma linha de tool concluída. |
| `▸` / `▾` | Chevron de seção colapsada / expandida (thinking, tools, subagents, seções do banner). Clique para alternar. |
| `▍` | Cursor de streaming — pisca enquanto o modelo ainda está emitindo texto. |
| `│` `├─` `└─` | Rails de árvore — conectam um parent (uma delegation, uma journey) às entradas filhas; `└─` marca o último filho. |
| `◈` | Evento de timeline só de display (notices de sessão renderizados inline, não mensagens do usuário). |
| `◇` | Um bloco de referência injetado, por exemplo `◇ Reference 1/2 — <label>`. |
| `↳` | O sticky prompt — ecoa a mensagem do usuário na qual o agente está trabalhando agora. |
| `☐` / `☑` / `•` | Itens de task-list markdown (abertos / feitos) e bullets de lista simples. |
| `▶` | Summary de `<details>` colapsado dentro de markdown renderizado. |

## Símbolos da status bar {#status-bar-symbols}

A linha única no rodapé da TUI. Segmentos aparecem só quando relevantes e caem primeiro em terminais estreitos.

| Símbolo | Significado |
|--------|---------|
| `⠋⠙⠹…` (padrões braille) | Spinner de busy. As fases thinking e tool usam sets de animação braille diferentes. |
| `⚕ 🌀 🤔 ✨ 🍵 🔮` | Frames do estilo de busy-indicator `emoji` (`/indicator emoji`). O estilo padrão rota faces kaomoji em vez disso. |
| <code>&#124; / - &#92;</code> | Frames do estilo de busy-indicator `ascii`. |
| `⏱` | Tempo decorrido por prompt enquanto o turno roda, por exemplo `⏱ 12s/3m 45s` (tempo do turno / tempo da sessão). |
| `⏲` | O mesmo timer, congelado depois que o turno completa. |
| `cmp N` | A sessão foi auto-comprimida N vezes. |
| `▶ N` | N tarefas `/background` rodando agora. |
| `⚠ YOLO` | Modo YOLO ligado (auto-aprovação). Também mostrado no banner de startup. |
| `⛓ N` | N subagents ativos agora. |
| `↩ resumes when subagent finishes` | Reassurance mostrado enquanto você está idle mas trabalho delegado ainda está em voo — o resultado volta sozinho. |
| `● REC` | Voice mode está gravando. |
| `◉ STT` | A gravação de voz parou; speech-to-text está transcrevendo. |
| `◉ focus` | Focus view ligado (output reduzido). Pinado para nunca cair de um terminal estreito. |
| `♥` | Flash de affection — o Hermes notou você sendo gentil com ele. |
| `⚡` / `🔋` | Indicador de bateria (opt-in): plugado / na bateria, com porcentagem. |
| `N bg` | N processos de terminal em background rastreados nesta sessão. |
| `N live sessions` | Sessões TUI abertas neste processo — clique para abrir o session switcher. |

## Notices {#notices}

Notices de curta duração na status bar carregam o próprio glyph líder, definido pela severidade:

| Símbolo | Significado |
|--------|---------|
| `✓` | Notice de sucesso. |
| `•` | Notice informativo. |
| `⚠` | Warning (também usado para avisos de crédito). |
| `✕` | Notice de erro. |

## Prompts de aprovação e confirmação {#approval-and-confirmation-prompts}

| Símbolo | Significado |
|--------|---------|
| `⚠ approval required` | Uma tool quer rodar algo que precisa do seu sim explícito (painel com borda e preview do comando). |
| `⚠` / `?` | Título do diálogo de confirmação: ação perigosa / pergunta ordinária. |
| `🔐` | Prompt de senha sudo (input mascarado). |
| `🔑` | Prompt de input de credential/secret (input mascarado). |

## Overlay de subagents (`/agents`) {#subagents-overlay-agents}

| Símbolo | Significado |
|--------|---------|
| `●` | Subagent rodando. |
| `○` | Em fila. |
| `✓` | Concluído. |
| `■` | Interrompido. |
| `✗` | Falhou. |
| `⌛` | Timed out. |
| `⚠` | Errored. |
| `⚡N` | N agents ativos agora numa linha de rollup. |
| `▁▂▃▄▅▆▇█` | Sparkline de atividade — volume recente de eventos por branch. |

## Session switcher (`Ctrl+X`) {#session-switcher-ctrlx}

| Símbolo | Significado |
|--------|---------|
| `✓` | Sessão idle. |
| `…` | Starting. |
| `?` | Esperando input. |
| `▶` | Working. |
| `✎ draft` | O composer naquela sessão segura um draft não enviado. |

## Pickers e hubs {#pickers-and-hubs}

| Símbolo | Significado |
|--------|---------|
| `▸` | Linha de seleção atual (model picker e afins). |
| `*` | O model / provider ativo agora. |
| `●` / `○` | Provider autenticado / não autenticado (model picker); fallback de estado de plugin (plugins hub). |
| `✓` / `✗` | Plugin enabled / disabled (plugins hub). |
| `↑ N more` / `↓ N more` | Mais linhas acima/abaixo da janela visível de uma lista. |
| `┃` | Thumb da scrollbar em overlays scrolláveis. |

## Goals {#goals}

Notices de lifecycle de goal (de [goals](../user-guide/features/goals.md)) começam com o estado:

| Símbolo | Significado |
|--------|---------|
| `✓` | Goal completo. |
| `↻` | Goal continuando — outra iteração foi agendada. |
| `⏸` | Goal pausado. |

## Veja também {#see-also}

- [TUI](../user-guide/tui.md) — status line, modos de detalhes, estilos de busy-indicator
- [Classic CLI](../user-guide/cli.md) — keybindings e slash commands compartilhados
- [Skins & Themes](../user-guide/features/skins.md) — quais glyphs e cores você pode customizar
