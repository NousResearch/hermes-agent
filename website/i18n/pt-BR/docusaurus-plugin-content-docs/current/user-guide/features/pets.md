---
sidebar_position: 11
title: "Pets (mascotes Petdex)"
description: "Adote um mascote animado que reage à atividade do agente no CLI, TUI e app desktop"
---

# Pets

O Hermes pode exibir um **pet** animado — um sprite mascote pequeno que reage ao que o agente está fazendo (idle, rodando ferramenta, pensando, finalizando, falhando) no **CLI**, **TUI** e **app desktop**. Pets vêm da galeria pública [petdex](https://github.com/crafter-station/petdex).

Pets são puramente cosméticos. **Não afetam prompt caching, tokens ou o comportamento do agente** — o sprite é só uma preocupação de display. A feature está **desligada por padrão** e fica dormente até você instalar e selecionar um pet.

## Como funciona {#how-it-works}

- Pets são instalados no diretório `pets/` do seu profile
  (`<HERMES_HOME>/pets/<slug>/`), então cada [profile](../profiles.md) mantém
  seu próprio conjunto.
- Selecionar um pet grava `display.pet.slug` e `display.pet.enabled` em
  `config.yaml` — nada é armazenado como secret ou env var.
- Cada superfície observa a atividade que já rastreia e mapeia para um dos seis
  estados de animação. O mapeamento vive em um lugar para toda superfície se
  comportar igual:

  | Atividade do agente | Estado do pet |
  | --- | --- |
  | Um tool/turno acabou de falhar | `failed` |
  | Um plano terminou (todos os todos concluídos) | `jump` (celebrar) |
  | Um turno terminou limpo | `wave` |
  | Uma ferramenta está executando | `run` |
  | O model está pensando/lendo | `review` |
  | Turno em andamento (não especificado) | `run` |
  | Bloqueado em você (prompt clarify/approval aberto) | `waiting` (cai para `idle` em sheets legacy de 8 linhas) |
  | Nada acontecendo | `idle` |

## Renderização {#rendering}

No terminal (CLI/TUI), o Hermes renderiza o sprite com fidelidade total quando seu terminal suporta um protocolo gráfico (**kitty**, **Ghostty**, **WezTerm**, **iTerm2** ou **sixel**). Caso contrário, cai automaticamente para renderização **half-block** Unicode truecolor. Dentro de pipe ou redirect (sem TTY), renderização de terminal é desabilitada por design.

O app desktop desenha o pet como sprite flutuante em um canvas e alterna em **Settings → Appearance**.

## Início rápido (CLI) {#quick-start-cli}

```bash
# Navegue a galeria (filtre por substring)
hermes pets list
hermes pets list cat

# Instale um pet e ative em um passo
hermes pets install boba --select

# Preview / anime no terminal (Ctrl+C para parar)
hermes pets show

# Verifique seu setup
hermes pets doctor
```

## Comandos `hermes pets` {#hermes-pets-commands}

| Objetivo | Comando |
| --- | --- |
| Navegar a galeria | `hermes pets list [query] [--limit N]` |
| Listar pets instalados | `hermes pets list --installed` |
| Instalar um pet | `hermes pets install <slug> [--select] [--force]` |
| Definir o pet ativo | `hermes pets select [slug]` (omit slug para um picker) |
| Redimensionar o pet em todo lugar | `hermes pets scale <factor>` (ex.: `0.5`, limitado 0.1–3.0) |
| Preview/animar | `hermes pets show [slug] [--state <s>] [--cycle] [--once] [--mode <m>] [--scale <f>]` |
| Desabilitar o pet | `hermes pets off` |
| Remover pet instalado | `hermes pets remove <slug>` |
| Diagnosticar setup | `hermes pets doctor` |

Flags de `hermes pets show`:

- `--state` — reproduz um único estado (`idle`, `wave`, `run`, `failed`, `review`,
  `jump`).
- `--cycle` — percorre todos os estados.
- `--once` — reproduz uma vez em vez de loop.
- `--mode` — sobrescreve o protocolo de render (`kitty`, `iterm`, `sixel`,
  `unicode`, `auto`).
- `--scale` — sobrescreve a escala na tela (`0` = usar config).

## Slash command `/pet` {#pet-slash-command}

Dentro do CLI e TUI você pode gerenciar o pet sem sair da sessão:

- `/pet` — alterna o pet on/off (adota o primeiro pet instalado se nenhum estiver
  ativo).
- `/pet list` — navega a galeria.
- `/pet scale <factor>` — redimensiona o pet em todo lugar (ex.: `/pet scale 0.5`).
- `/pet <slug>` — adota um pet específico.
- `/pet off` — desabilita o pet.

No TUI, `/pet list` abre um overlay picker interativo; no app desktop abre a paleta de pets Cmd+K.

## Gerar um pet (`/hatch`) {#generating-a-pet-hatch}

Além de instalar pets prontos da galeria, o Hermes pode **gerar um pet totalmente novo** a partir de uma descrição em texto — seu próprio pipeline de geração de sprite com IA.

- CLI/TUI: `/hatch <description>` (alias `/generate-pet`), ou `hermes pets` → fluxo de generate.
- App desktop: UI estilo Pokédex de **generate** — ovo animado, FX de hatch e picker de draft.

Como a geração funciona (fluxo de dois passos, com custo limitado):

1. **Drafts base** — um punhado de variantes baratas, só prompt, de "como este pet deve parecer" são geradas. Você escolhe uma, ou remix/retry para uma rodada nova.
2. **Hatch** — a base escolhida é usada como imagem de referência para gerar uma linha de animação grounded por estado Hermes (idle, thinking, tool use, etc.), que são fatiadas deterministicamente em frames e empacotadas em um atlas petdex/Codex padrão (grid 8×9 de células 192×208). O resultado é um spritesheet válido que você mantém — e poderia `petdex submit`.

### Backend de imagem {#image-backend}

A geração usa o [provider de geração de imagem](/user-guide/features/image-generation) ativo, mas exige **grounding por imagem de referência** para cada linha de animação manter o mesmo personagem da base. Backends com referência: **Nous Portal**, **OpenRouter**, **OpenAI** (`gpt-image-2`) e **Krea**. OpenRouter/Nous rodam uma cadeia de models quality-first por padrão.

- A ordem de resolução prefere Nous Portal → OpenAI → OpenRouter.
- Se nenhum backend com referência estiver configurado, a geração mostra erro acionável apontando para `hermes tools` → Image Generation. (Instalar/adotar pets existentes da galeria não precisa de backend de imagem.)
- Sobrescreva o backend com a env var `HERMES_PET_IMAGE_PROVIDER` (ex.: `HERMES_PET_IMAGE_PROVIDER=openrouter`).

## App desktop {#desktop-app}

No app desktop você pode gerenciar o pet de duas formas:

- **Cmd+K → "Pets…"** — navegue, busque, adote e alterne pets sem sair do
  teclado (espelha o theme picker).
- **Settings → Appearance** — a mesma galeria mais um **slider de tamanho** que
  redimensiona o mascote flutuante ao vivo enquanto você arrasta.

Ambos adotam/alternam/redimensionam o mascote flutuante no lugar — mudanças de tamanho aplicam
instantaneamente; adotar um pet novo acende em instantes.

### Roaming {#roaming}

Settings → Appearance tem um toggle **Roam**: quando habilitado, o pet vagueia pela
janela sozinho enquanto o agente está idle — andando superfícies, pausando e
saltando entre pontos. Roaming só roda enquanto o pet está na janela, ativo,
e o agente em repouso; qualquer estado driven pelo agente (trabalhando, celebrando)
assume imediatamente. O toggle está off por padrão e persiste entre
restarts.

### Redimensionar com Alt+wheel {#altwheel-resizing}

Segure **Alt** e role a roda do mouse sobre o pet para redimensioná-lo no lugar —
na janela do app e no overlay popped-out igualmente. O overlay dá zoom
em direção à posição do cursor e a escala resultante persiste, então
sobrevive a restarts e fica sincronizada com o pet in-app.

### Reações de vibe {#vibe-reactions}

Diga algo gentil ao agente — "good bot", "thank you", "ily", `<3`, ou um
emoji de coração — e o pet reage com corações flutuantes (desktop) ou um
flash de coração (CLI/TUI). A detecção é um léxico curado, sem tokens, casado localmente em
cada mensagem do usuário (sem chamada de model); dispara em afeto e gratidão direcionados ao
agente, não sentimento positivo geral. Todas as superfícies — pet CLI, TUI, pet flutuante desktop
e overlay pop-out — reagem ao mesmo sinal.

### Overlay pop-out {#pop-out-overlay}

**Shift-click** no pet flutuante para pop-out em sua própria janela desktop transparente,
always-on-top. Lá fora ele fica visível enquanto o Hermes está
minimizado (estilo Codex), então um olhar diz o que o agente está fazendo.

Gestos depois do pop-out:

| Gesto | Ação |
| --- | --- |
| **Drag** | Move o pet para qualquer lugar na tela, inclusive fora do app. Sua posição e estado in/out persistem entre restarts. |
| **Single-click** | Abre um mini composer para enviar prompt à sessão mais recente — sem trazer o app à frente. |
| **Double-click** | Alterna a janela do app: minimiza se estiver à frente, restaura se estiver oculta. |
| **Shift-click** | Pop o pet de volta para a janela. |
| **Ícone de mail** | Aparece só quando um turno terminou enquanto você estava ausente; clique para trazer o app na thread mais recente (e marcar como lida). |

Só o pet popped-out mostra **speech bubble** (`working…`, `thinking…`,
`your turn`, …) — in-window o app é a superfície, então o pet fica
quieto lá.

O overlay é um puro puppet do pet in-app — não carrega conexão gateway separada
e nunca aparece no dock ou app switcher.

## Configuração {#configuration}

Todas as configurações ficam em `display.pet` em `config.yaml`:

```yaml
display:
  pet:
    enabled: false        # master on/off (true quando você seleciona um pet)
    slug: ""              # pet ativo; vazio = primeiro instalado
    render_mode: auto      # auto | kitty | iterm | sixel | unicode | off
    scale: 0.33           # knob master de tamanho (relativo a frames nativos 192x208)
    unicode_cols: 0       # override hard de largura no terminal (0 = derivar de scale)
```

- **`scale`** é o knob master de tamanho. Um número encolhe toda superfície:
  o canvas desktop escala seus pixels por ele, e CLI/TUI derivam largura de colunas
  do terminal dele. O fallback half-block limita a um piso de legibilidade
  — não encolhe tanto quanto render true-pixel kitty/GUI sem virar papo,
  então o mesmo `scale` fica nítido no kitty mas é limitado em
  half-blocks.
- **`render_mode: auto`** detecta kitty/iTerm2/sixel e cai para half-blocks
  unicode. Defina explicitamente para forçar um protocolo ou `off` para desabilitar
  render de terminal mantendo o pet no desktop.
- **`unicode_cols`** fixa largura de colunas do terminal independente de `scale`;
  deixe em `0` para derivar largura de `scale`.

## Solução de problemas {#troubleshooting}

Execute `hermes pets doctor` — ele reporta:

- o diretório pets e quais pets estão instalados,
- `display.pet.enabled`, `display.pet.slug` e o pet ativo resolvido,
- o `render_mode` configurado, o protocolo gráfico de terminal detectado e o
  modo efetivo para um TTY,
- se Pillow (usado para decode de sprite) é importável.

Imprime `✓ ready` quando um pet está instalado, selecionado, habilitado e Pillow está
disponível.

Armadilhas comuns:

- Um pet só aparece quando está **instalado E selecionado** (`enabled: true`).
- Dentro de pipe/redirect (sem TTY), render de terminal é desabilitado por design.
- O CLI npm petdex instala em `~/.codex/pets`; o Hermes usa seu próprio
  `<HERMES_HOME>/pets/` scoped por profile — instale via `hermes pets`.

## Veja também {#see-also}

- A [skill `hermes-agent`](../skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent.md)
  deixa o agente instalar e trocar pets para você sob demanda (veja
  `references/petdex.md`).
