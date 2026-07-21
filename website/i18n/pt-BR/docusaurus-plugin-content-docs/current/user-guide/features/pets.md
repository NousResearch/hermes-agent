---
sidebar_position: 11
title: "Pets (Mascotes Petdex)"
description: "Adote um mascote animado que reage à atividade do agente na CLI, TUI e app desktop"
---

# Pets

O Hermes pode exibir um **pet** animado — um sprite mascote pequeno que reage ao que
o agente está fazendo (ocioso, executando ferramenta, pensando, finalizando, falhando) na
**CLI**, **TUI** e **app desktop**. Pets vêm da galeria pública
[petdex](https://github.com/crafter-station/petdex).

Pets são puramente cosméticos. **Não afetam prompt caching, tokens nem
o comportamento do agente** — o sprite é só uma preocupação de exibição. O recurso está
**desligado por padrão** e permanece inativo até você instalar e selecionar um pet.

## Como funciona {#how-it-works}

- Pets são instalados no diretório `pets/` do seu perfil
  (`<HERMES_HOME>/pets/<slug>/`), então cada [perfil](../profiles.md) mantém seu
  próprio conjunto.
- Selecionar um pet grava `display.pet.slug` e `display.pet.enabled` em
  `config.yaml` — nada é armazenado como segredo ou variável de ambiente.
- Cada superfície observa a atividade que já rastreia e a mapeia para um dos seis
  estados de animação. O mapeamento fica em um só lugar para toda superfície se comportar
  igual:

  | Atividade do agente | Estado do pet |
  | --- | --- |
  | Uma ferramenta/turn acabou de falhar | `failed` |
  | Um plano terminou (todos os todos concluídos) | `jump` (comemorar) |
  | Um turn terminou sem erros | `wave` |
  | Uma ferramenta está em execução | `run` |
  | O modelo está pensando/lendo | `review` |
  | Turn em andamento (não especificado) | `run` |
  | Bloqueado em você (prompt de clarify/aprovação aberto) | `waiting` (cai para `idle` em sheets legadas de 8 linhas) |
  | Nada acontecendo | `idle` |

## Renderização {#rendering}

No terminal (CLI/TUI), o Hermes renderiza o sprite em fidelidade total quando seu
terminal suporta um protocolo gráfico (**kitty**, **Ghostty**, **WezTerm**,
**iTerm2** ou **sixel**). Caso contrário, cai automaticamente para renderização
**half-block** Unicode em truecolor. Dentro de um pipe ou redirecionamento (sem TTY), a renderização
no terminal é desabilitada por design.

O app desktop desenha o pet como um sprite flutuante em um canvas e o alterna em
**Settings → Appearance**.

## Início rápido (CLI) {#quick-start-cli}

```bash
# Navegar a galeria (filtrar por substring)
hermes pets list
hermes pets list cat

# Instalar um pet e torná-lo ativo em um passo
hermes pets install boba --select

# Visualizar / animar no terminal (Ctrl+C para parar)
hermes pets show

# Verificar sua configuração
hermes pets doctor
```

## Comandos `hermes pets` {#hermes-pets-commands}

| Objetivo | Comando |
| --- | --- |
| Navegar a galeria | `hermes pets list [query] [--limit N]` |
| Listar pets instalados | `hermes pets list --installed` |
| Instalar um pet | `hermes pets install <slug> [--select] [--force]` |
| Definir o pet ativo | `hermes pets select [slug]` (omitir slug para um seletor) |
| Redimensionar o pet em todo lugar | `hermes pets scale <factor>` (ex.: `0.5`, limitado a 0.1–3.0) |
| Visualizar/animar | `hermes pets show [slug] [--state <s>] [--cycle] [--once] [--mode <m>] [--scale <f>]` |
| Desabilitar o pet | `hermes pets off` |
| Remover um pet instalado | `hermes pets remove <slug>` |
| Diagnosticar configuração | `hermes pets doctor` |

Flags de `hermes pets show`:

- `--state` — reproduz um único estado (`idle`, `wave`, `run`, `failed`, `review`,
  `jump`).
- `--cycle` — percorre todos os estados.
- `--once` — reproduz uma vez em vez de loop.
- `--mode` — sobrescreve o protocolo de render (`kitty`, `iterm`, `sixel`,
  `unicode`, `auto`).
- `--scale` — sobrescreve a escala na tela (`0` = usar config).

## Slash command `/pet` {#pet-slash-command}

Dentro da CLI e TUI você pode gerenciar o pet sem sair da sessão:

- `/pet` — alterna o pet ligado/desligado (adota o primeiro pet instalado se nenhum estiver
  ativo).
- `/pet list` — navega a galeria.
- `/pet scale <factor>` — redimensiona o pet em todo lugar (ex.: `/pet scale 0.5`).
- `/pet <slug>` — adota um pet específico.
- `/pet off` — desabilita o pet.

Na TUI, `/pet list` abre um overlay de seletor interativo; no app desktop
abre a paleta de pets Cmd+K.

## Gerar um pet (`/hatch`) {#generating-a-pet-hatch}

Além de instalar pets prontos da galeria, o Hermes pode **gerar um pet totalmente novo** a partir de uma descrição em texto — seu próprio pipeline de geração de sprite com IA.

- CLI/TUI: `/hatch <description>` (alias `/generate-pet`), ou `hermes pets` → fluxo de geração.
- App desktop: UI estilo Pokédex de **generate** — ovo animado, FX de hatch e seletor de rascunho.

Como a geração funciona (fluxo em duas etapas, com custo limitado):

1. **Rascunhos base** — um punhado de variantes baratas, só com prompt, de "como este pet deve parecer" são geradas. Você escolhe uma, ou remix/retry para uma rodada nova.
2. **Hatch** — a base escolhida é usada como imagem de referência para gerar uma linha de animação grounded por estado Hermes (idle, thinking, tool use, etc.), que são fatiadas deterministicamente em frames e empacotadas em um atlas petdex/Codex padrão (grade 8×9 de células 192×208). O resultado é um spritesheet válido que você mantém — e poderia `petdex submit`.

### Backend de imagem {#image-backend}

A geração usa o [provider de geração de imagem](/user-guide/features/image-generation) ativo, mas exige **grounding com imagem de referência** para cada linha de animação manter o mesmo personagem da base. Backends com referência: **Nous Portal**, **OpenRouter**, **OpenAI** (`gpt-image-2`) e **Krea**. OpenRouter/Nous rodam uma cadeia de modelos quality-first por padrão.

- A ordem de resolução prefere Nous Portal → OpenAI → OpenRouter.
- Se nenhum backend com referência estiver configurado, a geração mostra um erro acionável apontando para `hermes tools` → Image Generation. (Instalar/adotar pets existentes da galeria não precisa de backend de imagem.)
- Sobrescreva o backend com a variável de ambiente `HERMES_PET_IMAGE_PROVIDER` (ex.: `HERMES_PET_IMAGE_PROVIDER=openrouter`).

## App desktop {#desktop-app}

No app desktop você pode gerenciar o pet de duas formas:

- **Cmd+K → "Pets…"** — navega, busca, adota e alterna pets sem sair do
  teclado (espelha o seletor de tema).
- **Settings → Appearance** — a mesma galeria mais um **slider de tamanho** que
  redimensiona o mascote flutuante ao vivo enquanto você arrasta.

Ambos adotam/alternam/redimensionam o mascote flutuante no lugar — mudanças de tamanho aplicam
na hora; adotar um pet novo o acende em instantes.

### Overlay pop-out {#pop-out-overlay}

**Shift-clique** no pet flutuante para pop-out em uma janela desktop transparente,
sempre no topo. Lá ele permanece visível com o Hermes minimizado (estilo Codex), então um olhar
diz o que o agente está fazendo.

Gestos depois do pop-out:

| Gesto | Ação |
| --- | --- |
| **Arrastar** | Move o pet para qualquer lugar da tela, inclusive fora do app. Posição e estado in/out persistem entre reinícios. |
| **Clique simples** | Abre um mini compositor para enviar um prompt à sessão mais recente — sem trazer o app à frente. |
| **Clique duplo** | Alterna a janela do app: minimiza se estiver em primeiro plano, restaura se estiver oculta. |
| **Shift-clique** | Recoloca o pet de volta na janela. |
| **Ícone de mail** | Aparece só quando um turn terminou enquanto você estava ausente; clique para trazer o app na thread mais recente (e marcar como lida). |

Só o pet pop-out mostra um **balão de fala** (`working…`, `thinking…`,
`your turn`, …) — na janela o app em si é a superfície, então o pet fica
quieto lá.

O overlay é um puro puppet do pet in-app — não carrega conexão gateway separada
e nunca aparece no dock ou app switcher.

## Configuração {#configuration}

Todas as configurações ficam sob `display.pet` em `config.yaml`:

```yaml
display:
  pet:
    enabled: false        # master on/off (true once you select a pet)
    slug: ""              # active pet; empty = first installed
    render_mode: auto      # auto | kitty | iterm | sixel | unicode | off
    scale: 0.33           # master size knob (relative to native 192x208 frames)
    unicode_cols: 0       # hard override for terminal width (0 = derive from scale)
```

- **`scale`** é o único knob mestre de tamanho. Um número encolhe toda superfície:
  o canvas desktop escala pixels por ele, e CLI/TUI derivam a largura em colunas do terminal
  dele. O fallback half-block limita a um piso de legibilidade
  — não encolhe tanto quanto renderização true-pixel kitty/GUI sem virar
  papo, então o mesmo `scale` fica nítido no kitty mas tem piso em
  half-blocks.
- **`render_mode: auto`** detecta kitty/iTerm2/sixel e cai para unicode
  half-blocks. Defina explicitamente para forçar um protocolo ou `off` para desabilitar
  renderização no terminal mantendo o pet no desktop.
- **`unicode_cols`** fixa a largura em colunas do terminal independentemente de `scale`;
  deixe em `0` para derivar largura de `scale`.

## Solução de problemas {#troubleshooting}

Execute `hermes pets doctor` — ele reporta:

- o diretório pets e quais pets estão instalados,
- `display.pet.enabled`, `display.pet.slug` e o pet ativo resolvido,
- o `render_mode` configurado, o protocolo gráfico de terminal detectado e o
  modo efetivo para um TTY,
- se Pillow (usado para decodificação de sprite) é importável.

Imprime `✓ ready` quando um pet está instalado, selecionado, habilitado e Pillow está
disponível.

Armadilhas comuns:

- Um pet só aparece quando um está **instalado E selecionado** (`enabled: true`).
- Dentro de pipe/redirecionamento (sem TTY), renderização no terminal é desabilitada por design.
- A CLI npm petdex instala em `~/.codex/pets`; o Hermes usa seu próprio
  `<HERMES_HOME>/pets/` com escopo de perfil — instale via `hermes pets`.

## Veja também {#see-also}

- A [skill `petdex`](../skills/bundled/productivity/productivity-petdex.md)
  deixa o agente instalar e trocar pets para você sob demanda.
