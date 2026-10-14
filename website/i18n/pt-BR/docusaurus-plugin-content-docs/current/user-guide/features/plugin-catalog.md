---
sidebar_position: 13
sidebar_label: "Plugin Catalog"
title: "Catálogo de Plugins"
description: "Navegue e instale plugins Hermes revisados e pinados por SHA a partir do catálogo curado"
---

# Catálogo de Plugins

O catálogo de plugins é um diretório curado e revisado por humanos de plugins
Hermes que você pode instalar pelo nome com um único comando:

```bash
hermes plugins install <name>
```

Navegue visualmente em **[/docs/plugins](/plugins)** — busca, filtros de tier
(Official / Community), chips de capability e comandos de instalação
copiáveis para cada entrada.

O catálogo complementa — não substitui — o [sistema de plugins](plugins.md)
existente. Tudo que você pode instalar pelo catálogo é um plugin normal por
baixo; o catálogo só adiciona descoberta e uma camada de review por cima.

## O que tem em uma entrada {#whats-in-an-entry}

Cada entrada do catálogo é um pequeno arquivo YAML no diretório
[`plugin-catalog/`](https://github.com/NousResearch/hermes-agent/tree/main/plugin-catalog)
do repositório hermes-agent, declarando:

| Campo | Significado |
|---|---|
| `name` | A chave do catálogo que você passa para `hermes plugins install` |
| `repo` | O repositório git público do plugin |
| `sha` | O **commit exato de 40 hex** que foi revisado — as instalações fazem checkout deste pin, não da ponta do branch |
| `tier` | `official` (mantido pela NousResearch) ou `community` |
| `maintainer` | Quem é dono do plugin |
| `capabilities` | Tools, hooks, middleware declarados e env vars necessárias |
| `requires_hermes` | Versão mínima do Hermes, ex. `>=0.19` (opcional) |
| `platforms` | Restrições de SO, vazio = todos (opcional) |
| `docs_url` | Link de documentação externa (opcional) |

## Modelo de confiança {#trust-model}

O catálogo é desenhado para você saber exatamente o que está instalando:

- **Admissão mesclada por humano.** Toda entrada (e toda atualização de pin)
  entra via pull request revisado por um maintainer. Nada entra no catálogo
  automaticamente.
- **Pins de SHA exatos.** Entradas pinam um commit específico, não um branch.
  Um autor de plugin empurrando código novo no repo **não** muda o que o
  catálogo instala — atualizar o pin exige outro PR revisado.
- **Declarações de capability.** Entradas declaram de antemão quais tools,
  hooks e middleware o plugin fornece e quais variáveis de ambiente (API keys
  etc.) precisa, para você julgar o raio de impacto antes de instalar.
- **Lista de removidos.** Plugins retirados do catálogo (por exemplo após um
  incidente de segurança) vão para `plugin-catalog/removed.yaml` com motivo e
  data. O instalador se recusa a instalar qualquer coisa na lista de
  removidos.
- **Instalado ≠ habilitado.** Instalar um plugin do catálogo o coloca em
  disco; como qualquer plugin, ainda precisa ser habilitado antes de carregar.
  Veja [Plugins → Habilitando e desabilitando](plugins.md).

:::warning A review do catálogo é uma review pontual
Uma entrada no catálogo significa que o commit pinado foi olhado por um
humano, as declarações de capability foram checadas, e o repo atendeu a barra
de submissão. Não é uma auditoria de segurança, e não diz nada sobre outros
commits no mesmo repositório. Revise o código de qualquer coisa a que você
dê credenciais.
:::

## Instalando pelo catálogo {#installing-from-the-catalog}

```bash
# Install a reviewed catalog entry by name (checks out the pinned SHA)
hermes plugins install <name>

# Then enable it, as with any plugin
hermes plugins enable <name>
```

O prompt de instalação mostra o resumo de capabilities da entrada — tools
declaradas, hooks e env vars necessárias — antes de qualquer clone.

### Atualizando uma instalação do catálogo {#updating-a-catalog-install}

`hermes plugins update <name>` nunca roda `git pull` para instalações do
catálogo — compara o pin instalado com o pin atual do catálogo e, quando o
catálogo mudou (via PR revisado), força a reinstalação no novo SHA. Seu
estado enabled/disabled é preservado. `hermes plugins list` mostra
instalações do catálogo como `catalog:<tier>@<sha>` para você ver a
proveniência de relance.

### Nomes que não estão no catálogo {#names-not-in-the-catalog}

Um nome simples que não é entrada do catálogo é um erro: não há um segundo
índice de nomes sem review. Instale esses plugins por `owner/repo` ou URL Git
em vez disso (fonte customizada, veja abaixo), ou os submeta ao catálogo.

### Refresh ao vivo {#live-refresh}

O build da docs publica o catálogo como um documento JSON
(`https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json`).
`search`/`install`/`update` o buscam no máximo a cada seis horas e fazem
cache em `~/.hermes/cache/`, para que entradas novas e remoções cheguem a
clients instalados sem atualizar o Hermes. Offline, a cópia enviada com seu
checkout é usada. Remoções da lista in-tree e da lista ao vivo são sempre
ambas aplicadas.

### URLs git customizadas são diferentes {#custom-git-urls-are-different}

`hermes plugins install <git-url>` ainda funciona para qualquer repositório,
mas contorna o catálogo por completo:

- **Sem review** — você recebe o que estiver na ponta do branch, não um pin
  revisado.
- **Um banner de aviso** é mostrado para deixar claro que o código não foi
  vetado.
- A lista de removidos ainda é consultada (um repo conhecido como ruim é
  recusado pela URL).

Use o caminho de URL git para seus próprios plugins e repos em que você já
confia; use o catálogo para descoberta.

## Submetendo um plugin ao catálogo {#submitting-a-plugin-to-the-catalog}

Submissões são pull requests que adicionam um arquivo
`plugin-catalog/<name>.yaml`. A checklist completa vive no
[README do plugin-catalog](https://github.com/NousResearch/hermes-agent/tree/main/plugin-catalog);
em resumo, uma entrada precisa ser:

1. **Submetida pelo dono** — o autor do PR é dono ou maintainer do repo do
   plugin.
2. **Um repositório público** — a URL `repo` é clonável publicamente.
3. **Released** — o repo tem releases/tags reais, não só um branch padrão.
4. **Passando na validação** — a GitHub Action de validação do catálogo está
   verde no PR (schema, formato de SHA, reachability).
5. **Pinada em código assentado** — o SHA pinado tem pelo menos **2 semanas**,
   para o catálogo nunca apontar para código empurrado momentos antes da
   review.

Atualizações de pin (bump de `sha` para um commit mais novo) seguem o mesmo
processo de PR + review.

## Veja também {#see-also}

- [Plugins](plugins.md) — o sistema de plugins em si: formato do manifest,
  habilitação, configuração
- [Built-in Plugins](built-in-plugins.md) — plugins que vêm com o Hermes
- [Build a Hermes Plugin](/developer-guide/plugins) — escreva o seu
- [Página do Plugin Catalog](/plugins) — o catálogo navegável
