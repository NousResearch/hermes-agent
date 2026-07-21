---
title: "Nous Tool Gateway"
description: "Uma assinatura, todas as ferramentas. Web search, geração de imagem, TTS e navegadores na nuvem — tudo roteado pelo Nous Portal sem chaves de API extras."
sidebar_label: "Tool Gateway"
sidebar_position: 2
---

# Nous Tool Gateway

**Uma assinatura. Todas as ferramentas incluídas.**

O Tool Gateway está incluído em toda assinatura paga do [Nous Portal](https://portal.nousresearch.com). Ele roteia as chamadas de ferramentas do Hermes — web search, geração de imagem, text-to-speech e automação de navegador na nuvem — pela infraestrutura que a Nous já opera, para que você não precise se cadastrar no Firecrawl, FAL, OpenAI, Browser Use ou em mais ninguém só para tornar seu agente útil.

<div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', margin: '1.5rem 0'}}>
  <a href="https://portal.nousresearch.com/manage-subscription" style={{background: 'var(--ifm-color-primary)', color: 'white', padding: '0.75rem 1.5rem', borderRadius: '6px', textDecoration: 'none', fontWeight: 'bold'}}>Iniciar ou gerenciar assinatura →</a>
</div>

## O que está incluído {#whats-included}

| | Ferramenta | O que você obtém |
|---|---|---|
| 🔍 | **Web search & extract** | Web search e extração de página completa em nível de agente via Firecrawl. Sem rate limits para se preocupar — o gateway cuida da escala. |
| 🎨 | **Geração de imagem** | Nove modelos em um endpoint: **FLUX 2 Klein 9B**, **FLUX 2 Pro**, **Z-Image Turbo**, **Nano Banana Pro** (Gemini 3 Pro Image), **GPT Image 1.5**, **GPT Image 2**, **Ideogram V3**, **Recraft V4 Pro**, **Qwen Image**. Escolha por geração com uma flag, ou deixe o Hermes usar FLUX 2 Klein como padrão. |
| 🔊 | **Text-to-speech** | Vozes OpenAI TTS integradas à ferramenta `text_to_speech`. Envie notas de voz no Telegram, gere áudio para pipelines, narre qualquer coisa. |
| 🌐 | **Automação de navegador na nuvem** | Sessões Chromium headless via Browser Use. `browser_navigate`, `browser_click`, `browser_type`, `browser_vision` — todos os primitivos que dirigem o agente, sem conta Browserbase. |

Todos os quatro são cobrados pay-as-you-use contra sua assinatura Nous. Use qualquer combinação — rode o gateway para web e imagens mantendo sua própria chave ElevenLabs para TTS, ou roteie tudo pela Nous.

## Por que existe {#why-its-here}

Construir um agente que realmente *faz coisas* significa juntar 5+ assinaturas de API — cada uma com cadastro, rate limits, billing e peculiaridades próprias. O gateway colapsa isso em uma conta:

- **Uma fatura.** Você paga a Nous; nós cuidamos do resto.
- **Um cadastro.** Sem contas Firecrawl, FAL, Browser Use ou OpenAI audio para gerenciar.
- **Uma chave.** Seu OAuth do Nous Portal cobre todas as ferramentas.
- **Mesma qualidade.** Mesmos backends que a rota de chave direta — só fronteados por nós.

Traga suas próprias chaves a qualquer momento — por ferramenta, quando quiser. O gateway não é lock-in, é atalho.

## Começando {#get-started}

Há três caminhos — escolha o que encaixa onde você está:

```bash
hermes setup --portal     # Instalação nova: OAuth Nous + Nous como provider + liga o Tool Gateway de uma vez
```

```bash
hermes model              # Troque seu provider de inferência para Nous Portal — o Hermes então oferece ligar o gateway para todas as ferramentas
```

```bash
hermes tools              # Habilite o gateway por ferramenta — escolha "Nous Subscription" para qualquer ferramenta que quiser
```

`hermes setup --portal` e `hermes model` são os caminhos de uma vez: faça login uma vez, opcionalmente ligue todas as ferramentas ao gateway. `hermes tools` é o caminho à la carte — ligue só as ferramentas que quiser, uma por vez.

**Você não precisa fazer login primeiro.** Com `hermes tools`, os backends gerenciados pela Nous (Web search, Image, Video, TTS, Browser) estão sempre listados, mesmo se você nunca entrou no Nous Portal. Selecione um e o Hermes executa o login do Portal ali mesmo se você ainda não estiver autenticado — sem precisar rodar `hermes model` antes. Se seu OAuth Nous já estiver ativo, selecionar o backend habilita imediatamente sem prompt extra. Esse caminho só faz login e liga a ferramenta que você escolheu — **não** troca seu provider de inferência e **não** pergunta se quer habilitar o gateway para todas as outras ferramentas.

Confira o que está ativo a qualquer momento:

```bash
hermes portal info        # Auth Portal + resumo de roteamento do Tool Gateway
hermes portal tools       # Catálogo do gateway com roteamento atual por ferramenta
hermes status             # Status completo do sistema (Tool Gateway é uma seção)
```

`hermes portal info` mostra uma seção como:

```
◆ Nous Tool Gateway
  Nous Portal     ✓ managed tools available
  Web tools       ✓ active via Nous subscription
  Image gen       ✓ active via Nous subscription
  TTS             ✓ active via Nous subscription
  Browser         ○ active via Browser Use key
```

Ferramentas marcadas "active via Nous subscription" passam pelo gateway. Qualquer outra usa suas próprias chaves.

## Elegibilidade {#eligibility}

O Tool Gateway é recurso de **assinatura paga**. Contas Nous free-tier podem usar o Portal para inferência, mas não incluem ferramentas gerenciadas — [faça upgrade do seu plano](https://portal.nousresearch.com/manage-subscription) para desbloquear o gateway.

Algumas contas também têm direito a um **free tool pool** — uma pequena cota de ferramentas gerenciadas que cobre chamadas do gateway sem assinatura paga. Quando um free pool está disponível, o gateway o exibe e mostra um prompt de setup no primeiro uso, para você optar in e começar a usar ferramentas gerenciadas imediatamente.

## Misture como quiser {#mix-and-match}

O gateway é por ferramenta. Ligue só o que quiser:

- **Todas as ferramentas pela Nous** — mais fácil; uma assinatura, pronto.
- **Gateway para web + imagens, traga seu próprio TTS** — mantenha sua voz ElevenLabs, deixe a Nous cuidar do resto.
- **Gateway só para o que você não tem chave** — "Já pago Browserbase, mas não quero conta Firecrawl" funciona bem.

Troque qualquer ferramenta a qualquer momento via:

```bash
hermes tools          # Seletor interativo para cada categoria de ferramenta
```

Selecione a ferramenta, escolha **Nous Subscription** como provider (ou qualquer provider direto que preferir). Sem editar config manualmente. Se você ainda não estiver logado no Nous Portal, escolher **Nous Subscription** inicia o login do Portal inline — não precisa autenticar via `hermes model` antes.

## Usando modelos de imagem individuais {#using-individual-image-models}

A geração de imagem usa FLUX 2 Klein 9B por padrão por velocidade. Sobrescreva por chamada passando o ID do modelo à ferramenta `image_generate`:

| Modelo | ID | Melhor para |
|---|---|---|
| FLUX 2 Klein 9B | `fal-ai/flux-2/klein/9b` | Rápido, bom padrão |
| FLUX 2 Pro | `fal-ai/flux-2-pro` | FLUX de maior fidelidade |
| Z-Image Turbo | `fal-ai/z-image/turbo` | Estilizado, rápido |
| Nano Banana Pro | `fal-ai/nano-banana-pro` | Google Gemini 3 Pro Image |
| GPT Image 1.5 | `fal-ai/gpt-image-1.5` | OpenAI image gen, texto+imagem |
| GPT Image 2 | `fal-ai/gpt-image-2` | OpenAI mais recente |
| Ideogram V3 | `fal-ai/ideogram/v3` | Forte aderência ao prompt + tipografia |
| Recraft V4 Pro | `fal-ai/recraft/v4/pro/text-to-image` | Estilo vetorial, design gráfico |
| Qwen Image | `fal-ai/qwen-image` | Alibaba multimodal |

O conjunto evolui — `hermes tools` → Image Generation mostra a lista ao vivo atual.

---

## Referência de configuração {#configuration-reference}

A maioria dos usuários nunca precisa mexer nisso — `hermes model` e `hermes tools` cobrem todo fluxo interativamente. Esta seção é para escrever config.yaml diretamente ou automatizar setups.

### Flag `use_gateway` por ferramenta

Cada bloco de config de ferramenta aceita um booleano `use_gateway`:

```yaml
web:
  backend: firecrawl
  use_gateway: true

image_gen:
  use_gateway: true

tts:
  provider: openai
  use_gateway: true

browser:
  cloud_provider: browser-use
  use_gateway: true
```

Precedência: `use_gateway: true` roteia pela Nous independentemente de chaves diretas no `.env`. `use_gateway: false` (ou ausente) usa chaves diretas se disponíveis e só faz fallback ao gateway quando nenhuma existe.

### Desabilitando o gateway {#disabling-the-gateway}

```yaml
web:
  use_gateway: false   # Hermes agora usa FIRECRAWL_API_KEY do .env
```

`hermes tools` limpa a flag automaticamente quando você escolhe um provider não-gateway, então isso normalmente acontece por você.

### Gateway self-hosted (avançado) {#self-hosted-gateway-advanced}

Rodando seu próprio gateway compatível com Nous? Sobrescreva endpoints em `~/.hermes/.env`:

```bash
TOOL_GATEWAY_DOMAIN=your-domain.example.com
TOOL_GATEWAY_SCHEME=https
TOOL_GATEWAY_USER_TOKEN=your-token        # normalmente preenchido automaticamente pelo login Portal
FIRECRAWL_GATEWAY_URL=https://...         # sobrescreve um endpoint especificamente
```

Esses knobs existem para setups de infra customizada (deploy enterprise, ambientes dev). Assinantes regulares nunca os definem.

## FAQ {#faq}

### Funciona com Telegram / Discord / os outros gateways de mensagens?

Sim. O Tool Gateway opera na camada de execução de ferramentas, não no CLI. Toda interface que pode chamar uma ferramenta — CLI, Telegram, Discord, Slack, IRC, Teams, API server, qualquer coisa — se beneficia de forma transparente.

### O que acontece se minha assinatura expirar?

Ferramentas roteadas pelo gateway param de funcionar até você renovar ou trocar por chaves de API diretas via `hermes tools`. O Hermes mostra um erro claro apontando para o portal.

### Posso ver uso ou custos por ferramenta?

Sim — o [dashboard Nous Portal](https://portal.nousresearch.com) detalha uso por ferramenta para você ver o que está puxando sua fatura.

### Modal (terminal serverless) está incluído?

Modal está disponível como **add-on opcional** pela assinatura Nous, não faz parte do bundle padrão do Tool Gateway. Configure via `hermes setup terminal` ou diretamente em `config.yaml` quando quiser um sandbox remoto para execução de shell.

### Preciso apagar minhas chaves de API existentes ao habilitar o gateway?

Não — mantenha-as no `.env`. Com `use_gateway: true`, o Hermes pula chaves diretas e usa o gateway. Volte a flag para `false` e suas chaves voltam a ser a fonte. O gateway não é lock-in.
