---
title: "Nous Tool Gateway"
description: "Uma assinatura, todas as ferramentas. Busca web, geração de imagens, TTS e browsers em nuvem — tudo roteado pelo Nous Portal sem chaves de API extras."
sidebar_label: "Tool Gateway"
sidebar_position: 2
---

# Nous Tool Gateway

**Uma assinatura. Todas as ferramentas integradas.**

O Tool Gateway está incluído em toda assinatura paga do [Nous Portal](https://portal.nousresearch.com). Ele roteia as chamadas de ferramentas do Hermes — busca web, geração de imagens, conversão de texto em fala e automação de browsers em nuvem — pela infraestrutura que a Nous já opera, então você não precisa se cadastrar no Firecrawl, FAL, OpenAI, Browser Use ou em qualquer outro serviço só pra deixar seu agente útil.

<div style={{display: 'flex', gap: '1rem', flexWrap: 'wrap', margin: '1.5rem 0'}}>
  <a href="https://portal.nousresearch.com/manage-subscription" style={{background: 'var(--ifm-color-primary)', color: 'white', padding: '0.75rem 1.5rem', borderRadius: '6px', textDecoration: 'none', fontWeight: 'bold'}}>Iniciar ou gerenciar assinatura →</a>
</div>

## O que está incluído

| | Ferramenta | O que você ganha |
|---|---|---|
| 🔍 | **Busca e extração web** | Busca web e extração de página inteira de qualidade para agentes via Firecrawl. Sem se preocupar com rate limits — o gateway cuida do escalonamento. |
| 🎨 | **Geração de imagens** | Nove modelos sob um único endpoint: **FLUX 2 Klein 9B**, **FLUX 2 Pro**, **Z-Image Turbo**, **Nano Banana Pro** (Gemini 3 Pro Image), **GPT Image 1.5**, **GPT Image 2**, **Ideogram V3**, **Recraft V4 Pro**, **Qwen Image**. Escolha por geração com uma flag, ou deixe o Hermes usar FLUX 2 Klein como padrão. |
| 🔊 | **Texto para fala** | Vozes do OpenAI TTS conectadas à ferramenta `text_to_speech`. Mande áudios pelo Telegram, gere áudio pra pipelines, narre o que quiser. |
| 🌐 | **Automação de browser em nuvem** | Sessões headless do Chromium via Browser Use. `browser_navigate`, `browser_click`, `browser_type`, `browser_vision` — todos os primitivos pra dirigir o agente, sem precisar de conta no Browserbase. |

Os quatro são cobrados sob demanda na sua assinatura Nous. Use qualquer combinação — rode o gateway pra web e imagens enquanto mantém sua chave do ElevenLabs pra TTS, ou roteie tudo pela Nous.

## Por que isso existe

Construir um agente que de fato *faça coisas* significa juntar 5+ assinaturas de API — cada uma com seu próprio cadastro, rate limits, cobrança e particularidades. O gateway colapsa isso numa única conta:

- **Uma fatura.** Pague à Nous; nós cuidamos do resto.
- **Um cadastro.** Sem contas no Firecrawl, FAL, Browser Use ou OpenAI audio pra gerenciar.
- **Uma chave.** Seu OAuth do Nous Portal cobre todas as ferramentas.
- **Mesma qualidade.** Os mesmos backends usados pelo caminho de chave direta — só que com a Nous na frente.

Traga suas próprias chaves quando quiser — por ferramenta, a qualquer momento. O gateway não é lock-in, é atalho.

## Comece a usar

```bash
hermes model          # Escolha Nous Portal como provider
```

Ao selecionar o Nous Portal, o Hermes oferece ativar o Tool Gateway. Aceite, e pronto — toda ferramenta suportada fica viva na próxima execução.

Confira o que está ativo a qualquer momento:

```bash
hermes status
```

Você verá uma seção como:

```
◆ Nous Tool Gateway
  Nous Portal     ✓ managed tools available
  Web tools       ✓ active via Nous subscription
  Image gen       ✓ active via Nous subscription
  TTS             ✓ active via Nous subscription
  Browser         ○ active via Browser Use key
```

Ferramentas marcadas com "active via Nous subscription" estão indo pelo gateway. Qualquer outra está usando suas próprias chaves.

## Elegibilidade

O Tool Gateway é um recurso **de assinatura paga**. Contas Nous gratuitas podem usar o Portal pra inferência mas não incluem ferramentas gerenciadas — [faça upgrade do plano](https://portal.nousresearch.com/manage-subscription) pra desbloquear o gateway.

## Misture e combine

O gateway é por ferramenta. Ative só pra o que você quer:

- **Tudo via Nous** — mais fácil; uma assinatura, e acabou.
- **Gateway pra web + imagens, TTS próprio** — mantenha sua voz do ElevenLabs, deixe a Nous cuidar do resto.
- **Gateway só pra coisas que você não tem chave** — "já pago Browserbase, mas não quero conta no Firecrawl" funciona perfeitamente.

Troque qualquer ferramenta a qualquer momento via:

```bash
hermes tools          # Seletor interativo por categoria de ferramenta
```

Selecione a ferramenta, escolha **Nous Subscription** como provider (ou qualquer provider direto que preferir). Sem editar config na mão.

## Usando modelos individuais de imagem

A geração de imagens usa FLUX 2 Klein 9B como padrão, por velocidade. Sobrescreva por chamada passando o ID do modelo pra ferramenta `image_generate`:

| Modelo | ID | Melhor pra |
|---|---|---|
| FLUX 2 Klein 9B | `fal-ai/flux-2/klein/9b` | Rápido, bom padrão |
| FLUX 2 Pro | `fal-ai/flux-2-pro` | FLUX de fidelidade maior |
| Z-Image Turbo | `fal-ai/z-image/turbo` | Estilizado, rápido |
| Nano Banana Pro | `fal-ai/nano-banana-pro` | Google Gemini 3 Pro Image |
| GPT Image 1.5 | `fal-ai/gpt-image-1.5` | Geração de imagem da OpenAI, texto+imagem |
| GPT Image 2 | `fal-ai/gpt-image-2` | OpenAI mais recente |
| Ideogram V3 | `fal-ai/ideogram/v3` | Boa aderência a prompt + tipografia |
| Recraft V4 Pro | `fal-ai/recraft/v4/pro/text-to-image` | Estilo vetorial, design gráfico |
| Qwen Image | `fal-ai/qwen-image` | Multimodal da Alibaba |

O conjunto evolui — `hermes tools` → Image Generation mostra a lista atual ao vivo.

---

## Referência de configuração

A maioria dos usuários nunca precisa mexer aqui — `hermes model` e `hermes tools` cobrem todo workflow de forma interativa. Esta seção é pra quem edita `config.yaml` direto ou roteiriza setups.

### Flag `use_gateway` por ferramenta

O bloco de config de cada ferramenta aceita um booleano `use_gateway`:

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

Precedência: `use_gateway: true` roteia pela Nous independentemente de chaves diretas no `.env`. `use_gateway: false` (ou ausente) usa chaves diretas se disponíveis e só faz fallback pro gateway quando nenhuma existe.

### Desativando o gateway

```yaml
web:
  use_gateway: false   # Hermes agora usa FIRECRAWL_API_KEY do .env
```

`hermes tools` limpa a flag automaticamente quando você escolhe um provider que não é o gateway, então isso geralmente acontece sozinho.

### Gateway self-hosted (avançado)

Rodando seu próprio gateway compatível com a Nous? Sobrescreva os endpoints em `~/.hermes/.env`:

```bash
TOOL_GATEWAY_DOMAIN=seu-dominio.exemplo.com
TOOL_GATEWAY_SCHEME=https
TOOL_GATEWAY_USER_TOKEN=seu-token        # normalmente preenchido automaticamente pelo login do Portal
FIRECRAWL_GATEWAY_URL=https://...         # sobrescrever um endpoint específico
```

Esses parâmetros existem pra setups de infraestrutura customizada (deploys enterprise, ambientes de dev). Assinantes regulares nunca precisam definir.

## FAQ

### Funciona com Telegram / Discord / outros gateways de mensagem?

Sim. O Tool Gateway opera na camada de execução de ferramentas, não no CLI. Toda interface que pode chamar uma ferramenta — CLI, Telegram, Discord, Slack, IRC, Teams, o servidor de API, qualquer coisa — se beneficia dele de forma transparente.

### O que acontece se minha assinatura expirar?

Ferramentas roteadas pelo gateway param de funcionar até você renovar ou trocar por chaves de API diretas via `hermes tools`. O Hermes mostra um erro claro apontando pro portal.

### Dá pra ver uso ou custos por ferramenta?

Sim — o [dashboard do Nous Portal](https://portal.nousresearch.com) quebra o uso por ferramenta pra você ver o que está pesando na sua conta.

### O Modal (terminal serverless) está incluído?

O Modal está disponível como **add-on opcional** da assinatura Nous, não faz parte do bundle padrão do Tool Gateway. Configure via `hermes setup terminal` ou direto no `config.yaml` quando quiser uma sandbox remota pra execução de shell.

### Preciso apagar minhas chaves de API existentes ao ativar o gateway?

Não — mantenha elas no `.env`. Quando `use_gateway: true`, o Hermes ignora chaves diretas e usa o gateway. Volta a flag pra `false` e suas chaves voltam a ser a fonte. O gateway não é lock-in.
