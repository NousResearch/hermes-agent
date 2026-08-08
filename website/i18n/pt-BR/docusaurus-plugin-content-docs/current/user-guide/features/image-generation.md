---
title: Geração de Imagens
description: Gere imagens via FAL.ai — 11 modelos incluindo FLUX 2, GPT Image (1.5 & 2), Nano Banana Pro, Ideogram, Recraft V4 Pro, Krea 2 e mais, selecionáveis via `hermes tools`.
sidebar_label: Geração de Imagens
sidebar_position: 6
---

# Geração de Imagens

O Hermes Agent gera imagens a partir de prompts de texto via FAL.ai. Onze modelos são suportados out of the box, cada um com diferentes tradeoffs de velocidade, qualidade e custo. O modelo ativo é configurável pelo usuário via `hermes tools` e persiste em `config.yaml`.

## Modelos suportados {#supported-models}

| Modelo | Velocidade | Pontos fortes | Preço |
|---|---|---|---|
| `fal-ai/flux-2/klein/9b` *(padrão)* | `<1s` | Rápido, texto nítido | $0.006/MP |
| `fal-ai/flux-2-pro` | ~6s | Fotorrealismo de estúdio | $0.03/MP |
| `fal-ai/z-image/turbo` | ~2s | Bilíngue EN/CN, 6B params | $0.005/MP |
| `fal-ai/nano-banana-pro` | ~8s | Gemini 3 Pro, profundidade de raciocínio, renderização de texto | $0.15/image (1K) |
| `fal-ai/gpt-image-1.5` | ~15s | Aderência ao prompt | $0.034/image |
| `fal-ai/gpt-image-2` | ~20s | Renderização de texto SOTA + CJK, fotorrealismo world-aware | $0.04–0.06/image |
| `fal-ai/ideogram/v3` | ~5s | Melhor tipografia | $0.03–0.09/image |
| `fal-ai/recraft/v4/pro/text-to-image` | ~8s | Design, brand systems, pronto para produção | $0.25/image |
| `fal-ai/qwen-image` | ~12s | Baseado em LLM, texto complexo | $0.02/MP |
| `fal-ai/krea/v2/medium/text-to-image` | ~15-25s | Ilustração, anime, pintura, estilos expressivos/artísticos | $0.030–0.035/image |
| `fal-ai/krea/v2/large/text-to-image` | ~25-60s | Fotorrealismo, looks texturizados raw (motion blur, grain, film) | $0.060–0.065/image |

Os preços são da FAL no momento da escrita; confira [fal.ai](https://fal.ai/) para valores atuais.

## Configuração {#setup}

:::tip Assinantes Nous
Se você tem uma assinatura paga do [Nous Portal](https://portal.nousresearch.com), pode usar geração de imagens pelo **[Tool Gateway](tool-gateway.md)** sem uma chave de API FAL. Sua seleção de modelo persiste em ambos os caminhos. Instalações novas podem rodar `hermes setup --portal` para fazer login e ligar todas as ferramentas do gateway de uma vez; instalações existentes podem escolher **Nous Subscription** como backend de image-gen via `hermes tools`.

Se o gateway gerenciado retornar `HTTP 4xx` para um modelo específico, esse modelo ainda não está proxied no lado do portal — o agente informará, com passos de remediação (defina `FAL_KEY` para acesso direto, ou escolha outro modelo).
:::

### Obtenha uma chave de API FAL {#get-a-fal-api-key}

1. Cadastre-se em [fal.ai](https://fal.ai/)
2. Gere uma chave de API no seu dashboard

### Configure e escolha um modelo {#configure-and-pick-a-model}

Execute o comando de ferramentas:

```bash
hermes tools
```

Navegue até **🎨 Image Generation**, escolha seu backend (Nous Subscription ou FAL.ai), então o picker mostra todos os modelos suportados em uma tabela alinhada em colunas — setas para navegar, Enter para selecionar:

```
  Model                          Speed    Strengths                    Price
  fal-ai/flux-2/klein/9b         <1s      Fast, crisp text             $0.006/MP   ← currently in use
  fal-ai/flux-2-pro              ~6s      Studio photorealism          $0.03/MP
  fal-ai/z-image/turbo           ~2s      Bilingual EN/CN, 6B          $0.005/MP
  ...
```

Sua seleção é salva em `config.yaml`:

```yaml
image_gen:
  model: fal-ai/flux-2/klein/9b
  use_gateway: false            # true if using Nous Subscription
  max_parallel_requests: 4      # concurrent images in one tool-call batch
```

`max_parallel_requests` tem padrão `4`. O Hermes limita a pelo menos um e
ao limite global de tool-workers, então providers de imagem recebem requisições paralelas limitadas sem permitir que um batch de imagens contorne o cap de concorrência do agente.

### Qualidade GPT-Image {#gpt-image-quality}

A qualidade de requisição de `fal-ai/gpt-image-1.5` e `fal-ai/gpt-image-2` está fixada em `medium` (~$0.034–$0.06/image em 1024×1024). Não expomos os tiers `low` / `high` como opção voltada ao usuário para que a cobrança do Nous Portal permaneça previsível entre todos os usuários — a diferença de custo entre tiers é 3–22×. Se quiser uma opção mais barata, escolha Klein 9B ou Z-Image Turbo; se quiser qualidade maior, use Nano Banana Pro ou Recraft V4 Pro.

## Uso {#usage}

O schema voltado ao agente é intencionalmente mínimo — o modelo usa o que você configurou:

```
Generate an image of a serene mountain landscape with cherry blossoms
```

```
Create a square portrait of a wise old owl — use the typography model
```

```
Make me a futuristic cityscape, landscape orientation
```

## Image-to-image / Edição {#image-to-image--editing}

A mesma ferramenta `image_generate` também **edita imagens existentes** quando o modelo
ativo suporta — passe uma imagem fonte e o backend roteia automaticamente para seu endpoint
de edição (espelha como `video_generate` lida com image-to-video).
Omita a imagem fonte e é text-to-image simples.

```
Take this photo and make it a rainy Tokyo street at night → <image>
```

```
Blend these two product shots into one hero image → <image1> <image2>
```

Duas entradas dirigem a edição:

- **`image_url`** — a imagem fonte principal para editar/transformar (URL pública ou caminho local).
- **`reference_image_urls`** — referências adicionais de estilo/composição (limitadas por modelo).

### Quais backends suportam edição {#which-backends-support-editing}

| Backend | Image-to-image | Limite de referências | Como |
|---|---|---|---|
| **FAL.ai** (modelos edit-capable abaixo) | ✓ | até 9 | roteia para o endpoint `/edit` do modelo |
| **OpenAI** (`gpt-image-2`) | ✓ | até 16 | `images.edit()` |
| **xAI** (Grok Imagine) | ✓ | 1 | `/v1/images/edits` (`grok-imagine-image-quality`) |
| **Krea** (`Krea 2`) | ✓ | até 10 | geração guiada por referência (`image_style_references`) |
| **OpenAI (Codex auth)** | ✓ | até 16 | ferramenta Codex Responses `image_generation` com content parts `input_image` |

Modelos FAL com endpoint de edição: `flux-2/klein/9b`, `flux-2-pro`,
`nano-banana-pro`, `gpt-image-1.5`, `gpt-image-2`, `ideogram/v3` e
`qwen-image`. Modelos FAL puramente text-to-image (`z-image/turbo`, `recraft`,
`krea/*`) rejeitam entradas de imagem com um erro claro apontando para um
modelo edit-capable.

:::note OpenAI (Codex auth) é best-effort

A superfície Codex (`chatgpt.com/backend-api/codex`) hospeda `image_generation`
como ferramenta que o modelo de chat pode chamar, e o Hermes não consegue forçar a chamada — o
backend rejeita toda forma de `tool_choice` para ferramentas hospedadas, então a requisição
depende de instruções para direcionar o modelo. Quando o modelo host recusa
invocar a ferramenta, a chamada falha com `empty_response`. Se a ferramenta de imagem
hospedada é alcançável também foi reportado variar entre
contas. Se precisa que geração de imagens funcione deterministicamente, configure
o backend **OpenAI** (API key), **FAL** ou **xAI**.

:::

A capacidade de edição do modelo ativo é exposta na descrição da ferramenta em
runtime, para que o agente saiba se `image_url` será honrado antes de
chamar a ferramenta.

## Proporções de aspecto {#aspect-ratios}

Todo modelo aceita as mesmas três proporções de aspecto da perspectiva do agente. Internamente, o spec de tamanho nativo de cada modelo é preenchido automaticamente:

| Entrada do agente | image_size (flux/z-image/qwen/recraft/ideogram) | aspect_ratio (nano-banana-pro) | image_size (gpt-image-1.5) | image_size (gpt-image-2) |
|---|---|---|---|---|
| `landscape` | `landscape_16_9` | `16:9` | `1536x1024` | `landscape_4_3` (1024×768) |
| `square` | `square_hd` | `1:1` | `1024x1024` | `square_hd` (1024×1024) |
| `portrait` | `portrait_16_9` | `9:16` | `1024x1536` | `portrait_4_3` (768×1024) |

GPT Image 2 mapeia para presets 4:3 em vez de 16:9 porque sua contagem mínima de pixels é 655.360 — o preset `landscape_16_9` (1024×576 = 589.824) seria rejeitado.

Essa tradução acontece em `_build_fal_payload()` — código do agente nunca precisa saber sobre diferenças de schema por modelo.

## Upscaling automático {#automatic-upscaling}

Upscaling via **Clarity Upscaler** da FAL é limitado por modelo:

| Modelo | Upscale? | Por quê |
|---|---|---|
| `fal-ai/flux-2-pro` | ✓ | Backward-compat (era o default pré-picker) |
| Todos os outros | ✗ | Modelos rápidos perderiam sua proposta de valor sub-segundo; modelos hi-res não precisam |

Quando upscaling roda, usa estas configurações:

| Configuração | Valor |
|---|---|
| Fator de upscale | 2× |
| Criatividade | 0.35 |
| Semelhança | 0.6 |
| Guidance scale | 4 |
| Inference steps | 18 |

Se upscaling falhar (problema de rede, rate limit), a imagem original é retornada automaticamente.

## Como funciona internamente {#how-it-works-internally}

1. **Resolução de modelo** — `_resolve_fal_model()` lê `image_gen.model` de `config.yaml`, cai para a env var `FAL_IMAGE_MODEL`, depois para `fal-ai/flux-2/klein/9b`.
2. **Construção de payload** — `_build_fal_payload()` traduz seu `aspect_ratio` para o formato nativo do modelo (enum preset, enum aspect-ratio ou literal GPT), mescla os params padrão do modelo, aplica quaisquer overrides do caller e filtra pela whitelist `supports` do modelo para que chaves não suportadas nunca sejam enviadas.
3. **Submissão** — `_submit_fal_request()` roteia via credenciais FAL diretas ou gateway Nous gerenciado.
4. **Upscaling** — roda apenas se os metadados do modelo tiverem `upscale: True`.
5. **Entrega** — URL final da imagem retornada ao agente, que emite uma tag `MEDIA:<url>` que adapters de plataforma convertem em mídia nativa.

## Depuração {#debugging}

Habilite logging de debug:

```bash
export IMAGE_TOOLS_DEBUG=true
```

Logs de debug vão para `./logs/image_tools_debug_<session_id>.json` com detalhes por chamada (modelo, parâmetros, timing, erros).

## Entrega por plataforma {#platform-delivery}

| Plataforma | Entrega |
|---|---|
| **CLI** | URL da imagem impressa como markdown `![](url)` — clique para abrir |
| **Telegram** | Mensagem de foto com o prompt como legenda |
| **Discord** | Embutida em uma mensagem |
| **Slack** | URL desdobrada pelo Slack |
| **WhatsApp** | Mensagem de mídia |
| **Outras** | URL em texto simples |

## Limitações {#limitations}

- **Requer credenciais** para o backend ativo (FAL `FAL_KEY` / Nous Subscription, `OPENAI_API_KEY`, xAI OAuth, `KREA_API_KEY`)
- **Edição depende do modelo** — image-to-image funciona apenas em modelos edit-capable (veja a tabela acima); modelos só text-to-image rejeitam entradas de imagem com erro claro
- **URLs temporárias** — backends retornam URLs hospedadas que expiram após horas/dias; o Hermes materializa para o cache local para que a entrega funcione após expiração
- **Restrições por modelo** — alguns modelos não suportam `seed`, `num_inference_steps`, etc. O filtro `supports` / `edit_supports` descarta silenciosamente params não suportados; isso é comportamento esperado
