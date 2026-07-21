---
sidebar_position: 16
title: "Google Gemini"
description: "Use o Hermes Agent com o Google Gemini — API nativa do AI Studio, configuração de chave de API, chamadas de ferramentas, streaming e orientações sobre cotas"
---

# Google Gemini {#google-gemini}

O Hermes Agent oferece suporte ao Google Gemini como provedor nativo usando a **API do Google AI Studio / Gemini** — não o endpoint compatível com OpenAI. Isso permite que o Hermes traduza seu loop interno de mensagens e ferramentas no formato OpenAI para a API nativa `generateContent` do Gemini, preservando chamadas de ferramentas, streaming, entradas multimodais e metadados de resposta específicos do Gemini.

## Pré-requisitos {#prerequisites}

- **Chave de API do Google AI Studio** — crie uma em [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Projeto do Google Cloud com faturamento ativado** — recomendado para uso com agentes. O tier gratuito do Gemini é muito pequeno para sessões de agente de longa duração, pois o Hermes pode fazer várias chamadas ao modelo por turno do usuário.
- **Hermes instalado** — nenhum pacote Python extra é necessário para o provedor nativo do Gemini.

:::tip Caminho da chave de API
Defina `GOOGLE_API_KEY` ou `GEMINI_API_KEY`. O Hermes verifica ambos os nomes para o provedor `gemini`.
:::

## Início Rápido {#quick-start}

```bash
# Adicione sua chave de API do Gemini
echo "GOOGLE_API_KEY=..." >> ~/.hermes/.env

# Selecione o Gemini como seu provedor
hermes model
# → Escolha "More providers..." → "Google AI Studio"
# → O Hermes verifica o tier da sua chave e mostra os modelos do Gemini
# → Selecione um modelo

# Comece a conversar
hermes chat
```

Se preferir editar a configuração diretamente, use a URL base nativa da API do Gemini:

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## Configuração {#configuration}

Depois de executar `hermes model`, seu `~/.hermes/config.yaml` conterá:

```yaml
model:
  default: gemini-3-flash-preview
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

E em `~/.hermes/.env`:

```bash
GOOGLE_API_KEY=...
```

### API Nativa do Gemini {#native-gemini-api}

O endpoint recomendado é:

```text
https://generativelanguage.googleapis.com/v1beta
```

O Hermes detecta esse endpoint e cria seu adaptador nativo do Gemini. Internamente, o Hermes ainda mantém o loop do agente em mensagens no formato OpenAI, traduzindo cada requisição para o esquema nativo do Gemini:

- `messages[]` → `contents[]` do Gemini
- prompts de sistema → `systemInstruction` do Gemini
- esquemas de ferramentas → `functionDeclarations` do Gemini
- resultados de ferramentas → partes `functionResponse` do Gemini
- respostas em streaming → blocos de stream no formato OpenAI para o loop do Hermes

:::note Assinaturas de pensamento do Gemini 3
Para o uso de ferramentas do Gemini 3, o Hermes preserva os valores de `thoughtSignature` anexados às partes de chamada de função e os reproduz no próximo turno de ferramenta. Isso cobre o caminho crítico de validação para fluxos de trabalho de agente com várias etapas.

O Gemini 3 também pode anexar assinaturas de pensamento a outras partes da resposta. O adaptador nativo do Hermes é otimizado hoje para loops de ferramentas de agente, então ainda não reproduz todas as assinaturas fora de chamadas de ferramentas com fidelidade total no nível de parte.
:::

### Prefira o Endpoint Nativo {#prefer-the-native-endpoint}

O Google também expõe um endpoint compatível com OpenAI:

```text
https://generativelanguage.googleapis.com/v1beta/openai/
```

Para sessões de agente do Hermes, prefira o endpoint nativo do Gemini acima. O Hermes inclui um adaptador nativo do Gemini que permite mapear diretamente o uso de ferramentas em várias etapas, resultados de chamadas de ferramentas, streaming, entradas multimodais e metadados de resposta do Gemini para a API `generateContent` do Gemini. O endpoint compatível com OpenAI ainda é útil quando você precisa especificamente de compatibilidade com a API da OpenAI.

Se anteriormente você definiu `GEMINI_BASE_URL` para a URL `/openai`, remova-a ou altere-a:

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

## Modelos Disponíveis {#available-models}

O seletor do `hermes model` mostra os modelos do Gemini mantidos no registro de provedores do Hermes. As opções comuns incluem:

| Modelo | ID | Notas |
|-------|----|-------|
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | Modelo de preview mais capaz quando disponível |
| Gemini 3 Pro Preview | `gemini-3-pro-preview` | Modelo forte em raciocínio e código |
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | Equilíbrio padrão recomendado entre velocidade e capacidade |
| Gemini 3.1 Flash Lite Preview | `gemini-3.1-flash-lite-preview` | Opção mais rápida / de menor custo quando disponível |

A disponibilidade dos modelos muda com o tempo. Se um modelo desaparecer ou não estiver habilitado para sua chave, execute `hermes model` novamente e escolha um da lista atual.

:::info IDs de modelo
Use os IDs nativos de modelo do Gemini, como `gemini-3-flash-preview`, e não IDs no estilo OpenRouter, como `google/gemini-3-flash-preview`, quando `provider: gemini`.
:::

### Aliases Mais Recentes {#latest-aliases}

O Google publica aliases móveis para as famílias Pro e Flash do Gemini. `gemini-pro-latest` e `gemini-flash-latest` são úteis quando você quer que o Google avance o modelo automaticamente sem alterar sua configuração do Hermes.

| Alias | Atualmente aponta para | Notas |
|-------|------------------|-------|
| `gemini-pro-latest` | Modelo Gemini Pro mais recente | Melhor quando você quer o padrão Pro atual do Google |
| `gemini-flash-latest` | Modelo Gemini Flash mais recente | Melhor quando você quer o padrão Flash atual do Google |

```yaml
model:
  default: gemini-pro-latest
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

Se você precisar de reprodutibilidade estrita, prefira IDs de modelo explícitos, como `gemini-3.1-pro-preview` ou `gemini-3-flash-preview`.

### Gemma via a API do Gemini {#gemma-via-the-gemini-api}

O Google também expõe modelos Gemma por meio da API do Gemini. O Hermes os reconhece como modelos do Google, mas oculta entradas Gemma de baixo throughput do seletor de modelo padrão, para que novos usuários não selecionem acidentalmente um modelo de nível avaliação para uma sessão de agente de longa duração.

IDs úteis para avaliação incluem:

| Modelo | ID | Notas |
|-------|----|-------|
| Gemma 4 31B IT | `gemma-4-31b-it` | Modelo Gemma maior; útil para avaliação de compatibilidade e qualidade |
| Gemma 4 26B A4B IT | `gemma-4-26b-a4b-it` | Variante com menos parâmetros ativos, quando disponível |

Esses modelos são melhor tratados como opções de avaliação em chaves de API do Gemini. O preço da API do Gemma do Google é apenas de tier gratuito e os limites de uso são baixos em comparação com os modelos Gemini de produção, então o uso sustentado do agente Hermes normalmente deve migrar para um modelo Gemini pago, uma implantação auto-hospedada, ou outro provedor com a cota adequada.

Para usar um modelo Gemma que está oculto do seletor, defina-o diretamente:

```yaml
model:
  default: gemma-4-31b-it
  provider: gemini
  base_url: https://generativelanguage.googleapis.com/v1beta
```

## Trocando de Modelo Durante a Sessão {#switching-models-mid-session}

Use o comando `/model` durante uma conversa:

```text
/model gemini-3-flash-preview
/model gemini-flash-latest
/model gemini-3-pro-preview
/model gemini-pro-latest
/model gemma-4-31b-it
/model gemini-3.1-flash-lite-preview
```

Se você ainda não configurou o Gemini, saia da sessão e execute `hermes model` primeiro. O `/model` alterna entre provedores e modelos já configurados; ele não coleta novas chaves de API.

## Diagnóstico {#diagnostics}

```bash
hermes doctor
```

O doctor verifica:

- Se `GOOGLE_API_KEY` ou `GEMINI_API_KEY` está disponível
- Se as credenciais do provedor configurado podem ser resolvidas

## Gateway (Plataformas de Mensagens) {#gateway-messaging-platforms}

O Gemini funciona com todas as plataformas de gateway do Hermes (Telegram, Discord, Slack, WhatsApp, LINE, Feishu, etc.). Configure o Gemini como seu provedor e depois inicie o gateway normalmente:

```bash
hermes gateway setup
hermes gateway start
```

O gateway lê o `config.yaml` e usa a mesma configuração do provedor Gemini.

## Solução de Problemas {#troubleshooting}

### "Gemini native client requires an API key" {#gemini-native-client-requires-an-api-key}

O Hermes não conseguiu encontrar uma chave de API utilizável. Adicione uma destas ao `~/.hermes/.env`:

```bash
GOOGLE_API_KEY=...
# ou
GEMINI_API_KEY=...
```

Depois execute `hermes model` novamente.

### "This Google API key is on the free tier" {#this-google-api-key-is-on-the-free-tier}

O Hermes verifica as chaves de API do Gemini durante a configuração. As cotas do tier gratuito podem se esgotar depois de algumas interações do agente, porque o uso de ferramentas, retentativas, compressão e tarefas auxiliares podem exigir várias chamadas ao modelo.

Ative o faturamento no projeto do Google Cloud vinculado à sua chave, regenere a chave se necessário e depois execute:

```bash
hermes model
```

### "404 model not found" {#404-model-not-found}

O modelo selecionado não está disponível para sua conta, região ou chave. Execute `hermes model` novamente e escolha outro modelo Gemini da lista atual.

### O modelo Gemma não aparece em `hermes model` {#gemma-model-is-not-shown-in-hermes-model}

O Hermes pode ocultar modelos Gemma de baixo throughput do seletor por padrão. Se você intencionalmente quiser avaliar um deles, defina o ID do modelo diretamente em `~/.hermes/config.yaml`.

### "429 quota exceeded" no Gemma {#429-quota-exceeded-on-gemma}

Os modelos Gemma expostos pela API do Gemini são úteis para avaliação, mas seus limites de tier gratuito na API do Gemini são baixos. Use-os para testes de compatibilidade e depois migre para um modelo Gemini pago ou outro provedor para sessões de agente sustentadas.

### O endpoint compatível com OpenAI está configurado {#openai-compatible-endpoint-is-configured}

Verifique em `~/.hermes/.env`:

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

Altere-o para o endpoint nativo ou remova a substituição:

```bash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### Chamada de ferramentas falha com erros de esquema {#tool-calling-fails-with-schema-errors}

Atualize o Hermes e execute `hermes model` novamente. O adaptador nativo do Gemini sanitiza os esquemas de ferramentas para o formato mais estrito de declaração de função do Gemini; builds mais antigas ou endpoints personalizados podem não fazer isso.

## Relacionados {#related}

- [AI Providers](/integrations/providers)
- [Configuration](/user-guide/configuration)
- [Fallback Providers](/user-guide/features/fallback-providers)
- [AWS Bedrock](/guides/aws-bedrock) — integração nativa com provedor de nuvem usando credenciais AWS
