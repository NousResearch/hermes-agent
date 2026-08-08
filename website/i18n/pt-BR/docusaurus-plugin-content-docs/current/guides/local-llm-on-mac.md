---
sidebar_position: 2
title: "Execute LLMs Locais no Mac"
description: "Configure um servidor de LLM local, compatível com a API da OpenAI, no macOS com llama.cpp ou MLX, incluindo seleção de modelo, otimização de memória e benchmarks reais em Apple Silicon"
---

# Execute LLMs Locais no Mac

Este guia mostra como executar um servidor de LLM local no macOS com uma API compatível com a OpenAI. Você ganha privacidade total, custo zero de API e um desempenho surpreendentemente bom em Apple Silicon.

Cobrimos dois backends:

| Backend | Instalação | Melhor em | Formato |
|---------|---------|---------|--------|
| **llama.cpp** | `brew install llama.cpp` | Menor tempo até o primeiro token, cache KV quantizado para baixo uso de memória | GGUF |
| **omlx** | [omlx.ai](https://omlx.ai) | Geração de tokens mais rápida, otimização nativa do Metal | MLX (safetensors) |

Ambos expõem um endpoint `/v1/chat/completions` compatível com a OpenAI. O Hermes funciona com qualquer um dos dois — basta apontá-lo para `http://localhost:8080` ou `http://localhost:8000`.

:::info Apenas Apple Silicon
Este guia é voltado para Macs com Apple Silicon (M1 e posteriores). Macs Intel funcionarão com llama.cpp, mas sem aceleração de GPU — espere um desempenho significativamente mais lento.
:::

---

## Escolhendo um modelo {#choosing-a-model}

Para começar, recomendamos o **Qwen3.5-9B** — é um modelo de raciocínio forte que cabe confortavelmente em 8GB+ de memória unificada com quantização.

| Variante | Tamanho no disco | RAM necessária (contexto de 128K) | Backend |
|---------|-------------|---------------------------|---------|
| Qwen3.5-9B-Q4_K_M (GGUF) | 5,3 GB | ~10 GB com cache KV quantizado | llama.cpp |
| Qwen3.5-9B-mlx-lm-mxfp4 (MLX) | ~5 GB | ~12 GB | omlx |

**Regra prática de memória:** tamanho do modelo + cache KV. Um modelo de 9B em Q4 tem cerca de 5 GB. O cache KV em contexto de 128K com quantização Q4 adiciona ~4-5 GB. Com o cache KV padrão (f16), isso salta para ~16 GB. As flags de cache KV quantizado no llama.cpp são o truque principal para sistemas com memória limitada.

Para modelos maiores (27B, 35B), você vai precisar de 32 GB+ de memória unificada. O 9B é o ponto ideal para máquinas de 8-16 GB.

---

## Opção A: llama.cpp {#option-a-llamacpp}

O llama.cpp é o runtime de LLM local mais portátil. No macOS, ele usa o Metal para aceleração de GPU imediatamente.

### Instale {#install}

```bash
brew install llama.cpp
```

Isso te dá o comando `llama-server` globalmente.

### Baixe o modelo {#download-the-model}

Você precisa de um modelo no formato GGUF. A fonte mais fácil é o Hugging Face, via `huggingface-cli`:

```bash
brew install huggingface-cli
```

Depois baixe:

```bash
huggingface-cli download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir ~/models
```

:::tip Modelos com acesso restrito
Alguns modelos no Hugging Face exigem autenticação. Execute `huggingface-cli login` primeiro se você receber um erro 401 ou 404.
:::

### Inicie o servidor {#start-the-server}

```bash
llama-server -m ~/models/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 \
  -c 131072 \
  -np 1 \
  -fa on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --host 0.0.0.0
```

Veja o que cada flag faz:

| Flag | Finalidade |
|------|---------|
| `-ngl 99` | Descarrega todas as camadas para a GPU (Metal). Use um número alto para garantir que nada fique na CPU. |
| `-c 131072` | Tamanho da janela de contexto (128K tokens). Reduza se você tiver pouca memória. |
| `-np 1` | Número de slots paralelos. Mantenha em 1 para uso de um único usuário — mais slots dividem seu orçamento de memória. |
| `-fa on` | Flash attention. Reduz o uso de memória e acelera a inferência de contexto longo. |
| `--cache-type-k q4_0` | Quantiza o cache de chave (key) para 4 bits. **Este é o grande redutor de memória.** |
| `--cache-type-v q4_0` | Quantiza o cache de valor (value) para 4 bits. Junto com o anterior, isso corta a memória do cache KV em ~75% em relação ao f16. |
| `--host 0.0.0.0` | Escuta em todas as interfaces. Use `127.0.0.1` se você não precisar de acesso via rede. |

O servidor está pronto quando você vê:

```
main: server is listening on http://0.0.0.0:8080
srv  update_slots: all slots are idle
```

### Otimização de memória para sistemas restritos {#memory-optimization-for-constrained-systems}

As flags `--cache-type-k q4_0 --cache-type-v q4_0` são a otimização mais importante para sistemas com memória limitada. Aqui está o impacto em um contexto de 128K:

| Tipo de cache KV | Memória do cache KV (contexto de 128K, modelo de 9B) |
|---------------|--------------------------------------|
| f16 (padrão) | ~16 GB |
| q8_0 | ~8 GB |
| **q4_0** | **~4 GB** |

Em um Mac de 8 GB, use o cache KV `q4_0` e escolha um modelo menor que ainda caiba no mínimo de 64K de contexto do Hermes. Em 16 GB, você pode fazer confortavelmente contexto de 128K. Em 32 GB+, você pode rodar modelos maiores ou múltiplos slots paralelos.

Se você ainda estiver ficando sem memória, reduza o contexto apenas mantendo-se no mínimo de 64K do Hermes ou acima dele; caso contrário, mude para um modelo menor ou uma quantização menor (Q3_K_M em vez de Q4_K_M).

### Teste {#test-it}

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q4_K_M.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### Obtenha o nome do modelo {#get-the-model-name}

Se você esquecer o nome do modelo, consulte o endpoint de modelos:

```bash
curl -s http://localhost:8080/v1/models | jq '.data[].id'
```

---

## Opção B: MLX via omlx {#option-b-mlx-via-omlx}

O [omlx](https://omlx.ai) é um aplicativo nativo do macOS que gerencia e serve modelos MLX. O MLX é o próprio framework de machine learning da Apple, otimizado especificamente para a arquitetura de memória unificada do Apple Silicon.

### Instale {#install-1}

Baixe e instale a partir de [omlx.ai](https://omlx.ai). Ele fornece uma GUI para gerenciamento de modelos e um servidor integrado.

### Baixe o modelo {#download-the-model-1}

Use o aplicativo omlx para navegar e baixar modelos. Procure por `Qwen3.5-9B-mlx-lm-mxfp4` e baixe-o. Os modelos são armazenados localmente (normalmente em `~/.omlx/models/`).

### Inicie o servidor {#start-the-server-1}

O omlx serve modelos em `http://127.0.0.1:8000` por padrão. Inicie o serviço pela interface do aplicativo, ou use a CLI, se disponível.

### Teste {#test-it-1}

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-mlx-lm-mxfp4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }' | jq .choices[0].message.content
```

### Liste os modelos disponíveis {#list-available-models}

O omlx pode servir vários modelos simultaneamente:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'
```

---

## Benchmarks: llama.cpp vs MLX {#benchmarks-llamacpp-vs-mlx}

Ambos os backends testados na mesma máquina (Apple M5 Max, 128 GB de memória unificada), rodando o mesmo modelo (Qwen3.5-9B) em níveis de quantização comparáveis (Q4_K_M para GGUF, mxfp4 para MLX). Cinco prompts diversos, três execuções cada, backends testados sequencialmente para evitar disputa de recursos.

### Resultados {#results}

| Métrica | llama.cpp (Q4_K_M) | MLX (mxfp4) | Vencedor |
|--------|-------------------|-------------|--------|
| **TTFT (média)** | **67 ms** | 289 ms | llama.cpp (4,3x mais rápido) |
| **TTFT (p50)** | **66 ms** | 286 ms | llama.cpp (4,3x mais rápido) |
| **Geração (média)** | 70 tok/s | **96 tok/s** | MLX (37% mais rápido) |
| **Geração (p50)** | 70 tok/s | **96 tok/s** | MLX (37% mais rápido) |
| **Tempo total (512 tokens)** | 7,3s | **5,5s** | MLX (25% mais rápido) |

### O que isso significa {#what-this-means}

- **llama.cpp** se destaca no processamento de prompts — seu pipeline de flash attention + cache KV quantizado entrega o primeiro token em ~66ms. Se você está construindo aplicações interativas onde a responsividade percebida importa (chatbots, autocompletar), essa é uma vantagem significativa.

- **MLX** gera tokens ~37% mais rápido depois que começa. Para cargas de trabalho em lote, geração de formato longo, ou qualquer tarefa em que o tempo total de conclusão importa mais que a latência inicial, o MLX termina mais rápido.

- Ambos os backends são **extremamente consistentes** — a variação entre execuções foi insignificante. Você pode confiar nesses números.

### Qual você deve escolher? {#which-one-should-you-pick}

| Caso de uso | Recomendação |
|----------|---------------|
| Chat interativo, ferramentas de baixa latência | llama.cpp |
| Geração de formato longo, processamento em massa | MLX (omlx) |
| Memória restrita (8-16 GB) | llama.cpp (o cache KV quantizado é imbatível) |
| Servindo vários modelos simultaneamente | omlx (suporte multi-modelo integrado) |
| Compatibilidade máxima (Linux também) | llama.cpp |

---

## Conecte ao Hermes {#connect-to-hermes}

Uma vez que seu servidor local esteja rodando:

```bash
hermes model
```

Selecione **Custom endpoint** e siga as instruções. Ele vai pedir a URL base e o nome do modelo — use os valores do backend que você configurou acima.

---

## Tempos Limite {#timeouts}

O Hermes detecta automaticamente endpoints locais (localhost, IPs de rede local) e relaxa seus tempos limite de streaming. Nenhuma configuração é necessária na maioria dos casos.

Se você ainda tiver erros de tempo limite (por exemplo, contextos muito grandes em hardware lento), você pode sobrescrever o tempo limite de leitura de streaming:

```bash
# In your .env — raise from the 120s default to 30 minutes
HERMES_STREAM_READ_TIMEOUT=1800
```

| Tempo limite | Padrão | Ajuste automático local | Sobrescrita via variável de ambiente |
|---------|---------|----------------------|------------------|
| Leitura de streaming (nível de socket) | 120s | Aumentado para 1800s | `HERMES_STREAM_READ_TIMEOUT` |
| Detecção de stream obsoleto | 180s | Desativado completamente | `HERMES_STREAM_STALE_TIMEOUT` |
| Chamada de API (não streaming) | 1800s | Nenhuma mudança necessária | `HERMES_API_TIMEOUT` |

O tempo limite de leitura de streaming é o mais propenso a causar problemas — é o prazo no nível de socket para receber o próximo pedaço de dados. Durante o pré-processamento (prefill) em contextos grandes, modelos locais podem não produzir saída por minutos enquanto processam o prompt. A detecção automática trata isso de forma transparente.
