---
sidebar_position: 12
title: "Processamento em lote"
description: "Gere trajetórias de agente em escala — processamento paralelo, checkpointing e distribuições de toolsets"
---

# Processamento em lote {#batch-processing}

O processamento em lote permite executar o agente Hermes em centenas ou milhares de prompts em paralelo, gerando dados de trajetória estruturados. Isso é usado principalmente para **geração de dados de treinamento** — produzindo trajetórias no formato ShareGPT com estatísticas de uso de ferramentas que podem ser usadas para fine-tuning ou avaliação.

## Visão geral {#overview}

O batch runner (`batch_runner.py`) processa um dataset JSONL de prompts, executando cada um em uma sessão completa de agente com acesso a ferramentas. Cada prompt recebe seu próprio ambiente isolado. A saída são dados de trajetória estruturados com histórico completo de conversa, estatísticas de chamadas de ferramentas e métricas de cobertura de reasoning.

## Início rápido {#quick-start}

```bash
# Execução básica em lote
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=4

# Retomar uma execução interrompida
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --resume

# Listar distribuições de toolsets disponíveis
python batch_runner.py --list_distributions
```

:::tip Custo previsível em escala
Execuções em lote disparam muitas sessões de agente concorrentes, cada uma fazendo chamadas de model e de ferramentas. Uma assinatura [Nous Portal](/user-guide/features/tool-gateway) agrupa acesso a model plus web search, geração de imagem, TTS e browsers na nuvem em uma única conta — útil quando você quer custo estável por trajetória sem equilibrar rate limits entre cinco contas de vendor. Configure com `hermes setup --portal`, depois aponte `--model` para um model Nous.
:::

## Formato do dataset {#dataset-format}

O dataset de entrada é um arquivo JSONL (um objeto JSON por linha). Cada entrada deve ter um campo `prompt`:

```jsonl
{"prompt": "Write a Python function that finds the longest palindromic substring"}
{"prompt": "Create a REST API endpoint for user authentication using Flask"}
{"prompt": "Debug this error: TypeError: cannot unpack non-iterable NoneType object"}
```

Entradas podem opcionalmente incluir:
- `image` ou `docker_image`: Uma imagem de container para o sandbox deste prompt (funciona com backends Docker, Modal e Singularity)
- `cwd`: Override de diretório de trabalho para a sessão de terminal da tarefa

## Opções de configuração {#configuration-options}

| Parâmetro | Padrão | Descrição |
|-----------|---------|-------------|
| `--dataset_file` | (obrigatório) | Path para o dataset JSONL |
| `--batch_size` | (obrigatório) | Prompts por batch |
| `--run_name` | (obrigatório) | Nome desta execução (usado para dir de saída e checkpointing) |
| `--distribution` | `"default"` | Distribuição de toolsets para amostragem |
| `--model` | `claude-sonnet-4.6` | Model a usar |
| `--base_url` | `https://openrouter.ai/api/v1` | URL base da API |
| `--api_key` | (env var) | API key para o model |
| `--max_turns` | `10` | Máximo de iterações de tool-calling por prompt |
| `--num_workers` | `4` | Processos worker paralelos |
| `--resume` | `false` | Retomar do checkpoint |
| `--verbose` | `false` | Habilitar logging verbose |
| `--max_samples` | all | Processar apenas os primeiros N samples do dataset |
| `--max_tokens` | model default | Máximo de tokens por resposta do model |

### Roteamento de provider (OpenRouter) {#provider-routing-openrouter}

| Parâmetro | Descrição |
|-----------|-------------|
| `--providers_allowed` | Providers permitidos separados por vírgula (ex.: `"anthropic,openai"`) |
| `--providers_ignored` | Providers ignorados separados por vírgula (ex.: `"together,deepinfra"`) |
| `--providers_order` | Ordem preferida de providers separada por vírgula |
| `--provider_sort` | Ordenar por `"price"`, `"throughput"` ou `"latency"` |

### Controle de reasoning {#reasoning-control}

| Parâmetro | Descrição |
|-----------|-------------|
| `--reasoning_effort` | Esforço de reasoning: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra` |
| `--reasoning_disabled` | Desabilitar completamente tokens de reasoning/thinking |

### Opções avançadas {#advanced-options}

| Parâmetro | Descrição |
|-----------|-------------|
| `--ephemeral_system_prompt` | System prompt usado durante execução mas NÃO salvo nas trajetórias |
| `--log_prefix_chars` | Caracteres a mostrar em previews de log (padrão: 100) |
| `--prefill_messages_file` | Path para arquivo JSON com mensagens prefill para priming few-shot |

## Distribuições de toolsets {#toolset-distributions}

Cada prompt recebe um conjunto de toolsets amostrado aleatoriamente de uma **distribuição**. Isso garante que dados de treinamento cubram combinações diversas de ferramentas. Use `--list_distributions` para ver todas as distribuições disponíveis.

Na implementação atual, distribuições atribuem uma probabilidade a **cada toolset individual**. O sampler ativa cada toolset independentemente, depois garante que pelo menos um toolset esteja habilitado. Isso difere de uma tabela escrita à mão de combinações pré-montadas.

## Formato de saída {#output-format}

Toda saída vai para `data/<run_name>/`:

```text
data/my_run/
├── trajectories.jsonl    # Saída final combinada (todos os batches mergeados)
├── batch_0.jsonl         # Resultados de batch individuais
├── batch_1.jsonl
├── ...
├── checkpoint.json       # Checkpoint de retomada
└── statistics.json       # Estatísticas agregadas de uso de ferramentas
```

### Formato de trajetória {#trajectory-format}

Cada linha em `trajectories.jsonl` é um objeto JSON:

```json
{
  "prompt_index": 42,
  "conversations": [
    {"from": "human", "value": "Write a function..."},
    {"from": "gpt", "value": "I'll create that function...",
     "tool_calls": [...]},
    {"from": "tool", "value": "..."},
    {"from": "gpt", "value": "Here's the completed function..."}
  ],
  "metadata": {
    "batch_num": 2,
    "timestamp": "2026-01-15T10:30:00",
    "model": "anthropic/claude-sonnet-4.6"
  },
  "completed": true,
  "partial": false,
  "api_calls": 3,
  "toolsets_used": ["terminal", "file"],
  "tool_stats": {
    "terminal": {"count": 2, "success": 2, "failure": 0},
    "read_file": {"count": 1, "success": 1, "failure": 0}
  },
  "tool_error_counts": {
    "terminal": 0,
    "read_file": 0
  }
}
```

O campo `conversations` usa um formato tipo ShareGPT com campos `from` e `value`. Estatísticas de ferramentas são normalizadas para incluir todas as ferramentas possíveis com defaults zero, garantindo schema consistente entre entradas para compatibilidade com datasets HuggingFace.

## Checkpointing {#checkpointing}

O batch runner tem checkpointing robusto para tolerância a falhas:

- **Arquivo de checkpoint:** Salvo após cada batch completar, rastreando quais índices de prompt estão prontos
- **Retomada baseada em conteúdo:** Com `--resume`, o runner varre arquivos de batch existentes e combina prompts completados pelo texto real (não só índices), permitindo recuperação mesmo se a ordem do dataset mudar
- **Prompts falhos:** Apenas prompts completados com sucesso são marcados como done — prompts falhos serão retentados na retomada
- **Merge de batches:** Na conclusão, todos os arquivos de batch (incluindo de execuções anteriores) são mergeados em um único `trajectories.jsonl`

### Como a retomada funciona {#how-resume-works}

1. Varrer todos os arquivos `batch_*.jsonl` por prompts completados (por matching de conteúdo)
2. Filtrar o dataset para excluir prompts já completados
3. Re-batch dos prompts restantes
4. Processar apenas os prompts restantes
5. Merge de todos os arquivos de batch (antigos + novos) na saída final

## Filtragem de qualidade {#quality-filtering}

O batch runner aplica filtragem automática de qualidade:

- **Filtro no-reasoning:** Samples onde zero turns de assistant contêm reasoning (sem `<REASONING_SCRATCHPAD>` ou thinking tokens nativos) são descartados
- **Filtro de entrada corrompida:** Entradas com nomes de ferramenta alucinados (não na lista válida de ferramentas) são filtradas no merge final
- **Estatísticas de reasoning:** Rastreia porcentagem de turns com/sem reasoning em toda a execução

## Estatísticas {#statistics}

Após conclusão, o runner imprime estatísticas abrangentes:

- **Uso de ferramentas:** Contagens de chamadas, taxas de sucesso/falha por ferramenta
- **Cobertura de reasoning:** Porcentagem de turns de assistant com reasoning
- **Samples descartados:** Contagem de samples filtrados por falta de reasoning
- **Duração:** Tempo total de processamento

Estatísticas também são salvas em `statistics.json` para análise programática.

## Casos de uso {#use-cases}

### Geração de dados de treinamento {#training-data-generation}

Gere trajetórias diversas de uso de ferramentas para fine-tuning:

```bash
python batch_runner.py \
    --dataset_file=data/coding_prompts.jsonl \
    --batch_size=20 \
    --run_name=coding_v1 \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=8 \
    --distribution=default \
    --max_turns=15
```

### Avaliação de model {#model-evaluation}

Avalie quão bem um model usa ferramentas em prompts padronizados:

```bash
python batch_runner.py \
    --dataset_file=data/eval_suite.jsonl \
    --batch_size=10 \
    --run_name=eval_gpt4 \
    --model=openai/gpt-4o \
    --num_workers=4 \
    --max_turns=10
```

### Imagens de container por prompt {#per-prompt-container-images}

Para benchmarks que exigem ambientes específicos, cada prompt pode especificar sua própria imagem de container:

```jsonl
{"prompt": "Install numpy and compute eigenvalues of a 3x3 matrix", "image": "python:3.11-slim"}
{"prompt": "Compile this Rust program and run it", "image": "rust:1.75"}
{"prompt": "Set up a Node.js Express server", "image": "node:20-alpine", "cwd": "/app"}
```

O batch runner verifica se imagens Docker estão acessíveis antes de executar cada prompt.
