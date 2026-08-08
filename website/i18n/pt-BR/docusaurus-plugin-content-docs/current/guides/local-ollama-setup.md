---
sidebar_position: 9
title: "Execute o Hermes Localmente com Ollama — Custo Zero de API"
description: "Guia passo a passo para executar o Hermes Agent inteiramente na sua própria máquina com Ollama e modelos de peso aberto como o Gemma 4, sem chaves de API na nuvem ou assinaturas pagas"
---

# Execute o Hermes Localmente com Ollama — Custo Zero de API

## O Problema {#the-problem}

APIs de LLM na nuvem cobram por token. Uma sessão pesada de programação pode custar de US$ 5 a US$ 20. Para projetos pessoais, aprendizado ou trabalho sensível à privacidade, isso soma rápido — e você está enviando cada conversa para terceiros.

## O Que Este Guia Resolve {#what-this-guide-solves}

Você vai configurar o Hermes Agent rodando inteiramente no seu próprio hardware, usando o [Ollama](https://ollama.com) como backend de modelo. Sem chaves de API, sem assinaturas, sem dados saindo da sua máquina. Uma vez configurado, o Hermes funciona exatamente como funciona com OpenRouter ou Anthropic — comandos de terminal, edição de arquivos, navegação web, delegação — mas o modelo roda localmente.

Ao final, você terá:

- Ollama servindo um ou mais modelos de peso aberto
- Hermes conectado ao Ollama como um endpoint personalizado
- Um agente local funcional que pode editar arquivos, executar comandos e navegar na web
- Opcional: um bot de Telegram/Discord alimentado inteiramente pelo seu próprio hardware

## O Que Você Precisa {#what-you-need}

| Componente | Mínimo | Recomendado |
|-----------|---------|-------------|
| **RAM** | 8 GB (para modelos de 3B) | 32+ GB (para modelos de 27B+) |
| **Armazenamento** | 5 GB livres | 30+ GB (para vários modelos) |
| **CPU** | 4 núcleos | 8+ núcleos (AMD EPYC, Ryzen, Intel Xeon) |
| **GPU** | Não obrigatória | GPU NVIDIA com 8+ GB de VRAM acelera bastante |

:::tip Funciona só com CPU, mas espere respostas mais lentas
O Ollama roda em servidores apenas com CPU. Um modelo de 9B em uma CPU moderna de 8 núcleos entrega ~10 tokens/seg. Um modelo de 31B em CPU é mais lento (~2–5 tokens/seg) — cada resposta leva de 30 a 120 segundos, mas funciona. Uma GPU melhora isso drasticamente. Para configurações apenas com CPU, aumente o tempo limite da API via variável de ambiente (não é uma chave do `config.yaml`):

```bash
# ~/.hermes/.env
HERMES_API_TIMEOUT=1800   # 30 minutes — generous for slow local models
```
:::

## Passo 1: Instale o Ollama {#step-1-install-ollama}

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verifique se está rodando:

```bash
ollama --version
curl http://localhost:11434/api/tags   # Should return {"models":[]}
```

## Passo 2: Baixe um Modelo {#step-2-pull-a-model}

Escolha com base no seu hardware:

| Modelo | Tamanho no Disco | RAM Necessária | Chamada de Ferramentas | Melhor Para |
|-------|-------------|------------|:------------:|----------|
| `gemma4:31b` | ~20 GB | 24+ GB | Sim | Melhor qualidade — uso de ferramentas e raciocínio fortes |
| `gemma2:27b` | ~16 GB | 20+ GB | Não | Tarefas conversacionais, sem uso de ferramentas |
| `gemma2:9b` | ~5 GB | 8+ GB | Não | Chat rápido, perguntas e respostas — não pode chamar ferramentas |
| `llama3.2:3b` | ~2 GB | 4+ GB | Não | Apenas respostas rápidas e leves |

:::warning A chamada de ferramentas importa
O Hermes é um assistente **agêntico** — ele edita arquivos, executa comandos e navega na web através de chamadas de ferramentas. Modelos sem suporte a chamadas de ferramentas só conseguem conversar; eles não conseguem realizar ações. Para a experiência completa do Hermes, use um modelo que suporte ferramentas (como `gemma4:31b`).
:::

Baixe o modelo escolhido:

```bash
ollama pull gemma4:31b
```

:::info Vários modelos
Você pode baixar vários modelos e alternar entre eles dentro do Hermes com `/model`. O Ollama carrega o modelo ativo na memória sob demanda e descarrega os inativos automaticamente.
:::

Verifique se o modelo funciona:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4:31b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50
  }'
```

Você deve ver uma resposta JSON com a resposta do modelo.

## Passo 3: Configure o Hermes {#step-3-configure-hermes}

Execute o assistente de configuração do Hermes:

```bash
hermes setup
```

Quando solicitado a escolher um provedor, selecione **Custom Endpoint** e digite:

- **Base URL:** `http://localhost:11434/v1`
- **API Key:** Deixe vazio ou digite `no-key` (o Ollama não precisa de uma)
- **Model:** `gemma4:31b` (ou qualquer modelo que você tenha baixado)

Alternativamente, edite `~/.hermes/config.yaml` diretamente:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

## Passo 4: Comece a Usar o Hermes {#step-4-start-using-hermes}

```bash
hermes
```

É isso. Você agora está rodando um agente totalmente local. Experimente:

```
You: List all Python files in this directory and count the lines of code in each

You: Read the README.md and summarize what this project does

You: Create a Python script that fetches the weather for Ho Chi Minh City
```

O Hermes vai usar a ferramenta de terminal, operações de arquivo e seu modelo local — sem chamadas à nuvem.

## Passo 5: Escolha o Modelo Certo para Sua Tarefa {#step-5-pick-the-right-model-for-your-task}

Nem toda tarefa precisa do maior modelo. Aqui está um guia prático:

| Tarefa | Modelo Recomendado | Por Quê |
|------|-------------------|-----|
| Edição de arquivos, código, comandos de terminal | `gemma4:31b` | Único modelo com chamada de ferramentas confiável |
| Perguntas e respostas rápidas (sem uso de ferramentas) | `gemma2:9b` | Respostas rápidas para tarefas conversacionais |
| Chat leve | `llama3.2:3b` | O mais rápido, mas com capacidades muito limitadas |

:::note
Para trabalho agêntico completo (editar arquivos, executar comandos, navegar), o `gemma4:31b` é atualmente a melhor opção local com suporte a chamadas de ferramentas. Verifique a [biblioteca de modelos do Ollama](https://ollama.com/library) para modelos mais novos — o suporte a chamadas de ferramentas está se expandindo rapidamente.
:::

Troque de modelo em tempo real dentro de uma sessão:

```
/model gemma2:9b
```

## Passo 6: Otimize para Velocidade {#step-6-optimize-for-speed}

### Aumente a Janela de Contexto do Ollama {#increase-ollamas-context-window}

Por padrão, o Ollama usa um contexto de 2048 tokens. O Hermes exige pelo menos 64.000 tokens para trabalho agêntico com ferramentas:

```bash
# Create a Modelfile that extends context
cat > /tmp/Modelfile << 'EOF'
FROM gemma4:31b
PARAMETER num_ctx 64000
EOF

ollama create gemma4-64k -f /tmp/Modelfile
```

Depois atualize sua configuração do Hermes para usar `gemma4-64k` como o nome do modelo.

### Mantenha o Modelo Carregado {#keep-the-model-loaded}

Por padrão, o Ollama descarrega modelos após 5 minutos de inatividade. Para um bot de gateway persistente, mantenha-o carregado:

```bash
# Set keep-alive to 24 hours
curl http://localhost:11434/api/generate \
  -d '{"model": "gemma4:31b", "keep_alive": "24h"}'
```

Ou defina globalmente no ambiente do Ollama:

```bash
# /etc/systemd/system/ollama.service.d/override.conf
[Service]
Environment="OLLAMA_KEEP_ALIVE=24h"
```

### Use Descarregamento para GPU (Se Disponível) {#use-gpu-offloading-if-available}

Se você tem uma GPU NVIDIA, o Ollama descarrega automaticamente camadas para ela. Verifique com:

```bash
ollama ps   # Shows which model is loaded and how many GPU layers
```

Para um modelo de 31B em uma GPU de 12 GB, você terá um descarregamento parcial (~40 camadas na GPU, o resto na CPU), o que ainda dá uma aceleração significativa.

## Passo 7: Execute como um Bot de Gateway (Opcional) {#step-7-run-as-a-gateway-bot-optional}

Uma vez que o Hermes funcione localmente na CLI, você pode expô-lo como um bot de Telegram ou Discord — ainda rodando inteiramente no seu hardware.

### Telegram {#telegram}

1. Crie um bot via [@BotFather](https://t.me/BotFather) e obtenha o token
2. Adicione ao seu `~/.hermes/config.yaml`:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

platforms:
  telegram:
    enabled: true
    token: "YOUR_TELEGRAM_BOT_TOKEN"
```

3. Inicie o gateway:

```bash
hermes gateway
```

Agora envie uma mensagem para o seu bot no Telegram — ele responde usando seu modelo local.

### Discord {#discord}

1. Crie uma aplicação do Discord em [discord.com/developers](https://discord.com/developers/applications)
2. Adicione à configuração:

```yaml
platforms:
  discord:
    enabled: true
    token: "YOUR_DISCORD_BOT_TOKEN"
```

3. Inicie: `hermes gateway`

## Passo 8: Configure Fallbacks (Opcional) {#step-8-set-up-fallbacks-optional}

Modelos locais podem ter dificuldade com tarefas complexas. Configure um fallback na nuvem que só é ativado quando o modelo local falha:

```yaml
model:
  default: "gemma4:31b"
  provider: "custom"
  base_url: "http://localhost:11434/v1"

fallback_providers:
  - provider: openrouter
    model: anthropic/claude-sonnet-4
```

Dessa forma, 90% do seu uso é gratuito (local), e apenas as tarefas difíceis acessam a API paga.

## Solução de Problemas {#troubleshooting}

### "Connection refused" ao iniciar {#connection-refused-on-startup}

O Ollama não está rodando. Inicie-o:

```bash
sudo systemctl start ollama
# or
ollama serve
```

### Respostas lentas {#slow-responses}

- **Verifique o tamanho do modelo vs. RAM:** Se seu modelo precisa de mais RAM do que está disponível, ele faz swap para o disco. Use um modelo menor ou adicione RAM.
- **Verifique `ollama ps`:** Se nenhuma camada de GPU estiver descarregada, as respostas são limitadas pela CPU. Isso é normal para servidores apenas com CPU.
- **Reduza o contexto:** Conversas grandes deixam a inferência mais lenta. Use `/compress` regularmente, ou defina um limite de compressão menor na configuração.

### O modelo não segue as chamadas de ferramentas {#model-doesnt-follow-tool-calls}

Modelos menores (3B, 7B) às vezes ignoram instruções de chamada de ferramentas e produzem texto simples em vez de chamadas de função estruturadas. Soluções:

- **Use um modelo maior** — `gemma4:31b` ou `gemma2:27b` lidam com chamadas de ferramentas muito melhor do que modelos de 3B/7B.
- **O Hermes tem autorreparo** — ele detecta chamadas de ferramentas malformadas e tenta corrigi-las automaticamente.
- **Configure um fallback** — se o modelo local falhar 3 vezes, o Hermes recorre a um provedor na nuvem.

### Erros de janela de contexto {#context-window-errors}

O contexto padrão do Ollama (2048 tokens) é muito pequeno para trabalho agêntico. Veja o [Passo 6](#step-6-optimize-for-speed) para aumentá-lo.

## Comparação de Custos {#cost-comparison}

Aqui está o quanto rodar localmente economiza em comparação com APIs na nuvem, com base em uma sessão típica de programação (~100 mil tokens de entrada, ~20 mil tokens de saída):

| Provedor | Custo por Sessão | Mensal (uso diário) |
|----------|-----------------|---------------------|
| Anthropic Claude Sonnet | ~US$ 0,80 | ~US$ 24 |
| OpenRouter (GPT-4o) | ~US$ 0,60 | ~US$ 18 |
| **Ollama (local)** | **US$ 0,00** | **US$ 0,00** |

Seu único custo é energia elétrica — cerca de US$ 0,01 a US$ 0,05 por sessão, dependendo do hardware.

## O Que Funciona Bem Localmente {#what-works-well-locally}

- **Edição de arquivos e geração de código** — modelos de 9B+ lidam bem com isso
- **Comandos de terminal** — o Hermes encapsula o comando, executa e lê a saída independentemente do modelo
- **Navegação web** — a ferramenta de navegador faz a busca; o modelo apenas interpreta os resultados
- **Tarefas agendadas** — funcionam de forma idêntica às configurações na nuvem
- **Gateway multiplataforma** — Telegram, Discord, Slack, todos funcionam com modelos locais

## O Que É Melhor com Modelos na Nuvem {#whats-better-with-cloud-models}

- **Raciocínio muito complexo de múltiplas etapas** — modelos de 70B+ ou modelos na nuvem como o Claude Opus são notavelmente melhores
- **Janelas de contexto longas** — modelos na nuvem oferecem de 100 mil a 1 milhão de tokens; runtimes locais geralmente ficam por padrão abaixo do mínimo de 64K do Hermes, a menos que você os configure
- **Velocidade em respostas grandes** — a inferência na nuvem é mais rápida do que local apenas com CPU para gerações longas

O ponto ideal: use local para tarefas do dia a dia, configure um fallback na nuvem para as tarefas difíceis.
