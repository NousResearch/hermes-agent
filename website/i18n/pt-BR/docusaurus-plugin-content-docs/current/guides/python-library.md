---
sidebar_position: 5
title: "Usando o Hermes como uma Biblioteca Python"
description: "Incorpore o AIAgent nos seus próprios scripts Python, aplicações web ou pipelines de automação — sem precisar da CLI"
---

# Usando o Hermes como uma Biblioteca Python

O Hermes não é apenas uma ferramenta de linha de comando. Você pode importar `AIAgent` diretamente e usá-lo programaticamente nos seus próprios scripts Python, aplicações web ou pipelines de automação. Este guia mostra como.

---

## Instalação {#installation}

Instale o Hermes diretamente do repositório:

```bash
pip install git+https://github.com/NousResearch/hermes-agent.git
```

Ou com o [uv](https://docs.astral.sh/uv/):

```bash
uv pip install git+https://github.com/NousResearch/hermes-agent.git
```

Você também pode fixá-lo no seu `requirements.txt`:

```text
hermes-agent @ git+https://github.com/NousResearch/hermes-agent.git
```

:::tip
As mesmas variáveis de ambiente usadas pela CLI são necessárias ao usar o Hermes como uma biblioteca. No mínimo, defina `OPENROUTER_API_KEY` (ou `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` se estiver usando acesso direto a um provedor).
:::

---

## Uso Básico {#basic-usage}

A maneira mais simples de usar o Hermes é o método `chat()` — passe uma mensagem, receba uma string de volta:

```python
from run_agent import AIAgent

agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)
response = agent.chat("What is the capital of France?")
print(response)
```

`chat()` gerencia todo o loop de conversa internamente — chamadas de ferramentas, retentativas, tudo — e retorna apenas a resposta final em texto.

:::warning
Sempre defina `quiet_mode=True` ao incorporar o Hermes no seu próprio código. Sem isso, o agente imprime spinners de CLI, indicadores de progresso e outras saídas de terminal que vão poluir a saída da sua aplicação.
:::

---

## Controle Completo da Conversa {#full-conversation-control}

Para mais controle sobre a conversa, use `run_conversation()` diretamente. Ele retorna um dicionário com a resposta completa, o histórico de mensagens e metadados:

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)

result = agent.run_conversation(
    user_message="Search for recent Python 3.13 features",
    task_id="my-task-1",
)

print(result["final_response"])
print(f"Messages exchanged: {len(result['messages'])}")
```

O dicionário retornado contém:
- **`final_response`** — A resposta final em texto do agente
- **`messages`** — O histórico completo de mensagens (sistema, usuário, assistente, chamadas de ferramentas)

(O `task_id` que você passa é armazenado na instância do agente para isolamento de VM, mas não é retornado no dicionário de resultado.)

Você também pode passar uma mensagem de sistema personalizada que substitui o prompt de sistema efêmero para aquela chamada:

```python
result = agent.run_conversation(
    user_message="Explain quicksort",
    system_message="You are a computer science tutor. Use simple analogies.",
)
```

---

## Configurando Ferramentas {#configuring-tools}

Controle a quais toolsets o agente tem acesso usando `enabled_toolsets` ou `disabled_toolsets`:

```python
# Only enable web tools (browsing, search)
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    enabled_toolsets=["web"],
    quiet_mode=True,
)

# Enable everything except terminal access
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    disabled_toolsets=["terminal"],
    quiet_mode=True,
)
```

:::tip
Use `enabled_toolsets` quando você quiser um agente mínimo e bloqueado (por exemplo, apenas busca na web para um bot de pesquisa). Use `disabled_toolsets` quando quiser a maioria das capacidades, mas precisar restringir algumas específicas (por exemplo, sem acesso a terminal em um ambiente compartilhado).
:::

---

## Conversas com Múltiplos Turnos {#multi-turn-conversations}

Mantenha o estado da conversa entre vários turnos passando o histórico de mensagens de volta:

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    quiet_mode=True,
)

# First turn
result1 = agent.run_conversation("My name is Alice")
history = result1["messages"]

# Second turn — agent remembers the context
result2 = agent.run_conversation(
    "What's my name?",
    conversation_history=history,
)
print(result2["final_response"])  # "Your name is Alice."
```

O parâmetro `conversation_history` aceita a lista `messages` de um resultado anterior. O agente a copia internamente, então sua lista original nunca é alterada.

---

## Salvando Trajetórias {#saving-trajectories}

Ative o salvamento de trajetórias para capturar conversas no formato ShareGPT — útil para gerar dados de treinamento ou depurar:

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4.6",
    save_trajectories=True,
    quiet_mode=True,
)

agent.chat("Write a Python function to sort a list")
# Saves to trajectory_samples.jsonl in ShareGPT format
```

Cada conversa é adicionada como uma única linha JSONL, facilitando a coleta de datasets a partir de execuções automatizadas.

---

## Prompts de Sistema Personalizados {#custom-system-prompts}

Use `ephemeral_system_prompt` para definir um prompt de sistema personalizado que guia o comportamento do agente, mas **não** é salvo nos arquivos de trajetória (mantendo seus dados de treinamento limpos):

```python
agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    ephemeral_system_prompt="You are a SQL expert. Only answer database questions.",
    quiet_mode=True,
)

response = agent.chat("How do I write a JOIN query?")
print(response)
```

Isso é ideal para construir agentes especializados — um revisor de código, um redator de documentação, um assistente de SQL — todos usando as mesmas ferramentas subjacentes.

---

## Processamento em Lote {#batch-processing}

Para executar muitos prompts em paralelo, o Hermes inclui o `batch_runner.py`. Ele gerencia múltiplas instâncias de `AIAgent` concorrentes com isolamento adequado de recursos:

```bash
python batch_runner.py --input prompts.jsonl --output results.jsonl
```

Cada prompt recebe seu próprio `task_id` e ambiente isolado. Se você precisar de lógica de lote personalizada, pode construir a sua própria usando `AIAgent` diretamente:

```python
import concurrent.futures
from run_agent import AIAgent

prompts = [
    "Explain recursion",
    "What is a hash table?",
    "How does garbage collection work?",
]

def process_prompt(prompt):
    # Create a fresh agent per task for thread safety
    agent = AIAgent(
        model="anthropic/claude-sonnet-4",
        quiet_mode=True,
        skip_memory=True,
    )
    return agent.chat(prompt)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_prompt, prompts))

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

:::warning
Sempre crie uma **nova instância de `AIAgent` por thread ou tarefa**. O agente mantém estado interno (histórico de conversa, sessões de ferramentas, contadores de iteração) que não é seguro para compartilhar entre threads.
:::

---

## Exemplos de Integração {#integration-examples}

### Endpoint FastAPI {#fastapi-endpoint}

```python
from fastapi import FastAPI
from pydantic import BaseModel
from run_agent import AIAgent

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    model: str = "anthropic/claude-sonnet-4"

@app.post("/chat")
async def chat(request: ChatRequest):
    agent = AIAgent(
        model=request.model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    response = agent.chat(request.message)
    return {"response": response}
```

### Bot do Discord {#discord-bot}

```python
import discord
from run_agent import AIAgent

client = discord.Client(intents=discord.Intents.default())

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith("!hermes "):
        query = message.content[8:]
        agent = AIAgent(
            model="anthropic/claude-sonnet-4",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="discord",
        )
        response = agent.chat(query)
        await message.channel.send(response[:2000])

client.run("YOUR_DISCORD_TOKEN")
```

### Etapa de Pipeline de CI/CD {#cicd-pipeline-step}

```python
#!/usr/bin/env python3
"""CI step: auto-review a PR diff."""
import subprocess
from run_agent import AIAgent

diff = subprocess.check_output(["git", "diff", "main...HEAD"]).decode()

agent = AIAgent(
    model="anthropic/claude-sonnet-4",
    quiet_mode=True,
    skip_context_files=True,
    skip_memory=True,
    disabled_toolsets=["terminal", "browser"],
)

review = agent.chat(
    f"Review this PR diff for bugs, security issues, and style problems:\n\n{diff}"
)
print(review)
```

---

## Principais Parâmetros do Construtor {#key-constructor-parameters}

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|---------|-------------|
| `model` | `str` | `""` | Modelo no formato OpenRouter (padrão vazio; resolvido a partir da sua configuração do hermes em tempo de execução) |
| `quiet_mode` | `bool` | `False` | Suprime a saída da CLI |
| `enabled_toolsets` | `List[str]` | `None` | Lista de permissão de toolsets específicos |
| `disabled_toolsets` | `List[str]` | `None` | Lista de bloqueio de toolsets específicos |
| `save_trajectories` | `bool` | `False` | Salva conversas em JSONL |
| `ephemeral_system_prompt` | `str` | `None` | Prompt de sistema personalizado (não salvo nas trajetórias) |
| `max_iterations` | `int` | `90` | Máximo de iterações de chamada de ferramentas por conversa |
| `skip_context_files` | `bool` | `False` | Pula o carregamento dos arquivos AGENTS.md |
| `skip_memory` | `bool` | `False` | Desativa a leitura/escrita de memória persistente |
| `api_key` | `str` | `None` | Chave de API (recorre às variáveis de ambiente) |
| `base_url` | `str` | `None` | URL de endpoint de API personalizada |
| `platform` | `str` | `None` | Dica de plataforma (`"discord"`, `"telegram"`, etc.) |

---

## Notas Importantes {#important-notes}

:::tip
- Defina **`skip_context_files=True`** se você não quiser que arquivos `AGENTS.md` do diretório de trabalho sejam carregados no prompt de sistema.
- Defina **`skip_memory=True`** para impedir que o agente leia ou escreva memória persistente — recomendado para endpoints de API sem estado.
- O parâmetro `platform` (por exemplo, `"discord"`, `"telegram"`) injeta dicas de formatação específicas da plataforma, para que o agente adapte seu estilo de saída.
:::

:::warning
- **Segurança entre threads**: crie um `AIAgent` por thread ou tarefa. Nunca compartilhe uma instância entre chamadas concorrentes.
- **Limpeza de recursos**: o agente limpa automaticamente os recursos (sessões de terminal, instâncias de navegador) quando uma conversa termina. Se você estiver rodando em um processo de longa duração, garanta que cada conversa seja concluída normalmente.
- **Limites de iteração**: o valor padrão `max_iterations=90` é generoso. Para casos de uso simples de perguntas e respostas, considere reduzi-lo (por exemplo, `max_iterations=10`) para evitar loops descontrolados de chamadas de ferramentas e controlar custos.
:::
