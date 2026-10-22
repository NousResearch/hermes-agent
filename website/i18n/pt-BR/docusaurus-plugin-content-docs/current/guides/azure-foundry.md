---
sidebar_position: 15
title: "Microsoft Foundry"
description: "Use o Hermes Agent com o Microsoft Foundry — endpoints no estilo OpenAI e Anthropic, com detecção automática de transporte e modelos implantados"
---

# Microsoft Foundry {#microsoft-foundry}

O provedor `azure-foundry` do Hermes Agent oferece suporte ao Microsoft Foundry (anteriormente Azure AI Foundry) e ao Azure OpenAI. Um único recurso Foundry pode hospedar modelos com dois formatos de transporte diferentes:

- **Estilo OpenAI** — `POST /v1/chat/completions` em endpoints como `https://<resource>.openai.azure.com/openai/v1`. Usado para GPT-4.x, GPT-5.x, Llama, Mistral e a maioria dos modelos de peso aberto.
- **Estilo Anthropic** — `POST /v1/messages` em endpoints como `https://<resource>.services.ai.azure.com/anthropic`. Usado quando o Microsoft Foundry serve modelos Claude via o formato da API Anthropic Messages.

O assistente de configuração sonda seu endpoint e detecta automaticamente qual transporte ele usa, quais implantações estão disponíveis e o tamanho de contexto de cada modelo.

## Pré-requisitos {#prerequisites}

- Um recurso Microsoft Foundry ou Azure OpenAI com pelo menos uma implantação
- A URL do endpoint da implantação
- **Ou** uma chave de API (no Portal do Azure, em "Keys and Endpoint") **ou** a função RBAC **Azure AI User** no recurso Foundry, se você planeja usar o Microsoft Entra ID (o caminho sem chave que a Microsoft recomenda). Alguns locatários podem exibir a função como **Foundry User** durante a fase de transição de renomeação da Microsoft.

## Início Rápido {#quick-start}

```bash
hermes model
# → Selecione "Azure Foundry"
# → Digite a URL do seu endpoint
# → Escolha a autenticação:
#     1. Chave de API
#     2. Microsoft Entra ID (identidade gerenciada / identidade de workload / az login)
# → (Entra) O Hermes sonda o DefaultAzureCredential; em caso de sucesso, nunca pede uma chave
# → (Chave de API) Digite sua chave de API
# O Hermes sonda o endpoint e detecta automaticamente o transporte + modelos
# → Escolha um modelo da lista (ou digite manualmente um nome de implantação)
```

O assistente vai:

1. **Analisar o caminho da URL** — URLs terminando em `/anthropic` são reconhecidas como rotas Claude do Microsoft Foundry.
2. **Sondar `GET <base>/models`** — se o endpoint retornar uma lista de modelos no formato OpenAI, o Hermes muda para `chat_completions` e preenche um seletor com os IDs de implantação retornados.
3. **Sondar o formato Anthropic Messages** — alternativa para endpoints que não expõem `/models`, mas aceitam o formato Anthropic Messages.
4. **Recorrer à entrada manual** — endpoints privados/restritos que rejeitam toda sondagem ainda funcionam; você escolhe o modo de API e digita um nome de implantação manualmente.

O tamanho de contexto do modelo escolhido é resolvido pela cadeia padrão de metadados do Hermes (`models.dev`, metadados do provedor e substitutos codificados por família) e armazenado no `config.yaml`, para que o modelo possa dimensionar corretamente sua própria janela de contexto.

## Microsoft Entra ID (sem chave, RBAC) — recomendado {#microsoft-entra-id-keyless-rbac--recommended}

A Microsoft recomenda a [autenticação sem chave com o Microsoft Entra ID](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id) para cargas de trabalho de produção no Foundry. O Hermes oferece suporte ao Entra ID para **ambas** as superfícies de API:

- **Estilo OpenAI** (`api_mode: chat_completions` / `codex_responses`) — GPT-4/5, Llama, Mistral, DeepSeek, etc.
- **Estilo Anthropic** (`api_mode: anthropic_messages`) — modelos Claude no Microsoft Foundry.

O RBAC do Foundry é por recurso (`Azure AI User` concede ambas as superfícies; alguns locatários podem exibir `Foundry User`) e a Microsoft documenta o mesmo escopo de inferência (`https://ai.azure.com/.default`) para ambas. Por baixo dos panos:

- O estilo OpenAI usa o contrato nativo `api_key=` chamável do SDK Python da OpenAI — o SDK gera automaticamente um novo JWT por requisição.
- O estilo Anthropic usa um `httpx.Client` com um hook de evento de requisição instalado por `agent.azure_identity_adapter.build_bearer_http_client`, porque o SDK da Anthropic não aceita `auth_token` chamável nativamente. O hook reescreve `Authorization: Bearer <jwt-novo>` em cada requisição de saída. Mesmo RBAC da Microsoft, mesmo escopo do Foundry — o contrato do SDK é a única diferença.

### Por que usar o Entra ID? {#why-use-entra-id}

- Sem chaves de API de longa duração para rotacionar ou revogar.
- Acesso guiado por RBAC — conceda ou remova `Azure AI User` no recurso Foundry, sem reescrever a configuração.
- Logs de acesso e auditoria segmentados por responsável, em vez de todos os chamadores compartilhando uma única chave estática.
- Superfície de autenticação única para VMs do Azure, pods do AKS, App Service, Functions, Container Apps e Foundry Agent Service via identidade gerenciada.
- Fluxos de identidade de workload e entidade de serviço para pipelines de CI/CD.

### Configuração única (lado do Azure) {#one-time-setup-azure-side}

1. No Portal do Azure, abra seu recurso Foundry → **Access control (IAM)** → **Add → Add role assignment**.
2. Escolha a função **Azure AI User** (ou **Foundry User**, se seu locatário tiver a função renomeada).
3. Atribua a:
   - **Sua conta de usuário** para desenvolvimento local com `az login`.
   - **Uma identidade gerenciada ou identidade de workload** para computação hospedada no Azure (recomendado para produção).
   - **A identidade de agente de um agente hospedado no Foundry Agent Service**, quando o Hermes é executado dentro de um agente hospedado.
   - **Uma entidade de serviço** para pipelines de CI/CD quando a identidade de workload não estiver disponível.
4. Aguarde ~5 minutos para a função se propagar.

Equivalente na CLI do Azure:

```bash
az role assignment create \
  --assignee <principal-or-agent-identity-client-id> \
  --role "Azure AI User" \
  --scope <foundry-resource-id>
```

### Configuração única (lado do Hermes) {#one-time-setup-hermes-side}

```bash
hermes model
# → Selecione "Azure Foundry"
# → Digite a URL do seu endpoint
# → Autenticação: 2 (Microsoft Entra ID)
# → (opcional) ID do cliente da identidade gerenciada atribuída pelo usuário
# → (opcional) ID do locatário do Azure
# → O Hermes sonda o DefaultAzureCredential() e informa qual credencial
#    interna teve sucesso (ex.: AzureCliCredential, ManagedIdentityCredential)
```

O assistente executa uma sondagem preliminar limitada (timeout de 10 s). Em caso de falha, oferece a opção de "salvar mesmo assim, validar depois" — útil ao configurar em uma máquina que ainda não tem credenciais, mas terá em tempo de execução (por exemplo, preparando a configuração para uma implantação com identidade gerenciada).

O `azure-identity` é instalado automaticamente no primeiro uso, via caminho de instalação sob demanda do Hermes. Para pré-instalar:

```bash
pip install azure-identity
```

### Configuração gravada no `config.yaml` {#configuration-written-to-configyaml}

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  auth_mode: entra_id
  default: gpt-4o
  context_length: 128000
  entra:
    scope: https://ai.azure.com/.default        # apenas ao sobrescrever o padrão
```

O Hermes gerencia apenas um ajuste específico do Entra no `config.yaml`:

- **`scope`** — o escopo de recurso OAuth. O padrão é o escopo de inferência documentado pela Microsoft (`https://ai.azure.com/.default`). Substitua apenas se seu recurso foi provisionado com uma audiência não padrão.

Tudo o mais (locatário, segredo da entidade de serviço, arquivo de token federado, autoridade de nuvem soberana, preferências de broker) é lido pelo `azure-identity` diretamente das variáveis de ambiente `AZURE_*` padrão — veja a [ordem de resolução de credenciais](#credential-resolution-order) abaixo. Defina-as em `~/.hermes/.env` ou no ambiente de implantação, exatamente como descrito na referência do SDK da Microsoft.

Nenhum segredo é gravado em `~/.hermes/.env` no modo Entra — o `azure-identity` armazena tokens em cache no processo (e, quando disponível, no keychain do seu sistema operacional / `~/.IdentityService`).

### Ordem de resolução de credenciais {#credential-resolution-order}

A `DefaultAzureCredential` do `azure-identity` percorre esta cadeia em cada requisição de token, parando na primeira credencial que retorna um token:

1. **Credencial de ambiente** — `AZURE_TENANT_ID` + `AZURE_CLIENT_ID` + `AZURE_CLIENT_SECRET` (ou `AZURE_CLIENT_CERTIFICATE_PATH` / `AZURE_FEDERATED_TOKEN_FILE`).
2. **Workload Identity** — `AZURE_FEDERATED_TOKEN_FILE` (tokens federados do AKS / OIDC).
3. **Managed Identity** — endpoint IMDS (`169.254.169.254`) para máquinas virtuais; `IDENTITY_ENDPOINT` para App Service / Functions / Container Apps. Agentes hospedados no Foundry Agent Service usam a identidade de agente do próprio agente hospedado.
4. **Visual Studio Code** — extensão de conta do Azure.
5. **Azure CLI** — sessão de `az login`.
6. **Azure Developer CLI** — `azd auth login`.
7. **Azure PowerShell** — `Connect-AzAccount`.
8. **Broker** (somente Windows / WSL) — Web Account Manager.

A credencial de navegador interativo é excluída por padrão para execuções não interativas do Hermes; use CLI do Azure, Azure Developer CLI, identidade gerenciada, identidade de workload ou credenciais de entidade de serviço em vez disso.

### Padrões de implantação {#deployment-patterns}

**Desenvolvimento local:**
```bash
az login
hermes model   # escolha Azure Foundry → Entra ID
hermes         # usa seu token de login do az
```

**VM do Azure / Functions / App Service / Container Apps (identidade gerenciada atribuída pelo sistema):**
1. Ative a identidade atribuída pelo sistema no recurso de computação.
2. Conceda à identidade `Azure AI User` (ou `Foundry User`) no recurso Foundry.
3. Defina `model.auth_mode: entra_id` no config.yaml — sem variáveis de ambiente necessárias.

**VM do Azure / Functions / App Service / Container Apps (identidade gerenciada atribuída pelo usuário):**
- Defina `AZURE_CLIENT_ID` como o ID de cliente da identidade atribuída pelo usuário, para que o `DefaultAzureCredential` escolha a correta.

**Agente hospedado no Foundry Agent Service:**
- Crie o agente hospedado e conceda à identidade desse agente `Azure AI User` (ou `Foundry User`) no recurso Foundry. O Hermes usa o `ManagedIdentityCredential` de dentro do agente hospedado; a atribuição de função pertence à identidade do agente, não apenas ao projeto pai ou ao seu usuário.

**AKS Workload Identity (substitui o AAD Pod Identity):**
- Anote a conta de serviço do pod com o ID de cliente da identidade de workload.
- O arquivo de token federado do pod é detectado automaticamente via `AZURE_FEDERATED_TOKEN_FILE`.
- `model.auth_mode: entra_id` funciona sem alterações adicionais na configuração.

**Entidade de serviço em CI:**
- Defina `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` no ambiente do runner.

#### Nuvens soberanas (Governo, China) {#sovereign-clouds-government-china}

Exporte `AZURE_AUTHORITY_HOST` (por exemplo, `https://login.microsoftonline.us` para Azure Government, `https://login.partner.microsoftonline.cn` para Azure China). O `azure-identity` lê isso diretamente.

### Verificações de saúde {#health-checks}

`hermes doctor` executa uma sondagem de 10 s contra o `DefaultAzureCredential` quando `model.auth_mode: entra_id`, informando qual credencial interna venceu (variáveis de ambiente presentes, endpoint de identidade gerenciada acessível, etc.).

`hermes auth` mostra um bloco de status estruturado:

```
azure-foundry (Microsoft Entra ID):
  Endpoint: https://my-resource.openai.azure.com/openai/v1
  Scope: https://ai.azure.com/.default
  Status: configured; live token probe is skipped here
```

### Limitações {#limitations}

- **Endpoints no estilo Anthropic usam um hook de evento httpx.** O SDK Python da Anthropic não aceita nativamente um `auth_token` chamável (≤ 0.86.0). O Hermes instala um hook de evento de requisição em um `httpx.Client` personalizado que gera um novo JWT por requisição de saída e reescreve `Authorization: Bearer <jwt>`. Isso é funcionalmente equivalente ao contrato nativo `Callable[[], str]` do SDK da OpenAI, mas adiciona uma camada extra de indireção. Se o SDK da Anthropic adicionar suporte nativo a autenticação chamável em uma versão futura, o Hermes migrará para ela de forma transparente.
- **Jobs em lote e `multiprocessing.Pool`.** O provedor de token do Entra é um closure que não pode ser serializado (pickled) entre processos. O `batch_runner.py` remove automaticamente o chamável da configuração do worker e deixa cada processo worker reconstruir seu próprio provedor a partir do `config.yaml` — nenhuma ação do usuário é necessária, mas cada worker paga o custo de uma passagem pela cadeia de credenciais na inicialização.
- **Sem persistência do JWT bearer em `auth.json`.** O Hermes não duplica o cache de tokens interno do `azure-identity`; inicializações a frio percorrem a cadeia de credenciais na primeira inferência.

## Configuração (gravada no `config.yaml`) {#configuration-written-to-configyaml-1}

Depois de executar o assistente, você verá algo assim:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions         # ou "anthropic_messages"
  default: gpt-5.4-mini              # seu nome de implantação / modelo
  context_length: 400000             # detectado automaticamente
```

E em `~/.hermes/.env`:

```
AZURE_FOUNDRY_API_KEY=<your-azure-key>
```

## Endpoints no estilo OpenAI (GPT, Llama, etc.) {#openai-style-endpoints-gpt-llama-etc}

O endpoint GA v1 do Azure OpenAI aceita o cliente Python padrão da `openai` com alterações mínimas:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.openai.azure.com/openai/v1
  api_mode: chat_completions
  default: gpt-5.4
```

Comportamento importante:

- **GPT-5.x, codex e a série o roteiam automaticamente para a Responses API.** O Microsoft Foundry implanta modelos GPT-5 / codex / o1 / o3 / o4 apenas via Responses API — chamar `/chat/completions` contra eles retorna `400 "The requested operation is unsupported."`. O Hermes detecta essas famílias de modelo pelo nome e atualiza o `api_mode` para `codex_responses` de forma transparente, mesmo quando o `config.yaml` ainda tem `api_mode: chat_completions`. GPT-4, GPT-4o, Llama, Mistral e outras implantações permanecem em `/chat/completions`.
- **`max_completion_tokens` é usado automaticamente.** O Azure OpenAI (assim como a OpenAI diretamente) exige `max_completion_tokens` para modelos gpt-4o, da série o e gpt-5.x. O Hermes envia o parâmetro correto com base no endpoint.
- **Endpoints pré-v1 que exigem `api-version`.** Se você tiver uma URL base legada como `https://<resource>.openai.azure.com/openai?api-version=2025-04-01-preview`, o Hermes extrai a query string e a envia via `default_query` em cada requisição (o SDK da OpenAI, caso contrário, a descarta ao juntar os caminhos).

## Endpoints no estilo Anthropic (Claude via Microsoft Foundry) {#anthropic-style-endpoints-claude-via-microsoft-foundry}

Para implantações Claude, use a rota no estilo Anthropic:

```yaml
model:
  provider: azure-foundry
  base_url: https://my-resource.services.ai.azure.com/anthropic
  api_mode: anthropic_messages
  default: claude-sonnet-4-6
```

Comportamento importante:

- **`/v1` é removido da URL base.** O SDK da Anthropic anexa `/v1/messages` a cada URL de requisição — o Hermes remove qualquer `/v1` final antes de passar a URL para o SDK, evitando caminhos com `/v1` duplicado.
- **`api-version` é enviado via `default_query`, e não anexado à URL.** O Azure Anthropic exige uma query string `api-version`. Incorporá-la na URL base produz caminhos malformados como `/anthropic?api-version=.../v1/messages` e retorna 404. O Hermes passa `api-version=2025-04-15` via `default_query` do SDK da Anthropic.
- **Autenticação Bearer é usada em vez de `x-api-key`.** A rota compatível com Anthropic do Azure exige `Authorization: Bearer <key>` em vez do header nativo `x-api-key` da Anthropic. O Hermes detecta `azure.com` na URL base e roteia a chave de API pelo campo `auth_token` do SDK, para que o header correto chegue ao upstream.
- **O header beta da janela de contexto de 1M é mantido.** O Azure ainda restringe o contexto Claude de 1M tokens (Opus 4.6/4.7, Sonnet 4.6) por trás do header `anthropic-beta: context-1m-2025-08-07`. O Hermes mantém esse header beta nos caminhos do Azure (ele é removido das requisições OAuth nativas da Anthropic porque algumas assinaturas o rejeitam, mas o Azure o exige).
- **A atualização de token OAuth é desativada.** As implantações do Azure usam chaves de API estáticas. O loop de atualização de token OAuth do `~/.claude/.credentials.json`, aplicável ao Anthropic Console, é explicitamente ignorado para endpoints do Azure, para evitar que o token OAuth do Claude Code sobrescreva sua chave do Azure no meio da sessão.

## Alternativa: `provider: anthropic` + URL base do Azure {#alternative-provider-anthropic--azure-base-url}

Se você já tem `provider: anthropic` configurado e só quer apontá-lo para o Microsoft Foundry para usar Claude, pode pular totalmente o provedor `azure-foundry`:

```yaml
model:
  provider: anthropic
  base_url: https://my-resource.services.ai.azure.com/anthropic
  key_env: AZURE_ANTHROPIC_KEY
  default: claude-sonnet-4-6
```

Com `AZURE_ANTHROPIC_KEY` definido em `~/.hermes/.env`. O Hermes detecta `azure.com` na URL base e contorna a cadeia de token OAuth do Claude Code, para que a chave do Azure seja usada diretamente com autenticação `x-api-key`.

`key_env` é o nome canônico do campo em snake_case; `api_key_env` (e as variantes em camelCase `keyEnv` / `apiKeyEnv`) são aceitos como aliases. Se `key_env` e `AZURE_ANTHROPIC_KEY`/`ANTHROPIC_API_KEY` estiverem definidos, a variável de ambiente nomeada por `key_env` prevalece.

## Descoberta de modelos {#model-discovery}

O Azure **não** expõe um endpoint puramente por chave de API para listar suas implantações de modelo *implantadas*. A enumeração de implantações exige autenticação do Azure Resource Manager (`az cognitiveservices account deployment list`) com um principal do Azure AD, não a chave de API de inferência.

O que o Hermes pode fazer:

- Endpoints v1 do Azure OpenAI (`<resource>.openai.azure.com/openai/v1`) expõem `GET /models` com o catálogo de modelos **disponíveis** do recurso. O Hermes usa essa lista para preencher o seletor de modelo.
- Rotas `/anthropic` do Microsoft Foundry: detectadas pelo caminho da URL, com o nome do modelo digitado manualmente.
- Endpoints privados / com firewall: entrada manual com uma mensagem amigável de "não foi possível sondar".

Você sempre pode digitar um nome de implantação diretamente — o Hermes não valida contra a lista retornada.

## Variáveis de ambiente {#environment-variables}

| Variável | Finalidade |
|----------|---------|
| `AZURE_FOUNDRY_API_KEY` | Chave de API principal para Microsoft Foundry / Azure OpenAI (modo api_key) |
| `AZURE_FOUNDRY_BASE_URL` | URL do endpoint (definida via `hermes model`; a variável de ambiente é usada como alternativa) |
| `AZURE_ANTHROPIC_KEY` | Usada por `provider: anthropic` + URL base do Azure (alternativa a `ANTHROPIC_API_KEY`) |
| `AZURE_TENANT_ID` | Locatário do Entra ID para fluxos de entidade de serviço |
| `AZURE_CLIENT_ID` | ID de cliente do Entra ID (entidade de serviço, identidade de workload ou identidade gerenciada atribuída pelo usuário) |
| `AZURE_CLIENT_SECRET` | Segredo da entidade de serviço |
| `AZURE_CLIENT_CERTIFICATE_PATH` | Certificado da entidade de serviço (alternativa ao segredo) |
| `AZURE_FEDERATED_TOKEN_FILE` | Caminho do token federado do Workload Identity (AKS) |
| `AZURE_AUTHORITY_HOST` | Substituição da autoridade de nuvem soberana |
| `IDENTITY_ENDPOINT` / `MSI_ENDPOINT` | Endpoint do Managed Identity para App Service, Functions e Container Apps; VMs geralmente usam IMDS em vez disso |

O SDK do Azure lê as variáveis de ambiente `AZURE_*` diretamente. O Hermes nunca as inspeciona, exceto para informar quais fontes estão presentes na saída do `hermes doctor`.

## Solução de problemas {#troubleshooting}

**401 Unauthorized em implantações gpt-5.x.**
O Azure serve o gpt-5.x em `/chat/completions`, não em `/responses`. O Hermes trata isso automaticamente quando a URL contém `openai.azure.com`, mas se você vir um 401 com um corpo `Invalid API key`, verifique se `api_mode` no seu `config.yaml` é `chat_completions`.

**404 em `/v1/messages?api-version=.../v1/messages`.**
Esse é o bug de URL malformada de configurações Azure Anthropic anteriores à correção. Atualize o Hermes — o parâmetro `api-version` agora é passado via `default_query`, em vez de embutido na URL base, então o SDK não pode corrompê-lo ao juntar caminhos.

**O assistente diz "Auto-detection incomplete."**
O endpoint rejeitou tanto a sondagem `/models` quanto a sondagem Anthropic Messages. Isso é normal para endpoints privados atrás de um firewall ou com uma lista de permissões de IP. Recorra à seleção manual do modo de API e digite seu nome de implantação — tudo ainda funciona, o Hermes apenas não consegue preencher o seletor automaticamente.

**Transporte errado escolhido.**
Execute `hermes model` novamente e o assistente sondará de novo. Se a sondagem ainda escolher o modo errado, você pode editar o `config.yaml` diretamente:

```yaml
model:
  provider: azure-foundry
  api_mode: anthropic_messages   # ou chat_completions
```

**Entra ID: "credential chain exhausted" ou 401 Unauthorized após mudar para `auth_mode: entra_id`.**
- Execute `az login` para renovar sua sessão de desenvolvedor (o token em cache pode ter expirado).
- Verifique se a atribuição da função `Azure AI User` (ou `Foundry User`) teve efeito: `az role assignment list --assignee <user-or-identity-id>` deve listá-la no seu recurso Foundry. A propagação da função pode levar até 5 minutos.
- Para identidades gerenciadas atribuídas pelo usuário, confira novamente se `AZURE_CLIENT_ID` corresponde à identidade anexada ao recurso de computação.
- Execute `hermes doctor` — a sondagem do Entra do Azure informa se a obtenção do token teve sucesso e inclui uma dica de correção.

**Entra ID: o preflight do assistente trava ou expira.**
A sondagem preliminar de 10 s é uma verificação leve. Escolha "Save anyway and validate later" e execute `hermes doctor` após implantar no ambiente de destino. As causas comuns incluem um serviço de token inacessível ou estado de login local obsoleto — prefira identidade de workload em CI, defina `AZURE_TENANT_ID`+`AZURE_CLIENT_ID`+`AZURE_CLIENT_SECRET` ao usar uma entidade de serviço, ou execute `az login` para desenvolvimento local.

**401 em endpoint no estilo Anthropic com Entra ID.**
Verifique se a mesma função `Azure AI User` (ou `Foundry User`) está atribuída no recurso Foundry (ela cobre tanto os caminhos `/openai/v1` quanto `/anthropic`). Se a sondagem no estilo OpenAI funcionar durante o assistente, mas as requisições `claude-*` falharem em tempo de execução, a causa mais comum é um `model.entra.scope` obsoleto deixado de uma execução anterior do assistente — remova a linha `entra.scope` do `config.yaml` para que o tempo de execução volte ao escopo padrão `https://ai.azure.com/.default`.

## Relacionados {#related}

- [Environment variables](/reference/environment-variables)
- [Configuration](/user-guide/configuration)
- [AWS Bedrock](/guides/aws-bedrock) — a outra grande integração com provedor de nuvem
- [Microsoft: Configure Entra ID for Foundry](https://learn.microsoft.com/azure/ai-foundry/foundry-models/how-to/configure-entra-id) — documentação original do caminho sem chave
