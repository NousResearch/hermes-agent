---
sidebar_position: 11
title: "Integração com Host ACP"
description: "Use o Hermes Agent dentro de editores e plataformas de colaboração compatíveis com ACP"
---

# Integração com Host ACP

O Hermes Agent pode ser executado como um servidor ACP, permitindo que hosts
compatíveis com ACP se comuniquem com o Hermes via stdio. Os editores podem
renderizar:

- mensagens de chat
- atividade de ferramentas
- diffs de arquivos
- comandos de terminal
- prompts de aprovação
- streaming de pensamento / trechos de resposta

Outros hosts podem usar o mesmo protocolo para encaminhar eventos de
colaboração para o Hermes. O ACP é uma boa opção quando você quer que o
Hermes mantenha sua identidade, configuração de provedor, memória, skills e
ferramentas existentes, enquanto outro aplicativo controla o transporte da
conversa.

## O que o Hermes expõe no modo ACP {#what-hermes-exposes-in-acp-mode}

O Hermes é executado com um conjunto de ferramentas `hermes-acp` selecionado
especificamente para fluxos de trabalho em editores. Ele inclui:

- ferramentas de arquivo: `read_file`, `write_file`, `patch`, `search_files`
- ferramentas de terminal: `terminal`, `process`
- ferramentas de web/navegador
- memória, todo, busca de sessão
- skills
- execute_code e delegate_task
- vision

Ele exclui intencionalmente recursos que não se encaixam na UX típica de um
editor, como envio de mensagens e gerenciamento de cronjobs.

## Instalação {#installation}

Instale o Hermes normalmente e, em seguida, adicione o extra do ACP a partir
do checkout de instalação:

```bash
cd ~/.hermes/hermes-agent && uv pip install -e '.[acp]'
```

Isso instala a dependência `agent-client-protocol` e habilita:

- `hermes acp`
- `hermes-acp`
- `python -m acp_adapter`

## Iniciando o servidor ACP {#launching-the-acp-server}

Qualquer um dos comandos abaixo inicia o Hermes no modo ACP:

```bash
hermes acp
```

```bash
hermes-acp
```

```bash
python -m acp_adapter
```

O Hermes registra logs no stderr para que o stdout permaneça reservado para
o tráfego JSON-RPC do ACP.

Para verificações não interativas:

```bash
hermes acp --version
hermes acp --check
```

### Ferramentas de navegador (opcional) {#browser-tools-optional}

As ferramentas de navegador (`browser_navigate`, `browser_click`, etc.)
dependem do pacote npm `agent-browser` e do Chromium, que não fazem parte do
wheel Python. Instale-as com:

```bash
hermes acp --setup-browser           # interativo (pede confirmação antes de baixar ~400 MB)
hermes acp --setup-browser --yes     # aceita o download de forma não interativa
```

Este é o comando independente. O fluxo de autenticação via terminal
(`hermes acp --setup`) também oferece o bootstrap do navegador como uma
pergunta de acompanhamento após a seleção do modelo, então a maioria dos
usuários nunca precisa executar `--setup-browser` diretamente.

O que ele faz:

- Instala o Node.js 22 LTS em `~/.hermes/node/`, caso ainda não esteja instalado
- Executa `npm install -g agent-browser @askjo/camofox-browser` nesse prefixo (sem necessidade de sudo — o `--prefix` do `npm` aponta para o Node gerenciado pelo Hermes, que tem permissão de escrita do usuário)
- Instala o Chromium do Playwright, ou usa um Chrome/Chromium do sistema detectado, quando disponível

O bootstrap é idempotente — executá-lo novamente é rápido e pula o trabalho
que já foi feito.

## Configuração do host {#host-setup}

### Canais do Buzz (ponte de relay) {#buzz-channels-relay-bridge}

[Buzz](https://github.com/block/buzz) é uma plataforma de colaboração
baseada em Nostr para pessoas e agentes. Seu harness `buzz-acp` conecta
canais do Buzz a qualquer agente ACP via stdio:

```text
Buzz relay <-- WebSocket --> buzz-acp <-- ACP over stdio --> Hermes Agent
```

Trata-se de uma integração de transporte, não de uma segunda instalação do
Hermes. O subprocesso iniciado pelo `buzz-acp` usa a mesma configuração,
credenciais, memória, skills e estado do Hermes que o `hermes` naquele host.

(Isso é diferente do [runtime gerenciado do Buzz Desktop](#buzz-desktop),
que inicia o Hermes localmente como um harness pré-configurado. A ponte de
relay serve para entrar em *canais* do Buzz como uma identidade de agente,
tipicamente em um servidor.)

Pré-requisitos:

- Conclua a instalação do ACP e o `hermes acp --check` acima.
- Compile o `buzz-acp` e a CLI `buzz` a partir do
  [repositório do Buzz](https://github.com/block/buzz)
  (`cargo build --release -p buzz-acp`).
- Gere um par de chaves Nostr dedicado para o Hermes (`buzz-admin
  generate-key`) e registre-o como membro do relay (`buzz-admin
  add-member`). Cada agente precisa de sua própria identidade — não
  reutilize o par de chaves de uma pessoa.
- Adicione essa identidade aos canais do Buzz pretendidos.

Inicie uma ponte com:

```bash
export BUZZ_RELAY_URL="wss://community.example.com"
export BUZZ_PRIVATE_KEY="..."
export BUZZ_API_TOKEN="..."
export BUZZ_ACP_AGENT_COMMAND="hermes"
export BUZZ_ACP_AGENT_ARGS="acp"

buzz-acp
```

`BUZZ_API_TOKEN` só é necessário quando o relay exige autenticação por
token. Não faça commit nem cole a chave privada ou o token de API em nenhum
lugar.

Para uma implantação persistente em servidor, execute o `buzz-acp` sob um
gerenciador de serviços, usando o mesmo usuário do sistema operacional que
é dono do Hermes home pretendido. A configuração, a geração de chaves, a
descoberta de canais e as opções por agente estão documentadas no
[README do buzz-acp](https://github.com/block/buzz/tree/main/crates/buzz-acp).

A ponte descobre todos os canais do Buzz nos quais a identidade do Hermes é
membro e se inscreve automaticamente quando ela é adicionada a outro canal.
A associação a canais do Buzz continua sendo, portanto, o limite de acesso;
o Hermes não precisa de uma lista de canais separada em sua própria
configuração.

Para expor a atividade do ACP do Hermes no Buzz Desktop do proprietário,
adicione:

```bash
export BUZZ_ACP_RELAY_OBSERVER="true"
```

Isso publica frames de observador criptografados do tipo (`kind`) `24200`,
endereçados ao proprietário do agente (NIP-AO do Buzz). O Desktop renderiza,
em tempo real, o stream de ciclo de vida, ferramentas, respostas e uso no
**Activity log** do agente. O relay trata esses frames como efêmeros, então
o Desktop precisa estar online antes do turno começar; seu arquivo local de
observador é o histórico durável do lado do proprietário.

Pontes headless respondem sozinhas às solicitações de permissão do ACP,
porque não há nenhum editor presente para exibir os diálogos de aprovação —
veja [Mantenha os agentes do Buzz restritos ao proprietário](#keep-buzz-agents-owner-only).
Trate a ponte como automação privilegiada: use uma conta de sistema
operacional dedicada, restrinja quais usuários do Buzz podem enviar prompts
ao agente (o `buzz-acp` oferece suporte a uma trava de resposta restrita ao
proprietário via `BUZZ_ACP_AGENT_OWNER`), e conceda associação apenas nos
canais onde o Hermes deve realmente atuar.

### VS Code {#vs-code}

Instale a extensão [ACP Client](https://marketplace.visualstudio.com/items?itemName=formulahendry.acp-client).

Para conectar:

1. Abra o painel do ACP Client a partir da Activity Bar.
2. Selecione **Hermes Agent** na lista de agentes integrados.
3. Conecte-se e comece a conversar.

Se quiser definir o Hermes manualmente, adicione-o pelas configurações do
VS Code em `acp.agents`:

```json
{
  "acp.agents": {
    "Hermes Agent": {
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

### Zed {#zed}

Configure o Hermes como um servidor de agente personalizado nas
configurações do Zed:

1. Abra o Agent Panel.
2. Adicione um servidor de agente personalizado com a configuração a seguir:

```json
{
  "agent_servers": {
    "hermes-agent": {
      "type": "custom",
      "command": "hermes",
      "args": ["acp"]
    }
  }
}
```

3. Inicie uma nova thread de agente externo do Hermes.

Pré-requisitos:

- Configure primeiro as credenciais do provedor do Hermes com `hermes
  model`, ou defina-as em `~/.hermes/.env` / `~/.hermes/config.yaml`.

### JetBrains {#jetbrains}

Use um plugin compatível com ACP e aponte-o para `hermes acp` ou
`hermes-acp`.

### Buzz Desktop {#buzz-desktop}


O [Buzz](https://github.com/block/buzz) distribui o Hermes Agent como um
runtime pré-configurado. Com o Hermes instalado da forma normal, o Buzz o
descobre automaticamente — abra **Settings → Runtimes** e o Hermes
aparecerá entre seus runtimes.

Se a descoberta falhar (em instalações mais antigas), certifique-se de que
o launcher do ACP seja resolvido no PATH de um shell de login:

```bash
command -v hermes-acp || command -v hermes
```

Instalações recentes gravam os launchers `hermes` e `hermes-acp` em
`~/.local/bin`; executar `hermes update` adiciona o launcher `hermes-acp` a
instalações mais antigas. Como alternativa manual, configure o comando de
agente do Buzz como `hermes` com os argumentos `["acp"]`.

#### Seletor de modelo {#model-picker}

O Buzz Desktop (v0.5.1+) exibe o menu completo de modelos do Hermes nas
configurações de runtime do agente. A lista vem do próprio Hermes, via ACP:
ela mostra todos os modelos dos provedores que você autenticou no Hermes (o
mesmo inventário por trás do `hermes model` e do comando `/model`), então
um modelo ausente do menu significa que o provedor correspondente não tem
credenciais configuradas no lado do Hermes.

Os IDs das entradas seguem o formato `provider:model` (ex.:
`openrouter:z-ai/glm-5.1`), ou `custom:<name>:<model>` para endpoints
personalizados compatíveis com OpenAI definidos em `config.yaml`. Escolher
um modelo se aplica apenas à sessão daquele agente; isso não altera o seu
padrão geral do Hermes — use `hermes model` para isso.

#### Mantenha os agentes do Buzz restritos ao proprietário {#keep-buzz-agents-owner-only}

O Buzz cria todo agente com a opção **Who can talk to this agent** definida
como `Owner only`. Deixe-a assim quando o runtime for o Hermes.

Dois comportamentos se combinam nesse caminho. O conjunto de ferramentas
`hermes-acp` inclui `terminal` e `execute_code`, e a ponte ACP do Buzz
responde sozinha às solicitações de permissão do Hermes com `allow_once`,
em vez de exibi-las. Por isso, um agente Hermes no Buzz executa comandos de
shell no host sem pedir confirmação. Pedi a um deles que executasse
`rm -rf` em um diretório de teste, e ele apagou tudo, sem nenhum prompt.

Selecionar `Anyone` concede esse mesmo acesso de shell a qualquer autor que
consiga alcançar o canal. O Buzz não avisa quando você escolhe essa opção.

Nenhuma das mitigações óbvias funciona hoje:

- `approvals.mode: manual` de fato faz o Hermes levantar a solicitação de
  permissão, mas o Buzz a aprova automaticamente e o comando é executado
  mesmo assim.
- `platform_toolsets.acp` não restringe o conjunto de ferramentas do ACP,
  então não pode ser usado para remover o `terminal`.

`!shutdown` enviado pelo proprietário interrompe o agente em qualquer modo,
e o Buzz ignora esse comando vindo de qualquer outra pessoa.

## Configuração e credenciais {#configuration-and-credentials}

O modo ACP usa a mesma configuração do Hermes que a CLI:

- `~/.hermes/.env`
- `~/.hermes/config.yaml`
- `~/.hermes/skills/`
- `~/.hermes/state.db`

A resolução de provedor usa o resolvedor de runtime normal do Hermes, então
o ACP herda o provedor e as credenciais configurados atualmente. O Hermes
também anuncia um método de autenticação via terminal (`--setup`) para
clientes ACP em primeira execução; isso abre a configuração interativa de
modelo/provedor do Hermes.

## Integração com o host {#host-integration}

Essas variáveis são definidas por um **processo de host ACP** (um editor ou
outro harness de agente) no subprocesso do Hermes que ele inicia. Elas não
são configuração de usuário — não as defina manualmente em `.env` ou
`config.yaml`.

| Variável | Valor | Efeito |
|----------|-------|--------|
| `HERMES_ACP_SKIP_CONFIGURED_MCP` | `1` | Pula a inicialização dos servidores MCP **configurados globalmente** em `config.yaml` antes que o loop JSON-RPC do ACP comece. |

Normalmente, o Hermes inicia todos os servidores MCP configurados em
`config.yaml` antes de entrar no loop JSON-RPC do ACP. Um host que gerencia
o MCP por conta própria — passando os servidores da sessão explicitamente
via `session/new` — não precisa dessa inicialização global, e um servidor
MCP lento ou interativo, sem relação com a sessão, acabaria atrasando o
`initialize`. Definir o marcador como exatamente `1` permite que esse host
pule essa etapa.

Apenas a descoberta global via `config.yaml` é pulada. **Os servidores MCP
fornecidos pela sessão ACP através de `session/new` continuam sendo
registrados**, então o host não perde nenhuma capacidade que solicitou.
Qualquer outro valor (não definido, vazio, `0`, `false`) mantém o
comportamento padrão, de modo que uma string qualquer que pareça
"verdadeira" não consegue desabilitar o MCP silenciosamente.

## Comportamento de sessão {#session-behavior}

As sessões ACP são rastreadas pelo gerenciador de sessões em memória do
adaptador ACP enquanto o servidor está em execução.

Cada sessão armazena:

- ID da sessão
- diretório de trabalho
- modelo selecionado
- histórico atual da conversa
- evento de cancelamento

O `AIAgent` subjacente continua usando os caminhos normais de
persistência/log do Hermes, mas as operações `list/load/resume/fork` do ACP
ficam restritas ao processo do servidor ACP atualmente em execução.

## Comportamento do diretório de trabalho {#working-directory-behavior}

As sessões ACP vinculam o cwd (diretório atual) do editor ao ID de tarefa
do Hermes, para que as ferramentas de arquivo e terminal sejam executadas
em relação ao workspace do editor, e não ao cwd do processo do servidor.

## Aprovações {#approvals}

Comandos de terminal perigosos podem ser encaminhados de volta ao editor
como prompts de aprovação. As opções de aprovação do ACP são mais simples
do que o fluxo da CLI:

- permitir uma vez
- permitir sempre
- negar

Se você realmente vê um prompt ou não depende do host. Um host tem total
liberdade para responder à solicitação de forma programática, em vez de
exibi-la para você — nesse caso, essas opções existem no protocolo, mas
nunca chegam a um ser humano. É isso que o Buzz Desktop faz, então trate
esse caminho como execução não supervisionada, independentemente da sua
configuração de `approvals`.

Em caso de timeout ou erro, a ponte de aprovação nega a solicitação.

### Aprovação automática de edição por sessão {#session-scoped-edit-auto-approval}

O ACP expõe um terceiro nível entre *permitir uma vez* e *permitir sempre*:
**Allow for session** (permitir para a sessão). Ao escolher essa opção no
prompt de permissão do editor, a aprovação é registrada apenas dentro da
sessão ACP atual — todo comando correspondente subsequente naquela sessão
passa sem solicitar confirmação, mas uma nova sessão ACP (ou reiniciar o
editor) zera esse registro e volta a pedir confirmação na primeira vez.

| Opção | Rótulo no editor | Escopo | Persistido entre reinicializações |
|---|---|---|---|
| `allow_once` | Allow once | Esta única chamada de ferramenta | Não |
| `allow_session` | Allow for session | Todas as chamadas correspondentes nesta sessão ACP | Não — é limpo quando a sessão termina |
| `allow_always` | Allow always | Todas as sessões futuras | Sim (gravado na allowlist permanente do Hermes) |
| `deny` | Deny | Esta única chamada de ferramenta | Não |

`allow_session` é o padrão certo para um fluxo de trabalho em editor no
qual você confia no agente durante a duração de uma tarefa, mas não quer
conceder uma entrada de allowlist de longa duração. O trade-off de
segurança é direto: quanto mais amplo o escopo, menos o editor vai
interromper você, e mais dano um agente mal-comportado (ou uma injeção de
prompt) pode causar antes que você perceba. Comece com `allow_once` para
comandos pouco familiares; promova para `allow_session` depois de ver o
agente executar o mesmo padrão corretamente algumas vezes; reserve o
`allow_always` para comandos verdadeiramente idempotentes nos quais você
confia para sempre (ex.: `git status`).

A ponte ACP mapeia essas opções para a semântica interna de aprovação do
Hermes — o `allow_always` grava uma entrada de allowlist permanente da
mesma forma que a CLI faz, enquanto o `allow_session` afeta apenas o cache
de aprovação em processo da sessão ACP atual.

## Solução de problemas {#troubleshooting}

### O agente ACP não aparece no editor {#acp-agent-does-not-appear-in-the-editor}

Verifique:

- Para desenvolvimento manual/local, verifique se o comando do host aponta
  para `hermes acp`.
- O Hermes está instalado e presente no seu PATH.
- O extra do ACP está instalado (`cd ~/.hermes/hermes-agent && uv pip
  install -e '.[acp]'`).

### O ACP inicia mas gera erro imediatamente {#acp-starts-but-immediately-errors}

Tente estas verificações:

```bash
hermes acp --version
hermes acp --check
hermes doctor
hermes status
```

### Credenciais ausentes {#missing-credentials}

O modo ACP usa a configuração de provedor já existente do Hermes. Configure
as credenciais com:

```bash
hermes model
```

ou editando `~/.hermes/.env`. O fluxo de autenticação via terminal (`hermes
acp --setup`) também pode acionar a configuração interativa de
provedor/modelo.

## Veja também {#see-also}

- [Harness ACP do Buzz](https://github.com/block/buzz/tree/main/crates/buzz-acp)
- [Internals do ACP](../../developer-guide/acp-internals.md)
- [Resolução de Runtime de Provedor](../../developer-guide/provider-runtime.md)
- [Runtime de Ferramentas](../../developer-guide/tools-runtime.md)
