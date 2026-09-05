# Bitwarden Secrets Manager

Obtenha chaves de API do [Bitwarden Secrets Manager](https://bitwarden.com/products/secrets-manager/) na inicialização do processo, em vez de armazená-las em texto plano dentro de `~/.hermes/.env`. Um único segredo de bootstrap (token de acesso de machine account) substitui N chaves por provedor, e rotacionar uma credencial passa a ser uma única alteração no app web do Bitwarden.

## Como funciona

1. Você cria uma **machine account** no Bitwarden Secrets Manager, concede acesso de leitura a um projeto e gera um **access token**.
2. O Hermes armazena esse token único em `~/.hermes/.env` como `BWS_ACCESS_TOKEN`.
3. Sempre que `hermes` (ou o gateway, ou um job cron) inicia, depois que `~/.hermes/.env` foi carregado, o Hermes chama `bws secret list <project_id>` e define as chaves retornadas em `os.environ`.
4. Por padrão, o Hermes **sobrescreve** valores já presentes no seu ambiente, para que o Bitwarden seja a fonte da verdade — rotacione uma chave uma vez no app web e todo processo Hermes a pega na próxima inicialização. Defina `override_existing: false` na config se quiser que `.env` prevaleça.

O binário `bws` é baixado automaticamente em `~/.hermes/bin/` no primeiro uso — sem `apt`, sem `brew`, sem `sudo`.

## Por que machine accounts (e por que sem prompt de 2FA)

O Bitwarden Secrets Manager foi feito para cargas não interativas: machine accounts não podem exigir 2FA porque não há humano no loop. O access token é a credencial. Quem o tiver pode ler todo segredo a que a machine account tem acesso, então trate-o como um bearer token de alto valor — armazene em `.env` (não em `config.yaml`) e revogue + regenere no app web do Bitwarden se vazar.

Você configura a machine account *no app web*, onde seu 2FA normal se aplica. Depois disso, o token é autônomo.

## Setup

### 1. Crie uma machine account e access token

No [app web do Bitwarden](https://vault.bitwarden.com) (ou [vault.bitwarden.eu](https://vault.bitwarden.eu) para contas EU):

1. Mude para **Secrets Manager** no seletor de produto.
2. Crie ou escolha um **Project** (ex.: "Hermes keys").
3. Adicione suas chaves de provedor como secrets. O **Name** do secret vira o nome da variável de ambiente — use `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.
4. **Machine accounts → New machine account → My Hermes machine** → aba **Projects** → conceda Read access ao seu projeto.
5. Aba **Access tokens** → **Create access token** → **Never** expires (ou escolha uma data) → copie o token (começa com `0.`). O Bitwarden não consegue recuperá-lo de novo — guarde a cópia.

O Secrets Manager está incluído no plano gratuito do Bitwarden com limites; não precisa de plano pago para experimentar.

### 2. Execute o wizard

```bash
hermes secrets bitwarden setup
```

Ele irá:

1. Baixar e verificar `bws v2.0.0` em `~/.hermes/bin/bws`.
2. Pedir o access token (entrada oculta). Armazenado em `~/.hermes/.env` como `BWS_ACCESS_TOKEN`.
3. Perguntar a qual região Bitwarden sua machine account pertence — **US Cloud**, **EU Cloud** ou **self-hosted / custom URL**. Armazenado em `config.yaml` como `secrets.bitwarden.server_url` e passado ao `bws` como `BWS_SERVER_URL`.
4. Listar os projetos que a machine account pode ver; escolha um. Armazenado em `config.yaml` como `secrets.bitwarden.project_id`.
5. Testar fetch dos secrets do projeto e mostrar quais env vars serão resolvidas.
6. Definir `secrets.bitwarden.enabled: true`.

Setup não interativo também é suportado via flags:

```bash
hermes secrets bitwarden setup \
  --access-token "$BWS_ACCESS_TOKEN" \
  --server-url https://vault.bitwarden.eu \
  --project-id <project-uuid>
```

### 3. Confirme

```bash
hermes secrets bitwarden status
```

A partir daí, toda invocação `hermes` obtém secrets frescos na inicialização. Você verá um resumo de uma linha em stderr na primeira vez que os segredos forem aplicados em um processo.

## CLI

| Comando | O que faz |
|---|---|
| `hermes secrets bitwarden setup` | Wizard interativo (instala binary, pede token, escolhe projeto, test fetch) |
| `hermes secrets bitwarden status` | Mostra config + versão do binary + presença do token |
| `hermes secrets bitwarden token` | Rotaciona o access token: valida o novo token no Bitwarden, depois armazena em `.env` |
| `hermes secrets bitwarden sync` | Dry-run: puxa secrets agora e mostra o que seria aplicado |
| `hermes secrets bitwarden sync --apply` | Puxa e exporta para o ambiente do shell atual |
| `hermes secrets bitwarden install` | Apenas baixa o binary `bws` fixado (sem auth) |
| `hermes secrets bitwarden disable` | Define `enabled: false`; mantém token + project id |

## Rotacionando um token expirado ou revogado

Quando o token da machine account expira, é revogado ou a conta é excluída, a inicialização mostra:

```
Bitwarden Secrets Manager: Bitwarden rejected the machine-account access token (BWS_ACCESS_TOKEN) — it was likely revoked, expired, or belongs to another region.  (...)
Bitwarden Secrets Manager: → Run `hermes secrets bitwarden token` to paste a fresh access token ...
```

Corrija sem reexecutar o wizard inteiro:

```bash
hermes secrets bitwarden token                     # prompt mascarado
hermes secrets bitwarden token --access-token 0.…  # não interativo
```

O comando sonda o Bitwarden com o novo token **antes** de gravar qualquer coisa — um token rejeitado deixa seu `.env` atual intacto. Em sucesso, armazena o token, limpa os caches de fetch e avisa se o projeto configurado não está visível para a nova machine account.

## Configuração

Padrões em `~/.hermes/config.yaml`:

```yaml
secrets:
  bitwarden:
    enabled: false
    access_token_env: BWS_ACCESS_TOKEN
    project_id: ""
    server_url: ""
    cache_ttl_seconds: 300
    override_existing: true
    auto_install: true
```

| Chave | Padrão | O que faz |
|---|---|---|
| `enabled` | `false` | Interruptor principal. Quando false, Bitwarden nunca é contactado. |
| `access_token_env` | `BWS_ACCESS_TOKEN` | Nome da env var que guarda o token de bootstrap. Mude se você já usa `BWS_ACCESS_TOKEN` para outra coisa. |
| `project_id` | `""` | UUID do projeto de onde sincronizar. |
| `server_url` | `""` | Região Bitwarden ou endpoint self-hosted. Vazio = padrão do `bws` (US Cloud, `https://vault.bitwarden.com`). Defina `https://vault.bitwarden.eu` para EU Cloud, ou sua URL para self-hosted. Passado ao subprocesso `bws` como `BWS_SERVER_URL`. |
| `cache_ttl_seconds` | `300` | Por quanto tempo um resultado de fetch in-process é reutilizado. Defina `0` para desabilitar cache. Cache é por processo; novas invocações `hermes` começam do zero. |
| `override_existing` | `true` | Quando true, valores Bitwarden sobrescrevem o que já está no env (para rotação no app web ter efeito). Defina `false` se quiser `.env` / exports do shell prevalecerem localmente. |
| `auto_install` | `true` | Quando true, `bws` é baixado automaticamente em `~/.hermes/bin/` no primeiro uso. |

## Modos de falha

O Bitwarden nunca bloqueia a inicialização do Hermes. Se algo der errado, você verá um aviso de uma linha em stderr e o Hermes continua com as credenciais que `.env` já tinha:

| Sintoma | Causa | Correção |
|---|---|---|
| `BWS_ACCESS_TOKEN is not set` | Habilitado na config, mas token removido de `.env` | Reexecute `hermes secrets bitwarden setup` |
| `Bitwarden rejected the machine-account access token … invalid_client` | Token revogado, expirado, machine account excluída — ou token de outra região (ex.: token EU no endpoint US) | Execute `hermes secrets bitwarden token` para colar token novo; para mismatch de região, reexecute setup e escolha EU/self-hosted (ou defina `secrets.bitwarden.server_url`) |
| `bws exited 1: invalid access token` | Token revogado ou errado | Execute `hermes secrets bitwarden token` com token novo |
| `bws timed out` | Rede bloqueada ou API Bitwarden lenta | Verifique conectividade com `api.bitwarden.com` (ou seu `server_url`) |
| `bws binary not available` | `auto_install: false` e `bws` não no PATH | Instale manualmente de [github.com/bitwarden/sdk-sm/releases](https://github.com/bitwarden/sdk-sm/releases) ou reative `auto_install` |
| `Checksum mismatch` | Download corrompido ou adulterado | Reexecute, tentará de novo; se persistir, abra issue |

Avisos na inicialização agora incluem uma linha de remediação `→` indicando exatamente qual comando corrige a falha.

## Notas de segurança

- O token de bootstrap (`BWS_ACCESS_TOKEN`) é sensível — quem o tiver pode ler todo segredo a que a machine account tem acesso. Trate-o como qualquer outra API key.
- O Hermes recusa deixar o Bitwarden sobrescrever o próprio token de bootstrap, mesmo com `override_existing: true`. Se você armazenar `BWS_ACCESS_TOKEN` como secret dentro do projeto, ele é ignorado silenciosamente na aplicação.
- O download do binary `bws` é verificado contra o checksum SHA-256 publicado na mesma release do GitHub. Mismatch aborta a instalação.
- A versão fixada (`bws v2.0.0` no momento da escrita) é atualizada via PRs neste repositório — o Hermes não faz auto-upgrade do `bws` para "latest" porque formatos de release upstream podem mudar.

## Quando NÃO usar

- **Setups pessoais de uma máquina** em que `~/.hermes/.env` basta. Você troca uma credencial por outra e adiciona dependência de rede na inicialização.
- **Ambientes air-gapped** que não alcançam `api.bitwarden.com`.
- **CI/CD** em que o mecanismo de injeção de segredos existente (GitHub Actions secrets, Vault, etc.) já está configurado — escolha um caminho, não dois.

O bom caso é frotas multi-máquina, boxes de dev compartilhadas, VPS de gateway ou qualquer setup em que você queira rotação e revogação centralizadas em várias instalações Hermes.
