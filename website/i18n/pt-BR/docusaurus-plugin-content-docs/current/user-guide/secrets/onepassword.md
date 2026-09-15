# 1Password

Resolva chaves de API de provedores a partir do [1Password](https://1password.com/) na inicialização do processo, em vez de armazená-las em texto plano dentro de `~/.hermes/.env`. Você mantém suas chaves como itens do 1Password e as referencia com `op://vault/item/field`; rotacionar uma credencial passa a ser uma única alteração no 1Password.

## Como funciona

1. Você instala o [1Password CLI](https://developer.1password.com/docs/cli/get-started/) oficial (`op`) e o autentica — com um **token de service account** (servidores headless) ou uma **sessão interativa/desktop** (seu laptop).
2. Você mapeia nomes de variáveis de ambiente para referências `op://` em `~/.hermes/config.yaml`.
3. Sempre que `hermes` (ou o gateway, ou um job cron) inicia, depois que `~/.hermes/.env` foi carregado, o Hermes executa `op read` para cada referência e define os valores resolvidos em `os.environ`.
4. Por padrão, o Hermes **sobrescreve** valores já presentes no seu ambiente, para que o 1Password seja a fonte da verdade — rotacione uma credencial uma vez e todo processo Hermes a pega na próxima inicialização. Defina `override_existing: false` se quiser que `.env` prevaleça.

O Hermes nunca autentica em seu nome e nunca baixa o `op`: ele invoca o CLI que você já instalou e já confia. Se `op` estiver ausente, sua sessão estiver bloqueada ou uma referência estiver errada, o Hermes imprime um aviso de uma linha e continua com as credenciais que `.env` já tinha — nunca bloqueia a inicialização.

## Autenticação

O `op` suporta dois modos amigáveis a não interação; o Hermes funciona com qualquer um:

- **Service accounts** (recomendado para servidores/CI): crie uma service account no 1Password, conceda acesso de leitura ao cofre relevante e exporte o token como `OP_SERVICE_ACCOUNT_TOKEN` em `~/.hermes/.env`. O token é a credencial — trate-o como qualquer outro bearer token.
- **Sessões desktop / interativas** (laptops): execute `op signin` (ou habilite a integração CLI no app 1Password). O Hermes repassa suas variáveis `OP_SESSION_*` ao processo filho `op`. A chave de cache do 1Password inclui essas variáveis de sessão, então entrar em outra conta nunca serve um valor cacheado sob a identidade anterior.

## Token de bootstrap

Quando você autentica com um **token de service account**, esse token é a própria credencial de bootstrap que o Hermes precisa *antes* de resolver qualquer referência `op://`. Ele deve estar presente em `os.environ` de todo processo que resolve segredos — incluindo jobs cron (`kanban.dispatch_in_gateway: false`), invocações de subprocesso, execuções CLI, agents launchd do macOS e containers Docker — não apenas o gateway interativo. Há três formas de disponibilizá-lo, em ordem de precedência:

1. **Em `~/.hermes/.env` (recomendado).** `hermes secrets onepassword setup --token <token>` grava o token em `~/.hermes/.env`, exatamente como o `BWS_ACCESS_TOKEN` do Bitwarden. Como `load_hermes_dotenv()` sempre carrega `.env`, o token fica disponível em todo lugar sem configuração extra. Esta é a opção simples e confiável.

2. **Em `~/.hermes/.op.env` (gitignored).** Se preferir manter o token de service account fora de `.env` — por exemplo, para que `.env` possa ir para um repositório privado de dotfiles enquanto o token fica fora do controle de versão — coloque-o em `~/.hermes/.op.env`:

   ```bash
   echo 'OP_SERVICE_ACCOUNT_TOKEN=ops_...' > ~/.hermes/.op.env
   chmod 600 ~/.hermes/.op.env
   ```

   O Hermes carrega `.op.env` automaticamente na inicialização, **depois** de `.env`, e **nunca** sobrescreve um token já presente no ambiente. `.op.env` está no gitignore para que o token nunca entre em um arquivo commitado.

3. **Via systemd `EnvironmentFile` (gateway Linux).** Se você executa o gateway sob systemd, pode injetar o token diretamente no ambiente do serviço:

   ```ini
   [Service]
   EnvironmentFile=-/home/youruser/.hermes/.op.env
   ```

   Um token injetado assim tem precedência — o Hermes detecta que `OP_SERVICE_ACCOUNT_TOKEN` já está definido e pula carregar `.op.env` por completo.

Se o token só estiver acessível por um shell interativo (`op signin`, exports `OP_SESSION_*` em `.bashrc`, etc.), ele **não** será herdado por jobs cron ou subprocessos recém-criados, e esses contextos registrarão um aviso e voltarão às credenciais que `.env` já tinha. Use uma das três opções acima para qualquer carga de trabalho não interativa.

## Setup

### 1. Instale e faça sign-in no `op`

Siga o [guia de getting started do 1Password CLI](https://developer.1password.com/docs/cli/get-started/). Verifique se funciona:

```bash
op whoami
```

### 2. Habilite a integração

```bash
hermes secrets onepassword setup
```

Isso verifica se `op` está no `PATH` (ou use `--binary-path`), registra suas configurações de conta/token, verifica uma sessão ativa e define `secrets.onepassword.enabled: true`. Flags não interativas:

```bash
hermes secrets onepassword setup \
  --account my.1password.com \
  --token-env OP_SERVICE_ACCOUNT_TOKEN \
  --token "$OP_SERVICE_ACCOUNT_TOKEN"
```

### 3. Mapeie suas credenciais

O formato de referência é `op://<vault>/<item>/<field>`:

```bash
hermes secrets onepassword set OPENAI_API_KEY    "op://Private/OpenAI/api key"
hermes secrets onepassword set ANTHROPIC_API_KEY "op://Private/Anthropic/credential"
```

### 4. Visualize e confirme

```bash
hermes secrets onepassword sync     # dry-run: resolve agora, mostra o que seria aplicado
hermes secrets onepassword status   # config + binary + referências + auth
```

A partir daí, toda invocação `hermes` resolve as referências na inicialização. Você verá um resumo de uma linha em stderr na primeira vez que os segredos forem aplicados em um processo.

## CLI

| Comando | O que faz |
|---|---|
| `hermes secrets onepassword setup` | Verifica `op`, define conta / env var de token, habilita |
| `hermes secrets onepassword status` | Mostra config, binary, auth e referências configuradas |
| `hermes secrets onepassword token` | Rotaciona o token de service account: valida com `op whoami`, depois armazena em `.env` |
| `hermes secrets onepassword set ENV_VAR "op://…"` | Mapeia uma env var a uma referência (armazenada stripped + validada) |
| `hermes secrets onepassword remove ENV_VAR` | Remove um mapeamento |
| `hermes secrets onepassword sync` | Dry-run: resolve referências agora e mostra o que seria aplicado |
| `hermes secrets onepassword sync --apply` | Resolve e exporta para o ambiente do shell atual |
| `hermes secrets onepassword disable` | Define `enabled: false`; mantém mapeamentos |

`op` e `1password` são aceitos como aliases de `onepassword`.

## Configuração

Padrões em `~/.hermes/config.yaml`:

```yaml
secrets:
  onepassword:
    enabled: false
    env:
      OPENAI_API_KEY: "op://Private/OpenAI/api key"
      ANTHROPIC_API_KEY: "op://Private/Anthropic/credential"
    account: ""
    service_account_token_env: OP_SERVICE_ACCOUNT_TOKEN
    binary_path: ""
    cache_ttl_seconds: 300
    override_existing: true
```

| Chave | Padrão | O que faz |
|---|---|---|
| `enabled` | `false` | Interruptor principal. Quando false, `op` nunca é invocado. |
| `env` | `{}` | Mapeamento de nome de env var → referência `op://vault/item/field`. Entradas cujo nome não é um nome válido de env var, ou cujo valor não é referência `op://`, são ignoradas com aviso. |
| `account` | `""` | Atalho de conta / endereço de sign-in passado como `op read --account`. Vazio usa a conta padrão do `op`. |
| `service_account_token_env` | `OP_SERVICE_ACCOUNT_TOKEN` | Env var da qual o Hermes lê o token de service account. Seu valor é exportado ao filho `op` como `OP_SERVICE_ACCOUNT_TOKEN` (o nome que o `op` espera). Deixe a var indefinida para usar sessão desktop/interativa. |
| `binary_path` | `""` | Caminho absoluto para `op`. Quando definido, é usado verbatim e `PATH` **não** é consultado — fixe para não confiar no primeiro `op` no `PATH`. |
| `cache_ttl_seconds` | `300` | Por quanto tempo valores resolvidos são reutilizados (in-process e em disco). Defina `0` para desabilitar **ambas** as camadas de cache — nenhum valor é gravado em disco. |
| `override_existing` | `true` | Quando true, valores resolvidos sobrescrevem o que já está no env (para rotação ter efeito). Defina `false` para `.env` / exports do shell prevalecerem; essas referências são ignoradas *antes* de invocar `op`. |

## Modos de falha

O 1Password nunca bloqueia a inicialização do Hermes. Se algo der errado, você verá um aviso de uma linha em stderr e o Hermes continua:

| Sintoma | Causa | Correção |
|---|---|---|
| `the op CLI was not found on PATH` | `op` não instalado / não no PATH | Instale o CLI ou defina `secrets.onepassword.binary_path` |
| `op read failed for 'op://…': …` | Sessão bloqueada, token expirado ou sem acesso ao cofre | `op signin`, execute `hermes secrets onepassword token` para rotacionar o token de service account, ou conceda acesso à service account |
| `op read returned an empty value for 'op://…'` | O campo referenciado existe, mas está vazio | Corrija o item/campo no 1Password (valor vazio nunca é aplicado — sua env var existente permanece intacta) |
| `… is not an op:// secret reference` | Um valor de mapeamento não é referência `op://` | Redefina com o formato correto `op://vault/item/field` |
| `op read timed out` | Rede bloqueada ou 1Password lento | Verifique conectividade / integração com o app desktop |

Avisos na inicialização agora incluem uma linha de remediação `→` indicando exatamente qual comando corrige a falha.

## Cache

Pulls bem-sucedidos e completos são cacheados in-process e em disco em `<hermes_home>/cache/op_cache.json` (escrita atômica, modo `0600`), para que invocações curtas consecutivas de `hermes` não reinvocarem `op` para cada referência. O cache:

- armazena apenas **valores** de segredos resolvidos — nunca o token de service account nem material de auth bruto (auth entra na chave de cache como fingerprint);
- é invalidado quando o token, conta, variáveis `OP_SESSION_*` ou o conjunto de referências mudam;
- **não** é gravado quando um pull teve erro por referência, para que falha transitória de auth não fique congelada pelo TTL;
- fica totalmente desabilitado — leituras *e* gravações — quando `cache_ttl_seconds: 0`.

## Notas de segurança

- Um token de service account do 1Password pode ler todo segredo a que a conta tem acesso. Armazene-o em `~/.hermes/.env` (não em `config.yaml`) e revogue + regenere no 1Password se vazar.
- O Hermes recusa deixar um valor resolvido sobrescrever a própria env var do token, mesmo com `override_existing: true`.
- O processo filho `op` recebe um ambiente mínimo allowlisted (vars de auth/sessão + `PATH`/`HOME`), não uma cópia de `os.environ` completo, para que credenciais de provedor pós-dotenv não sejam todas herdadas pelo filho.
- Referências são validadas para começar com `op://`, e a referência é passada após um terminador de opção `--` para que um valor craftado não seja interpretado como flag do `op`.

## Quando NÃO usar

- **Setups pessoais de uma máquina** em que `~/.hermes/.env` basta.
- **Ambientes air-gapped** que não alcançam o 1Password.
- **CI/CD** em que um mecanismo de injeção de segredos já está configurado — escolha um caminho, não dois.

O bom caso é frotas multi-máquina, boxes de dev compartilhadas, VPS de gateway ou qualquer lugar em que você queira rotação e revogação centralizadas em várias instalações Hermes.
