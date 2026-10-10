# Proxy de injeção de credenciais no egresso (iron-proxy)

Quando o Hermes executa seu agente dentro de um sandbox de terminal Docker, esse sandbox normalmente mantém suas chaves de API reais upstream (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc.). Um agente com prompt injetado nesse sandbox pode rodar `cat ~/.config/openrouter/auth.json` ou `printenv | grep -i key` e exfiltrá-las.

O proxy de egresso resolve isso: o sandbox mantém **tokens de proxy** opacos, nunca as chaves reais. Todo o tráfego de saída do sandbox é roteado por um daemon [iron-proxy](https://github.com/ironsh/iron-proxy) local (Apache-2.0, Go) no host, que termina o TLS e troca o token de proxy pela credencial real antes de encaminhar a requisição upstream. Se o sandbox for comprometido, o atacante fica apenas com tokens que só funcionam dentro do **limite de proxy confiável configurado** — a chave privada da CA e a integridade do endpoint do proxy fazem parte desse limite. Se o tráfego puder ser redirecionado para uma infraestrutura de proxy controlada pelo atacante (por exemplo, uma chave privada de CA roubada ou um endpoint de proxy sequestrado), a garantia do token deixa de valer.

Esta versão conecta o proxy de egresso apenas ao backend Docker. Modal, Daytona, SSH e Singularity **ainda não** recebem variáveis de ambiente de proxy nem montagens de CA.

## O que ele é {#what-it-is}

- Um subprocesso `iron-proxy` gerenciado no host, instalado sob demanda em `~/.hermes/bin/iron-proxy`
- Uma CA local em `~/.hermes/proxy/ca.crt` na qual o sandbox confia, para que o iron-proxy possa fazer MITM de TLS e reescrever cabeçalhos
- Uma configuração `proxy.yaml` em `~/.hermes/proxy/proxy.yaml` listando os hosts upstream permitidos e o mapeamento de transformação de segredos
- Um `mappings.json` registrando qual token de proxy corresponde a qual variável de ambiente real

O sandbox recebe `HTTPS_PROXY=http://host.docker.internal:9090`, `HTTP_PROXY=http://host.docker.internal:9091` e variáveis de ambiente padrão de provedor, como `OPENROUTER_API_KEY`, definidas com tokens de proxy opacos. Aliases correspondentes `HERMES_PROXY_TOKEN_<ENV_NAME>` também são exportados para diagnóstico. Os SDKs de provedor existentes leem os nomes de variável de ambiente usuais, enviam o token de proxy em `Authorization`, e a transformação `secrets` do iron-proxy substitui pelo valor real obtido do ambiente do daemon no lado do host.

## O que ele não é {#what-it-is-not}

- Ele **não é** o comando de entrada `hermes proxy`, que é um proxy reverso agregador de OAuth. Comando diferente (`hermes egress`), direção diferente.
- Ele **não** fica entre seu terminal local e os provedores — apenas entre o sandbox e os provedores.
- Ele **não** reescreve credenciais para chamadas de LLM feitas em processo pelo host. Essas continuam usando suas chaves do `.env` diretamente. O modelo de ameaça é o *sandbox*, não o host.

## Início rápido {#quick-start}

```bash
# 1. Instale o binário do iron-proxy (versão fixada, verificada por SHA-256)
hermes egress install

# 2. Rode o assistente: gera a CA, cria tokens de proxy para cada chave de provedor
#    no seu ambiente, escreve proxy.yaml.
hermes egress setup

# 3. Inicie o daemon do proxy
hermes egress start

# 4. Verifique o status
hermes egress status
```

`hermes egress setup` descobre as chaves de provedor no seu ambiente. Se suas chaves só existem em `~/.hermes/.env` (e não foram exportadas para o seu shell), o setup lê esse arquivo automaticamente — você não precisa dar `export` nelas antes.

Quando você roda `setup` novamente mais tarde (novo host na allowlist, tokens rotacionados, troca da fonte de credenciais), ele para o daemon em execução porque sua configuração fica em memória, e depois **oferece para reiniciá-lo para você** para que a mudança tenha efeito imediatamente. Em um tty ele pergunta; passe `--restart` para sempre reiniciar ou `--no-restart` para deixá-lo parado. Para aplicar mudanças em qualquer outro momento, `hermes egress restart` é o comando único de parar-e-iniciar.

Uma vez em execução, o backend de terminal Docker automaticamente:

- Monta `~/.hermes/proxy/ca.crt` no sandbox em `/etc/ssl/certs/hermes-egress-ca.crt`
- Define `HTTPS_PROXY`, `HTTP_PROXY`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS` para fazer todo runtime HTTP comum passar pelo proxy e confiar na CA
- Define `NODE_OPTIONS=--use-openssl-ca` (anexado ao que você já tiver em `docker_env.NODE_OPTIONS`) para que o Node.js passe pelo armazenamento OpenSSL que as outras variáveis de bundle de CA controlam — veja a [ressalva da CA assimétrica no Node.js](#nodejs-asymmetric-ca-caveat) abaixo para a lacuna residual
- Adiciona `--add-host=host.docker.internal:host-gateway` para que o sandbox consiga alcançar o proxy do lado do host no Linux (o Docker Desktop já cuida disso automaticamente no macOS/Windows)
- Exporta o token de proxy sob o nome de variável de ambiente padrão do provedor (por exemplo, `OPENROUTER_API_KEY`) mais um alias de diagnóstico `HERMES_PROXY_TOKEN_<ENV_NAME>` por mapeamento criado

## Configuração {#configuration}

A configuração completa fica em `~/.hermes/config.yaml`, na seção `proxy:`. Os padrões estão documentados inline; tudo é opcional.

```yaml
proxy:
  # Master switch. When false the feature is a complete no-op — no
  # binaries downloaded, no docker mounts added, no subprocess started.
  enabled: false

  # Tunnel listener port. Sandboxes hit http://host.docker.internal:<port>.
  tunnel_port: 9090

  # Auto-download the pinned iron-proxy binary on first use.
  auto_install: true

  # Where iron-proxy looks up the real upstream secrets at egress time.
  #   env       — process env (default). Whatever is in your ~/.hermes/.env
  #               at proxy-start time is the source of truth.
  #   bitwarden — refetch from Bitwarden Secrets Manager on each proxy
  #               restart. Rotation in the BW web app propagates without
  #               touching .env. Requires `secrets.bitwarden.enabled: true`.
  credential_source: env

  # When true (default), the Docker backend refuses to start a sandbox if
  # the proxy is enabled but not running. Set to false to fall back to the
  # legacy "real credentials inside the sandbox" posture when the proxy
  # is unavailable.
  enforce_on_docker: true

  # When `credential_source: bitwarden` but the BWS access token /
  # project_id is missing OR the bws fetch returns no values for mapped
  # providers, the daemon raises by default (matches the spirit of "I
  # asked for rotation — don't silently use stale env values").  Set
  # to true to opt back into the legacy host-env fallback — useful for
  # migrations where you want to start switching to BW mode but haven't
  # wired every secret yet.
  allow_env_fallback: false

  # SSRF deny list applied to outbound traffic.  Omit / leave null to
  # use the safe default: loopback (v4 + v6), link-local (incl. cloud
  # metadata IPs at 169.254.169.254), RFC1918, IPv6 ULA, IPv4-mapped-v6,
  # CGNAT, and the RFC2544 benchmark range.  Set to an explicit `[]`
  # to opt out entirely (only sensible in hermetic tests).
  upstream_deny_cidrs: null

  # Extra allowed upstream hosts beyond the bundled defaults.
  # Wildcards (`*.foo.com`) are supported. The defaults cover OpenRouter,
  # OpenAI, Anthropic, Google, xAI, Mistral, Groq, Together, DeepSeek,
  # and Nous Research.
  extra_allowed_hosts: []
```

### Hosts upstream permitidos por padrão {#default-allowed-upstream-hosts}

```
openrouter.ai           *.openrouter.ai
api.openai.com          api.anthropic.com
generativelanguage.googleapis.com
api.x.ai                api.mistral.ai
api.groq.com            api.together.xyz
api.deepseek.com        inference.nousresearch.com
```

Se seu agente precisa de um upstream que não está na lista — um endpoint de inferência auto-hospedado, um LLM de nuvem adicional, um servidor MCP — adicione-o a `proxy.extra_allowed_hosts`. Wildcards são comparados com o hostname completo (`*.example.com` casa com `api.example.com` e `staging.example.com`, mas não com o próprio `example.com`).

### CIDRs de negação SSRF padrão {#default-ssrf-deny-cidrs}

Aplicados independentemente da allowlist. Esses intervalos são recusados pelo iron-proxy na fronteira de rede, de modo que um ataque de DNS rebinding via um hostname na allowlist não consegue alcançar o IMDS ou sua rede interna:

| CIDR | Finalidade |
|---|---|
| `127.0.0.0/8`, `::1/128` | Loopback (v4 + v6) |
| `169.254.0.0/16`, `fe80::/10` | Link-local — **incl. IMDS da AWS/GCP/Azure em `169.254.169.254`** |
| `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` | RFC1918 |
| `fc00::/7` | IPv6 ULA |
| `::ffff:0:0/96` | IPv6 mapeado para IPv4 — fecha o bypass de IMDS via dual-stack |
| `100.64.0.0/10` | RFC6598 CGNAT (usado pela AWS VPC, redes de pods K8s) |
| `198.18.0.0/15` | Intervalo de benchmark RFC2544 |

Para sobrescrever: defina `proxy.upstream_deny_cidrs` com sua própria lista. Para desativar completamente (por exemplo, para um teste hermético que precisa alcançar um upstream em loopback): defina uma lista vazia `[]`.

### Política de bind {#bind-policy}

O proxy nunca faz bind em `0.0.0.0`. O bind padrão é específico da plataforma porque o iron-proxy v0.39 suporta apenas **um bind por processo de daemon**:

- **Linux:** o gateway da bridge do docker (`172.17.0.1:<tunnel_port>` por padrão). Os containers alcançam o proxy via `host.docker.internal`, que `--add-host=host.docker.internal:host-gateway` resolve exatamente para esse IP de gateway da bridge — um bind apenas em loopback seria inalcançável de dentro dos sandboxes. O IP da bridge é um endereço na interface `docker0` do host, então não fica exposto à LAN; ele É alcançável por outros containers na rede bridge padrão, mas as requisições ainda exigem um token de proxy criado e um upstream na allowlist. Se nenhuma bridge do docker for detectada (docker não instalado/em execução), o bind volta para loopback com um aviso.
- **macOS/Windows com Docker Desktop:** loopback (`127.0.0.1:<tunnel_port>`). O VPNkit do Desktop roteia `host.docker.internal` para o host, então o loopback é alcançável a partir dos containers e é a escolha menos exposta.

Um peer na LAN com um token de proxy vazado não consegue usar o proxy — nenhum dos dois binds é alcançável a partir da rede externa.

Também fixamos `metrics.listen: 127.0.0.1:0`, para que o servidor de métricas embutido do daemon receba uma porta efêmera de loopback em vez de sua porta padrão `:9090` — do contrário ele disputaria a mesma porta com `tunnel_port: 9090` e o daemon recusaria iniciar com "address already in use". Note que a porta efêmera `:0` é aleatória a cada início e não é exposta em lugar nenhum, então as métricas ficam efetivamente desativadas por causa dessa fixação.

Se um shim hostil de `ip` mais cedo no PATH tivesse conseguido injetar um IPv4 não privado como endereço da bridge (`0.0.0.0`, um endereço público, multicast, link-local etc.), o fallback para loopback ainda se aplicaria — nunca fazemos bind em nada que não pudéssemos validar via `ipaddress.IPv4Address` + checagens `is_*`.

## Esquemas de autenticação cobertos {#covered-auth-schemes}

A transformação `secrets` troca o token de proxy onde quer que ele apareça em um local reconhecido — e ela cobre mais do que apenas `Authorization: Bearer`:

| Provedor | Variável de ambiente | Trocado em |
|---|---|---|
| OpenRouter, OpenAI, Groq, Together, DeepSeek, Mistral, xAI, Nous | `*_API_KEY` | Cabeçalho `Authorization` |
| Anthropic nativo | `ANTHROPIC_API_KEY` | `x-api-key` + `Authorization` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | `api-key` + `Authorization` (`*.openai.azure.com`, `*.cognitiveservices.azure.com`, `*.services.ai.azure.com`) |
| Google AI Studio (Gemini) | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Cabeçalho `x-goog-api-key` ou parâmetro de query `?key=` |

`GEMINI_API_KEY` e `GOOGLE_API_KEY` são tratadas como uma única credencial: um único token de proxy é criado e injetado no sandbox sob **ambos** os nomes, e qualquer um dos dois no seu ambiente de host satisfaz a descoberta.

## Provedores não cobertos {#uncovered-providers}

Esquemas de autenticação que envolvem assinatura de requisição ou OAuth criado pelo SDK não podem ser trocados por uma substituição estática de cabeçalho — se suas variáveis de ambiente estiverem presentes, o sandbox mantém **credenciais reais** para esses provedores, e a garantia de isolamento de egresso fica incompleta para eles:

| Variável de ambiente | Provedor | Motivo |
|---|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | AWS Bedrock/SageMaker | Requisições assinadas com SigV4 |
| `GOOGLE_APPLICATION_CREDENTIALS` | GCP Vertex AI | OAuth criado a partir de um arquivo de service account |

Essas variáveis de ambiente estão presentes na maioria dos laptops de desenvolvedores por causa de outras ferramentas não relacionadas (terraform, gcloud, aws CLI, push para ECR). Elas aparecem como avisos no assistente e em `hermes egress status`, mas nunca impedem o proxy de iniciar. Se você não usa esses provedores a partir de sandboxes, dê `unset` nas variáveis para eliminar o aviso.

## Integração com Bitwarden {#bitwarden-integration}

Se você já usa o Bitwarden Secrets Manager via [`hermes secrets bitwarden setup`](../secrets/bitwarden), o proxy de egresso pode buscar as credenciais reais de lá em vez de `os.environ`:

```bash
hermes egress setup --from-bitwarden
```

Isso define `proxy.credential_source: bitwarden` e descobre os nomes de variável de ambiente de provedor a partir do seu projeto no BW.

### Semântica de rotação {#rotation-semantics}

Quando `credential_source: bitwarden`, o daemon do iron-proxy busca novamente os segredos do BWS via `bws secret list <project_id>` **toda vez que inicia**. Então o fluxo de rotação é:

1. Rotacione uma chave no app web do Bitwarden.
2. `hermes egress stop && hermes egress start` no host.
3. Sandboxes iniciados a partir desse ponto trocam os tokens de proxy pelo novo valor.

Sem edições no `.env`. Sem reiniciar o Hermes no host. O daemon do proxy é a única coisa que toca no novo valor — seu processo host e o `os.environ` permanecem intocados.

### Falha ruidosa no início {#fail-loud-at-start}

Quando `credential_source: bitwarden`, `hermes egress start` faz uma pré-checagem na camada do assistente E `_build_proxy_subprocess_env` faz uma checagem adicional na camada do daemon:

- A variável de ambiente do token de acesso do BWS não está definida → recusa iniciar, com uma dica para dar `unset` e rodar de novo, ou usar `hermes egress setup --no-bitwarden` para voltar ao modo env
- `secrets.bitwarden.project_id` está vazio → recusa iniciar, com uma dica para rodar `hermes secrets bitwarden setup`
- `bws secret list` não retorna valores para um ou mais provedores mapeados → recusa iniciar, listando os nomes que faltam

Isso é intencional. Voltar para o ambiente do host no modo BW reintroduz exatamente o bug de dados desatualizados que o caminho do BW deveria eliminar (o operador escolheu o BW pela garantia de rotação; um fallback silencioso quebra essa garantia).

A flag de configuração `proxy.allow_env_fallback: true` reativa o comportamento legado de "voltar silenciosamente para o ambiente do host se o BWS estiver inacessível", para cenários de migração. Use-a quando estiver movendo segredos para o BW um de cada vez e quiser que o daemon inicie com os valores que estiverem disponíveis.

### Trocando a fonte de credenciais {#switching-credential-source}

| De | Para | Comando |
|---|---|---|
| env | bitwarden | `hermes egress setup --from-bitwarden` |
| bitwarden | env | `hermes egress setup --no-bitwarden` |

**Rodar `hermes egress setup` novamente SEM nenhuma das duas flags preserva o `credential_source` existente** — o assistente se recusa a rebaixá-lo silenciosamente de volta para env. Isso importa porque, uma vez configurado o modo bitwarden, a garantia de rotação é o que você escolheu; você precisa dizer explicitamente "quero env de novo" para mudar isso.

## Comandos slash {#slash-commands}

A árvore de subcomandos da CLI:

```
hermes egress install                  # download the pinned iron-proxy binary
hermes egress install --force          # re-download even if a managed copy exists

hermes egress setup                    # interactive wizard
hermes egress setup --tunnel-port N    # override the tunnel listener port
hermes egress setup --from-bitwarden   # use BWS as credential source (fail-loud)
hermes egress setup --no-bitwarden     # explicitly switch back to env mode
hermes egress setup --rotate-tokens    # mint fresh tokens for every provider
                                       #   (default preserves existing)

hermes egress start                    # spawn the managed proxy daemon
hermes egress stop                     # SIGTERM (then SIGKILL after 5s grace)
hermes egress restart                  # stop (if running) then start — needed when
                                       #   upstream SECRETS change (rotation, new provider)
hermes egress reload                   # hot-reload the ruleset from proxy.yaml via the
                                       #   management API — no restart, no dropped
                                       #   connections (allowlist / mapping edits)

hermes egress status                   # binary + config + pid + listening state + mappings
hermes egress status --show-tokens     # print proxy tokens in full
                                       #   (default: redacted prefix + suffix only)

hermes egress disable                  # flip proxy.enabled = false
                                       #   (does not stop a running proxy)

hermes egress config                   # print the path to proxy.yaml for debugging
```

### Rotação de tokens {#token-rotation}

Por padrão, `hermes egress setup` **preserva** os tokens de proxy dos provedores que já os possuem. Adicionar um novo provedor cria um token novo apenas para ele; os tokens existentes ficam inalterados. Isso evita que sandboxes em execução recebam 401 quando você roda o assistente novamente.

`--rotate-tokens` renova todos os tokens:

```bash
hermes egress setup --rotate-tokens
```

Quando já existem tokens E o stdin é um tty, o assistente pede confirmação:

```
⚠  --rotate-tokens will invalidate proxy tokens in every running
   Hermes sandbox.  They will start 401-ing against upstreams until restarted.
Type 'rotate' to confirm:
```

Invocações sem tty (CI, scripts) pulam a confirmação — a flag é tratada como intencional. Antes de qualquer sobrescrita, o `mappings.json` atual é copiado para um arquivo irmão com timestamp, para permitir recuperação manual:

```
backup: ~/.hermes/proxy/mappings.json.rotated-20260524T143012
```

`hermes egress setup` para um daemon em execução ao reescrever a configuração ou os mapeamentos de token, porque o daemon mantém o YAML antigo em memória. Depois de `--rotate-tokens`:

```bash
hermes egress start
```

Containers já em execução mantêm os tokens antigos e precisarão ser reiniciados para usar os novos. Novos containers Docker persistentes incluem um rótulo de postura de egresso, então o Hermes não reutilizará um container anterior ao egresso ou anterior à rotação para novas sessões.

## Layout do diretório de estado {#state-directory-layout}

Tudo o que o iron-proxy mantém fica em `~/.hermes/proxy/`:

| Caminho | Modo | Finalidade |
|---|---|---|
| `~/.hermes/proxy/` (dir) | `0o700` | Pertence e é percorrível apenas por você |
| `ca.crt` | `0o644` | Certificado público da CA distribuído para os sandboxes |
| `ca.key` | `0o600` | Chave de assinatura da CA — nunca sai do host |
| `proxy.yaml` | `0o600` | Configuração do iron-proxy; reescrita a cada `setup` |
| `mappings.json` | `0o600` | Token de proxy do sandbox → variável de ambiente upstream |
| `mappings.json.rotated-*` | `0o600` | Backups criados por `--rotate-tokens` |
| `iron-proxy.pid` | `0o600` | PID do daemon em execução |
| `iron-proxy.nonce` | `0o600` | Nonce por início, para defesa contra reciclagem de PID |
| `iron-proxy.log` | `0o600` | Stdout/stderr do daemon — **inclui registros por requisição na v0.39** |
| `audit.log` | `0o600` | Reservado para o fluxo dedicado de auditoria por requisição em versões futuras do binário; pré-criado para que o contrato de privacidade se mantenha quando o upstream o implementar |

A chave privada da CA é o arquivo mais sensível. Ela é criada com `0o600` desde o primeiro byte (sem janela de TOCTOU via umask) e com `O_NOFOLLOW`, para que um atacante com o mesmo uid não possa redirecioná-la via um symlink plantado. O arquivo de pid, o arquivo de nonce, o log do daemon e o log de auditoria recebem o mesmo tratamento.

### Logging na v0.39 do iron-proxy {#logging-on-iron-proxy-v039}

Na versão de binário atualmente fixada (**v0.39.0**), o iron-proxy escreve TODA a saída — diagnósticos em nível de daemon E registros por requisição — em **`~/.hermes/proxy/iron-proxy.log`**. A struct `config.Log` da v0.39 não tem um campo `audit_path` separado, então não conseguimos rotear os registros por requisição para um fluxo dedicado nela.

Ainda assim pré-criamos `~/.hermes/proxy/audit.log` com `0o600` e `O_NOFOLLOW` porque:

1. Isso reserva o caminho para a futura atualização de versão: quando a versão fixada mudar para uma que suporte `log.audit_path`, os registros por requisição começarão a fluir para lá sem reconfiguração do lado do operador. **Até lá o arquivo permanece com 0 bytes — não aponte ferramentas de monitoramento, alertas ou forense para ele ainda.** Use `iron-proxy.log` para tudo hoje.
2. A garantia de `0o600` desde o primeiro byte protege contra o dia da correção upstream, em que a v0.40+ cria o arquivo sob seu umask padrão caso ele ainda não exista.

Até essa atualização de versão chegar, trate `iron-proxy.log` como a fonte da verdade para ambos os públicos:

- Eventos em nível de daemon (banner de início, erros de bind, motivo do encerramento, erros de transformação). Operações + troubleshooting.
- Registros por requisição (CONNECT para upstream na allowlist, troca de segredo disparada, negação de allowlist). Forense + compliance.

Ambos os arquivos são acrescidos entre reinícios. Rotacione-os com logrotate se você se preocupar com uso de disco em hosts de longa duração.

## Como funciona {#how-it-works}

```
┌──────────────┐                ┌──────────────┐                ┌─────────────┐
│ Docker       │ CONNECT /     │ iron-proxy    │ HTTPS w/       │ OpenRouter  │
│ sandbox      ├──────────────▶│ (host:9090)   ├───────────────▶│ / OpenAI /  │
│              │ HTTP forward  │               │ real API key   │ Anthropic …  │
│ has:         │ w/ proxy tok  │ mints leaf    │                │             │
│ - proxy tok  │ in Auth hdr   │ cert from CA  │                │             │
│ - CA cert    │               │ matches token │                │             │
│ - HTTPS_PROXY│               │ swaps secret  │                │             │
└──────────────┘               └──────────────┘                └─────────────┘
                                       │
                                       │ daemon + per-request log (combined on v0.39)
                                       ▼
                              ~/.hermes/proxy/iron-proxy.log
                              (~/.hermes/proxy/audit.log reserved for v0.40+ split stream)
```

1. O sandbox faz uma requisição HTTPS, por exemplo `POST https://openrouter.ai/v1/chat/completions` com `Authorization: Bearer hermes-proxy-openrouter-…` (o token de proxy, não a chave real).
2. Como `HTTPS_PROXY` está definido, a requisição vai para o iron-proxy como um túnel CONNECT.
3. O iron-proxy verifica a allowlist. `openrouter.ai` é permitido.
4. O iron-proxy cria um certificado folha assinado pela nossa CA para `openrouter.ai`, termina a conexão TLS e inspeciona a requisição.
5. A transformação `secrets` encontra a string do token de proxy no cabeçalho `Authorization` e a substitui pelo valor real de `OPENROUTER_API_KEY`, obtido do próprio ambiente do iron-proxy.
6. A requisição é recriptografada e encaminhada para a OpenRouter.
7. A requisição é registrada em `~/.hermes/proxy/iron-proxy.log` na v0.39. Quando a versão fixada do binário suportar o fluxo dividido (v0.40+), os registros por requisição passarão a fluir para `~/.hermes/proxy/audit.log`, e os diagnósticos em nível de daemon permanecerão em `iron-proxy.log`. Veja [Logging na v0.39 do iron-proxy](#logging-on-iron-proxy-v039).

Uma requisição para um host fora da allowlist (por exemplo, `https://attacker.example.com/leak?key=...`) é rejeitada com HTTP 403 antes que qualquer byte saia do host. A negação é registrada em `iron-proxy.log` com o host upstream e o sandbox de origem.

### Distribuição da CA para o sandbox {#ca-distribution-into-the-sandbox}

Quando o backend Docker inicia um container com `proxy.enabled: true` e o daemon está escutando, ele adiciona estes argumentos ao `docker run`:

| Argumento | Finalidade |
|---|---|
| `-v ~/.hermes/proxy/ca.crt:/etc/ssl/certs/hermes-egress-ca.crt:ro` | Montagem somente leitura da CA |
| `-e HTTPS_PROXY=http://host.docker.internal:9090` | Python httpx / curl / transporte padrão do go / fetch do Node |
| `-e HTTP_PROXY=http://host.docker.internal:9091` | curl + wget para HTTP puro — o listener de encaminhamento HTTP puro fica em `tunnel_port + 1` |
| `-e NO_PROXY=127.0.0.1,localhost,::1` | Servidores de desenvolvimento em loopback dentro do sandbox ignoram o proxy |
| `-e REQUESTS_CA_BUNDLE=…ca.crt` | `requests` do Python |
| `-e SSL_CERT_FILE=…ca.crt` | Módulo `ssl` do Python / OpenSSL — **substitui** o armazenamento do sistema |
| `-e CURL_CA_BUNDLE=…ca.crt` | curl — **substitui** o armazenamento do sistema |
| `-e NODE_EXTRA_CA_CERTS=…ca.crt` | Node.js — **adiciona** ao armazenamento do sistema |
| `-e NODE_OPTIONS="<your value> --use-openssl-ca"` | Node.js — roteia pelo armazenamento OpenSSL (anexado; seu `--max-old-space-size` etc. são preservados) |
| `-e HERMES_EGRESS_PROXY=1` | Sentinela que o agente pode ler para saber que está ciente do proxy |
| `-e OPENROUTER_API_KEY=<proxy-token>` | Nomes padrão de variável de ambiente de provedor recebem tokens de proxy, então os SDKs existentes continuam funcionando |
| `-e HERMES_PROXY_TOKEN_<NAME>=…` | Alias de diagnóstico para cada mapeamento; mesmo valor da variável de ambiente padrão do provedor |
| `--add-host=host.docker.internal:host-gateway` | Apenas Linux; o Docker Desktop mapeia isso automaticamente |

#### Ressalva da CA assimétrica no Node.js {#nodejs-asymmetric-ca-caveat}

`REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE` **substituem** o armazenamento de CA do sistema dentro do sandbox. `NODE_EXTRA_CA_CERTS` **adiciona** a ele. Um processo Node.js dentro do sandbox poderia, em princípio, contornar o proxy abrindo um `net.Socket` bruto e iniciando seu próprio handshake TLS — o armazenamento de CA do sistema ainda confiaria nos certificados reais dos upstreams, então a requisição teria sucesso onde Python/curl falhariam na validação.

`NODE_OPTIONS=--use-openssl-ca` é anexado ao que você já tiver em `docker_env.NODE_OPTIONS`. Isso força o Node a passar pelo armazenamento OpenSSL que `SSL_CERT_FILE` controla, reduzindo a assimetria. Isso NÃO cobre código que passa explicitamente sua própria opção `ca` para `tls.connect()` ou `https.request()`, mas fecha o caso mais fácil.

Essa é uma limitação conhecida da v1. Acompanhe [github.com/ironsh/iron-proxy/issues](https://github.com/ironsh/iron-proxy/issues) para uma solução upstream; enquanto isso, não rode código Node não confiável que abra sockets brutos em um sandbox que dependa do isolamento de egresso.

### Colisões de docker\_env {#docker_env-collisions}

Se você definir variáveis de ambiente que controlam o proxy no seu bloco de configuração `docker_env:` (raro, mas possível), o Hermes se recusa a iniciar o sandbox quando `enforce_on_docker: true` está definido. Isso inclui tanto:

- Variáveis de controle de egresso: `HTTPS_PROXY`, `HTTP_PROXY`, `NO_PROXY`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS`
- Variáveis de ambiente reais de provedor: todo nome presente em `mappings.json` (por exemplo, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`)

Exemplo de erro:

```
docker_env in config.yaml overrides egress-proxy variables
['HTTPS_PROXY', 'OPENROUTER_API_KEY']; enforce_on_docker is enabled.
Remove these keys from docker_env or disable enforce_on_docker to
opt out of egress isolation.
```

Com `enforce_on_docker: false`, a mesma situação aparece como um aviso e seus valores de `docker_env` prevalecem — útil para migrações ou testes, mas você está explicitamente optando por sair da garantia de isolamento.

## Defesa por PID e nonce {#pid-and-nonce-defense}

O arquivo de pid do daemon é escrito com `O_EXCL` + `O_NOFOLLOW` + verificação de propriedade. Chamadas concorrentes a `hermes egress start` produzem um de dois resultados:

- O arquivo de pid existente aponta para um iron-proxy vivo → o segundo início se recusa com "another start in progress" + uma dica para rodar `hermes egress stop`
- O arquivo de pid existente está obsoleto (daemon que travou) → o segundo início o remove e tenta novamente uma vez

Além disso, todo `start_proxy` planta um novo nonce aleatório em dois lugares:

- `HERMES_IRON_PROXY_NONCE=<nonce>` no ambiente do daemon
- `~/.hermes/proxy/iron-proxy.nonce` (arquivo irmão do pidfile, com `0o600`)

Quando `hermes egress stop` (ou qualquer outra checagem `_pid_alive`) quer confirmar que um PID ainda se refere ao *nosso* daemon — e não a um processo não relacionado que recebeu o mesmo PID depois que o iron-proxy travou —, ele lê `/proc/<pid>/environ` e procura o nonce. A cópia em disco é o que faz isso funcionar entre invocações da CLI (o `_proxy_nonce` em memória é por processo e reinicia a cada invocação do `hermes`).

Se a checagem de nonce falhar, o código recorre a comparar o basename de `argv[0]` com `iron-proxy`. `stop_proxy` também captura o starttime de `/proc/<pid>/stat` antes do SIGTERM e o revalida após a janela de 5s de tolerância — se o starttime mudou, o PID foi reciclado durante a espera e o SIGKILL é suprimido com um aviso.

## Modelo de segurança {#security-model}

**Contra o que isso protege:**

- Um agente com prompt injetado em um sandbox Docker lendo `printenv`/arquivos de credenciais e exfiltrando as chaves reais.
- Uma dependência comprometida no sandbox se comunicando com um host arbitrário — a allowlist com negação por padrão bloqueia destinos desconhecidos.
- Um agente discando para endpoints de metadados de nuvem (`169.254.169.254`) — o iron-proxy nega isso por padrão via `upstream_deny_cidrs`, incluindo a forma mapeada IPv4-para-IPv6 `::ffff:169.254.169.254`.
- DNS rebinding através de um hostname na allowlist apontando para um IP privado — os CIDRs de negação são checados no momento da conexão, não no momento da allowlist.
- Processos locais com o mesmo uid lendo o ambiente do daemon iron-proxy para coletar segredos — apenas os nomes de variável de ambiente referenciados pelos mapeamentos são encaminhados, não o ambiente completo do host.
- Um peer na LAN com um token de proxy vazado do sandbox gastando sua cota de API — o proxy faz bind no gateway da bridge do docker (Linux) ou em loopback (Docker Desktop), nunca em `0.0.0.0`, então é inalcançável a partir da rede externa.

**Contra o que isso NÃO protege:**

- Um processo host comprometido. Se o próprio processo do agente for comprometido, as chaves reais no `~/.hermes/.env` do host ficam expostas de qualquer forma. Este é um recurso de defesa em profundidade para o comprometimento do *sandbox*, não do host.
- **Perda do próprio limite de proxy confiável.** A garantia de troca de token pressupõe que o sandbox confia no certificado de CA montado (`/etc/ssl/certs/hermes-egress-ca.crt`) e que o tráfego realmente chega ao *nosso* iron-proxy. Se a chave privada da CA for roubada, ou se o egresso do sandbox for redirecionado para uma infraestrutura de proxy controlada pelo atacante, um adversário no meio pode apresentar um certificado folha válido, e os tokens de proxy deixam de ser um limite significativo (cf. [MITRE ATT&CK T1588.004](https://attack.mitre.org/techniques/T1588/004/) — material de certificado TLS obtido que habilita AiTM). Proteja a chave da CA (ela é `0600`, apenas no host) e o endpoint do proxy de acordo.
- Processos do sandbox que contornam `HTTPS_PROXY` usando um socket bruto. O proxy não consegue interceptar o que não é roteado até ele. O Node.js é parcialmente mitigado via `NODE_OPTIONS=--use-openssl-ca` (veja a ressalva acima).
- Arquivos de credenciais explicitamente montados no Docker (`terminal.credential_files` ou montagens registradas por skills). O egresso protege variáveis de ambiente de provedor; ele não inspeciona arquivos montados arbitrários. Não monte credenciais reais de provedor em um sandbox com egresso reforçado.
- Exfiltração de dados para um host na allowlist. Se `api.openai.com` é permitido, um agente poderia embutir dados de exfiltração no corpo de uma requisição para esse host. O log do daemon registra que a requisição aconteceu, mas não a impede.
- Provedores não cobertos (AWS Bedrock SigV4, OAuth de service account do GCP Vertex). As variáveis de ambiente deles permanecem no sandbox; se você os habilitar, essas credenciais contornam o proxy completamente. Veja [Provedores não cobertos](#uncovered-providers).
- Zeragem de segredos em memória do iron-proxy. O binário Go mantém as credenciais reais trocadas na memória do processo; um core-dump ou uma leitura de `/proc/<pid>/mem` por um atacante com o mesmo uid as exporia. Fora do escopo desta camada.

## Modos de falha {#failure-modes}

- **Binário não instalado, `auto_install: true`** — o primeiro `hermes egress setup` ou `hermes egress start` faz o download. Verificado por SHA-256 contra o `checksums.txt` upstream.
- **Binário não instalado, `auto_install: false`** — `start` falha com uma mensagem clara apontando para instalação manual.
- **`enabled: true` mas o proxy não está em execução** — com `enforce_on_docker: true` (padrão), a criação do sandbox Docker se recusa a iniciar com um erro explicativo. Com `enforce_on_docker: false`, cai de volta para saída direta com credenciais reais e registra um aviso.
- **Colisão de porta** — o iron-proxy sai imediatamente; `hermes egress start` reporta as últimas 20 linhas de log e falha com código de saída diferente de zero.
- **Host upstream negado** — o sandbox recebe HTTP 403 do proxy, com um corpo explicando qual host não foi permitido. O agente vê o erro e o reporta.
- **IP de metadados de nuvem (169.254.169.254) solicitado** — recusado por `upstream_deny_cidrs`, independentemente da allowlist.
- **`docker_env` colide com uma variável que controla o proxy (reforço ativado)** — a criação do sandbox se recusa, com os nomes das chaves em colisão.
- **`docker_forward_env` tenta encaminhar uma chave de provedor protegida (reforço ativado)** — a criação do sandbox se recusa; remova a chave de `docker_forward_env` ou desative com `proxy.enforce_on_docker: false`.
- **`docker_extra_args` sobrescreve os controles de ambiente/rede do proxy (reforço ativado)** — a criação do sandbox se recusa; argumentos fornecidos pelo usuário como `-e HTTPS_PROXY=...`, `--env-file` ou `--network` rodam depois dos argumentos gerados pelo Hermes e podem contornar o egresso.
- **Token de acesso do BWS ausente com `credential_source: bitwarden`** — `hermes egress start` se recusa, com `--no-bitwarden` como dica de recuperação.
- **iron-proxy não faz bind em 5 segundos** — o processo é finalizado, o pidfile é removido, o erro nomeia a porta + o final de `iron-proxy.log`.
- **Chamadas concorrentes a `hermes egress start`** — a segunda chamada se recusa com "another start in progress" se o daemon da primeira estiver ativo; caso contrário, a segunda remove o pidfile obsoleto e prossegue.

## Solução de problemas {#troubleshooting}

### "Refusing to start: BWS_ACCESS_TOKEN is not set"

Você habilitou `credential_source: bitwarden`, mas a variável de ambiente do token de acesso não está no seu shell. Faça uma das opções:

```bash
export BWS_ACCESS_TOKEN=…   # one-shot
hermes egress start
```

Ou coloque-a em `~/.hermes/.env`. Ou volte ao modo env:

```bash
hermes egress setup --no-bitwarden
```

### "iron-proxy exited immediately"

Olhe as últimas 20 linhas de `~/.hermes/proxy/iron-proxy.log`. Causas comuns:

- Porta já em uso → mude `proxy.tunnel_port` ou finalize o que quer que esteja usando a 9090
- `proxy.yaml` inválido → rode `hermes egress setup` para regenerá-lo
- Permissões erradas no certificado/chave da CA → `chmod 0o600 ~/.hermes/proxy/ca.key`

### "iron-proxy did not bind \<bind-host\>:9090 within 5s"

O daemon iniciou, mas nunca fez bind no listener. Geralmente significa que o binário travou ou está fazendo algo custoso na inicialização. Verifique `~/.hermes/proxy/iron-proxy.log`. O processo órfão é finalizado automaticamente e o pidfile é limpo, então você pode simplesmente tentar `hermes egress start` de novo.

### O sandbox expira ao tentar conectar ao proxy (Linux) {#sandbox-times-out-connecting-to-the-proxy-linux}

O container resolve `host.docker.internal` para o gateway da bridge do docker, e o proxy está vinculado ali, mas um firewall no host (comumente `ufw` com INPUT com negação por padrão) descarta o tráfego container→host na `docker0`. Verifique a partir de um container:

```bash
docker run --rm --add-host host.docker.internal:host-gateway busybox \
  nc -zv -w 3 host.docker.internal 9090
```

Se isso expirar enquanto `hermes egress status` mostra `listening`, permita a sub-rede da bridge no seu firewall, por exemplo para o ufw:

```bash
sudo ufw allow in on docker0 to any port 9090 proto tcp
sudo ufw allow in on docker0 to any port 9091 proto tcp
```

(9091 = o listener de encaminhamento HTTP puro em `tunnel_port + 1`.)

### O sandbox recebe `HTTP 403` do proxy {#sandbox-sees-http-403-from-the-proxy}

O agente dentro do sandbox tentou acessar um host que não está em `proxy.extra_allowed_hosts`. O corpo do 403 explica qual host. Se você quiser permiti-lo, adicione à sua configuração:

```yaml
proxy:
  extra_allowed_hosts:
    - api.example.com
    - "*.staging.example.com"
```

Depois rode `hermes egress setup` (para regenerar `proxy.yaml`) e `hermes egress stop && hermes egress start`.

### O sandbox apresenta erros de verificação SSL {#sandbox-sees-ssl-verification-errors}

Ou a CA não está montada no sandbox (raro; o backend docker faz isso automaticamente quando `proxy.enabled: true`), ou o cliente HTTP da sua imagem está lendo de uma variável de ambiente não padrão.

```bash
# Inside the sandbox:
cat /etc/ssl/certs/hermes-egress-ca.crt | head -1
# Should print: -----BEGIN CERTIFICATE-----
env | grep -E "^(REQUESTS|CURL|SSL|NODE).*CA"
# Should list all four CA-bundle env vars pointing at /etc/ssl/certs/hermes-egress-ca.crt
```

Se o certificado não estiver lá, verifique se `proxy.enabled: true` E `hermes egress status` mostra `Listening yes`. Se as variáveis de ambiente estiverem ausentes, a imagem do sandbox pode estar rodando um entrypoint que as remove — verifique sua configuração de `docker_env`.

### O sandbox recebe `HTTP 401` dos upstreams {#sandbox-sees-http-401-from-upstreams}

Duas causas comuns:

1. **Sobrescrita de token em um novo setup.** Você rodou `hermes egress setup --rotate-tokens` (ou rotacionou os tokens de outra forma) e os sandboxes em execução ainda mantêm os tokens antigos. Reinicie os sandboxes.
2. **Falha silenciosa na atualização do Bitwarden.** Não deveria acontecer com o novo comportamento de falha ruidosa, mas se você tiver `proxy.allow_env_fallback: true` definido, o daemon pode ter iniciado com valores de ambiente desatualizados. Verifique o ambiente do daemon (`/proc/<iron-proxy-pid>/environ`) para o `OPENROUTER_API_KEY` esperado, etc.

### "Address in use" depois que o processo pai morreu

O processo pai do Hermes morreu durante `hermes egress start` (Ctrl-C durante a sondagem de escuta, OOM, panic). A nova lógica de correção escreve o pidfile imediatamente após o `Popen`, para que o órfão seja recuperável:

```bash
hermes egress stop   # finds the orphan via the pidfile, kills it
hermes egress start
```

Se `hermes egress stop` disser "iron-proxy was not running", mas você ainda conseguir ver o daemon no `ps`, o pidfile ficou dessincronizado. Recuperação manual:

```bash
pkill -TERM iron-proxy
rm -f ~/.hermes/proxy/iron-proxy.pid ~/.hermes/proxy/iron-proxy.nonce
hermes egress start
```

### Inspecionando o comportamento por requisição {#inspecting-per-request-behavior}

Na versão de binário fixada (**v0.39**), tanto os eventos em nível de daemon quanto os registros por requisição vão para `~/.hermes/proxy/iron-proxy.log`. O formato é JSON delimitado por linha. Filtre por um upstream específico:

```bash
grep '"upstream":"openrouter.ai"' ~/.hermes/proxy/iron-proxy.log | tail -20
```

Ou acompanhe em tempo real:

```bash
tail -f ~/.hermes/proxy/iron-proxy.log | jq
```

Quando a versão fixada avançar para v0.40+ (que adiciona `log.audit_path`), os registros por requisição passarão para `~/.hermes/proxy/audit.log`, e `iron-proxy.log` conterá apenas eventos em nível de daemon. Até essa atualização, `audit.log` é um placeholder vazio (pré-criado com `0o600` para que o futuro daemon herde permissões restritas) — aponte hoje suas ferramentas de logrotate/monitoramento para `iron-proxy.log`, e planeje adicionar `audit.log` depois da atualização de versão.

## Limitações (v1) {#limitations-v1}

- Apenas backend Docker. A integração com Modal, Daytona e SSH virá em PRs separados.
- Provedores com autenticação baseada em assinatura (AWS SigV4, OAuth de service account do GCP) contornam o proxy completamente — veja [Provedores não cobertos](#uncovered-providers). Provedores baseados em token de cabeçalho (bearer, `x-api-key`, `api-key`, `x-goog-api-key`) são todos cobertos.
- Não há binário nativo para Windows upstream. Rode em Linux/macOS/WSL.
- A CA é um certificado autoassinado de 10 anos na primeira geração. A rotação exige `openssl genrsa ...` manualmente (ou aguarde um follow-up que adicione `hermes egress rotate-ca`).
- Rodar o setup novamente interrompe um daemon em execução depois de reescrever a configuração ou os mapeamentos; reinicie-o (ou use `hermes egress reload` para mudanças apenas de regras) e reinicie os sandboxes já em execução após a rotação de tokens.
- A zeragem de segredos em memória do iron-proxy é controlada pelo upstream. Atacantes com o mesmo uid e acesso de leitura a `/proc/<pid>/mem` conseguem ler segredos trocados da memória do daemon.
- O iron-proxy v0.39 suporta apenas **um bind por daemon** (fazemos bind no gateway da bridge do docker no Linux, em loopback no Docker Desktop) e combina os registros de daemon e por requisição em um único fluxo de log. Quando o upstream adicionar `proxy.http_listens` (plural) e `log.audit_path`, uma atualização de versão poderá habilitar múltiplos binds e o fluxo de auditoria dedicado.

## Veja também {#see-also}

- Projeto upstream: [github.com/ironsh/iron-proxy](https://github.com/ironsh/iron-proxy)
- Documentação upstream: [docs.iron.sh](https://docs.iron.sh/)
- Integração com Bitwarden: [`hermes secrets bitwarden`](../secrets/bitwarden)
- Backend de terminal Docker do Hermes: [Docker](../docker)
- Referência para desenvolvedores/colaboradores: [Internals do proxy de egresso](../../developer-guide/egress-internals)
