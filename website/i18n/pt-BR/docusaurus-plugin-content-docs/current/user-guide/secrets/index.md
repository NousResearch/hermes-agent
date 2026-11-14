# Secrets {#secrets}

O Hermes pode obter chaves de API de gerenciadores de secrets externos na inicialização do processo, em vez de armazená-las em `~/.hermes/.env`. O token de bootstrap do gerenciador de secrets fica no `.env`; todas as outras chaves de provider (OpenAI, Anthropic, OpenRouter, etc.) podem permanecer no gerenciador e ser rotacionadas centralmente.

Suportados:

- [Bitwarden Secrets Manager](./bitwarden) — CLI `bws`, instalada sob demanda, tier gratuito funciona.
- [1Password](./onepassword) — referências `op://` via a CLI oficial `op`; autenticação por service account ou sessão desktop.
- [Command helper](./command) — qualquer vault CLI (`keepassxc-cli`, `secret-tool`, `pass`, scripts customizados) via helper configurado pelo usuário que imprime linhas `KEY=VALUE`.

## Várias fontes ao mesmo tempo {#multiple-sources-at-once}

Você pode habilitar mais de uma fonte de secrets simultaneamente — por exemplo, um projeto Bitwarden de equipe junto com um plugin de vault pessoal. As fontes se compõem por variável de ambiente com uma escada de precedência determinística:

1. **Seu `.env` / shell vence por padrão.** Uma fonte só substitui um valor pré-existente quando seu próprio `override_existing: true` está definido (Bitwarden usa `true` por padrão para rotação central funcionar).
2. **Fontes mapeadas vencem fontes em massa.** Uma fonte em que você vincula explicitamente env vars a referências (mapa `env:`) supera uma fonte que injeta um projeto inteiro de secrets implicitamente, independentemente da ordem.
3. **Primeira fonte vence.** Dentro da mesma forma, a ordem da lista opcional `secrets.sources` (ou ordem de registro) decide. Reivindicações posteriores em uma var já reivindicada são ignoradas — com aviso na inicialização, nunca silenciosamente.

`override_existing` nunca deixa uma fonte sobrescrever uma var que outra fonte já reivindicou, e nenhuma fonte pode sobrescrever o token de bootstrap de outra fonte (ex.: `BWS_ACCESS_TOKEN`).

```yaml
secrets:
  sources: [bitwarden]     # optional explicit ordering
  bitwarden:
    enabled: true
    project_id: "..."
```

Toda credencial injetada por uma fonte é rotulada com sua origem — fluxos de setup e `hermes model` mostram `(from Bitwarden)` ao lado de chaves detectadas para você sempre saber de onde veio o valor.

## Perfis e vaults compartilhados {#profiles-and-shared-vaults}

Dois controles no nível do orquestrador tornam um vault compartilhado seguro entre [perfis](../profiles):

- **`secrets.preserve_existing`** — lista de nomes de env vars cujo valor existente no `.env` / shell sempre vence, mesmo contra uma fonte com `override_existing: true`. Use para secrets de plataforma por perfil (ex.: `FEISHU_APP_SECRET`) que diferem intencionalmente entre perfis enquanto todo o resto rotaciona centralmente:

  ```yaml
  secrets:
    preserve_existing: [FEISHU_APP_SECRET, TELEGRAM_BOT_TOKEN]
  ```

- **Alias de perfil** (ativado por padrão, `secrets.profile_alias: false` para desabilitar) — quando o Hermes roda sob um perfil nomeado, um secret no vault chamado `FOO_<PROFILE>` (sufixos apenas no formato de credencial: `*_API_KEY`, `*_TOKEN`, `*_SECRET`, `*_KEY`, `*_PASSWORD`) também hidrata o `FOO` canônico. Armazene `TELEGRAM_BOT_TOKEN_MILLA` no projeto compartilhado e os adaptadores do perfil `milla` — que leem o nome fixo `TELEGRAM_BOT_TOKEN` — recebem o valor correto automaticamente. Uma var que o vault fornece diretamente sob o nome canônico sempre vence um alias.

Ambos se aplicam a toda fonte — bundled e plugin — porque vivem no orquestrador, não nos backends.

## Adicionando seu próprio backend {#adding-your-own-backend}

Gerenciadores de secrets de terceiros são enviados como plugins standalone, não PRs no core. Um backend estende `agent.secret_sources.base.SecretSource` (um método obrigatório: `fetch(cfg, home_path) -> FetchResult`) e se registra via `ctx.register_secret_source(MySource())` no `register(ctx)` do plugin. O orquestrador cuida de precedência, tratamento de conflitos, timeouts e proveniência — sua fonte só busca. Guia completo com regras de contrato, helper de segurança para subprocessos e kit de conformidade: [Building a Secret Source Plugin](/developer-guide/secret-source-plugin).

O conjunto bundled é deliberadamente fechado (mesma política dos memory providers): Bitwarden e 1Password vêm na árvore. Todo o resto — Infisical, Proton Pass, HashiCorp Vault, AWS Secrets Manager, keystores do SO — pertence a repositórios de plugin; compartilhe no Discord da Nous Research (`#plugins-skills-and-skins`).
