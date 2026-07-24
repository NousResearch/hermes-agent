# Segredos

O Hermes pode obter chaves de API de gerenciadores de segredos externos na inicialização do processo, em vez de armazená-las em `~/.hermes/.env`. O token de bootstrap do gerenciador de segredos fica em `.env`; todas as outras chaves de provedor (OpenAI, Anthropic, OpenRouter, etc.) podem permanecer no gerenciador e ser rotacionadas centralmente.

Suportados:

- [Bitwarden Secrets Manager](./bitwarden) — CLI `bws`, instalado sob demanda, o plano gratuito funciona.
- [1Password](./onepassword) — referências `op://` via o CLI oficial `op`; autenticação por service account ou sessão desktop.

## Várias fontes ao mesmo tempo

Você pode habilitar mais de uma fonte de segredos simultaneamente — por exemplo, um projeto Bitwarden de equipe junto com um plugin de cofre pessoal. As fontes se compõem por variável de ambiente com uma escada de precedência determinística:

1. **Seu `.env` / shell prevalece por padrão.** Uma fonte só substitui um valor pré-existente quando seu próprio `override_existing: true` está definido (Bitwarden usa `true` por padrão para que a rotação central funcione).
2. **Fontes mapeadas vencem fontes em massa.** Uma fonte em que você vincula explicitamente variáveis de ambiente a referências (um mapa `env:`) supera uma fonte que injeta implicitamente um projeto inteiro de segredos, independentemente da ordem.
3. **A primeira fonte vence.** Dentro da mesma forma, a ordem da lista opcional `secrets.sources` (ou ordem de registro) decide. Reivindicações posteriores sobre uma variável já reivindicada são ignoradas — com um aviso na inicialização, nunca em silêncio.

`override_existing` nunca permite que uma fonte sobrescreva uma variável que outra fonte já reivindicou, e nenhuma fonte pode sobrescrever o token de bootstrap de outra fonte (por exemplo, `BWS_ACCESS_TOKEN`).

```yaml
secrets:
  sources: [bitwarden]     # ordenação explícita opcional
  bitwarden:
    enabled: true
    project_id: "..."
```

Toda credencial injetada por uma fonte é rotulada com sua origem — fluxos de setup e `hermes model` mostram `(from Bitwarden)` ao lado das chaves detectadas, para que você sempre saiba de onde veio um valor.

## Adicionando seu próprio backend

Gerenciadores de segredos de terceiros são distribuídos como plugins standalone, não como PRs no core. Um backend estende `agent.secret_sources.base.SecretSource` (um método obrigatório: `fetch(cfg, home_path) -> FetchResult`) e se registra via `ctx.register_secret_source(MySource())` no `register(ctx)` do plugin. O orquestrador cuida de precedência, tratamento de conflitos, timeouts e proveniência — sua fonte só busca. Guia completo com regras de contrato, helper de segurança para subprocessos e kit de conformidade: [Building a Secret Source Plugin](/developer-guide/secret-source-plugin).

O conjunto embutido é deliberadamente fechado (mesma política dos provedores de memória): Bitwarden e 1Password vêm na árvore. Todo o resto — Infisical, Proton Pass, HashiCorp Vault, AWS Secrets Manager, keystores do SO — pertence a repositórios de plugins; compartilhe-os no Discord da Nous Research (`#plugins-skills-and-skins`).
