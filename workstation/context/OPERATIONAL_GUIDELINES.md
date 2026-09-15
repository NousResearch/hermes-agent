# Diretrizes Operacionais para Tarefas Longas de Navegador — Hermes Workstation

> **Status:** Normativo / Canônico (V3.5+)
> **Aplica-se a:** Agentes autônomos, desenvolvedores, engenheiros de prompts e operadores do Hermes Work.
> **Origem:** Consolidação pós-teste de estresse de 86 itens (`H-050` / `ChatGPT-Análise de melhorias do Hermes Work-20260915-0015.md`).

---

## 1. Regra de Ouro da Extração (Batch Data Plane First)

> **Invariante:** Tarefas envolvendo repetição de mais de 5 itens (produtos de e-commerce, posts de redes sociais, linhas de tabela, resultados de busca, issues de repositórios) **NUNCA devem rodar em loops no prompt do LLM**.

1. **Anti-pattern proibido:**
   O agente fazer turnos de LLM no estilo `for item in items: call browser_navigate; call browser_snapshot; call browser_console; record to prompt`. Isso consome centenas de chamadas de ferramentas, inunda a janela de contexto com JSONs brutos, destrói o prompt caching e quebra sem recuperação em caso de falha de conexão.
2. **Padrão Obrigatório:**
   O agente deve acionar o plano de dados determinístico:
   - Usar a ferramenta nativa de alto nível `browser_extract_items(selector=..., limit=...)`.
   - Ou despachar um `WorkPlan` gerenciado pelo `DurableBatchRunner` (`workstation/batch_runner.py`).
   - O loop de iteração roda 100% no Python/runtime com persistência atômica transacional no SQLite canônico (`~/.hermes/kanban.db`).
   - O LLM recebe apenas um extrato executivo compacto (`to_compact_context()`), preservando o prompt cache.

---

## 2. Pre-flight Audit (Auditoria Prévia de Dependências)

> **Invariante:** Sempre audite o `RuntimeCapabilityRegistry` antes de iniciar uma tarefa que demande dependências externas do sistema ou do Python.

1. Se a instrução do usuário solicitar a geração de um arquivo Excel (`.xlsx`), o agente ou ferramenta deve verificar previamente se o módulo `openpyxl` está disponível via `RuntimeCapabilityRegistry.is_python_module_available("openpyxl")`.
2. Se a dependência não estiver instalada, o agente deve:
   - Instalar imediatamente no ambiente isolado (`pip install openpyxl`), ou
   - Alertar o usuário no início do turno, antes de gastar tempo coletando dados.
3. Isso evita a falha catastrófica de coletar dezenas de itens com sucesso e falhar no último segundo por ausência de biblioteca de formatação.

---

## 3. Respeito ao Lease Humano (Human Takeover Fail-Closed)

> **Invariante:** Ao encontrar barreiras de verificação humana (Cloudflare Turnstile, reCAPTCHA, autenticação 2FA, verificação por SMS, login com sessão expirada), o sistema deve adquirir um lease `HUMAN` no `BrowserControlLeaseManager` e solicitar a intervenção do usuário, mantendo as ferramentas do agente pausadas e proibidas de fechar a janela.

1. **Comportamento Fail-Closed:**
   Quando o lease `HUMAN` estiver ativo, qualquer chamada de ferramenta concorrente ou destrutiva (`close_preview`, `browser_navigate`, `click`) deve ser imediatamente rejeitada com `HumanTakeoverActiveError`.
2. **Preservação de Janela e Estado:**
   A viewport e o perfil de navegação (`%LOCALAPPDATA%/HermesWorkstation/Browser/User Data`) permanecem abertos e disponíveis para o usuário operar no Desktop. O agente não pode desanexar ou fechar o preview enquanto o usuário estiver autenticando.
3. **Retomada Explícita:**
   O controle só retorna ao agente quando o usuário clicar em "Devolver Controle ao Agente" no banner da UI ou quando o lease temporizado for explicitamente liberado (`releaseControl()`).

---

## 4. Desacoplamento de Artefatos (Artifact Store Offload)

> **Invariante:** Dados volumosos (JSONs de extração, listas de itens, logs de terminal, traces de execução) devem ser gravados diretamente no `ArtifactStore` com hash SHA-256 (`artifact://<sha256>`), retornando ao modelo apenas o resumo executivo e a referência.

1. **Zero Raw Payload no Prompt:**
   Nenhum JSON contendo dezenas de registros completos deve ser colocado como texto no corpo da mensagem da ferramenta.
2. **Representação por Referência:**
   A ferramenta deve gravar no disco via `ArtifactStore.store_json()` ou `browser_bridge.browser_save_json()` e retornar:
   ```json
   {
     "status": "SUCCESS",
     "total_extracted": 86,
     "anomalies_detected": 0,
     "artifact_ref": "artifact://4a8f9c1b...json",
     "sample_items": [ ...apenas 2 itens para inspeção... ]
   }
   ```
3. Se o usuário solicitar a exibição ou transformação dos dados, o agente utiliza ferramentas de leitura seletiva ou scripts de processamento apontando para o arquivo no disco.
