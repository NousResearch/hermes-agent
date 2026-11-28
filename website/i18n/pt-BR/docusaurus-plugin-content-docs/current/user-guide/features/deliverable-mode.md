---
title: Modo entregável (artefatos no chat)
sidebar_label: Modo entregável
description: Como o agente envia gráficos, PDFs, planilhas e outros arquivos gerados como anexos nativos nas plataformas de mensagens.
---

# Modo entregável {#deliverable-mode}

Quando o Hermes Agent roda dentro de um gateway de mensagens (Slack, Discord, Telegram,
WhatsApp, Signal, etc.), ele pode entregar arquivos gerados diretamente no
chat — não como paths que você precisa copiar, mas como anexos nativos.

Um gráfico aparece como imagem inline. Um relatório PDF aparece como download de
arquivo. Uma planilha sobe como `.xlsx`. O agente não precisa
escrever uma tag `MEDIA:` nem fazer nada especial — ele só gera o arquivo
e menciona o path absoluto na resposta. O gateway extrai o path
do texto, remove-o da mensagem visível e faz upload do
arquivo nativamente.

## Como funciona {#how-it-works}

Três peças se encaixam:

1. **O agente tem ferramentas que produzem arquivos.** `execute_code` para gráficos via
   matplotlib, a skill `docx` para documentos Word, a skill `xlsx` para
   planilhas, as skills `pdf` e `latex-pdf-report` para PDFs, a
   skill `powerpoint` para apresentações, `image_generate` para imagens,
   `text_to_speech` para áudio, e assim por diante.

2. **O gateway varre as respostas do agente em busca de paths de arquivo.** Qualquer path absoluto
   (`/tmp/...`) ou path relativo ao home (`~/...`) terminando em uma extensão
   suportada é extraído. Paths dentro de blocos de código e código inline são
   ignorados para que amostras de código nunca sejam mutiladas.

3. **O gateway despacha por tipo de arquivo.** Imagens são embutidas inline onde a
   plataforma suporta; vídeos são embutidos inline; áudio vai para anexos de voz/áudio;
   todo o resto sobe como anexo de arquivo.

## Extensões de arquivo suportadas {#supported-file-extensions}

| Categoria | Extensões | Entrega |
|---|---|---|
| Imagens | `.png .jpg .jpeg .gif .webp .bmp .tiff .svg` | Embed inline |
| Vídeo | `.mp4 .mov .avi .mkv .webm` | Embed inline (onde suportado) |
| Áudio | `.mp3 .wav .ogg .m4a .flac` | Anexo de voz/áudio |
| Documentos | `.pdf .docx .doc .odt .rtf .txt .md` | Upload de arquivo |
| Dados | `.xlsx .xls .csv .tsv .json .xml .yaml .yml` | Upload de arquivo |
| Apresentações | `.pptx .ppt .odp` | Upload de arquivo |
| Arquivos | `.zip .tar .gz .tgz .bz2 .7z` | Upload de arquivo |
| Web | `.html .htm` | Upload de arquivo |

Extensões como `.py`, `.log` e outros arquivos-fonte são intencionalmente excluídas para que
o agente não envie automaticamente arquivos-fonte arbitrários; se quiser enviar código
ao usuário, use um bloco de código.

## Incentivando o agente a produzir artefatos {#encouraging-the-agent-to-produce-artifacts}

O agente não recorre a artefatos por padrão — ele precisa saber que deve.
Duas formas de orientá-lo:

**Por sessão:** peça explicitamente ("me envie a comparação como gráfico",
"devolva os dados como CSV") ou escreva suas próprias instruções personalizadas /
entrada de personalidade que incline respostas no estilo artefato em
plataformas de mensagens.

**Por projeto:** adicione o viés em `AGENTS.md` / `CLAUDE.md` /
`.cursorrules` em um projeto do qual o agente trabalha, à sua persona global
em `~/.hermes/SOUL.md`, ou como preset nomeado em
`agent.personalities` em `~/.hermes/config.yaml` (alternável por sessão
via `/personality`).

A mecânica que o agente precisa usar é simples: renderizar o arquivo em um
path absoluto (ex.: `/tmp/q3-revenue.png`) e mencionar esse path como
texto simples na resposta. O gateway faz o resto. Paths dentro de
blocos de código cercados ou backticks são ignorados para que amostras de código nunca sejam
mutiladas.

## Kanban: artefatos vão nas notificações de conclusão {#kanban-artifacts-ride-completion-notifications}

Se você usa o fluxo multiagente kanban do Hermes, workers podem anexar
arquivos entregáveis à chamada `kanban_complete`:

```python
kanban_complete(
    summary="rendered Q3 revenue chart and report",
    artifacts=[
        "/tmp/q3-revenue.png",
        "/tmp/q3-report.pdf",
    ],
)
```

Quando o notificador do gateway entrega a mensagem "task completed" para quem
se inscreveu na tarefa no Slack/Telegram/etc., ele também faz upload de cada artefato
como anexo nativo naquele chat. O humano recebe o entregável e o
resumo no mesmo lugar.

Arquivos que não existem no disco quando o notificador roda são silenciosamente ignorados.

## Conectando mais serviços com MCP {#connecting-more-services-with-mcp}

Além do pipeline de entrega de artefatos, o agente pode alcançar outros
serviços via MCP (Model Context Protocol). O ecossistema MCP oferece
servidores da comunidade para a maioria das ferramentas populares — instale os que precisar:

| Serviço | O que desbloqueia |
|---|---|
| **Notion** | Ler/escrever páginas Notion, databases, consultar workspace |
| **GitHub** | Issues, PRs, comentários, busca em repo além do gh CLI |
| **Linear** | Tickets, projetos, cycles |
| **Slack** | Busca em todo o workspace, ler outros canais |
| **Gmail** | Triagem de inbox, enviar e-mail, gerenciamento de labels |
| **Salesforce** | Leads, oportunidades, dados de conta |
| **Snowflake / BigQuery** | SQL contra data warehouses |
| **Google Drive** | Busca de arquivos, conteúdos, gerenciamento de compartilhamento |

Instale servidores MCP via `~/.hermes/config.yaml` na seção `mcp_servers`.
Veja [Integração MCP](./mcp.md) para o guia completo de configuração.

## Comparação com Perplexity Computer no Slack {#comparison-to-perplexity-computer-in-slack}

A integração Slack do Perplexity Computer é construída em torno da mesma ideia:
o agente gera um entregável (gráfico, PDF, slide deck) e publica de volta
na thread como anexo nativo. O modo entregável do Hermes Agent oferece o
mesmo padrão voltado ao usuário localmente:

- A geração acontece no venv / sandbox do próprio usuário (sem tenant remoto).
- Arquivos chegam ao chat via a mesma API Slack `files.uploadV2`.
- A amplitude de conectores vem via MCP em vez de um catálogo curado de 400
  integrações hospedadas — instale as que você realmente usa.

Tokens OAuth ficam na máquina do usuário em `auth.json` / `.env`. Sem armazenamento
de token hospedado. Sem microVM multi-tenant. Mesmo resultado final.
