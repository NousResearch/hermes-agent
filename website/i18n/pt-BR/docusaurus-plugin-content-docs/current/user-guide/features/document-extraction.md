---
sidebar_position: 3
title: "Extração de Documentos"
description: "Como o read_file converte PDFs, documentos do Office e notebooks em texto — e o que fazer quando um PDF é feito de imagens digitalizadas"
---

# Extração de Documentos {#document-extraction}

A ferramenta `read_file` converte automaticamente formatos comuns de documento em texto legível, para que o agente possa inspecionar um PDF ou planilha da mesma forma que lê código-fonte.

## Formatos suportados {#supported-formats}

| Formato | Extensões | Conversor | Disponibilidade |
|--------|-----------|-----------|--------------|
| Notebooks Jupyter | `.ipynb` | Embutido (stdlib) | Sempre |
| Documentos Word | `.docx` | Embutido (stdlib) | Sempre |
| Planilhas Excel | `.xlsx` | Embutido (stdlib) | Sempre |
| PDF | `.pdf` | Conversor opcional `anydoc` | Autoinstalado no primeiro uso* |
| Office legado | `.doc`, `.ppt`, `.xls`, `.pptx` e variantes | Conversor opcional `anydoc` | Autoinstalado no primeiro uso* |
| OpenDocument | `.odt`, `.ods`, `.odp` | Conversor opcional `anydoc` | Autoinstalado no primeiro uso* |
| Rich text / eBooks | `.rtf`, `.epub` | Conversor opcional `anydoc` | Autoinstalado no primeiro uso* |

\* O conversor opcional é o pacote `firecrawl-anydoc`, instalado sob demanda onde instalações são permitidas (`security.allow_lazy_installs` em `config.yaml`). Sem ele, os três formatos da stdlib continuam funcionando; os demais caem na proteção de arquivo binário.

A saída da conversão é Markdown, paginada pela janela normal de `offset`/`limit` do `read_file`. Documentos com mais de 50 MB são recusados para manter as turnos das ferramentas limitados.

A extração funciona com backends de terminal remotos (Docker, Modal, SSH): os bytes do arquivo são transferidos pela fronteira do backend e convertidos no host, então um documento dentro de um sandbox lê igual a um local.

## PDFs digitalizados: o aviso de cobertura {#scanned-pdfs-the-coverage-warning}

A conversão de PDF lê **somente a camada de texto**. Páginas que são imagens digitalizadas — comuns em documentos jurídicos, pacotes de revenda, contratos assinados, faxes — não têm camada de texto e silenciosamente convertem para nada. A assinatura típica são cabeçalhos de seção com corpos vazios.

Quando uma parcela significativa de páginas não produz texto (mais de 20% do documento, ou 10+ páginas em absoluto), o `read_file` antepõe um aviso à extração. Cada lacuna ilegível é rotulada com o último texto extraído antes dela — em geral um divisor de seção — para que o agente possa mirar só as lacunas de que realmente precisa, em vez de fazer OCR do documento inteiro:

```
[EXTRACTION COVERAGE WARNING: 198 of 311 pages in this PDF yielded no
text. ... Unreadable gaps, each labeled with the last text extracted
before it:
  pages 42-77 (36 pages) — after "Antigua Maintenance Corp Bylaws" (p41)
  pages 92-213 (122 pages) — after "... Covenants, Codes and Regulations" (p91)
  page 224 (1 page) — after "... Insurance Declaration Pages" (p223)
Decide which gaps you actually need — do NOT OCR or render everything. ...]
```

O aviso lista as faixas de páginas exatas e os caminhos de recuperação:

1. **Poucas páginas — renderizar + visão.** Converta as páginas em imagens e leia-as com a ferramenta de visão:
   ```bash
   pdftoppm -jpeg -r 150 -f 92 -l 94 document.pdf /tmp/page
   ```
   Depois inspecione cada imagem com `vision_analyze`. Zero dependências extras (o poppler já é necessário para a própria detecção).
2. **Muitas páginas — OCR.** A skill `ocr-and-documents` cobre OCR em lote com marker-pdf (90+ idiomas, lida com equações e tabelas; instalação de ~3–5 GB).

A detecção usa o `pdftotext` do poppler para contagens de texto por página. Se o poppler não estiver instalado, a extração ainda funciona — a verificação de cobertura é silenciosamente ignorada.

:::tip
O agente lida com o aviso sozinho — ele oferece renderizar ou fazer OCR das páginas faltantes. Se você estiver lendo as extrações por conta própria, trate "cabeçalho com corpo vazio" como uma seção digitalizada, não como uma seção ausente.
:::
