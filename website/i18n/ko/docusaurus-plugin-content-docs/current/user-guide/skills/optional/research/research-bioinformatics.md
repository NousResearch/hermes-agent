---
title: "Bioinformatics — bioSkills 및 ClawBio의 400개 이상의 생물정보학 스킬에 대한 관문"
sidebar_label: "Bioinformatics"
description: "bioSkills 및 ClawBio의 400개 이상의 생물정보학 스킬에 대한 관문"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Bioinformatics

bioSkills 및 ClawBio의 400개 이상의 생물정보학 스킬에 대한 관문입니다. 유전체학(genomics), 전사체학(transcriptomics), 단일 세포(single-cell), 변이 호출(variant calling), 약물유전체학(pharmacogenomics), 메타유전체학(metagenomics), 구조 생물학(structural biology) 등을 다룹니다. 도메인 특정 참조 자료를 필요에 따라 가져옵니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — install with `hermes skills install official/research/bioinformatics` |
| Path | `optional-skills/research/bioinformatics` |
| Version | `1.0.0` |
| Platforms | linux, macos |
| Tags | `bioinformatics`, `genomics`, `sequencing`, `biology`, `research`, `science` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Bioinformatics Skills Gateway

생물정보학, 유전체학, 시퀀싱, 변이 호출, 유전자 발현, 단일 세포 분석, 단백질 구조, 약물유전체학, 메타유전체학, 계통발생학 또는 기타 계산 생물학 작업에 대해 질문을 받았을 때 사용합니다.

이 스킬은 두 가지 오픈소스 생물정보학 스킬 라이브러리로 가는 관문입니다. 수백 개의 도메인 특정 스킬을 번들로 제공하는 대신 색인화하여 필요에 따라 스킬을 가져옵니다.

## Sources

◆ **bioSkills** — 385개의 참조 스킬 (코드 패턴, 매개변수 가이드, 결정 트리)
  Repo: https://github.com/GPTomics/bioSkills
  Format: 코드 예제가 포함된 주제별 SKILL.md. Python/R/CLI.

◆ **ClawBio** — 33개의 실행 가능한 파이프라인 스킬 (실행 가능한 스크립트, 재현성 번들)
  Repo: https://github.com/ClawBio/ClawBio
  Format: 데모가 포함된 Python 스크립트. 각 분석은 report.md + commands.sh + environment.yml을 내보냅니다.

## How to fetch and use a skill

1. 아래 색인에서 도메인과 스킬 이름을 식별합니다.
2. 관련 리포지토리를 복제합니다 (시간 절약을 위해 얕은 복제 사용):
   ```bash
   # bioSkills (참조 자료)
   git clone --depth 1 https://github.com/GPTomics/bioSkills.git /tmp/bioSkills

   # ClawBio (실행 가능한 파이프라인)
   git clone --depth 1 https://github.com/ClawBio/ClawBio.git /tmp/ClawBio
   ```
3. 특정 스킬을 읽습니다:
   ```bash
   # bioSkills — 각 스킬의 위치: <category>/<skill-name>/SKILL.md
   cat /tmp/bioSkills/variant-calling/gatk-variant-calling/SKILL.md

   # ClawBio — 각 스킬의 위치: skills/<skill-name>/
   cat /tmp/ClawBio/skills/pharmgx-reporter/README.md
   ```
4. 가져온 스킬을 참조 자료로 따릅니다. 이것들은 Hermes 형식의 스킬이 아닙니다 — 전문가 도메인 가이드로 취급하십시오. 여기에는 올바른 매개변수, 적절한 도구 플래그 및 검증된 파이프라인이 포함되어 있습니다.

## Skill Index by Domain

### Sequence Fundamentals
bioSkills:
  sequence-io/ — read-sequences, write-sequences, format-conversion, batch-processing, compressed-files, fastq-quality, filter-sequences, paired-end-fastq, sequence-statistics
  sequence-manipulation/ — seq-objects, reverse-complement, transcription-translation, motif-search, codon-usage, sequence-properties, sequence-slicing
ClawBio:
  seq-wrangler — 서열 QC, 정렬 및 BAM 처리 (FastQC, BWA, SAMtools 래핑)

### Read QC & Alignment
bioSkills:
  read-qc/ — quality-reports, fastp-workflow, adapter-trimming, quality-filtering, umi-processing, contamination-screening, rnaseq-qc
  read-alignment/ — bwa-alignment, star-alignment, hisat2-alignment, bowtie2-alignment
  alignment-files/ — sam-bam-basics, alignment-sorting, alignment-filtering, bam-statistics, duplicate-handling, pileup-generation

### Variant Calling & Annotation
bioSkills:
  variant-calling/ — gatk-variant-calling, deepvariant, variant-calling (bcftools), joint-calling, structural-variant-calling, filtering-best-practices, variant-annotation, variant-normalization, vcf-basics, vcf-manipulation, vcf-statistics, consensus-sequences, clinical-interpretation
ClawBio:
  vcf-annotator — 혈통을 고려한 VEP + ClinVar + gnomAD 주석(annotation)
  variant-annotation — 변이 주석 파이프라인

### Differential Expression (Bulk RNA-seq)
bioSkills:
  differential-expression/ — deseq2-basics, edger-basics, batch-correction, de-results, de-visualization, timeseries-de
  rna-quantification/ — alignment-free-quant (Salmon/kallisto), featurecounts-counting, tximport-workflow, count-matrix-qc
  expression-matrix/ — counts-ingest, gene-id-mapping, metadata-joins, sparse-handling
ClawBio:
  rnaseq-de — QC, 정규화 및 시각화가 포함된 전체 발현 차이 분석(DE) 파이프라인
  diff-visualizer — DE 결과를 위한 풍부한 시각화 및 리포팅

### Single-Cell RNA-seq
bioSkills:
  single-cell/ — preprocessing, clustering, batch-integration, cell-annotation, cell-communication, doublet-detection, markers-annotation, trajectory-inference, multimodal-integration, perturb-seq, scatac-analysis, lineage-tracing, metabolite-communication, data-io
ClawBio:
  scrna-orchestrator — 전체 Scanpy 파이프라인 (QC, 군집화, 마커, 주석)
  scrna-embedding — scVI 기반 잠재 임베딩 및 배치 통합

### Spatial Transcriptomics
bioSkills:
  spatial-transcriptomics/ — spatial-data-io, spatial-preprocessing, spatial-domains, spatial-deconvolution, spatial-communication, spatial-neighbors, spatial-statistics, spatial-visualization, spatial-multiomics, spatial-proteomics, image-analysis

### Epigenomics
bioSkills:
  chip-seq/ — peak-calling, differential-binding, motif-analysis, peak-annotation, chipseq-qc, chipseq-visualization, super-enhancers
  atac-seq/ — atac-peak-calling, atac-qc, differential-accessibility, footprinting, motif-deviation, nucleosome-positioning
  methylation-analysis/ — bismark-alignment, methylation-calling, dmr-detection, methylkit-analysis
  hi-c-analysis/ — hic-data-io, tad-detection, loop-calling, compartment-analysis, contact-pairs, matrix-operations, hic-visualization, hic-differential
ClawBio:
  methylation-clock — 후성유전학적 연령 추정

### Pharmacogenomics & Clinical
bioSkills:
  clinical-databases/ — clinvar-lookup, gnomad-frequencies, dbsnp-queries, pharmacogenomics, polygenic-risk, hla-typing, variant-prioritization, somatic-signatures, tumor-mutational-burden, myvariant-queries
ClawBio:
  pharmgx-reporter — 23andMe/AncestryDNA를 통한 PGx 리포트 (유전자 12개, SNP 31개, 약물 51개)
  drug-photo — 투약 사진 → 맞춤형 PGx 복용량 카드 (비전 사용)
  clinpgx — 유전자-약물 데이터 및 CPIC 가이드라인을 위한 ClinPGx API
  gwas-lookup — 9개의 유전체 데이터베이스 전반에 걸친 연합 변이 조회
  gwas-prs — 소비자 유전자 데이터를 사용한 다유전자 위험 점수(PRS)
  nutrigx_advisor — 소비자 유전자 데이터를 사용한 맞춤형 영양 조언

### Population Genetics & GWAS
bioSkills:
  population-genetics/ — association-testing (PLINK GWAS), plink-basics, population-structure, linkage-disequilibrium, scikit-allel-analysis, selection-statistics
  causal-genomics/ — mendelian-randomization, fine-mapping, colocalization-analysis, mediation-analysis, pleiotropy-detection
  phasing-imputation/ — haplotype-phasing, genotype-imputation, imputation-qc, reference-panels
ClawBio:
  claw-ancestry-pca — SGDP 참조 패널을 사용한 혈통 PCA

### Metagenomics & Microbiome
bioSkills:
  metagenomics/ — kraken-classification, metaphlan-profiling, abundance-estimation, functional-profiling, amr-detection, strain-tracking, metagenome-visualization
  microbiome/ — amplicon-processing, diversity-analysis, differential-abundance, taxonomy-assignment, functional-prediction, qiime2-workflow
ClawBio:
  claw-metagenomics — 샷건 메타유전체학 프로파일링 (분류, 내성체, 기능 경로)

### Genome Assembly & Annotation
bioSkills:
  genome-assembly/ — hifi-assembly, long-read-assembly, short-read-assembly, metagenome-assembly, assembly-polishing, assembly-qc, scaffolding, contamination-detection
  genome-annotation/ — eukaryotic-gene-prediction, prokaryotic-annotation, functional-annotation, ncrna-annotation, repeat-annotation, annotation-transfer
  long-read-sequencing/ — basecalling, long-read-alignment, long-read-qc, clair3-variants, structural-variants, medaka-polishing, nanopore-methylation, isoseq-analysis

### Structural Biology & Chemoinformatics
bioSkills:
  structural-biology/ — alphafold-predictions, modern-structure-prediction, structure-io, structure-navigation, structure-modification, geometric-analysis
  chemoinformatics/ — molecular-io, molecular-descriptors, similarity-searching, substructure-search, virtual-screening, admet-prediction, reaction-enumeration
ClawBio:
  struct-predictor — 로컬 AlphaFold/Boltz/Chai 구조 예측 및 비교

### Proteomics
bioSkills:
  proteomics/ — data-import, peptide-identification, protein-inference, quantification, differential-abundance, dia-analysis, ptm-analysis, proteomics-qc, spectral-libraries
ClawBio:
  proteomics-de — 단백질체학 발현 차이 분석

### Pathway Analysis & Gene Networks
bioSkills:
  pathway-analysis/ — go-enrichment, gsea, kegg-pathways, reactome-pathways, wikipathways, enrichment-visualization
  gene-regulatory-networks/ — scenic-regulons, coexpression-networks, differential-networks, multiomics-grn, perturbation-simulation

### Immunoinformatics
bioSkills:
  immunoinformatics/ — mhc-binding-prediction, epitope-prediction, neoantigen-prediction, immunogenicity-scoring, tcr-epitope-binding
  tcr-bcr-analysis/ — mixcr-analysis, scirpy-analysis, immcantation-analysis, repertoire-visualization, vdjtools-analysis

### CRISPR & Genome Engineering
bioSkills:
  crispr-screens/ — mageck-analysis, jacks-analysis, hit-calling, screen-qc, library-design, crispresso-editing, base-editing-analysis, batch-correction
  genome-engineering/ — grna-design, off-target-prediction, hdr-template-design, base-editing-design, prime-editing-design

### Workflow Management
bioSkills:
  workflow-management/ — snakemake-workflows, nextflow-pipelines, cwl-workflows, wdl-workflows
ClawBio:
  repro-enforcer — 모든 분석을 재현성 번들(Conda env + Singularity + 체크섬)로 내보냅니다.
  galaxy-bridge — usegalaxy.org에서 8,000개 이상의 Galaxy 도구에 액세스합니다.

### Specialized Domains
bioSkills:
  alternative-splicing/ — splicing-quantification, differential-splicing, isoform-switching, sashimi-plots, single-cell-splicing, splicing-qc
  ecological-genomics/ — edna-metabarcoding, landscape-genomics, conservation-genetics, biodiversity-metrics, community-ecology, species-delimitation
  epidemiological-genomics/ — pathogen-typing, variant-surveillance, phylodynamics, transmission-inference, amr-surveillance
  liquid-biopsy/ — cfdna-preprocessing, ctdna-mutation-detection, fragment-analysis, tumor-fraction-estimation, methylation-based-detection, longitudinal-monitoring
  epitranscriptomics/ — m6a-peak-calling, m6a-differential, m6anet-analysis, merip-preprocessing, modification-visualization
  metabolomics/ — xcms-preprocessing, metabolite-annotation, normalization-qc, statistical-analysis, pathway-mapping, lipidomics, targeted-analysis, msdial-preprocessing
  flow-cytometry/ — fcs-handling, gating-analysis, compensation-transformation, clustering-phenotyping, differential-analysis, cytometry-qc, doublet-detection, bead-normalization
  systems-biology/ — flux-balance-analysis, metabolic-reconstruction, gene-essentiality, context-specific-models, model-curation
  rna-structure/ — secondary-structure-prediction, ncrna-search, structure-probing

### Data Visualization & Reporting
bioSkills:
  data-visualization/ — ggplot2-fundamentals, heatmaps-clustering, volcano-customization, circos-plots, genome-browser-tracks, interactive-visualization, multipanel-figures, network-visualization, upset-plots, color-palettes, specialized-omics-plots, genome-tracks
  reporting/ — rmarkdown-reports, quarto-reports, jupyter-reports, automated-qc-reports, figure-export
ClawBio:
  profile-report — 분석 프로파일 리포팅
  data-extractor — 과학 그림 이미지에서 수치 데이터 추출 (비전 사용)
  lit-synthesizer — PubMed/bioRxiv 검색, 요약, 인용 그래프
  pubmed-summariser — 구조화된 브리핑이 포함된 유전자/질환 PubMed 검색

### Database Access
bioSkills:
  database-access/ — entrez-search, entrez-fetch, entrez-link, blast-searches, local-blast, sra-data, geo-data, uniprot-access, batch-downloads, interaction-databases, sequence-similarity
ClawBio:
  ukb-navigator — 12,000개 이상의 UK Biobank 필드 전반에 걸친 의미론적 검색
  clinical-trial-finder — 임상 시험 검색

### Experimental Design
bioSkills:
  experimental-design/ — power-analysis, sample-size, batch-design, multiple-testing

### Machine Learning for Omics
bioSkills:
  machine-learning/ — omics-classifiers, biomarker-discovery, survival-analysis, model-validation, prediction-explanation, atlas-mapping
ClawBio:
  claw-semantic-sim — 질환 문헌을 위한 의미론적 유사도 인덱스 (PubMedBERT)
  omics-target-evidence-mapper — 오믹스 소스 전반의 타깃 수준 증거를 집계합니다.

## Environment Setup

이 스킬들은 생물정보학 워크스테이션을 가정합니다. 공통 의존성:

```bash
# Python
pip install biopython pysam cyvcf2 pybedtools pyBigWig scikit-allel anndata scanpy mygene

# R/Bioconductor
Rscript -e 'BiocManager::install(c("DESeq2","edgeR","Seurat","clusterProfiler","methylKit"))'

# CLI 도구 (Ubuntu/Debian)
sudo apt install samtools bcftools ncbi-blast+ minimap2 bedtools

# CLI 도구 (macOS)
brew install samtools bcftools blast minimap2 bedtools

# 또는 Conda를 통해 (재현성을 위해 권장됨)
conda install -c bioconda samtools bcftools blast minimap2 bedtools fastp kraken2
```

## Pitfalls

- 가져온 스킬은 Hermes SKILL.md 형식이 아닙니다. 스킬들은 고유한 구조를 사용합니다 (bioSkills: 코드 패턴 쿡북, ClawBio: README + Python 스크립트). 전문가 참조 자료로 읽으십시오.
- bioSkills는 참조 가이드입니다 — 올바른 매개변수와 코드 패턴을 보여주지만 실행 가능한 파이프라인은 아닙니다.
- ClawBio 스킬은 실행 가능합니다 — 대부분 `--demo` 플래그가 있으며 직접 실행할 수 있습니다.
- 두 리포지토리 모두 생물정보학 도구가 설치되어 있다고 가정합니다. 파이프라인을 실행하기 전에 전제 조건을 확인하세요.
- ClawBio의 경우 먼저 복제된 리포지토리에서 `pip install -r requirements.txt`를 실행하세요.
- 유전체 데이터 파일은 매우 클 수 있습니다. 참조 유전체, SRA 데이터셋을 다운로드하거나 인덱스를 빌드할 때 디스크 공간에 유의하세요.
