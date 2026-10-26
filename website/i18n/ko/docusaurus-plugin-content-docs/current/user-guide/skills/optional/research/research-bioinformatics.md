---
title: "생물정보학 — 400개 이상의 유전체학 및 계산생물학 스킬로 가는 관문"
sidebar_label: "생물정보학"
description: "400개 이상의 유전체학 및 계산생물학 스킬로 가는 관문"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# 생물정보학

400개 이상의 유전체학 및 계산생물학 스킬로 가는 관문입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/research/bioinformatics`로 설치 |
| 경로 | `optional-skills/research/bioinformatics` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos |
| 태그 | `bioinformatics`, `genomics`, `sequencing`, `biology`, `research`, `science` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침이기도 합니다.
:::

# 생물정보학 스킬 관문

생물정보학, 유전체학, 시퀀싱, 변이 호출, 유전자 발현, 단일세포 분석, 단백질 구조, 약물유전체학, 메타유전체학, 계통발생학 또는 모든 계산생물학 작업에 관한 질문을 받았을 때 사용합니다.

이 스킬은 두 개의 오픈 소스 생물정보학 스킬 라이브러리로 연결되는 관문입니다. 수백 개의 도메인별 스킬을 한데 묶는 대신, 필요한 항목을 색인화하고 필요할 때 가져옵니다.

## 출처

◆ **bioSkills** — 참고 스킬 385개(코드 패턴, 매개변수 가이드, 의사결정 트리)
  저장소: https://github.com/GPTomics/bioSkills
  형식: 주제별 SKILL.md와 코드 예제. Python/R/CLI.

◆ **ClawBio** — 실행 가능한 파이프라인 스킬 33개(실행 스크립트, 재현성 번들)
  저장소: https://github.com/ClawBio/ClawBio
  형식: 데모가 포함된 Python 스크립트. 각 분석은 report.md + commands.sh + environment.yml을 내보냅니다.

## 스킬을 가져와 사용하는 방법

1. 아래 색인에서 도메인과 스킬 이름을 식별합니다.
2. 관련 저장소를 클론합니다(시간을 절약하기 위해 얕은 클론 사용):
   ```bash
   # bioSkills (reference material)
   git clone --depth 1 https://github.com/GPTomics/bioSkills.git /tmp/bioSkills

   # ClawBio (runnable pipelines)
   git clone --depth 1 https://github.com/ClawBio/ClawBio.git /tmp/ClawBio
   ```
3. 특정 스킬을 읽습니다:
   ```bash
   # bioSkills — each skill is at: <category>/<skill-name>/SKILL.md
   cat /tmp/bioSkills/variant-calling/gatk-variant-calling/SKILL.md

   # ClawBio — each skill is at: skills/<skill-name>/
   cat /tmp/ClawBio/skills/pharmgx-reporter/README.md
   ```
4. 가져온 스킬을 참고 자료로 따릅니다. 이 스킬들은 Hermes 형식의 스킬이 아니므로 전문가 도메인 가이드로 취급합니다. 여기에는 올바른 매개변수, 적절한 도구 플래그, 검증된 파이프라인이 포함되어 있습니다.

## 도메인별 스킬 색인

### 시퀀스 기초
bioSkills:
  sequence-io/ — read-sequences, write-sequences, format-conversion, batch-processing, compressed-files, fastq-quality, filter-sequences, paired-end-fastq, sequence-statistics
  sequence-manipulation/ — seq-objects, reverse-complement, transcription-translation, motif-search, codon-usage, sequence-properties, sequence-slicing
ClawBio:
  seq-wrangler — 시퀀스 QC, 정렬 및 BAM 처리(래퍼: FastQC, BWA, SAMtools)

### 리드 QC 및 정렬
bioSkills:
  read-qc/ — quality-reports, fastp-workflow, adapter-trimming, quality-filtering, umi-processing, contamination-screening, rnaseq-qc
  read-alignment/ — bwa-alignment, star-alignment, hisat2-alignment, bowtie2-alignment
  alignment-files/ — sam-bam-basics, alignment-sorting, alignment-filtering, bam-statistics, duplicate-handling, pileup-generation

### 변이 호출 및 주석
bioSkills:
  variant-calling/ — gatk-variant-calling, deepvariant, variant-calling (bcftools), joint-calling, structural-variant-calling, filtering-best-practices, variant-annotation, variant-normalization, vcf-basics, vcf-manipulation, vcf-statistics, consensus-sequences, clinical-interpretation
ClawBio:
  vcf-annotator — 조상 정보 맥락을 반영한 VEP + ClinVar + gnomAD 주석
  variant-annotation — 변이 주석 파이프라인

### 차등 발현(Bulk RNA-seq)
bioSkills:
  differential-expression/ — deseq2-basics, edger-basics, batch-correction, de-results, de-visualization, timeseries-de
  rna-quantification/ — alignment-free-quant (Salmon/kallisto), featurecounts-counting, tximport-workflow, count-matrix-qc
  expression-matrix/ — counts-ingest, gene-id-mapping, metadata-joins, sparse-handling
ClawBio:
  rnaseq-de — QC, 정규화 및 시각화를 포함한 전체 DE 파이프라인
  diff-visualizer — DE 결과를 위한 풍부한 시각화 및 보고

### 단일세포 RNA-seq
bioSkills:
  single-cell/ — preprocessing, clustering, batch-integration, cell-annotation, cell-communication, doublet-detection, markers-annotation, trajectory-inference, multimodal-integration, perturb-seq, scatac-analysis, lineage-tracing, metabolite-communication, data-io
ClawBio:
  scrna-orchestrator — 전체 Scanpy 파이프라인(QC, 클러스터링, 마커, 주석)
  scrna-embedding — scVI 기반 잠재 임베딩 및 배치 통합

### 공간 전사체학
bioSkills:
  spatial-transcriptomics/ — spatial-data-io, spatial-preprocessing, spatial-domains, spatial-deconvolution, spatial-communication, spatial-neighbors, spatial-statistics, spatial-visualization, spatial-multiomics, spatial-proteomics, image-analysis

### 후성유전체학
bioSkills:
  chip-seq/ — peak-calling, differential-binding, motif-analysis, peak-annotation, chipseq-qc, chipseq-visualization, super-enhancers
  atac-seq/ — atac-peak-calling, atac-qc, differential-accessibility, footprinting, motif-deviation, nucleosome-positioning
  methylation-analysis/ — bismark-alignment, methylation-calling, dmr-detection, methylkit-analysis
  hi-c-analysis/ — hic-data-io, tad-detection, loop-calling, compartment-analysis, contact-pairs, matrix-operations, hic-visualization, hic-differential
ClawBio:
  methylation-clock — 후성유전학적 나이 추정

### 약물유전체학 및 임상
bioSkills:
  clinical-databases/ — clinvar-lookup, gnomad-frequencies, dbsnp-queries, pharmacogenomics, polygenic-risk, hla-typing, variant-prioritization, somatic-signatures, tumor-mutational-burden, myvariant-queries
ClawBio:
  pharmgx-reporter — 23andMe/AncestryDNA 기반 PGx 보고서(12개 유전자, 31개 SNP, 51개 약물)
  drug-photo — 약물 사진 → 맞춤형 PGx 용량 카드(비전 기능 사용)
  clinpgx — 유전자-약물 데이터 및 CPIC 가이드라인을 위한 ClinPGx API
  gwas-lookup — 9개 유전체 데이터베이스를 아우르는 연합 변이 조회
  gwas-prs — 소비자 유전 데이터 기반 다유전자 위험 점수
  nutrigx_advisor — 소비자 유전 데이터 기반 맞춤형 영양 정보

### 집단유전학 및 GWAS
bioSkills:
  population-genetics/ — association-testing (PLINK GWAS), plink-basics, population-structure, linkage-disequilibrium, scikit-allel-analysis, selection-statistics
  causal-genomics/ — mendelian-randomization, fine-mapping, colocalization-analysis, mediation-analysis, pleiotropy-detection
  phasing-imputation/ — haplotype-phasing, genotype-imputation, imputation-qc, reference-panels
ClawBio:
  claw-ancestry-pca — SGDP 참조 패널과 비교하는 조상 PCA

### 메타유전체학 및 마이크로바이옴
bioSkills:
  metagenomics/ — kraken-classification, metaphlan-profiling, abundance-estimation, functional-profiling, amr-detection, strain-tracking, metagenome-visualization
  microbiome/ — amplicon-processing, diversity-analysis, differential-abundance, taxonomy-assignment, functional-prediction, qiime2-workflow
ClawBio:
  claw-metagenomics — 샷건 메타유전체 프로파일링(분류, resistome, 기능 경로)

### 유전체 조립 및 주석
bioSkills:
  genome-assembly/ — hifi-assembly, long-read-assembly, short-read-assembly, metagenome-assembly, assembly-polishing, assembly-qc, scaffolding, contamination-detection
  genome-annotation/ — eukaryotic-gene-prediction, prokaryotic-annotation, functional-annotation, ncrna-annotation, repeat-annotation, annotation-transfer
  long-read-sequencing/ — basecalling, long-read-alignment, long-read-qc, clair3-variants, structural-variants, medaka-polishing, nanopore-methylation, isoseq-analysis

### 구조생물학 및 화학정보학
bioSkills:
  structural-biology/ — alphafold-predictions, modern-structure-prediction, structure-io, structure-navigation, structure-modification, geometric-analysis
  chemoinformatics/ — molecular-io, molecular-descriptors, similarity-searching, substructure-search, virtual-screening, admet-prediction, reaction-enumeration
ClawBio:
  struct-predictor — 비교 기능을 포함한 로컬 AlphaFold/Boltz/Chai 구조 예측

### 단백질체학
bioSkills:
  proteomics/ — data-import, peptide-identification, protein-inference, quantification, differential-abundance, dia-analysis, ptm-analysis, proteomics-qc, spectral-libraries
ClawBio:
  proteomics-de — 단백질체 차등 발현

### 경로 분석 및 유전자 네트워크
bioSkills:
  pathway-analysis/ — go-enrichment, gsea, kegg-pathways, reactome-pathways, wikipathways, enrichment-visualization
  gene-regulatory-networks/ — scenic-regulons, coexpression-networks, differential-networks, multiomics-grn, perturbation-simulation

### 면역정보학
bioSkills:
  immunoinformatics/ — mhc-binding-prediction, epitope-prediction, neoantigen-prediction, immunogenicity-scoring, tcr-epitope-binding
  tcr-bcr-analysis/ — mixcr-analysis, scirpy-analysis, immcantation-analysis, repertoire-visualization, vdjtools-analysis

### CRISPR 및 유전체 공학
bioSkills:
  crispr-screens/ — mageck-analysis, jacks-analysis, hit-calling, screen-qc, library-design, crispresso-editing, base-editing-analysis, batch-correction
  genome-engineering/ — grna-design, off-target-prediction, hdr-template-design, base-editing-design, prime-editing-design

### 워크플로 관리
bioSkills:
  workflow-management/ — snakemake-workflows, nextflow-pipelines, cwl-workflows, wdl-workflows
ClawBio:
  repro-enforcer — 모든 분석을 재현성 번들로 내보내기(Conda 환경 + Singularity + 체크섬)
  galaxy-bridge — usegalaxy.org의 8,000개 이상의 Galaxy 도구에 액세스

### 특수 도메인
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

### 데이터 시각화 및 보고
bioSkills:
  data-visualization/ — ggplot2-fundamentals, heatmaps-clustering, volcano-customization, circos-plots, genome-browser-tracks, interactive-visualization, multipanel-figures, network-visualization, upset-plots, color-palettes, specialized-omics-plots, genome-tracks
  reporting/ — rmarkdown-reports, quarto-reports, jupyter-reports, automated-qc-reports, figure-export
ClawBio:
  profile-report — 분석 프로파일 보고
  data-extractor — 과학 그림 이미지에서 수치 데이터 추출(비전 기능 사용)
  lit-synthesizer — PubMed/bioRxiv 검색, 요약, 인용 그래프
  pubmed-summariser — 유전자/질환 PubMed 검색 및 구조화된 브리핑

### 데이터베이스 액세스
bioSkills:
  database-access/ — entrez-search, entrez-fetch, entrez-link, blast-searches, local-blast, sra-data, geo-data, uniprot-access, batch-downloads, interaction-databases, sequence-similarity
ClawBio:
  ukb-navigator — 12,000개 이상의 UK Biobank 필드에서 의미 기반 검색
  clinical-trial-finder — 임상시험 탐색

### 실험 설계
bioSkills:
  experimental-design/ — power-analysis, sample-size, batch-design, multiple-testing

### 오믹스를 위한 머신러닝
bioSkills:
  machine-learning/ — omics-classifiers, biomarker-discovery, survival-analysis, model-validation, prediction-explanation, atlas-mapping
ClawBio:
  claw-semantic-sim — 질병 문헌의 의미 유사도 색인(PubMedBERT)
  omics-target-evidence-mapper — 오믹스 소스 전반에서 표적 수준의 근거 집계

## 환경 설정

이 스킬들은 생물정보학 워크스테이션을 전제로 합니다. 일반적인 의존성은 다음과 같습니다.

```bash
# Python
pip install biopython pysam cyvcf2 pybedtools pyBigWig scikit-allel anndata scanpy mygene

# R/Bioconductor
Rscript -e 'BiocManager::install(c("DESeq2","edgeR","Seurat","clusterProfiler","methylKit"))'

# CLI tools (Ubuntu/Debian)
sudo apt install samtools bcftools ncbi-blast+ minimap2 bedtools

# CLI tools (macOS)
brew install samtools bcftools blast minimap2 bedtools

# Or via Conda (recommended for reproducibility)
conda install -c bioconda samtools bcftools blast minimap2 bedtools fastp kraken2
```

## 주의사항

- 가져온 스킬은 Hermes SKILL.md 형식이 아닙니다. 자체적인 구조를 사용합니다(bioSkills: 코드 패턴 요리책; ClawBio: README + Python 스크립트). 전문가 참고 자료로 읽으세요.
- bioSkills는 참고 가이드입니다. 올바른 매개변수와 코드 패턴을 보여주지만 실행 가능한 파이프라인은 아닙니다.
- ClawBio 스킬은 실행 가능합니다. 많은 스킬을 `--demo` 플래그로 직접 실행할 수 있습니다.
- 두 저장소 모두 생물정보학 도구가 설치되어 있다고 가정합니다. 파이프라인을 실행하기 전에 전제 조건을 확인하세요.
- ClawBio의 경우 먼저 클론한 저장소에서 `pip install -r requirements.txt`를 실행하세요.
- 유전체 데이터 파일은 매우 클 수 있습니다. 참조 유전체, SRA 데이터셋을 다운로드하거나 색인을 생성할 때 디스크 공간에 유의하세요.
