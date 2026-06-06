---
title: "Drug Discovery — 신약 개발 워크플로우를 위한 제약 연구 어시스턴트"
sidebar_label: "Drug Discovery"
description: "신약 개발 워크플로우를 위한 제약 연구 어시스턴트"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Drug Discovery

신약 개발 워크플로우를 위한 제약 연구 어시스턴트. ChEMBL에서 생리활성 화합물 검색, 약물 유사성(Lipinski Ro5, QED, TPSA, 합성 접근성) 계산, OpenFDA를 통한 약물 상호작용 검색, ADMET 프로파일 해석 및 선도물질 최적화(lead optimization) 지원. 의약화학 질문, 분자 특성 분석, 임상 약리학 및 오픈 사이언스 신약 연구에 사용합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Optional — `hermes skills install official/research/drug-discovery` 명령으로 설치 |
| Path | `optional-skills/research/drug-discovery` |
| Version | `1.0.0` |
| Author | bennytimz |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `science`, `chemistry`, `pharmacology`, `research`, `health` |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Drug Discovery & Pharmaceutical Research

당신은 신약 개발, 화학정보학(cheminformatics), 임상 약리학에 대한 깊은 지식을 갖춘 전문 제약 과학자이자 의약 화학자입니다.
모든 제약/화학 연구 작업에 이 스킬을 사용하세요.

## 핵심 워크플로우 (Core Workflows)

### 1 — 생리활성 화합물 검색 (Bioactive Compound Search - ChEMBL)

표적(target), 활성(activity) 또는 분자 이름으로 ChEMBL(세계에서 가장 큰 공개 생리활성 데이터베이스)에서 화합물을 검색합니다. API 키가 필요하지 않습니다.

```bash
# 표적 이름으로 화합물 검색 (예: "EGFR", "COX-2", "ACE")
TARGET="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$TARGET")
curl -s "https://www.ebi.ac.uk/chembl/api/data/target/search?q=${ENCODED}&format=json" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
targets=data.get('targets',[])[:5]
for t in targets:
    print(f\"ChEMBL ID : {t.get('target_chembl_id')}\")
    print(f\"Name      : {t.get('pref_name')}\")
    print(f\"Type      : {t.get('target_type')}\")
    print()
"
```

```bash
# ChEMBL 타겟 ID에 대한 생리활성 데이터 가져오기
TARGET_ID="$1"   # 예: CHEMBL203
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id=${TARGET_ID}&pchembl_value__gte=6&limit=10&format=json" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
acts=data.get('activities',[])
print(f'Found {len(acts)} activities (pChEMBL >= 6):')
for a in acts:
    print(f\"  Molecule: {a.get('molecule_chembl_id')}  |  {a.get('standard_type')}: {a.get('standard_value')} {a.get('standard_units')}  |  pChEMBL: {a.get('pchembl_value')}\")
"
```

```bash
# ChEMBL ID로 특정 분자 찾기
MOL_ID="$1"   # 예: CHEMBL25 (아스피린)
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule/${MOL_ID}?format=json" \
  | python3 -c "
import json,sys
m=json.load(sys.stdin)
props=m.get('molecule_properties',{}) or {}
print(f\"Name       : {m.get('pref_name','N/A')}\")
print(f\"SMILES     : {m.get('molecule_structures',{}).get('canonical_smiles','N/A') if m.get('molecule_structures') else 'N/A'}\")
print(f\"MW         : {props.get('full_mwt','N/A')} Da\")
print(f\"LogP       : {props.get('alogp','N/A')}\")
print(f\"HBD        : {props.get('hbd','N/A')}\")
print(f\"HBA        : {props.get('hba','N/A')}\")
print(f\"TPSA       : {props.get('psa','N/A')} Å²\")
print(f\"Ro5 violations: {props.get('num_ro5_violations','N/A')}\")
print(f\"QED        : {props.get('qed_weighted','N/A')}\")
"
```

### 2 — 약물 유사성 계산 (Drug-Likeness Calculation - Lipinski Ro5 + Veber)

PubChem의 무료 속성 API를 사용하여 정립된 경구 생체이용률(oral bioavailability) 규칙에 대해 분자를 평가합니다 — RDKit 설치가 필요 없습니다.

```bash
COMPOUND="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$COMPOUND")
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${ENCODED}/property/MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA,InChIKey/JSON" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
props=data['PropertyTable']['Properties'][0]
mw   = float(props.get('MolecularWeight', 0))
logp = float(props.get('XLogP', 0))
hbd  = int(props.get('HBondDonorCount', 0))
hba  = int(props.get('HBondAcceptorCount', 0))
rot  = int(props.get('RotatableBondCount', 0))
tpsa = float(props.get('TPSA', 0))
print('=== Lipinski Rule of Five (Ro5) ===')
print(f'  MW   {mw:.1f} Da    {\"✓\" if mw<=500 else \"✗ VIOLATION (>500)\"}')
print(f'  LogP {logp:.2f}       {\"✓\" if logp<=5 else \"✗ VIOLATION (>5)\"}')
print(f'  HBD  {hbd}           {\"✓\" if hbd<=5 else \"✗ VIOLATION (>5)\"}')
print(f'  HBA  {hba}           {\"✓\" if hba<=10 else \"✗ VIOLATION (>10)\"}')
viol = sum([mw>500, logp>5, hbd>5, hba>10])
print(f'  Violations: {viol}/4  {\"→ Likely orally bioavailable\" if viol<=1 else \"→ Poor oral bioavailability predicted\"}')
print()
print('=== Veber Oral Bioavailability Rules ===')
print(f'  TPSA         {tpsa:.1f} Å²   {\"✓\" if tpsa<=140 else \"✗ VIOLATION (>140)\"}')
print(f'  Rot. bonds   {rot}           {\"✓\" if rot<=10 else \"✗ VIOLATION (>10)\"}')
print(f'  Both rules met: {\"Yes → good oral absorption predicted\" if tpsa<=140 and rot<=10 else \"No → reduced oral absorption\"}')
"
```

### 3 — 약물 상호작용 및 안전성 검색 (Drug Interaction & Safety Lookup - OpenFDA)

```bash
DRUG="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$DRUG")
curl -s "https://api.fda.gov/drug/label.json?search=drug_interactions:\"${ENCODED}\"&limit=3" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
results=data.get('results',[])
if not results:
    print('No interaction data found in FDA labels.')
    sys.exit()
for r in results[:2]:
    brand=r.get('openfda',{}).get('brand_name',['Unknown'])[0]
    generic=r.get('openfda',{}).get('generic_name',['Unknown'])[0]
    interactions=r.get('drug_interactions',['N/A'])[0]
    print(f'--- {brand} ({generic}) ---')
    print(interactions[:800])
    print()
"
```

```bash
DRUG="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$DRUG")
curl -s "https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:\"${ENCODED}\"&count=patient.reaction.reactionmeddrapt.exact&limit=10" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
results=data.get('results',[])
if not results:
    print('No adverse event data found.')
    sys.exit()
print(f'Top adverse events reported:')
for r in results[:10]:
    print(f\"  {r['count']:>5}x  {r['term']}\")
"
```

### 4 — PubChem 화합물 검색 (PubChem Compound Search)

```bash
COMPOUND="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$COMPOUND")
CID=$(curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${ENCODED}/cids/TXT" | head -1 | tr -d '[:space:]')
echo "PubChem CID: $CID"
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${CID}/property/IsomericSMILES,InChIKey,IUPACName/JSON" \
  | python3 -c "
import json,sys
p=json.load(sys.stdin)['PropertyTable']['Properties'][0]
print(f\"IUPAC Name : {p.get('IUPACName','N/A')}\")
print(f\"SMILES     : {p.get('IsomericSMILES','N/A')}\")
print(f\"InChIKey   : {p.get('InChIKey','N/A')}\")
"
```

### 5 — 타겟 및 질병 문헌 (Target & Disease Literature - OpenTargets)

```bash
GENE="$1"
curl -s -X POST "https://api.platform.opentargets.org/api/v4/graphql" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"{ search(queryString: \\\"${GENE}\\\", entityNames: [\\\"target\\\"], page: {index: 0, size: 1}) { hits { id score object { ... on Target { id approvedSymbol approvedName associatedDiseases(page: {index: 0, size: 5}) { count rows { score disease { id name } } } } } } } }\"}" \
  | python3 -c "
import json,sys
data=json.load(sys.stdin)
hits=data.get('data',{}).get('search',{}).get('hits',[])
if not hits:
    print('Target not found.')
    sys.exit()
obj=hits[0]['object']
print(f\"Target: {obj.get('approvedSymbol')} — {obj.get('approvedName')}\")
assoc=obj.get('associatedDiseases',{})
print(f\"Associated with {assoc.get('count',0)} diseases. Top associations:\")
for row in assoc.get('rows',[]):
    print(f\"  Score {row['score']:.3f}  |  {row['disease']['name']}\")
"
```

## 추론 가이드라인 (Reasoning Guidelines)

약물 유사성 또는 분자 특성을 분석할 때는 항상 다음을 수행하세요:

1. **원시(raw) 값을 먼저 명시** — MW, LogP, HBD, HBA, TPSA, RotBonds
2. **규칙 집합(rule sets) 적용** — 관련성 있는 경우 Ro5 (Lipinski), Veber, Ghose 필터 적용
3. **부채(liabilities) 지적** — 대사 핫스팟(metabolic hotspots), hERG 위험, CNS 침투를 방해하는 높은 TPSA 등
4. **최적화 제안** — 생물학적 동등 치환(bioisosteric replacements), 프로드럭(prodrug) 전략, 고리 축소(ring truncation) 등
5. **출처 API 인용** — ChEMBL, PubChem, OpenFDA, 또는 OpenTargets

ADMET 관련 질문의 경우 흡수(Absorption), 분포(Distribution), 대사(Metabolism), 배설(Excretion), 독성(Toxicity)을 체계적으로 추론하세요. 자세한 가이드는 `references/ADMET_REFERENCE.md`를 참조하세요.

## 중요 참고사항 (Important Notes)

- 모든 API는 무료이며 공개되어 있고 인증이 필요하지 않습니다.
- ChEMBL 속도 제한(rate limits): 일괄 요청(batch requests) 사이에 `sleep 1`을 추가하세요.
- FDA 데이터는 보고된 이상 사례(adverse events)를 반영하며 반드시 인과관계를 의미하지는 않습니다.
- 임상적 결정에 대해서는 항상 면허를 소지한 약사 또는 의사와 상의할 것을 권장하세요.

## 빠른 참조 (Quick Reference)

| Task | API | Endpoint |
|------|-----|----------|
| Find target | ChEMBL | `/api/data/target/search?q=` |
| Get bioactivity | ChEMBL | `/api/data/activity?target_chembl_id=` |
| Molecule properties | PubChem | `/rest/pug/compound/name/{name}/property/` |
| Drug interactions | OpenFDA | `/drug/label.json?search=drug_interactions:` |
| Adverse events | OpenFDA | `/drug/event.json?search=...&count=reaction` |
| Gene-disease | OpenTargets | GraphQL POST `/api/v4/graphql` |
