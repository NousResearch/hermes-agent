#!/bin/bash
# 0-Token PR Factory — Hermes Agent
# Uso: ./zero_pr_factory.sh [padrao]
# padrao: len|fstring|noqa|all
# Cria PRs sem gastar 1 token de LLM

set -e
REPO="/Users/wesleysimplicio/Projetos/ai/hermes-agent"
FORK="wesleysimplicio"
BRANCH_PREFIX="zero/$(date +%H%M)"
COUNT=0

cd "$REPO"
git checkout main
git pull origin main

criar_pr() {
    local branch="$1" titulo="$2" arquivos="$3"
    # VERIFICAR DUPLICATA ANTES
    local existente=$(gh search prs --repo NousResearch/hermes-agent --author wesleysimplicio "$titulo" --json number --jq '.[0].number' 2>/dev/null)
    if [ -n "$existente" ] && [ "$existente" != "null" ]; then
        echo "  DUPLICATA: PR #$existente já existe para '$titulo'"
        return
    fi
    git checkout -b "$branch"
    git add $arquivos
    git commit -m "$titulo" --allow-empty
    git push fork "$branch" 2>/dev/null
    gh pr create --repo NousResearch/hermes-agent \
        --head "$FORK:$branch" --base main \
        --title "$titulo" --body "Zero-token PR. Batch automatizado."
    git checkout main
    COUNT=$((COUNT+1))
    echo "PR #$COUNT: $titulo"
}

# ============ PADRÃO: len(x) == 0 / len(x) > 0 ============
fix_len() {
    echo "=== Padrão: len() == 0 / > 0 ==="
    # Herança de PRs anteriores - apenas se encontrar novos
}

# ============ PADRÃO: # noqa RUF100 ============
fix_noqa() {
    echo "=== Padrão: noqa RUF100 ==="
    local DIRS=$(find . -name "*.py" -not -path "*/tests/*" -not -path "*/website/*" \
        -not -path "*/node_modules/*" -not -path "*/.venv/*" -not -path "*/__pycache__/*" \
        -not -path "*/.git/*" | head -200)
    
    for dir in agent tools gateway hermes_cli plugins; do
        local files=$(find "./$dir" -name "*.py" 2>/dev/null | head -50)
        [ -z "$files" ] && continue
        local changed=0
        for f in $files; do
            if grep -q "# noqa" "$f" 2>/dev/null; then
                ruff check --select RUF100 --fix "$f" 2>/dev/null && changed=1
            fi
        done
        if [ "$changed" -eq 1 ]; then
            criar_pr "$BRANCH_PREFIX/noqa-$dir" \
                "fix: remove stale noqa from $dir/" \
                "$(git diff --name-only)"
        fi
    done
}

# ============ PADRÃO: import * ============
fix_wildcard() {
    echo "=== Padrão: wildcard imports ==="
    local files=$(grep -rln "from.*import \*" --include="*.py" \
        | grep -v tests/ | grep -v website/ | grep -v node_modules/ | head -10)
    [ -z "$files" ] && { echo "  Nada encontrado"; return; }
    echo "  Encontrado em $(echo "$files" | wc -l) arquivos"
}

# ============ EXECUÇÃO ============
case "${1:-all}" in
    len)    fix_len ;;
    fstring) echo "=== Padrão: f-string (usar script separado) ===" ;;
    noqa)   fix_noqa ;;
    all)
        fix_noqa
        fix_wildcard
        fix_len
        ;;
    *)
        echo "Uso: $0 [len|fstring|noqa|all]"
        exit 1
        ;;
esac

echo "=== FINALIZADO ==="
echo "Total PRs criadas: $COUNT"
echo "Tokens gastos: 0"
