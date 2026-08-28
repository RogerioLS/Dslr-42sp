#!/bin/bash
# ==============================================================================
#           42 DSLR — STEP 2: SETUP KANBAN TASKS (IDEMPOTENT)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISSUES_DIR="$SCRIPT_DIR/../issues"

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)

if [ -z "$REPO" ]; then
    echo "❌ Erro: Não foi possível detectar o repositório GitHub via gh cli."
    exit 1
fi

echo "🚀 [ETAPA 2] Populando o Kanban com as tasks no repositório: $REPO..."

create_issue_if_missing() {
    local title="$1"
    local milestone="$2"
    local labels="$3"
    local body_file="$4"

    local existing=$(gh issue list --search "$title in:title" --json number -q '.[0].number' 2>/dev/null || true)

    if [ -n "$existing" ]; then
        echo "🔄 Atualizando corpo da Issue existente (#$existing): $title"
        gh issue edit "$existing" \
            --milestone "$milestone" \
            --add-label "$labels" \
            --body-file "$body_file"
    else
        echo "➕ Criando issue: $title"
        gh issue create \
            --title "$title" \
            --milestone "$milestone" \
            --label "$labels" \
            --body-file "$body_file"
    fi
}

# --- MILESTONE 1: Data Exploration & Handcrafted Stats ---

create_issue_if_missing \
  "[DSLR-01] Data Pipeline: Carregamento do CSV e Tratamento de NaNs" \
  "01. Data Exploration & Handcrafted Stats" \
  "area: stats,type: implementation,type: pedagogical,priority: high" \
  "$ISSUES_DIR/dslr-01-data-pipeline.md"

create_issue_if_missing \
  "[DSLR-02] Core Math: Count, Mean, Std, Min e Max do Zero" \
  "01. Data Exploration & Handcrafted Stats" \
  "area: stats,type: math-heavy,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-02-stats-math.md"

create_issue_if_missing \
  "[DSLR-03] Core Math: Cálculo Exato de Percentis (25%, 50%, 75%)" \
  "01. Data Exploration & Handcrafted Stats" \
  "area: stats,type: math-heavy,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-03-quantile-interpolation.md"

create_issue_if_missing \
  "[DSLR-04] CLI describe.py: Formatação Visual e Alinhamento de Tabela" \
  "01. Data Exploration & Handcrafted Stats" \
  "area: stats,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-04-cli-describe.md"

# --- MILESTONE 2: Data Visualization & Feature Analysis ---

create_issue_if_missing \
  "[DSLR-05] Viz: Histograma de Cursos Homogêneos (histogram.py)" \
  "02. Data Visualization & Feature Analysis" \
  "area: visualization,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-05-histogram-analysis.md"

create_issue_if_missing \
  "[DSLR-06] Viz: Scatter Plot de Cursos Similares (scatter_plot.py)" \
  "02. Data Visualization & Feature Analysis" \
  "area: visualization,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-06-scatter-plot.md"

create_issue_if_missing \
  "[DSLR-07] Viz: Pair Plot Completo de Features (pair_plot.py)" \
  "02. Data Visualization & Feature Analysis" \
  "area: visualization,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-07-pair-plot.md"

# --- MILESTONE 3: Logistic Regression & Classification Engine ---

create_issue_if_missing \
  "[DSLR-08] Preprocessing: StandardScaler Artesanal e Imputação Numérica" \
  "03. Logistic Regression & Classification Engine" \
  "area: model,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-08-preprocessing-scaler.md"

create_issue_if_missing \
  "[DSLR-09] Core Math: Sigmoide, Log-Loss e Gradiente Logístico" \
  "03. Logistic Regression & Classification Engine" \
  "area: model,type: math-heavy,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-09-core-math-logistic.md"

create_issue_if_missing \
  "[DSLR-10] Training Engine: One-vs-Rest e CLI logreg_train.py" \
  "03. Logistic Regression & Classification Engine" \
  "area: model,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-10-training-engine.md"

create_issue_if_missing \
  "[DSLR-11] Inference Engine: Classificador Final e CLI logreg_predict.py" \
  "03. Logistic Regression & Classification Engine" \
  "area: model,type: implementation,priority: high" \
  "$ISSUES_DIR/dslr-11-inference-engine.md"

# --- MILESTONE 4: Bonuses, Accuracy & Peer Defense ---

create_issue_if_missing \
  "[DSLR-12] Bonus: Algoritmos Otimizadores Extras (SGD / Mini-batch / Early Stopping)" \
  "04. Bonuses, Accuracy & Peer Defense" \
  "area: model,type: bonus,priority: medium" \
  "$ISSUES_DIR/dslr-12-bonuses.md"

create_issue_if_missing \
  "[DSLR-13] Peer Defense: Validação de Acurácia >= 98% e Checklist de Avaliação" \
  "04. Bonuses, Accuracy & Peer Defense" \
  "area: defense,type: defense,priority: high" \
  "$ISSUES_DIR/dslr-13-peer-defense.md"

echo "🎉 Todas as 13 Tasks de DSLR foram configuradas com sucesso!"
