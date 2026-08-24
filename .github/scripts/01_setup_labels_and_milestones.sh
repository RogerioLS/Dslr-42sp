#!/bin/bash
# ==============================================================================
#           42 DSLR — STEP 1: SETUP LABELS & MILESTONES
# ==============================================================================

set -e

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)

if [ -z "$REPO" ]; then
    echo "❌ Erro: Não foi possível detectar o repositório GitHub via gh cli."
    echo "💡 Certifique-se de estar autenticado com 'gh auth login'."
    exit 1
fi

echo "🚀 [ETAPA 1] Configurando Milestones e Labels no repositório: $REPO..."

# ------------------------------------------------------------------------------
# 1. CRIAR MILESTONES
# ------------------------------------------------------------------------------
echo "🎯 Criando Milestones..."

gh api repos/$REPO/milestones -f title="01. Data Exploration & Handcrafted Stats" \
  -f description="Implementação manual do motor estatístico e do programa describe.py sem bibliotecas prontas." 2>/dev/null || true

gh api repos/$REPO/milestones -f title="02. Data Visualization & Feature Analysis" \
  -f description="Exploração gráfica dos dados com histogramas, gráficos de dispersão e pair plots." 2>/dev/null || true

gh api repos/$REPO/milestones -f title="03. Logistic Regression Engine & Math" \
  -f description="Implementação do One-vs-Rest, Gradiente Descendente e pipeline de predição com acurácia >= 98%." 2>/dev/null || true

gh api repos/$REPO/milestones -f title="04. Defense Readiness & Bonuses" \
  -f description="Otimizadores extras (SGD/Mini-batch), métricas adicionais e simulação de peer-evaluation." 2>/dev/null || true

# ------------------------------------------------------------------------------
# 2. CRIAR LABELS
# ------------------------------------------------------------------------------
echo "🏷️ Criando Labels completas..."

# Labels de Áreas
gh label create "area: stats" --color "3498db" --description "Estatística descritiva e matemática pura" --force
gh label create "area: visualization" --color "9b59b6" --description "Gráficos e visualização de dados" --force
gh label create "area: model" --color "e67e22" --description "Machine Learning e Regressão Logística" --force
gh label create "area: preprocessing" --color "1abc9c" --description "Tratamento de dados, NaNs e Scaler" --force
gh label create "area: defense" --color "2ecc71" --description "Preparação e checklist para avaliação presencial" --force
gh label create "area: devops" --color "34495e" --description "CI/CD, Makefiles, Linters e automação" --force

# Labels de Tipos
gh label create "type: implementation" --color "27ae60" --description "Desenvolvimento de código" --force
gh label create "type: math-heavy" --color "e74c3c" --description "Foco em Álgebra Linear e Cálculo" --force
gh label create "type: pedagogical" --color "f1c40f" --description "Conceitos fundamentais explicados para estudo em dupla" --force
gh label create "type: defense" --color "16a085" --description "Foco em defesa e critérios de avaliação 42" --force
gh label create "type: bonus" --color "95a5a6" --description "Funcionalidades bônus" --force
gh label create "type: test" --color "d35400" --description "Testes unitários e suites de validação" --force
gh label create "type: docs" --color "7f8c8d" --description "Documentação técnica e científica" --force

# Labels de Prioridade
gh label create "priority: high" --color "b91c1c" --description "Prioridade Alta / Bloqueante" --force
gh label create "priority: medium" --color "f59e0b" --description "Prioridade Média" --force
gh label create "priority: low" --color "10b981" --description "Prioridade Baixa / Melhoria" --force

echo "✅ Milestones e Labels configuradas com sucesso!"
