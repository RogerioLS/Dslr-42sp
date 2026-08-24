#!/bin/bash
# ==============================================================================
#                 42 DSLR — GITHUB KANBAN MASTER RUNNER
# ==============================================================================

set -e

SCRIPT_DIR=$(dirname "$0")

echo "=================================================="
echo " 🚀 INICIANDO SETUP DO KANBAN DSLR NO GITHUB      "
echo "=================================================="

# 1. Configurar Labels e Milestones
bash "$SCRIPT_DIR/01_setup_labels_and_milestones.sh"

echo ""

# 2. Configurar Tasks (Idempotente)
bash "$SCRIPT_DIR/02_setup_kanban_tasks.sh"

echo ""
echo "=================================================="
echo " ✅ SETUP DO KANBAN CONCLUÍDO COM SUCESSO!        "
echo "=================================================="
