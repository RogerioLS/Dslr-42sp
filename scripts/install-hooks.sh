#!/usr/bin/env bash
# ==============================================================================
#                      42 DSLR - GIT HOOKS INSTALLER
# ==============================================================================

set -e

GIT_DIR=$(git rev-parse --git-dir 2>/dev/null || true)

if [ -z "$GIT_DIR" ]; then
    echo "❌ Error: Not a git repository."
    exit 1
fi

echo "🔧 Configuring custom git hooks path..."
git config core.hooksPath .githooks

chmod +x .githooks/* 2>/dev/null || true

echo "✅ Git hooks configured to '.githooks'."
echo "ℹ️  Note: Pre-commit checks are automatically executed via .githooks/pre-commit."
