#!/usr/bin/env bash
set -euo pipefail

# Blocks commits that change core API/infrastructure without updating README.md
# Skip with SKIP_README_CHECK=1

if [[ "${SKIP_README_CHECK:-}" == "1" ]]; then
  exit 0
fi

CHANGED=$(git diff --cached --name-only)

needs_docs_regex='^(backend/app/|backend/Dockerfile|backend/requirements|backend/alembic/|infrastructure/compose|infrastructure/.env|tests/|.github/workflows/)'

if echo "$CHANGED" | grep -Eq "$needs_docs_regex"; then
  if ! echo "$CHANGED" | grep -q '^README.md$'; then
    echo "\nREADME.md was not updated. Please reflect your changes in README or export SKIP_README_CHECK=1 to bypass.\n" >&2
    exit 1
  fi
fi

exit 0

