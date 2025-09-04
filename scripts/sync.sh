#!/usr/bin/env bash
set -euo pipefail

MSG=${1:-"chore: sync $(date -u +'%Y-%m-%dT%H:%M:%SZ')"}

git add -A
if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

git commit -m "$MSG"

# Push using gh token if available (works when HTTPS prompts aren't configured)
TOKEN=$(gh auth token 2>/dev/null || true)
REMOTE_URL=$(git remote get-url --push origin)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ -n "${TOKEN}" && "${REMOTE_URL}" =~ ^https://github.com/ ]]; then
  PUSH_URL="https://x-access-token:${TOKEN}@github.com/${REMOTE_URL#https://github.com/}"
  git push -u "$PUSH_URL" "$BRANCH"
else
  git push -u origin "$BRANCH"
fi

echo "Synced to remote: $BRANCH"

