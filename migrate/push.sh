#!/bin/bash
# Convenient one-shot "commit + push" for the Hawkeye paper repo.
#
#   bash migrate/push.sh "your commit message here"
#   bash migrate/push.sh                     # uses auto-message with timestamp
#
# Requires:
#   - git remote origin already set to github.com/graphuofm/hawkeye
#   - GitHub auth already working (SSH key or stored PAT — see comments
#     near the top of the paper README)
set -e
cd "$(dirname "$0")/.."

MSG="${1:-paper: update at $(date '+%Y-%m-%d %H:%M')}"

echo "== staging =="
git add -A
git status --short

if git diff --cached --quiet; then
  echo "nothing to commit, working tree clean"
  exit 0
fi

echo
echo "== commit =="
git commit -m "$MSG"

echo
echo "== push =="
git push origin main
echo "done."
