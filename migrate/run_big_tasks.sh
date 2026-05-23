#!/bin/bash
# Dispatcher for big_tasks.txt — submits the next PENDING task to iTiger as a
# 1-hour single-pass job, marks it RUNNING.  Skips DONE/FAILED/SKIP.
#
# Usage:
#   bash migrate/run_big_tasks.sh         # submit next PENDING task
#   bash migrate/run_big_tasks.sh --all   # submit every PENDING task
#
# Requires:
#   ITGR_PW   env var with iTiger password
#   pexpect installed in the local Python env
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASKS="$ROOT/migrate/big_tasks.txt"
PYBIN=/home/jding/miniconda3/envs/tgnn/bin/python

if [ -z "${ITGR_PW:-}" ]; then
  echo "ERROR: set ITGR_PW env var first" >&2; exit 1
fi

submit_one() {
  local DS=$1 TAG=$2 INDICATORS=$3 PAIRWISE=$4 BATCH=$5
  # Replace commas in INDICATORS for sbatch --export
  local IND_ESC=$(echo "$INDICATORS" | sed 's/,/\\,/g')
  $PYBIN <<PY
import os, sys, pexpect
PW = os.environ["ITGR_PW"]
cmd = ("ssh -o StrictHostKeyChecking=accept-new -o PreferredAuthentications=password "
       "-o PubkeyAuthentication=no -o GSSAPIAuthentication=no "
       "-o NumberOfPasswordPrompts=1 -tt jding2@itiger bash -s")
p = pexpect.spawn(cmd, encoding='utf-8', timeout=60)
p.logfile_read = sys.stdout
p.expect('[Pp]assword:', timeout=30); p.sendline(PW)
p.expect(r'\$', timeout=20)
remote = (f'sbatch --job-name=tgb_sp_{"$DS".replace("tgbl-","")}_{"$TAG"} '
          f'--export=DS="$DS",TAG="$TAG",INDICATORS="$IND_ESC",'
          f'PAIRWISE="$PAIRWISE",BATCH_SIZE="$BATCH" '
          f'/project/jding2/hawkeye/migrate/tgb_singlepass_1h.slurm')
p.sendline(remote)
p.expect(r'\$', timeout=30)
p.sendline("exit"); p.expect(pexpect.EOF, timeout=30)
PY
}

mode=${1:-one}
submitted=0
while IFS= read -r line; do
  # skip comments and blank lines
  [[ -z "$line" || "$line" =~ ^# ]] && continue
  read -r STATUS DS TAG INDICATORS PAIRWISE BATCH <<<"$line"
  [ "$STATUS" != "PENDING" ] && continue

  echo "[dispatch] submitting $DS $TAG (indicators=$INDICATORS pairwise=$PAIRWISE batch=$BATCH)"
  submit_one "$DS" "$TAG" "$INDICATORS" "$PAIRWISE" "$BATCH"

  # mark RUNNING in-place
  sed -i "s|^PENDING $DS $TAG|RUNNING $DS $TAG|" "$TASKS"
  submitted=$((submitted + 1))

  [ "$mode" = "--all" ] || break
done < "$TASKS"

if [ "$submitted" -eq 0 ]; then
  echo "[dispatch] no PENDING tasks to submit"
fi
echo "[dispatch] submitted $submitted task(s)"
