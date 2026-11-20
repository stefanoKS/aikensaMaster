#!/usr/bin/env bash
# AIKensaMaster - machine configuration backup (RUN BEFORE git pull)
# The backup is written OUTSIDE the work tree (default: ../aikensa_backups).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/machine_config_backup.py backup --reason prepull
echo
echo "Backup verified. It is now safe to run: git pull"
