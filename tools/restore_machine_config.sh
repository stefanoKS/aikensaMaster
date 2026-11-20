#!/usr/bin/env bash
# Restore from the newest verified backup. Use --dry-run first.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/machine_config_backup.py restore "$@"
