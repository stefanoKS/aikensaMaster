#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/machine_config_backup.py verify
