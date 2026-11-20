#!/usr/bin/env bash
# Software-only test suite (no cameras / GPU / PLC required).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec python3 -m pytest "$@"
