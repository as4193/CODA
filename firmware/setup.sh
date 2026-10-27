#!/usr/bin/env bash
# One-time setup + build of all CODA firmware. Usage: firmware/setup.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .pio-venv/bin/pio ]; then
  if command -v uv >/dev/null; then
    UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/Library/Caches/uv}" uv venv --python 3.11 .pio-venv
    UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/Library/Caches/uv}" uv pip install --python .pio-venv/bin/python "platformio==6.2.0"
  else
    python3 -m venv .pio-venv && .pio-venv/bin/pip install "platformio==6.2.0"
  fi
fi
[ -f lib/tflm/library.json ] || ../scripts/fetch_tflm.sh
for p in stm32 rp2350 esp32; do
  echo "=== building $p ==="
  .pio-venv/bin/pio run -d "$p"
done
