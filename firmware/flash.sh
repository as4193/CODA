#!/usr/bin/env bash
# Build + flash one board. Usage: firmware/flash.sh f446re | h743zi | pico2 | esp32s3 [serial-port]
set -euo pipefail
cd "$(dirname "$0")"
PIO=.pio-venv/bin/pio
[ -x "$PIO" ] && [ -f lib/tflm/library.json ] || ./setup.sh
PORT=${2:+--upload-port $2}
case "${1:-}" in
  f446re)  $PIO run -d stm32 -e nucleo_f446re -t upload ;;
  h743zi)  $PIO run -d stm32 -e nucleo_h743zi -t upload ;;
  pico2)   $PIO run -d rp2350 -e pico2 -t upload $PORT ;;
  esp32s3) $PIO run -d esp32 -e esp32s3 -t upload $PORT ;;
  *) echo "usage: $0 f446re|h743zi|pico2|esp32s3 [port]"; exit 1 ;;
esac
