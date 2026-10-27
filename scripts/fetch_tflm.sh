#!/usr/bin/env bash
# Generate the pinned TensorFlow Lite Micro + CMSIS-NN source tree used by the
# STM32 and RP2350 firmware (firmware/lib/tflm). The ESP32-S3 build gets TFLM +
# ESP-NN from the espressif/esp-tflite-micro component instead.
#
# Usage: scripts/fetch_tflm.sh          (about 1 minute; needs git, curl, make, python3)
set -euo pipefail

TFLM_COMMIT=06f0b46177562d00c4b2e7b11ec377b5104f582e   # tflite-micro main, 2026-09-30
# CMSIS-NN and CMSIS_6 commits are pinned (with MD5) inside that TFLM commit:
#   tensorflow/lite/micro/tools/make/ext_libs/cmsis_nn_download.sh
#   tensorflow/lite/micro/tools/make/ext_libs/cmsis_download.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/firmware/third_party/tflite-micro"
OUT="$ROOT/firmware/lib/tflm"

if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/tensorflow/tflite-micro.git "$SRC"
fi
git -C "$SRC" fetch -q origin "$TFLM_COMMIT" 2>/dev/null || true
git -C "$SRC" checkout -q "$TFLM_COMMIT"

# The TFLM download scripts call wget; macOS has only curl.
SHIM="$(mktemp -d)"
cat > "$SHIM/wget" <<'EOF'
#!/usr/bin/env bash
# minimal "wget URL -O FILE" -> curl
url=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in -O) out="$2"; shift 2;; -*) shift;; *) url="$1"; shift;; esac
done
exec curl -fsSL "$url" -o "${out:-$(basename "$url")}"
EOF
chmod +x "$SHIM/wget"

# TFLM needs GNU make >= 3.82 (macOS ships 3.81): brew install make
GNUBIN=/opt/homebrew/opt/make/libexec/gnubin
[ -d "$GNUBIN" ] && SHIM="$SHIM:$GNUBIN"
PY="$(command -v python3.11 || command -v python3)"
rm -rf "$OUT.tmp"
(cd "$SRC" && PATH="$SHIM:$PATH" "$PY" tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
   --makefile_options="TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn" \
   "$OUT.tmp")
rm -rf "${SHIM%%:*}"

# Drop example code and the generic platform hooks (firmware/common provides its own).
rm -rf "$OUT.tmp/examples"
cp "$ROOT/firmware/lib/tflm-library.json" "$OUT.tmp/library.json"
echo "$TFLM_COMMIT" > "$OUT.tmp/TFLM_COMMIT"
rm -rf "$OUT" && mv "$OUT.tmp" "$OUT"
echo "TFLM tree ready: $OUT ($(du -sh "$OUT" | cut -f1))"
