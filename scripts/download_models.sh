#!/usr/bin/env bash
# One-time model download. After this, `vox` runs fully offline.
# Usage: scripts/download_models.sh [models_dir]
set -euo pipefail

DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/models}"
HF="https://huggingface.co"
mkdir -p "$DIR/kokoro/voices" "$DIR/piper"

fetch() { # url dest
  if [[ -s "$2" ]]; then echo "  ok   $(basename "$2")"; return; fi
  echo "  get  $(basename "$2")"
  curl -fL --retry 3 -sS -o "$2.part" "$1" && mv "$2.part" "$2"
}

echo "Kokoro-82M (neural, 24 kHz) -> $DIR/kokoro"
K="$HF/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main"
fetch "$K/onnx/model.onnx" "$DIR/kokoro/model.onnx"
fetch "$K/tokenizer.json" "$DIR/kokoro/tokenizer.json"
for v in af_heart af_bella af_nicole af_sky am_michael am_fenrir am_puck ff_siwis; do
  fetch "$K/voices/$v.bin" "$DIR/kokoro/voices/$v.bin"
done

echo "Piper VITS voices (neural, 22.05 kHz) -> $DIR/piper"
P="$HF/rhasspy/piper-voices/resolve/main"
for spec in en/en_US/lessac/high en/en_US/ryan/high fr/fr_FR/tom/medium fr/fr_FR/siwis/medium fr/fr_FR/upmc/medium; do
  IFS=/ read -r lang loc name q <<<"$spec"
  base="${loc}-${name}-${q}"
  fetch "$P/$spec/$base.onnx" "$DIR/piper/$base.onnx"
  fetch "$P/$spec/$base.onnx.json" "$DIR/piper/$base.onnx.json"
done
echo "Chatterbox Multilingual (emotional neural TTS, 24 kHz, FR/EN + 21 langs) -> $DIR/chatterbox"
C="$HF/onnx-community/chatterbox-multilingual-ONNX/resolve/main"
mkdir -p "$DIR/chatterbox/onnx"
for f in tokenizer.json generation_config.json default_voice.wav; do fetch "$C/$f" "$DIR/chatterbox/$f"; done
for f in speech_encoder embed_tokens conditional_decoder language_model; do
  fetch "$C/onnx/$f.onnx" "$DIR/chatterbox/onnx/$f.onnx"
  fetch "$C/onnx/$f.onnx_data" "$DIR/chatterbox/onnx/$f.onnx_data"
done

echo "Done."
