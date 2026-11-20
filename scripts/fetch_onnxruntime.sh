#!/usr/bin/env bash
# Offline builds only: fetch a static ONNX Runtime 1.22 into lib/onnxruntime, then build with
#   ORT_LIB_LOCATION=$PWD/lib/onnxruntime cargo build --release --no-default-features
# (Normal builds don't need this: cargo downloads ONNX Runtime automatically.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)  T=aarch64-apple-darwin;      SHA=00FBFD6F08BAC2A4E28C66723AF900D58D1B4B1C73EFBA6290637CD3019883D5 ;;
  Darwin-x86_64) T=x86_64-apple-darwin;       SHA=E0538783248BBB77D2E97556134B8CC47CFC35AC3D762067D51EAF8355C0F48E ;;
  Linux-x86_64)  T=x86_64-unknown-linux-gnu;  SHA=ED1716DE95974BF47AB0223CA33734A0B5A5D09A181225D0E8ED62D070AEA893 ;;
  Linux-aarch64) T=aarch64-unknown-linux-gnu; SHA=24E4760207136FC50B854BB5012AB81DE6189039CF6D4FD3F5B8D3DB7E929F1E ;;
  *) echo "unsupported platform"; exit 1 ;;
esac
[[ -s "$ROOT/lib/onnxruntime/lib/libonnxruntime.a" ]] && { echo "already present"; exit 0; }
mkdir -p "$ROOT/lib"
curl -fL --retry 3 -sS -o "$ROOT/lib/ort.tgz" "https://cdn.pyke.io/0/pyke:ort-rs/ms@1.22.0/$T.tgz"
GOT=$(shasum -a 256 "$ROOT/lib/ort.tgz" | cut -d' ' -f1 | tr a-f A-F)
[[ "$GOT" == "$SHA" ]] || { echo "checksum mismatch"; rm -f "$ROOT/lib/ort.tgz"; exit 1; }
tar xzf "$ROOT/lib/ort.tgz" -C "$ROOT/lib" && rm "$ROOT/lib/ort.tgz"
echo "ok: lib/onnxruntime"
