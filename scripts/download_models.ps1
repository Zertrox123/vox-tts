# One-time model download for Windows. After this, vox runs fully offline.
# Usage:  powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1 [-Dir models]
param([string]$Dir = (Join-Path (Split-Path -Parent $PSScriptRoot) "models"))
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar
$HF = "https://huggingface.co"

function Fetch([string]$Url, [string]$Dest) {
    if ((Test-Path $Dest) -and ((Get-Item $Dest).Length -gt 0)) { Write-Host "  ok   $(Split-Path -Leaf $Dest)"; return }
    Write-Host "  get  $(Split-Path -Leaf $Dest)"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dest) | Out-Null
    for ($i = 1; $i -le 3; $i++) {
        try { Invoke-WebRequest -Uri $Url -OutFile "$Dest.part" -UseBasicParsing; Move-Item -Force "$Dest.part" $Dest; return }
        catch { if ($i -eq 3) { throw } ; Start-Sleep -Seconds 2 }
    }
}

Write-Host "Kokoro-82M -> $Dir\kokoro"
$K = "$HF/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main"
Fetch "$K/onnx/model.onnx" "$Dir\kokoro\model.onnx"
Fetch "$K/tokenizer.json" "$Dir\kokoro\tokenizer.json"
foreach ($v in "af_heart","af_bella","af_nicole","af_sky","am_michael","am_fenrir","am_puck","ff_siwis") {
    Fetch "$K/voices/$v.bin" "$Dir\kokoro\voices\$v.bin"
}

Write-Host "Piper voices -> $Dir\piper"
$P = "$HF/rhasspy/piper-voices/resolve/main"
foreach ($spec in "en/en_US/lessac/high","en/en_US/ryan/high","fr/fr_FR/tom/medium","fr/fr_FR/siwis/medium","fr/fr_FR/upmc/medium") {
    $lang, $loc, $name, $q = $spec.Split("/")
    $base = "$loc-$name-$q"
    Fetch "$P/$spec/$base.onnx" "$Dir\piper\$base.onnx"
    Fetch "$P/$spec/$base.onnx.json" "$Dir\piper\$base.onnx.json"
}

Write-Host "Chatterbox Multilingual -> $Dir\chatterbox"
$C = "$HF/onnx-community/chatterbox-multilingual-ONNX/resolve/main"
foreach ($f in "tokenizer.json","generation_config.json","default_voice.wav") { Fetch "$C/$f" "$Dir\chatterbox\$f" }
foreach ($f in "speech_encoder","embed_tokens","conditional_decoder","language_model") {
    Fetch "$C/onnx/$f.onnx" "$Dir\chatterbox\onnx\$f.onnx"
    Fetch "$C/onnx/$f.onnx_data" "$Dir\chatterbox\onnx\$f.onnx_data"
}
Write-Host "Done."
