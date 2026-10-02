param(
    [string]$Python = '',
    [int]$Port = 8501
)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $candidates = @(
        (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
        (Join-Path $env:USERPROFILE 'Desktop/huggface_material/mtl1/.venv/Scripts/python.exe')
    )
    $Python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Python) { $Python = (Get-Command python).Source }
}
$runtime = Join-Path $PSScriptRoot '.runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
while (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { $Port++ }
$appPath = Join-Path $PSScriptRoot 'app.py'
$outLog = Join-Path $runtime "streamlit-$Port.log"
$errLog = Join-Path $runtime "streamlit-$Port.err.log"
$arguments = @('-m', 'streamlit', 'run', ('"' + $appPath + '"'),
    '--server.address', '127.0.0.1', '--server.port', "$Port", '--server.headless', 'true',
    '--browser.gatherUsageStats', 'false')
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $PSScriptRoot `
    -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$url = "http://127.0.0.1:$Port"
@{ pid = $process.Id; url = $url; stdout = $outLog; stderr = $errLog } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtime 'server.json') -Encoding utf8
$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    if ($process.HasExited) { throw "Streamlit exited. See $errLog" }
    try {
        $result = Invoke-WebRequest "$url/_stcore/health" -TimeoutSec 2
        if ($result.StatusCode -eq 200) { $healthy = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $healthy) { throw "Streamlit did not become healthy. See $errLog" }
Write-Output "URL: $url"
Write-Output "PID: $($process.Id)"
Write-Output "Logs: $runtime"
