param(
  [string]$ComposeFile = "docker-compose.rl_hard.yml",
  [int]$HealthTimeoutSec = 150,
  [switch]$ForceRecreate,
  [switch]$SkipPause
)

$ErrorActionPreference = "Stop"

$services = @(
  "tfm-server-1", "tfm-server-2", "tfm-server-3", "tfm-server-4", "tfm-server-5",
  "tfm-server-6", "tfm-server-7", "tfm-server-8", "tfm-server-9", "tfm-server-10"
)
$ports = 8081..8090

function Invoke-TrainingControl([string]$action) {
  $url = "http://localhost:5000/control/$action"
  try {
    Invoke-RestMethod -Method Post -Uri $url -TimeoutSec 6 | Out-Null
    Write-Host "[ok] coordinator $action"
    return $true
  } catch {
    Write-Warning "Coordinator $action failed ($url): $($_.Exception.Message)"
    return $false
  }
}

function Assert-LastExit([string]$step) {
  if ($LASTEXITCODE -ne 0) {
    throw "$step failed with exit code $LASTEXITCODE"
  }
}

$paused = $false
if (-not $SkipPause) {
  $paused = Invoke-TrainingControl "pause"
}

if ($ForceRecreate) {
  Write-Host "[info] force recreating tfm servers..."
  & docker compose -f $ComposeFile up -d --force-recreate --no-deps @services
  Assert-LastExit "docker compose up --force-recreate"
} else {
  Write-Host "[info] restarting tfm servers..."
  & docker compose -f $ComposeFile restart @services
  Assert-LastExit "docker compose restart"
}

$pending = @{}
for ($i = 0; $i -lt $services.Length; $i++) {
  $pending[$services[$i]] = $ports[$i]
}

$deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
while ((Get-Date) -lt $deadline -and $pending.Count -gt 0) {
  foreach ($service in @($pending.Keys)) {
    $port = $pending[$service]
    try {
      $response = Invoke-WebRequest -Uri "http://localhost:$port/" -UseBasicParsing -TimeoutSec 2
      if ([int]$response.StatusCode -eq 200) {
        Write-Host "[ok] $service healthy on :$port"
        $pending.Remove($service)
      }
    } catch {
      # Keep waiting.
    }
  }
  if ($pending.Count -gt 0) {
    Start-Sleep -Seconds 2
  }
}

if ($pending.Count -gt 0) {
  $failed = ($pending.Keys | Sort-Object) -join ", "
  throw "Health checks timed out for: $failed"
}

if ($paused) {
  Invoke-TrainingControl "resume" | Out-Null
}

Write-Host "[done] all tfm servers restarted and healthy."
