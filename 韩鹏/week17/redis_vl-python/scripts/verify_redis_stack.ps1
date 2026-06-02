param(
    [string]$ContainerName = "redis-stack"
)

$running = docker ps --filter "name=$ContainerName" --filter "status=running" --format "{{.Names}}"
if (-not $running) {
    Write-Error "Container '$ContainerName' is not running. Start it with: docker compose up -d"
    exit 1
}

$pingResult = docker exec $ContainerName redis-cli ping
if ($pingResult -ne "PONG") {
    Write-Error "Container '$ContainerName' is running, but redis-cli ping failed: $pingResult"
    exit 1
}

Write-Host "Redis Stack verification passed for container '$ContainerName'."

