# Docker Desktop 실행 경로 설정
$dockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

if (Test-Path $dockerDesktopPath) {
    Start-Process $dockerDesktopPath
} else {
    Write-Host "Cannot find Docker Desktop."
    exit
}

Write-Host "Waiting for Docker service to start..."
Start-Sleep -Seconds 10

### docker image execution
docker run -p 8000:8000 -d --rm --name unstructured-api -e UNSTRUCTURED_PARALLEL_MODE_THREADS=3 downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0
Write-Host "Docker 'unstructured-api' image has been successfully launched."

## if conda
conda activate dspy
python documenthandle.py

### 정리
docker kill unstructured-api
Write-Host "Deleted used Docker image"
$dockerProcessName = "Docker Desktop"
$dockerProcess = Get-Process -Name $dockerProcessName -ErrorAction SilentlyContinue
if ($dockerProcess) {
    $dockerProcess | Stop-Process -Force
    Write-Host "Docker Desktop process has been terminated."
} else {
    Write-Host "Cannot find Docker Desktop process."
}