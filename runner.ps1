# Docker Desktop 실행 경로 설정
$dockerDesktopPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"

if (Test-Path $dockerDesktopPath) {
    Start-Process $dockerDesktopPath
} else {
    Write-Host "Docker Desktop 설치 경로를 찾을 수 없습니다."
    exit
}

Write-Host "Docker 서비스 시작을 기다리는 중..."
Start-Sleep -Seconds 5

docker run -p 8000:8000 -d --rm --name unstructured-api -e UNSTRUCTURED_PARALLEL_MODE_THREADS=3 downloads.unstructured.io/unstructured-io/unstructured-api:latest --port 8000 --host 0.0.0.0

Write-Host "Docker에서 'unstructured-api' 이미지 실행 완료."

## if conda
conda activate dspy
python documenthandle.py