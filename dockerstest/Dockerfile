FROM python:3.12.2-slim

WORKDIR /app
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# 컨테이너 시작 시 실행될 명령
CMD ["python", "./helloworld.py"]