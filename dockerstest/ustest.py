## Unstructured api test
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared

api_key = "your_api_key"

s = UnstructuredClient(
    server_url="http://localhost:8000",
    api_key_auth=api_key,
)

filename = "dockerstest/pdf/sample_file.pdf"  # 전송할 파일의 경로

with open(filename, "rb") as f:
    files = shared.Files(
        content=f.read(),
        file_name=filename,
    )

req = shared.PartitionParameters(
    files=files,
    chunking_strategy="by_title",
    strategy='hi_res',
)

try:
    resp = s.general.partition(req)
    print("처리된 요소:", resp.elements)
except Exception as e:
    print("오류 발생:", e)