import os
import platform
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="MTEB Model name from HuggingFace")
    parser.add_argument("-p", "--path", help="Model Save path")
    args = parser.parse_args()
    return args

class EmbeddingDownLoader() :
    def __init__ (
    self,
    model = None,
    path = None,
    ) -> None :
        try :
            from sentence_transformers import SentenceTransformer
        except ImportError :
            raise ImportError(
                "package not found. try install sentence_transformers."
                "try following command : pip install sentence-transformers"
            )
        
        self.model = model
        self.path = path
        
    def download(self) -> None :
        from sentence_transformers import SentenceTransformer

        # 경로 지정 안해놨으면 현재 스크립트 실행되는 곳에 모델명으로 파일 생성되게 만듦.
        if self.path is None :
            self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)),self.model)
        
        ### 다운로드 함수.
        downloader = SentenceTransformer(model_name_or_path=self.model)
        os.makedirs(self.path)
        downloader.save(self.path)

        print(f'model {self.model} download at path {self.path}.')

        return None

if __name__ == "__main__":
    args = get_args()
    model_name = args.model
    path = args.path if args.path else os.path.join(os.getcwd(), args.model)
    os.makedirs(path, exist_ok=True)

    loader = EmbeddingDownLoader(model=model_name, path=path)
    loader.download()