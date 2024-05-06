import fitz  # PyMuPDF
import layoutparser as lp
import numpy as np

fname = "example/sample_file.pdf"

def extract_text_within_bounds(page, bounds):
    """주어진 경계 내의 텍스트를 추출하는 함수"""
    text = page.get_text("dict", clip=fitz.Rect(bounds))
    full_text = " ".join([block['text'] for block in text['blocks']])
    return full_text

def process_text_element(text):
    """추출된 텍스트를 처리하는 함수"""
    print("Processed Text Element:")
    print(text)

def process_table_element(text):
    """추출된 테이블을 처리하는 함수"""
    print("Processed Table Element:")
    print(text)
    # 테이블 관련 처리 로직 추가

# PDF 파일 열기
document = fitz.open(fname)

# layoutparser 모델 로드

if lp.is_detectron2_available():
    model = lp.Detectron2LayoutModel(
            config_path ='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3],
        )
    
    