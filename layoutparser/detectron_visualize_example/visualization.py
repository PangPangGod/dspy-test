import cv2
import layoutparser as lp
import numpy as np
from PIL import Image

import fitz  # PyMuPDF
import numpy as np
from datetime import datetime

def save_layout_to_file(layout, file_path):
    with open(file_path, 'a') as file:
        file.write("="*30+"\n")
        file.write(str(datetime.now())+"\n")
        for element in layout:
            x1, y1, x2, y2 = element.coordinates
            element_info = f"Type: {element.type}, " \
                           f"Coordinates: ({x1}, {y1}, {x2}, {y2}), " \
                           f"Score: {element.score}\n"
            file.write(element_info)
        file.write("="*30)

def pdf_to_images(pdf_path):
    """PDF 파일을 페이지별 이미지로 변환합니다."""
    document = fitz.open(pdf_path)  # PDF 문서 열기
    images = []
    for page in document:  # 각 페이지에 대해 반복
        # 페이지를 이미지로 변환 (300 DPI)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        # 이미지 데이터를 numpy 배열로 변환
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:  # 포맷이 RGBA인 경우 RGB로 변환 (알파 채널 제거)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        images.append(img)  # 이미지 리스트에 추가
    return images

def visualize_layout_detection(image, output_path):
    if image.shape[2] == 3:  # 이미지가 RGB인 경우 직접 사용
        image_rgb = image
    else:  # 그 외의 경우 BGR에서 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.3]
    )

    layout = model.detect(image_rgb)
    ## layout 정보 저장
    save_layout_to_file(layout, "result.txt")

    vis_image = lp.draw_box(image_rgb, layout, box_width=3, show_element_type=True)

    # PIL 이미지 객체를 NumPy 배열로 변환하고 BGR로 변경 후 파일로 저장
    if isinstance(vis_image, Image.Image):
        vis_image = np.array(vis_image)
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    else:
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, vis_image_bgr)
    print("Image saved to", output_path)

if __name__ == '__main__':
    pdf_path = 'example/sample_file.pdf'
    images = pdf_to_images(pdf_path)
    for idx, image in enumerate(images):
        output_path = f'output_layout_{idx}.png'
        visualize_layout_detection(image, output_path)
