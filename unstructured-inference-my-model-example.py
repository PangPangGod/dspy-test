import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from pathlib import Path
import argparse
import platform
import json
from unstructured_inference.models.base import UnstructuredModel
from unstructured_inference.inference.layout import LayoutElement

class TATRTableDetectionModel(UnstructuredModel):
    def initialize(self):
        self.model = torch.load('path_to/tatr_model.pth')
        self.model.eval()

    def predict(self, image: Image.Image):
        input_tensor = self.preprocess(image)
        with torch.no_grad():
            predictions = self.model(input_tensor)
        return self.convert_to_layout_elements(predictions)

    def preprocess(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image).unsqueeze(0)

    def convert_to_layout_elements(self, predictions):
        elements = []
        for pred in predictions:
            bbox = pred['boxes'].tolist()
            score = pred['scores'].item()
            elements.append(LayoutElement(
                category="Table",
                bbox=bbox,
                score=score,
                metadata={"additional_info": "Detected table"}
            ))
        return elements

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input PDF data directory')
    parser.add_argument('--output_dir', help='output image directory')
    parser.add_argument('--postfix', choices=('.jpg', '.png'))
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    root = Path(args.input_dir)
    res = Path(args.output_dir)
    res_image = res / 'images'
    res_image.mkdir(exist_ok=True, parents=True)
    res_word = res / 'words'
    res_word.mkdir(exist_ok=True, parents=True)
    postfix = args.postfix
    os_type = platform.system()
    separator = '/' if os_type == 'Linux' else '\\'

    model = TATRTableDetectionModel()
    model.initialize()

    pdf_files = sorted(list(root.glob(f'**{separator}*.pdf'))) + sorted(list(root.glob(f'**{separator}*.PDF')))
    for fname in pdf_files:
        try:
            pages = convert_from_path(fname, dpi=300)
            pages_fitz = fitz.open(fname)
            for page_num, (page, page_fitz) in enumerate(zip(pages, pages_fitz)):
                image_path = res_image / f'{fname.stem}_page{page_num}{postfix}'
                page.save(image_path)
                page_image = Image.open(image_path)
                layout_elements = model.predict(page_image)
                # Optional: Draw bounding boxes on the image
                draw = ImageDraw.Draw(page_image)
                for element in layout_elements:
                    draw.rectangle(element.bbox, outline="red", width=2)
                page_image.save(image_path)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")
