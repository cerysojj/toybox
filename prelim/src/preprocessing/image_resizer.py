import sys
from PIL import Image
import os

class ImageResizer:
    def __init__(self, input_dir, output_dir, target_size=(227, 227)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_size = target_size

    def resize_and_save_images(self):
        for root, _, files in os.walk(self.input_dir):
            for filename in files:
                if filename.lower().endswith(".jpeg") or filename.lower().endswith(".jpg"):
                    img_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(root, self.input_dir)
                    target_dir = os.path.join(self.output_dir, relative_path)
                    os.makedirs(target_dir, exist_ok=True)

                    self._resize_and_save(img_path, target_dir, filename)
                    
    def _resize_and_save(self, img_path, target_dir, filename):
        # Internal helper method to resize and save individual images
        with Image.open(img_path) as img:
            img_resized = img.resize(self.target_size, Image.LANCZOS)
            output_path = os.path.join(target_dir, filename)
            img_resized.save(output_path, format="JPEG")
            print(f"Resized and saved: {output_path}")
