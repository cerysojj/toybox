import sys
import os
sys.path.append(os.path.abspath("scripts"))
from data_processing.toybox_dataset import ToyboxDataset, get_unique_labels
from torchvision import transforms
from PIL import Image
import numpy as np

def debug_toybox_dataset(data_dir):
    """Debug the ToyboxDataset"""
    transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor()
        ])
    flatten = False

    # Print unique labels
    unique_labels = get_unique_labels(data_dir)
    print("Unique labels identified:", unique_labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    print("Label-to-index mapping:", label_to_index)

    dataset = ToyboxDataset(data_dir, 'val', transform, label_to_index, flatten)

    # Print dataset length
    print(f"Dataset size: {len(dataset)}")

    # Iterate through the dataset and print details
    for idx in range(1):
        try:
            image, label, obj_id, video = dataset[idx]
            print(f"Index: {idx}, Label: {label}, Object ID: {obj_id}, Video: {video}")
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            print(f"Detailed traceback:", exc_info=True)

if __name__ == "__main__":
    
    # This dummy also produces a type error: TypeError: expected np.ndarray (got numpy.ndarray)
    img = Image.fromarray((np.random.rand(227, 227, 3) * 255).astype(np.uint8))  # Random image
    transform = transforms.ToTensor()
    tensor = transform(img)
    print(f"Transformed random image tensor shape: {tensor.shape}")
    
    dataset_path = "/home/s2186747/data/project/toybox_sample_v2"
    debug_toybox_dataset(dataset_path)
