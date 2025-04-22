from image_resizer import ImageResizer
from split_train_val import TrainValSplitter

# Resize images
input_directory = '/home/s2186747/data/project/toybox_sample'
output_directory = '/home/s2186747/data/project/toybox_sample_v2'
resizer = ImageResizer(input_directory, output_directory)
resizer.resize_and_save_images()

# Split train and val
train_dir = '/home/s2186747/data/project/toybox_sample_v2/train'
val_dir = '/home/s2186747/data/project/toybox_sample_v2/val'
split_ratio = 0.2
splitter = TrainValSplitter(train_dir, val_dir, split_ratio)
splitter.split()
