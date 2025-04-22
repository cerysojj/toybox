import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor
import csv
import cv2
import pickle

TOYBOX_MEAN = (0.5199, 0.4374, 0.3499)
TOYBOX_STD = (0.1775, 0.1894, 0.1623)

TOYBOX_CLASSES = ["airplane", "ball", "car", "cat", "cup", "duck", "giraffe", "helicopter", "horse", "mug", "spoon",
                  "truck"]
TOYBOX_VIDEOS = ("rxplus", "rxminus", "ryplus", "ryminus", "rzplus", "rzminus")

def get_unique_labels(data_dir):
    """Identify all unique labels across all folds in the dataset."""
    unique_labels = set()
    
    # Iterate through subsets like 'train' and 'val'
    for subset in ['train', 'val']:
        subset_path = os.path.join(data_dir, subset)
        
        # Check if the subset path exists
        if os.path.exists(subset_path):
            # Go one level down to identify categories (e.g., 'animals', 'vehicles')
            for category_folder in os.listdir(subset_path):
                category_path = os.path.join(subset_path, category_folder)
                
                # Check if it's a directory (e.g., 'animals')
                if os.path.isdir(category_path):
                    # Now, go another level down to individual objects (e.g., 'cat_14_pivothead')
                    for object_folder in os.listdir(category_path):
                        object_path = os.path.join(category_path, object_folder)
                        
                        # Check if it's a directory (e.g., 'cat_14_pivothead')
                        if os.path.isdir(object_path):
                            # Extract the label by splitting the folder name
                            label = object_folder.split('_')[0]
                            unique_labels.add(label)
    
    return sorted(unique_labels)

class ToyboxDataset(Dataset):
    """Custom dataset class for toybox_sample_resized dataset."""
    def __init__(self, data_dir, subset, transform, label_to_index, flatten):
        self.data = []
        self.transform = transform
        self.label_to_index = label_to_index
        dataset_path = os.path.join(data_dir, subset)
        self.flatten = flatten

        print(f"Initializing ToyboxDataset for subset: {subset}, Dataset path: {dataset_path}")
        
        for root, _, files in os.walk(dataset_path):
            for frame_file in files:      
                if frame_file.endswith('.jpg'):
                    try:
                        relative_path = os.path.relpath(root, dataset_path)  # animals/cat_04_pivothead/cat_04_pivothead_hodgepodge
                        object_name = relative_path.split('/')[-1]           # cat_04_pivothead_hodgepodge
                        label = object_name.split('_')[0]                    # cat
                        object_number = object_name.split('_')[1]            # 04
                        object_id = f"{label}_{object_number}"               # cat_04
                        video_type = object_name.split('_')[-1]              # hodgepodge
                        video = f"{label}_{video_type}"                      # cat_hodgepodge
                        if label in self.label_to_index:
                            label_idx = self.label_to_index[label]
                            frame_path = os.path.join(root, frame_file)  # animals/cat_04_pivothead/cat_04_pivothead_hodgepodge/frame_0000.jpg
                            self.data.append((frame_path, label_idx, object_id, video))
                        else:
                            print(f"Skipping frame: {frame_file}, Label '{label}' not in label_to_index")
                    except Exception as e:
                        print(f"Error while processing file: {frame_file} in directory {root}: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label, object_id, video = self.data[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)
        except Exception as e:
            print(f"Error opening image at {image_path}: {e}")
            return None, label, object_id, video

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to image at {image_path}: {e}")
                return None, label, object_id, video
           
        if self.flatten:
            try:
                image = image.flatten() # flatten for MLP compatability
            except Exception as e:
                print(f"Error flattening image at {image_path}: {e}")
                return None, label, object_id, video
                
        return image, label, object_id, video

class ToyboxDatasetPT(Dataset):
    """
    Class for loading Toybox data from pytorch for classification. Contains bounding boxes.
    The user can specify the number of instances per class and the number of images per class.
    If number of images per class is -1, all images are selected.
    """
    def __init__(self, rng, data_path, train=True, transform=None, size=224, hypertune=True, num_instances=-1,
                 num_images_per_class=-1, views=TOYBOX_VIDEOS):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        self.size = size
        self.rng = rng
        self.num_instances = num_instances
        self.num_images_per_class = num_images_per_class
        self.views = []
        for view in views:
            assert view in TOYBOX_VIDEOS
            self.views.append(view)
        try:
            assert os.path.isdir(self.data_path)
        except AssertionError:
            raise AssertionError("Data directory not found:", self.data_path)
        self.label_key = 'Class ID'
        if self.hypertune:
            self.trainImagesFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_dev.pickle")
            self.trainLabelsFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_dev.csv")
            self.testImagesFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_val.pickle")
            self.testLabelsFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_val.csv")
        else:
            self.trainImagesFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_train.pickle")
            self.trainLabelsFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_train.csv")
            self.testImagesFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_test.pickle")
            self.testLabelsFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_test.csv")
        
        super().__init__()
        
        if self.train:
            self.indicesSelected = []
            with open(self.trainImagesFile, "rb") as pickleFile:
                self.train_data = pickle.load(pickleFile)
            with open(self.trainLabelsFile, "r") as csvFile:
                self.train_csvFile = list(csv.DictReader(csvFile))
            self.set_train_indices()
            self.verify_train_indices()
        else:
            with open(self.testImagesFile, "rb") as pickleFile:
                self.test_data = pickle.load(pickleFile)
            with open(self.testLabelsFile, "r") as csvFile:
                self.test_csvFile = list(csv.DictReader(csvFile))
    
    def __len__(self):
        if self.train:
            return len(self.indicesSelected)
        else:
            return len(self.test_data)
    
    def __getitem__(self, index):
        if self.train:
            actual_index = self.indicesSelected[index]
            img = cv2.imdecode(self.train_data[actual_index], 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)
            label = int(self.train_csvFile[actual_index][self.label_key])
        else:
            actual_index = index
            img = cv2.imdecode(self.test_data[index], 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.array(img)
            label = int(self.test_csvFile[index][self.label_key])
        
        if self.transform is not None:
            imgs = self.transform(img)
        else:
            imgs = img
        return (index, actual_index), imgs, label
    
    def __str__(self):
        return "Toybox"
    
    def verify_train_indices(self):
        """
        This method verifies that the indices chosen for training has the same number of instances
        per class as specified in self.num_instances.
        """
        unique_objs = {}
        for idx_selected in self.indicesSelected:
            cl = self.train_csvFile[idx_selected]['Class']
            if cl not in unique_objs.keys():
                unique_objs[cl] = []
            obj = int(self.train_csvFile[idx_selected]['Object'])
            if obj not in unique_objs[cl]:
                unique_objs[cl].append(obj)
            view = self.train_csvFile[idx_selected]['Transformation']
            assert view in self.views
        for cl in TOYBOX_CLASSES:
            assert len(unique_objs[cl]) == self.num_instances
    
    def set_train_indices(self):
        """
        This method sets the training indices based on the settings provided in init().
        """
        obj_dict = {}
        obj_id_dict = {}
        for row in self.train_csvFile:
            cl = row['Class']
            if cl not in obj_dict.keys():
                obj_dict[cl] = []
            obj = int(row['Object'])
            if obj not in obj_dict[cl]:
                obj_dict[cl].append(obj)
                obj_start_id = int(row['Obj Start'])
                obj_end_id = int(row['Obj End'])
                obj_id_dict[(cl, obj)] = (obj_start_id, obj_end_id)
        
        if self.num_instances < 0:
            self.num_instances = len(obj_dict['airplane'])
        
        assert self.num_instances <= len(obj_dict['airplane']), "Number of instances must be less than number " \
                                                                "of objects in CSV: {}".format(len(obj_dict['ball']))
        
        if self.num_images_per_class < 0:
            num_images_per_instance = [-1 for _ in range(self.num_instances)]
        else:
            num_images_per_instance = [int(self.num_images_per_class / self.num_instances) for _ in
                                       range(self.num_instances)]
            remaining = max(0, self.num_images_per_class - num_images_per_instance[0] * self.num_instances)
            idx_instance = 0
            while remaining > 0:
                num_images_per_instance[idx_instance] += 1
                idx_instance = (idx_instance + 1) % self.num_instances
                remaining -= 1
        
        for cl in obj_dict.keys():
            obj_list = obj_dict[cl]
            selected_objs = self.rng.choice(obj_list, self.num_instances, replace=False)
            assert len(selected_objs) == len(set(selected_objs))
            for idx_obj, obj in enumerate(selected_objs):
                start_row = obj_id_dict[(cl, obj)][0]
                end_row = obj_id_dict[(cl, obj)][1]
                all_possible_rows = [obj_row for obj_row in range(start_row, end_row + 1)]
                
                rows_with_specified_views = []
                for obj_row in all_possible_rows:
                    view_row = self.train_csvFile[obj_row]['Transformation']
                    if view_row in self.views:
                        rows_with_specified_views.append(obj_row)
                num_images_obj = len(rows_with_specified_views)
                
                num_required_images = num_images_per_instance[idx_obj]
                if num_required_images < 0:
                    num_required_images = num_images_obj
                
                selected_indices_obj = []
                while num_required_images >= num_images_obj:
                    for idx_row in rows_with_specified_views:
                        selected_indices_obj.append(idx_row)
                    num_required_images -= num_images_obj
                additional_rows = self.rng.choice(rows_with_specified_views, num_required_images,
                                                  replace=False)
                assert len(additional_rows) == len(set(additional_rows))
                
                for idx_row in additional_rows:
                    selected_indices_obj.append(idx_row)
                for idx_row in selected_indices_obj:
                    assert start_row <= idx_row <= end_row
                    row_video = self.train_csvFile[idx_row]['Transformation']
                    assert row_video in self.views
                    self.indicesSelected.append(idx_row)

class ToyboxDatasetObjectLevel(Dataset):
    """
    Class for loading Toybox data for object-level classification.
    Each object is uniquely identified using a combination of Class ID (0-11) and Object ID (1-30).
    """
    def __init__(self, rng, data_path, train=True, transform=None, size=224, hypertune=True, num_images_per_object=-1, val_split=0.2):
        self.data_path = data_path
        self.train = train
        self.transform = transform
        self.size = size
        self.rng = rng
        self.hypertune = hypertune
        self.num_images_per_object = num_images_per_object

        try:
            assert os.path.isdir(self.data_path)
        except AssertionError:
            raise AssertionError("Data directory not found:", self.data_path)

        if self.hypertune:
            self.trainImagesFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_dev.pickle")
            self.trainLabelsFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_dev.csv")
            self.testImagesFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_val.pickle")
            self.testLabelsFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_val.csv")
        else:
            self.trainImagesFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_train.pickle")
            self.trainLabelsFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_train.csv")
            self.testImagesFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_test.pickle")
            self.testLabelsFile = os.path.join(self.data_path, "toybox_data_interpolated_cropped_test.csv")

        if self.train:
            with open(self.trainImagesFile, "rb") as f:
                self.data = pickle.load(f)
            with open(self.trainLabelsFile, "r") as f:
                self.csvFile = list(csv.DictReader(f))
        else:
            with open(self.testImagesFile, "rb") as f:
                self.data = pickle.load(f)
            with open(self.testLabelsFile, "r") as f:
                self.csvFile = list(csv.DictReader(f))
        
        self.indicesSelected = self.select_images_per_object() if self.num_images_per_object != -1 else list(range(len(self.data)))

    def select_images_per_object(self):
        """
        Selects a fixed number of images per object.
        Groups frames by (Class ID, Object ID) and for each group,
        randomly samples up to num_images_per_object indices.
        """
        obj_dict = {}
        for idx, row in enumerate(self.csvFile):
            class_id = int(row["Class ID"])
            object_id = int(row["Object"])
            key = (class_id, object_id)
            if key not in obj_dict:
                obj_dict[key] = []
            obj_dict[key].append(idx)
        
        selected = []
        for key, indices in obj_dict.items():
            # Randomly sample without replacement
            num_to_select = min(len(indices), self.num_images_per_object)
            selected.extend(self.rng.choice(indices, num_to_select, replace=False))
        return selected

    def get_unique_object_label(self, class_id, object_id):
        return (class_id * 30) + (object_id - 1)

    def __len__(self):
        return len(self.indicesSelected)

    def __getitem__(self, index):
        actual_index = self.indicesSelected[index]

        img = cv2.imdecode(self.data[actual_index], 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        class_id = int(self.csvFile[actual_index]['Class ID'])
        object_id = int(self.csvFile[actual_index]['Object'])
        unique_label = self.get_unique_object_label(class_id, object_id)

        if self.transform:
            img = self.transform(img)

        return (index, actual_index), img, unique_label

    def __str__(self):
        return "Toybox Object-Level Classification"
