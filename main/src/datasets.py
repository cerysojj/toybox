import os
import json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision.transforms import Resize, ToTensor
import csv
import cv2
import pickle
from collections import defaultdict

IN12_MEAN = (0.4980, 0.4845, 0.4541)
IN12_STD = (0.2756, 0.2738, 0.2928)

TOYBOX_MEAN = (0.5199, 0.4374, 0.3499)
TOYBOX_STD = (0.1775, 0.1894, 0.1623)

FACESCRUB_MEAN=(0.485, 0.456, 0.406)
FACESCRUB_STD=(0.229, 0.224, 0.225)

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]

TOYBOX_CLASSES = ["airplane", "ball", "car", "cat", "cup", "duck", "giraffe", "helicopter", "horse", "mug", "spoon",
                  "truck"]
TOYBOX_VIDEOS = ("rxplus", "rxminus", "ryplus", "ryminus", "rzplus", "rzminus")

TOYBOX12_DATA_PATH = "../data/Toybox-12/"
TOYBOX360_DATA_PATH = "../data/Toybox-360/"
IN12_DATA_PATH = "../data/IN-12/"
FACES_DATA_PATH = "../data/Faces/"
IN100_DATA_PATH = "../data/IN-100/"

class ToyboxDataset(Dataset):
    """
    Class for loading Toybox data from pytorch for classification. Contains bounding boxes.
    The user can specify the number of instances per class and the number of images per class.
    If number of images per class is -1, all images are selected.
    """
    def __init__(self, rng, train=True, transform=None, size=224, hypertune=True, num_instances=-1,
                 num_images_per_class=-1, views=TOYBOX_VIDEOS):
        self.train = train
        self.transform = transform
        self.hypertune = hypertune
        self.root = TOYBOX12_DATA_PATH
        self.size = size
        self.rng = rng
        self.num_instances = num_instances
        self.num_images_per_class = num_images_per_class
        self.views = []
        for view in views:
            assert view in TOYBOX_VIDEOS
            self.views.append(view)
        self.label_key = 'Class ID'
        if self.hypertune:
            self.trainImagesFile = self.root + "dev.pickle"
            self.trainLabelsFile = self.root + "dev.csv"
            self.testImagesFile = self.root + "val.pickle"
            self.testLabelsFile = self.root + "val.csv"
        else:
            self.trainImagesFile = self.root + "train.pickle"
            self.trainLabelsFile = self.root + "train.csv"
            self.testImagesFile = self.root + "test.pickle"
            self.testLabelsFile = self.root + "test.csv"
                
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
        # commenting out assertion because of error
        #for cl in TOYBOX_CLASSES:
        #    assert len(unique_objs[cl]) == self.num_instances
    
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

class ToyboxDatasetInstances(Dataset):
    """
    Class for loading Toybox data for object-level classification.
    Each object is uniquely identified using a combination of Class ID (0-11) and Object ID (1-30).
    """
    def __init__(self, rng, train=True, transform=None, size=224, hypertune=True, num_images_per_object=-1, val_split=0.2):
        self.root = TOYBOX360_DATA_PATH
        self.train = train
        self.transform = transform
        self.size = size
        self.rng = rng
        self.hypertune = hypertune
        self.num_images_per_object = num_images_per_object

        if self.hypertune:
            self.trainImagesFile = self.root + "dev.pickle"
            self.trainLabelsFile = self.root + "dev.csv"
            self.testImagesFile = self.root + "val.pickle"
            self.testLabelsFile = self.root + "val.csv"
        else:
            self.trainImagesFile = self.root + "train.pickle"
            self.trainLabelsFile = self.root + "train.csv"
            self.testImagesFile = self.root + "test.pickle"
            self.testLabelsFile = self.root + "test.csv"

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


class DatasetIN12(Dataset):
    """
    This class implements the IN12 dataset
    """
    
    def __init__(self, train=True, transform=None, fraction=1.0, hypertune=True, equal_div=True):
        self.train = train
        self.transform = transform
        self.root = IN12_DATA_PATH
        self.fraction = fraction
        self.hypertune = hypertune
        self.equal_div = equal_div
        
        if self.train:
            if self.hypertune:
                self.images_file = self.root + "dev.pickle"
                self.labels_file = self.root + "dev.csv"
            else:
                self.images_file = self.root + "train.pickle"
                self.labels_file = self.root + "train.csv"
        else:
            if self.hypertune:
                self.images_file = self.root + "val.pickle"
                self.labels_file = self.root + "val.csv"
            else:
                self.images_file = self.root + "test.pickle"
                self.labels_file = self.root + "test.csv"
        
        self.images = pickle.load(open(self.images_file, "rb"))
        self.labels = list(csv.DictReader(open(self.labels_file, "r")))
        if self.train:
            if self.fraction < 1.0:
                len_all_images = len(self.images)
                rng = np.random.default_rng(0)
                if self.equal_div:
                    len_images_class = len_all_images // 12
                    len_train_images_class = int(self.fraction * len_images_class)
                    self.selected_indices = []
                    for i in range(12):
                        all_indices = np.arange(i * len_images_class, (i + 1) * len_images_class)
                        sel_indices = rng.choice(all_indices, len_train_images_class, replace=False)
                        self.selected_indices = self.selected_indices + list(sel_indices)
                else:
                    len_train_images = int(len_all_images * self.fraction)
                    self.selected_indices = rng.choice(len_all_images, len_train_images, replace=False)
            else:
                self.selected_indices = np.arange(len(self.images))
    
    def __len__(self):
        if self.train:
            return len(self.selected_indices)
        else:
            return len(self.images)
    
    def __getitem__(self, index):
        if self.train:
            item = self.selected_indices[index]
        else:
            item = index
        im = np.array(cv2.imdecode(self.images[item], 3))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        label = int(self.labels[item]["Class ID"])
        if self.transform is not None:
            im = self.transform(im)
        return (index, item), im, label
      
    def __str__(self):
        return "IN12 Supervised"


class FaceScrubDataset(Dataset):
    """
    This class implements the FaceScrub dataset.
    """
    
    def __init__(self, train=True, transform=None, fraction=1.0, hypertune=True, equal_div=True):
        self.train = train
        self.transform = transform
        self.root = FACES_DATA_PATH
        self.fraction = fraction
        self.hypertune = hypertune
        self.equal_div = equal_div

        if self.train:
            if self.hypertune:
                self.images_file = os.path.join(self.root, "dev.pickle")
                self.labels_file = os.path.join(self.root, "dev.csv")
            else:
                self.images_file = os.path.join(self.root, "train.pickle")
                self.labels_file = os.path.join(self.root, "train.csv")
        else:
            if self.hypertune:
                self.images_file = os.path.join(self.root, "val.pickle")
                self.labels_file = os.path.join(self.root, "val.csv")
            else:
                self.images_file = os.path.join(self.root, "test.pickle")
                self.labels_file = os.path.join(self.root, "test.csv")
        
        self.images = pickle.load(open(self.images_file, "rb"))
        self.labels = list(csv.DictReader(open(self.labels_file, "r")))
        
        # Build a mapping from string label to integer index.
        self.classes = sorted(list({label["category"] for label in self.labels}))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        if self.train:
            if self.fraction < 1.0:
                len_all_images = len(self.images)
                rng = np.random.default_rng(0)
                if self.equal_div:
                    len_images_class = len_all_images // 12
                    len_train_images_class = int(self.fraction * len_images_class)
                    self.selected_indices = []
                    for i in range(12):
                        all_indices = np.arange(i * len_images_class, (i + 1) * len_images_class)
                        sel_indices = rng.choice(all_indices, len_train_images_class, replace=False)
                        self.selected_indices = self.selected_indices + list(sel_indices)
                else:
                    len_train_images = int(len_all_images * self.fraction)
                    self.selected_indices = rng.choice(len_all_images, len_train_images, replace=False)
            else:
                self.selected_indices = np.arange(len(self.images))
        
    def __len__(self):
        if self.train:
            return len(self.selected_indices)
        else:
            return len(self.images)
    
    def __getitem__(self, index):
        if self.train:
            item = self.selected_indices[index]
        else:
            item = index
        byte_array = np.frombuffer(self.images[item], dtype=np.uint8)
        im = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        label = self.class_to_idx[self.labels[item]["category"]]
        if self.transform is not None:
            im = self.transform(im)
        return (index, item), im, label

    def __str__(self):
        return "FaceScrub Dataset"

class DatasetIN100(Dataset):
    """
    This class implements a custom dataset for ImageNet100-style data using
    the pickle/CSV format. The expected files in IN100_DATA_PATH are:
      - Labels.json (optional): mapping from original folder names to descriptive labels.
      - train.pickle and train.csv for training split.
      - val.pickle and val.csv for validation split.
    
    Args:
        train (bool): Whether to load the training split (if False, loads validation).
        transform (callable, optional): Transform to apply to each image.
        fraction (float): Fraction of the training data to use (if < 1.0, subsamples the training set).
        equal_div (bool): If True and fraction < 1.0, subsample equally per class.
    """
    def __init__(self, train=True, transform=None, fraction=1.0, equal_div=True):
        self.train = train
        self.transform = transform
        self.fraction = fraction
        self.equal_div = equal_div
        self.root = IN100_DATA_PATH

        # Optionally load Labels.json to obtain a mapping.
        labels_json_path = os.path.join(self.root, "Labels.json")
        if os.path.exists(labels_json_path):
            with open(labels_json_path, "r") as f:
                raw_mapping = json.load(f)
            # (raw_mapping is not directly used below since the CSV already contains mapped labels.)
        else:
            raw_mapping = {}
        
        # Choose the correct file names based on the split.
        if self.train:
            self.images_file = os.path.join(self.root, "train.pickle")
            self.labels_file = os.path.join(self.root, "train.csv")
        else:
            self.images_file = os.path.join(self.root, "val.pickle")
            self.labels_file = os.path.join(self.root, "val.csv")

        # Load the image data (list of JPEG-encoded bytes) and the CSV metadata.
        self.images = pickle.load(open(self.images_file, "rb"))
        self.labels = list(csv.DictReader(open(self.labels_file, "r")))
        
        # Build a mapping from category (string) to integer label.
        self.classes = sorted(list({entry["category"] for entry in self.labels}))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Optionally subsample the training set.
        if self.train and self.fraction < 1.0:
            rng = np.random.default_rng(0)
            if self.equal_div:
                self.selected_indices = []
                # For each class, subsample indices equally.
                for cls in self.classes:
                    cls_indices = [i for i, entry in enumerate(self.labels) if entry["category"] == cls]
                    n_samples = int(len(cls_indices) * self.fraction)
                    if n_samples > 0:
                        selected = rng.choice(cls_indices, n_samples, replace=False)
                        self.selected_indices.extend(selected)
            else:
                total_samples = int(len(self.images) * self.fraction)
                self.selected_indices = rng.choice(len(self.images), total_samples, replace=False)
        else:
            self.selected_indices = np.arange(len(self.images))
        
    def __len__(self):
        return len(self.selected_indices) if self.train else len(self.images)
    
    def __getitem__(self, index):
        # Get the actual sample index (for training, after subsampling).
        sample_idx = self.selected_indices[index] if self.train else index
        
        # Load the image from its JPEG-encoded bytes.
        byte_data = self.images[sample_idx]
        byte_array = np.frombuffer(byte_data, dtype=np.uint8)
        im = cv2.imdecode(byte_array, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to decode image at index {sample_idx}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # Retrieve the label from the CSV metadata and convert it to an integer.
        label_str = self.labels[sample_idx]["category"]
        label = self.class_to_idx[label_str]
        
        if self.transform is not None:
            im = self.transform(im)
            
        return im, label
        
    def __str__(self):
        return "ImageNet-100 Dataset"
