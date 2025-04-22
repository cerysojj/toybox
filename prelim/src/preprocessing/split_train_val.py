import os
import shutil
import random
from collections import defaultdict

class TrainValSplitter:
    def __init__(self, train_dir, val_dir, split_ratio=0.2):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.split_ratio = split_ratio

    def get_objects(self):
        # Get all objects under each supercategory in the train directory
        objects = []
        for supercategory in os.listdir(self.train_dir):
            supercategory_path = os.path.join(self.train_dir, supercategory)
            if os.path.isdir(supercategory_path):
                for obj in os.listdir(supercategory_path):
                    obj_path = os.path.join(supercategory_path, obj)
                    if os.path.isdir(obj_path):
                        objects.append((supercategory, obj))  # Store supercategory and object together
        return objects

    def random_split(self):
        # Ensure validation directory exists
        os.makedirs(self.val_dir, exist_ok=True)
        
        objects = self.get_objects()

        # Shuffle and split objects
        num_val_objects = int(len(objects) * self.split_ratio)
        val_objects = random.sample(objects, num_val_objects)
        train_objects = [obj for obj in objects if obj not in val_objects]

        # Move selected objects to the validation directory, maintaining the structure
        for supercategory, obj in val_objects:
            src_path = os.path.join(self.train_dir, supercategory, obj)
            dst_path = os.path.join(self.val_dir, supercategory, obj)
            os.makedirs(os.path.join(self.val_dir, supercategory), exist_ok=True)
            shutil.move(src_path, dst_path)
            print(f"Moved {supercategory}/{obj} to validation set.")

        print(f"Random split completed: {len(train_objects)} objects in train, {len(val_objects)} objects in validation.")
        
    def stratified_split(self):
        os.makedirs(self.val_dir, exist_ok=True)
        
        class_groups = defaultdict(list)
        objects = self.get_objects()
        
        for supercategory, obj in objects:
            class_name = obj.split('_')[0]  # Extract class name (e.g., "cat")
            class_groups[class_name].append((supercategory, obj))
    
        train_objects = []
        val_objects = []
    
        for class_name, class_objects in class_groups.items():
            num_val_objects = int(len(class_objects) * self.split_ratio)
            val_samples = random.sample(class_objects, num_val_objects)  # Randomly select validation objects
            train_samples = [obj for obj in class_objects if obj not in val_samples]  # Remaining go to training
    
            val_objects.extend(val_samples)
            train_objects.extend(train_samples)

            print(f"Class {class_name}: {len(val_samples)} objects moved to validation.")
    
        # Move selected objects to the validation directory
        for supercategory, obj in val_objects:
            src_path = os.path.join(self.train_dir, supercategory, obj)
            dst_path = os.path.join(self.val_dir, supercategory, obj)
            os.makedirs(os.path.join(self.val_dir, supercategory), exist_ok=True)
            shutil.move(src_path, dst_path)
            print(f"Moved {supercategory}/{obj} to validation set.")

        print(f"Stratified split completed: {len(train_objects)} objects in train, {len(val_objects)} objects in validation.")

    def split(self, strategy="stratified"):
        if strategy == "stratified":
            self.stratified_split()
        elif strategy == "random":
            self.random_split()

