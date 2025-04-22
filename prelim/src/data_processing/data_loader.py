import os
import random
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from data_processing import toybox_dataset as td
from data_processing.mnist_dataset import CustomMNIST

class RandomColorTransform:
    def __init__(self, color_transform, grayscale_transform, initial_prob=0.1, increase_step=0.1):
        """
        color_transform: A PyTorch transformation pipeline for applying color image transformations.
        grayscale_transform: A PyTorch transformation pipeline for converting images to grayscale.
        initial_prob: The initial probability of images being in color.
        increase_step: The amount by which to increase the probability of choosing the color transform after each epoch.
        """
        self.color_transform = color_transform
        self.grayscale_transform = grayscale_transform
        self.prob = initial_prob
        self.increase_step = increase_step

    def __call__(self, img):
        if random.random() < self.prob:
            return self.color_transform(img)
        else:
            return self.grayscale_transform(img)

    def update_prob(self):
        self.prob = min(1, self.prob + self.increase_step)
        
def get_data_loaders(dataset_name="toybox", data_dir="data", flatten=False, batch_size=100, subset=None):

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.2, saturation=0.8)

    transform_color = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Resize(256),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=td.TOYBOX_MEAN, std=td.TOYBOX_STD)
    ])
    transform_grayscale = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(256),
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=td.TOYBOX_MEAN, std=td.TOYBOX_STD)
    ])
    transform_blur = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Resize(256),
        transforms.RandomResizedCrop(size=224),
        transforms.GaussianBlur(kernel_size=9, sigma=3.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=td.TOYBOX_MEAN, std=td.TOYBOX_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=td.TOYBOX_MEAN, std=td.TOYBOX_STD)
    ])

    random_color_transform = RandomColorTransform(transform_color, transform_grayscale)
    active_transform = None 

    if dataset_name == "MNIST":

        # Load the training dataset
        full_train = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform_grayscale)
        full_train = CustomMNIST(full_train)
        # Split the training dataset into training and validation subsets
        train_dataset, val_dataset = random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(1))

        # Load the test dataset
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform_grayscale)
        test_dataset = CustomMNIST(test_dataset)
        unique_labels = list(range(10))

    elif dataset_name == "toybox":

        unique_labels = td.get_unique_labels(data_dir)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        train_dataset = td.ToyboxDataset(data_dir=data_dir, subset='train', transform=transform_color, label_to_index=label_to_index, flatten=flatten)
        val_dataset = td.ToyboxDataset(data_dir=data_dir, subset='val', transform=transform_color, label_to_index=label_to_index, flatten=flatten)
        test_dataset = td.ToyboxDataset(data_dir=data_dir, subset='test', transform=transform_color, label_to_index=label_to_index, flatten=flatten)

    elif dataset_name == "toybox_grayscale":

        unique_labels = td.get_unique_labels(data_dir)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        train_dataset = td.ToyboxDataset(data_dir=data_dir, subset='train', transform=transform_grayscale, label_to_index=label_to_index, flatten=flatten)
        val_dataset = td.ToyboxDataset(data_dir=data_dir, subset='val', transform=transform_grayscale, label_to_index=label_to_index, flatten=flatten)
        test_dataset = td.ToyboxDataset(data_dir=data_dir, subset='test', transform=transform_grayscale, label_to_index=label_to_index, flatten=flatten)

    elif dataset_name == "toybox_random_color":

        unique_labels = td.get_unique_labels(data_dir)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        active_transform = random_color_transform

        train_dataset = td.ToyboxDataset(data_dir=data_dir, subset='train', transform=random_color_transform, label_to_index=label_to_index, flatten=flatten)
        val_dataset = td.ToyboxDataset(data_dir=data_dir, subset='val', transform=transform_color, label_to_index=label_to_index, flatten=flatten)
        test_dataset = td.ToyboxDataset(data_dir=data_dir, subset='test', transform=transform_color, label_to_index=label_to_index, flatten=flatten)

    elif dataset_name == "toybox_blur":

        unique_labels = td.get_unique_labels(data_dir)
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

        train_dataset = td.ToyboxDataset(data_dir=data_dir, subset='train', transform=transform_blur, label_to_index=label_to_index, flatten=flatten)
        val_dataset = td.ToyboxDataset(data_dir=data_dir, subset='val', transform=transform_blur, label_to_index=label_to_index, flatten=flatten)
        test_dataset = td.ToyboxDataset(data_dir=data_dir, subset='test', transform=transform_blur, label_to_index=label_to_index, flatten=flatten)

    elif dataset_name == "toybox_pt":
        rng = np.random.default_rng(seed=5)
        unique_labels = td.TOYBOX_CLASSES
        train_dataset = td.ToyboxDatasetPT(rng=rng, data_path=data_dir, train=True, hypertune=True, transform=transform_color, num_images_per_class = 1500)
        val_dataset = td.ToyboxDatasetPT(rng=rng, data_path=data_dir, train=False, hypertune=True, transform=transform_test, num_images_per_class = 1500)
        test_dataset = td.ToyboxDatasetPT(rng=rng, data_path=data_dir, train=False, hypertune=False, transform=transform_test, num_images_per_class = 1500)

    elif dataset_name in ("toybox_objects", "objects_blur"):
        rng = np.random.default_rng(seed=5)
        unique_labels = list(range(360))
        train_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=True, hypertune=True, transform=transform_color, num_images_per_object = 100)
        val_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=False, hypertune=True, transform=transform_test, num_images_per_object = 100)
        test_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=False, hypertune=False, transform=transform_test, num_images_per_object = -1)    
  
    elif dataset_name == "toybox_objects_grayscale":
        rng = np.random.default_rng(seed=5)
        unique_labels = list(range(360))
        train_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=True, hypertune=True, transform=transform_grayscale, num_images_per_object = 100)
        val_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=False, hypertune=True, transform=transform_grayscale, num_images_per_object = 100)
        test_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=False, hypertune=False, transform=transform_grayscale, num_images_per_object = -1)
        
    elif dataset_name == "toybox_objects_blur":
        rng = np.random.default_rng(seed=5)
        unique_labels = list(range(360))
        train_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=True, hypertune=True, transform=transform_blur, num_images_per_object = 100)
        val_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=False, hypertune=True, transform=transform_blur, num_images_per_object = 100)
        test_dataset = td.ToyboxDatasetObjectLevel(rng=rng, data_path=data_dir, train=False, hypertune=False, transform=transform_blur, num_images_per_object = -1)

    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    if subset is not None:
        total = len(train_dataset)
        num_samples = int(total * subset)
        indices = np.random.choice(total, num_samples, replace=False).tolist()
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=500, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    return train_loader, val_loader, test_loader, unique_labels, active_transform
