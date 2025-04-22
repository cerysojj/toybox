import data_processing.dataset_toybox as dt
import numpy as np
import torchvision.transforms.v2 as v2
import torch.utils.data as torchdata
import utils
import torchvision.models as models
import torch.nn as nn
"""
rng = np.random.default_rng(seed=5)
data_dir = "/home/s2186747/ug-project/data/"

transform = v2.Compose([
  v2.ToPILImage(),
  v2.ToTensor(),
  v2.Normalize(mean=dt.TOYBOX_MEAN, std=dt.TOYBOX_STD)
])

toybox_dev = dt.ToyboxDataset(data_path=data_dir, rng=rng, train=True, hypertune=True, transform=transform)
toybox_val = dt.ToyboxDataset(data_path=data_dir, rng=rng, train=False, hypertune=True, transform=transform)
toybox_test = dt.ToyboxDataset(data_path=data_dir, rng=rng, train=False, hypertune=False, transform=transform)
toybox_devval = dt.ToyboxDataset(data_path=data_dir, rng=rng, train=True, hypertune=False, transform=transform)

toybox_loader = torchdata.DataLoader(toybox_dev, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
idxs, images, labels = next(iter(toybox_loader))

# Prep the model 
rn18 = models.resnet18(weights=None)
fc_size = rn18.fc.in_features
rn18.fc = nn.Linear(fc_size, 12)

# Make predictions
preds = rn18.forward(images)
print(f"Model output size: {preds.shape}")
"""

file = "/home/s2186747/ug-project/data/toybox_data_interpolated_cropped_dev.pickle"
with open(file, "rb") as f:
    print(f.read(4))
