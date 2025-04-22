import os
import numpy as np
import pickle
import csv
import random

DATA_PATH = "/home/s2186747/data/project/Toybox"
NEW_DATA_PATH = "/home/s2186747/data/project/toybox_random"
os.makedirs(NEW_DATA_PATH, exist_ok=True)

all_data = []
all_csv = []
splits = ["dev", "val", "train", "test"]

for split in splits:
    pickle_path = os.path.join(DATA_PATH, f"toybox_data_interpolated_cropped_{split}.pickle")
    csv_path = os.path.join(DATA_PATH, f"toybox_data_interpolated_cropped_{split}.csv")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    with open(csv_path, "r") as f:
        csv_data = list(csv.DictReader(f))
    all_data.extend(data)
    all_csv.extend(csv_data)

object_groups = {}
for idx, row in enumerate(all_csv):
    class_id = int(row["Class ID"])
    object_id = int(row["Object"])
    key = (class_id, object_id)
    object_groups.setdefault(key, []).append(idx)

random.seed(42)

# 60% dev, 20% val, and 20% test.
dev_indices = []
val_indices = []
test_indices = []

for key, indices in object_groups.items():
    random.shuffle(indices)
    n = len(indices)
    # Ensure at least one sample per split (if possible)
    n_dev = max(1, int(0.6 * n))
    n_val = max(1, int(0.2 * n)) if n - n_dev > 0 else 0
    n_test = n - n_dev - n_val
    # If there aren’t enough samples to get a test split, adjust n_val and n_test
    if n_test <= 0 and n - n_dev > 1:
        n_val = n - n_dev - 1
        n_test = 1
    dev_indices.extend(indices[:n_dev])
    val_indices.extend(indices[n_dev:n_dev+n_val])
    test_indices.extend(indices[n_dev+n_val:])

def subset_list(lst, indices):
    return [lst[i] for i in indices]

new_splits = {}
new_splits["dev"] = {"data": subset_list(all_data, dev_indices), "csv": subset_list(all_csv, dev_indices)}
new_splits["val"] = {"data": subset_list(all_data, val_indices), "csv": subset_list(all_csv, val_indices)}
new_splits["test"] = {"data": subset_list(all_data, test_indices), "csv": subset_list(all_csv, test_indices)}
# Final training set is the union of dev and val (as in your original code)
new_splits["train"] = {
    "data": new_splits["dev"]["data"] + new_splits["val"]["data"],
    "csv": new_splits["dev"]["csv"] + new_splits["val"]["csv"]
}

for split in splits:
    pickle_out = os.path.join(NEW_DATA_PATH, f"toybox_data_interpolated_cropped_{split}.pickle")
    csv_out = os.path.join(NEW_DATA_PATH, f"toybox_data_interpolated_cropped_{split}.csv")
    with open(pickle_out, "wb") as f:
        pickle.dump(new_splits[split]["data"], f)
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_splits["dev"]["csv"][0].keys())
        writer.writeheader()
        writer.writerows(new_splits[split]["csv"])

pickle_out = os.path.join(NEW_DATA_PATH, "toybox_data_interpolated_cropped_train.pickle")
csv_out = os.path.join(NEW_DATA_PATH, "toybox_data_interpolated_cropped_train.csv")
with open(pickle_out, "wb") as f:
    pickle.dump(new_splits["train"]["data"], f)
with open(csv_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=new_splits["dev"]["csv"][0].keys())
    writer.writeheader()
    writer.writerows(new_splits["train"]["csv"])

print("New dataset splits created successfully!")
