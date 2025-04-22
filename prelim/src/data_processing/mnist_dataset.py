from torch.utils.data import Dataset

class CustomMNIST(Dataset):
    def __init__(self, mnist_dataset):
        """
        Initializes the custom dataset object.
        :param mnist_dataset: The original MNIST dataset.
        """
        self.mnist_dataset = mnist_dataset

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        """
        Retrieves an item from the MNIST dataset and appends dummy obj_ids and vid.
        """
        x, y = self.mnist_dataset[idx]
        obj_ids = 0   # Placeholder for object IDs
        vid = 'NA'    # Placeholder for video identifier
        return x, y, obj_ids, vid
