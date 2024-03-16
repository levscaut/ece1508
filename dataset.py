from torchvision import transforms, datasets
import torch
from torch.utils.data import Subset
import numpy as np

np.random.seed(0)


class SimCLRViewGenerator:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if self.n_views == 1:
            return self.base_transform(x)
        else:
            return [self.base_transform(x) for i in range(self.n_views)]
    
class SimCLRDataset:
    
    def __init__(self, dataset_name):
        dataset_map = {
            "cifar10": datasets.CIFAR10,
        }
        kernel_sizes = {
            "cifar10": 32,
        }
        num_classes_map = {
            "cifar10": 10,
        }
        self.data_fn = dataset_map[dataset_name]
        self.kernel_size = kernel_sizes[dataset_name]
        self.num_classes = num_classes_map[dataset_name]

    
    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms
    
    def _get_dataset(self, n_views, is_train, sample_rate=1):
        """Return a dataset with the SimCLR pipeline."""
        data_transforms = self.get_simclr_pipeline_transform(self.kernel_size)
        full_dataset = self.data_fn("data/", train=is_train, transform=SimCLRViewGenerator(data_transforms, n_views), download=True)
        if sample_rate == 1:
            return full_dataset
        else:
            indices = torch.randperm(len(full_dataset))
            subset_indices = indices[:int(sample_rate * len(full_dataset))]
            sampled_dataset = Subset(full_dataset, subset_indices)
            return sampled_dataset
    
    def get_train_dataset(self, n_views, sample_rate=1):
        return self._get_dataset(n_views, is_train=True, sample_rate=sample_rate)

    def get_test_dataset(self, n_views):
        return self._get_dataset(n_views, is_train=False)

    def get_train_val_datasets(self, n_views, val_split=0.2):
        full_set = self._get_dataset(n_views, is_train=True)
        split = torch.utils.data.random_split(full_set, [1-val_split, val_split])
        return split[0], split[1]

