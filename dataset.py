from torchvision import transforms, datasets
import numpy as np

np.random.seed(0)


class SimCLRViewGenerator:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]
    
class SimCLRDataset:
    
    def __init__(self, dataset_name):
        dataset_map = {
            "cifar10": datasets.CIFAR10,
        }
        kernel_sizes = {
            "cifar10": 32,
        }
        self.data_fn = dataset_map[dataset_name]
        self.kernel_size = kernel_sizes[dataset_name]

    
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
    
    def get_dataset(self, n_views):
        """Return a dataset with the SimCLR pipeline."""
        data_transforms = self.get_simclr_pipeline_transform(self.kernel_size)
        return self.data_fn("data/", train=True, transform=SimCLRViewGenerator(data_transforms, n_views), download=True)
