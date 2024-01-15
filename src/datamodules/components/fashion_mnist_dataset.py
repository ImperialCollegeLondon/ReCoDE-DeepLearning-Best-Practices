from torchvision import datasets, transforms
from torch.utils.data import Dataset


class CustomFashionMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        """
        Initialize the FashionMNIST dataset.

        Args:
            root (str): Root directory of dataset, if doesn't exists it will download the data.
            train (bool, optional): If True, creates dataset from train-images-idx3-ubyte,
                otherwise from t10k-images-idx3-ubyte.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a transformed version.
            target_transform (callable, optional) – A function/transform that takes in
                the target and transforms it.
            download (bool, optional): If True, downloads the dataset from the internet
                and puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
        """
        self.data = datasets.FashionMNIST(
            root=root, train=train, download=download, transform=transform, target_transform=target_transform
        )
        # Add any additional initialization if necessary

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get the data sample associated with the given index.

        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        image, label = self.data[idx]

        return image, label


if __name__ == "__main__":
    # Example usage
    transform = transforms.Compose([transforms.ToTensor()])  # Add any additional transformations here

    dataset = CustomFashionMNIST(root="./data", train=True, transform=transform, download=True)

    # Check the dataset loads correctly
    image, label = dataset[0]

    # Check the dataset has the correct shape
    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min()}, {image.max()}]")
    print(f"Label: {label}")
    print(f"Total number of images: {len(dataset)}")
