import torch.cuda
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader


################
# MNIST Dataset
################
def get_mnist_dataloader(args):
    transform = Compose([ToTensor()])  # , Normalize([0.1307], [0.3081])])
    ds_train = MNIST(args.data_root, True, transform, download=args.download)
    ds_valid = MNIST(args.data_root, False, transform, download=args.download)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers,
              "drop_last": True, "pin_memory": torch.cuda.is_available()}
    ds_train = DataLoader(ds_train, shuffle=True, **kwargs)
    ds_valid = DataLoader(ds_valid, shuffle=False, **kwargs)
    return ds_train, ds_valid


def get_cifar10_dataset(args):
    image_size = 32
    transforms = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }
    ds_train = CIFAR10(args.data_root, True, transforms["train"], download=args.download)
    ds_valid = CIFAR10(args.data_root, False, transforms["test"], download=args.download)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers,
              "drop_last": True, "pin_memory": torch.cuda.is_available()}

    ds_train = DataLoader(ds_train, shuffle=True, **kwargs)
    ds_valid = DataLoader(ds_valid, shuffle=False, **kwargs)
    return ds_train, ds_valid


def get_input_channels(args):
    if args.dataset == 0:
        in_channels = 1
    elif args.dataset == 1:
        in_channels = 3
    else:
        raise ValueError("args.dataset must be 0 or 1")
    return in_channels


def get_dataloaders(args):
    if args.dataset == 0:
        ds_train, ds_valid = get_mnist_dataloader(args)
    elif args.dataset == 1:
        ds_train, ds_valid = get_cifar10_dataset(args)
    else:
        raise ValueError("args.dataset must be 0 or 1")
    return ds_train, ds_valid
