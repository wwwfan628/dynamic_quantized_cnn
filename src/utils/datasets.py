from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from torchvision import transforms
import torch


def load_dataset(args):
    # load dataset
    if args.dataset_name == 'MNIST':
        num_workers = 0
        batch_size = args.batch_size
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        data_train = MNIST(root='../datasets', train=True, download=True, transform=transform)
        data_test = MNIST(root='../datasets', train=False, download=True, transform=transform)
    elif args.dataset_name == 'CIFAR10':
        num_workers = 8
        batch_size = args.batch_size
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        data_train = CIFAR10(root='../datasets', train=True, download=True, transform=transform_train)
        data_test = CIFAR10(root='../datasets', train=False, download=True, transform=transform_test)
    elif args.dataset_name == 'ImageNet':
        num_workers = 16
        batch_size = args.batch_size
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        data_train = ImageFolder(root='/itet-stor/yiflu/net_scratch/imagenet/train', transform=transform_train)
        data_test = ImageFolder(root='/itet-stor/yiflu/net_scratch/imagenet/val', transform=transform_val)
    elif args.dataset_name == 'ImageNet_mini':
        num_workers = 8
        batch_size = args.batch_size
        transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        data_train = ImageFolder(root='../datasets/imagenet/train', transform=transform_train)
        data_test = ImageFolder(root='../datasets/imagenet/val', transform=transform_val)
    else:
        print('Dataset is not supported! Please choose from: MNIST, CIFAR10 and ImageNet.')
    in_channels = data_train[0][0].shape[0]
    num_classes = len(data_train.classes)
    dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return in_channels, num_classes, dataloader_train, dataloader_test