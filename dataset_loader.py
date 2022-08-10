import pickle
import numpy as np
import torch
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.datasets.svhn import SVHN
from torchvision.datasets.cifar import CIFAR10
import torchnet as tnt
from torch.utils.data import Dataset, DataLoader
from skimage import io
from PIL import Image
import scipy.io


from torch.utils.data import DataLoader
from torchvision import transforms
from Affnist_dataset import *

USE_ALBUMENTATION = False

mean_arr = []
std_arr = []


from norb_dataset import NorbDataset


def visualize_dataloader(data, labels):
    from matplotlib import pyplot as plt
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(labels), size=(1,)).item()
        img = data[sample_idx]
        
        figure.add_subplot(rows, cols, i)
        plt.title(labels[sample_idx])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
    



# transforms_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform = transforms.Compose([transforms.ToTensor()])


# train': transforms.Compose(
#             [transforms.Resize(256),
#              transforms.RandomCrop(224),
#              transforms.RandomHorizontalFlip(),
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225]),
#              ]),
#         'test': transforms.Compose(
#             [transforms.Resize(224),
#              transforms.ToTensor(),
#              transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                   std=[0.229, 0.224, 0.225]),
#              ])}


def get_transform(img_width, mean_arr, std_arr, mode, is_norb=False):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    if is_norb:
        if mode:
            return A.Compose(
                [
                    # A.SmallestMaxSize(max_size=160),
                    A.Resize(48,48),
                    A.RandomCrop(32,32),
                    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    # A.RandomCrop(width=round(0.75*(img_width)), height=round(0.75*(img_width))),
                    # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    # A.RandomBrightnessContrast(p=0.5),
                    A.HorizontalFlip(),
                    A.Normalize(mean=tuple(mean_arr), std=tuple(std_arr)),
                    ToTensorV2(),
                ]
            )

        else:
            return A.Compose(
            [
                A.Resize(48,48),
                A.CenterCrop(32,32),
                A.Normalize(mean=tuple(mean_arr), std=tuple(std_arr)),
                ToTensorV2(),
            ])
        
    else:
        if mode:
            return A.Compose(
                [
                    # A.SmallestMaxSize(max_size=160),
                    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.RandomCrop(width=round(0.75*(img_width)), height=round(0.75*(img_width))),
                    # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    # A.RandomBrightnessContrast(p=0.5),
                    A.HorizontalFlip(),
                    A.Normalize(mean=tuple(mean_arr), std=tuple(std_arr)),
                    ToTensorV2(),
                ]
            )
        
        else:
            return A.Compose(
            [
                A.CenterCrop(width=round(0.75*(img_width)), height=round(0.75*(img_width))),
                A.Normalize(mean=tuple(mean_arr), std=tuple(std_arr)),
                ToTensorV2(),
            ])



def get_mean_std(dset):
    loader=DataLoader(dset, batch_size=len(dset), num_workers=1)
    data=next(iter(loader))
    ich=data[0].size()[1]
    img_width=data[0].size()[2]
    if ich == 3:
        mean_arr=[torch.mean(data[0][:, 0, :, :]), torch.mean(
            data[0][:, 1, :, :]), torch.mean(data[0][:, 2, :, :])]
        std_arr=[torch.std(data[0][:, 0, :, :]), torch.std(
            data[0][:, 1, :, :]), torch.std(data[0][:, 2, :, :])]
    elif ich == 1:
        mean_arr=[torch.mean(data[0][:, 0, :, :])]
        std_arr=[torch.std(data[0][:, 0, :, :])]

    return img_width, mean_arr, std_arr

def load_affnist_trans_test():
    transformed_path = 'affnist/transformed'
    x_test = []
    y_test = []
    for i in range(32):
        test = scipy.io.loadmat(f'{transformed_path}/test_batches/{i + 1}.mat')
        x_test_temp = np.transpose(test['affNISTdata'][0][0][2], (1, 0)).reshape(10000, 40, 40)
        y_test_temp = test['affNISTdata'][0][0][5].reshape(10000)
        x_test.append(x_test_temp)
        y_test.append(y_test_temp)

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    return (x_test, y_test)

def get_iterator(dset, dpath, bsize, mode, albumentation=False):
    if dset == 'aff_expanded':
        tensor_dataset = AffNISTDataset('affnist/expanded', mode)
        return DataLoader(tensor_dataset, batch_size=bsize, num_workers=4, shuffle=False)

    elif dset == 'aff_trans_test':
        (data, labels) = load_affnist_trans_test()
         
        
    elif dset == 'svhn':
        if mode is True:
            if albumentation:
                dataset=SVHN(root='./data', download=True, split="train",
                            transform=get_transform(w, mean_arr, std_arr, mode))
            else:
                dataset = SVHN(root='./data', download=True, split="train")

        elif mode is False:
            if albumentation:
                dataset=SVHN(root='./data', download=True, split="test",
                            transform=get_transform(w, mean_arr, std_arr, mode))
            else:
                dataset = SVHN(root='./data', download=True, split="test")

        data=torch.tensor(dataset.data)
        labels=torch.tensor(dataset.labels)


    elif dset == 'mnist':
        if albumentation:
            dataset=MNIST(root='./data', download=True, train=True,transform=transforms.Compose([transforms.ToTensor()]))
            w, mean_arr, std_arr=get_mean_std(dataset)
            dataset=MNIST(root='./data', download=True, train=mode,
                        transform=get_transform(w, mean_arr, std_arr, mode))
        else:
            dataset = MNIST(root='./data', download=True, train=mode)
            
        data=getattr(dataset, 'train_data' if mode else 'test_data')
        labels=getattr(dataset, 'train_labels' if mode else 'test_labels')

    elif dset == 'fmnist':
        if albumentation:
            dataset=FashionMNIST(root='./data', download=True, train=True,transform=transforms.Compose([transforms.ToTensor()]))
            w, mean_arr, std_arr=get_mean_std(dataset)
            dataset=FashionMNIST(root='./data', download=True, train=mode,
                                transform=get_transform(w, mean_arr, std_arr, mode))
        else:
            dataset = FashionMNIST(root='./data', download=True, train=mode)
            
        data=getattr(dataset, 'train_data' if mode else 'test_data')
        labels=getattr(dataset, 'train_labels' if mode else 'test_labels')

    elif dset == 'cifar':
        if albumentation:
            dataset=CIFAR10(root='./dataF', download=True, train=mode,transform=transforms.Compose([transforms.ToTensor()]))
            w, mean_arr, std_arr=get_mean_std(dataset)
            dataset=CIFAR10(root='./dataF', download=True, train=mode,
                            transform=get_transform(w, mean_arr, std_arr, mode))
        else:
            dataset = CIFAR10(root='./dataF', download=True, train=mode)
        
        data = np.transpose(getattr(dataset, 'data'), (0, 3, 1, 2))
        labels=getattr(dataset, 'targets')




    if dset != 'norb':
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        return tensor_dataset.parallel(batch_size=bsize, num_workers=4, shuffle=mode)
    else:
        norbdset = NorbDataset(mode)
        if albumentation:
            w, mean_arr, std_arr=get_mean_std(norbdset)
            norbdset = NorbDataset(mode, get_transform(w, mean_arr, std_arr,mode, is_norb=True))
        
    return DataLoader(norbdset, batch_size=bsize, num_workers=4, shuffle=mode)
