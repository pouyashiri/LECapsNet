
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

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


_CITATION = r"""\
@article{LeCun2004LearningMF,
  title={Learning methods for generic object recognition with invariance to pose and lighting},
  author={Yann LeCun and Fu Jie Huang and L{\'e}on Bottou},
  journal={Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
  year={2004},
  volume={2},
  pages={II-104 Vol.2}
}
"""

_TRAINING_URL_TEMPLATE = (
    "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    "smallnorb-5x46789x9x18x6x2x96x96-training-{type}.mat.gz")
_TESTING_URL_TEMPLATE = (
    "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    "smallnorb-5x01235x9x18x6x2x96x96-testing-{type}.mat.gz")

_DESCRIPTION = r"""\
This database is intended for experiments in 3D object recognition from shape. It contains images of 50 toys belonging to 5 generic categories: four-legged animals, human figures, airplanes, trucks, and cars. The objects were imaged by two cameras under 6 lighting conditions, 9 elevations (30 to 70 degrees every 5 degrees), and 18 azimuths (0 to 340 every 20 degrees).

The training set is composed of 5 instances of each category (instances 4, 6, 7, 8 and 9), and the test set of the remaining 5 instances (instances 0, 1, 2, 3, and 5).
"""

def _load_chunk(dat_path, cat_path, info_path):
  """Loads a data chunk as specified by the paths.

  Args:
    dat_path: Path to dat file of the chunk.
    cat_path: Path to cat file of the chunk.
    info_path: Path to info file of the chunk.

  Returns:
    Tuple with the dat, cat, info_arrays.
  """
  dat_array = read_binary_matrix(dat_path)
  # Even if the image is gray scale, we need to add an extra channel dimension
  # to be compatible with tfds.features.Image.
  dat_array = np.expand_dims(dat_array, -1)

  cat_array = read_binary_matrix(cat_path)

  info_array = read_binary_matrix(info_path)
  info_array = np.copy(info_array)  # Make read-only buffer array writable.
  # Azimuth values are 0, 2, 4, .., 34. We divide by 2 to get proper labels.
  info_array[:, 2] = info_array[:, 2] / 2

  return dat_array, cat_array, info_array


def read_binary_matrix(filename):
  """Reads and returns binary formatted matrix stored in filename.

  The file format is described on the data set page:
  https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/

  Args:
    filename: String with path to the file.

  Returns:
    Numpy array contained in the file.
  """
  with tf.io.gfile.GFile(filename, "rb") as f:
    s = f.read()

    # Data is stored in little-endian byte order.
    int32_dtype = np.dtype("int32").newbyteorder("<")

    # The first 4 bytes contain a magic code that specifies the data type.
    magic = int(np.frombuffer(s, dtype=int32_dtype, count=1))
    if magic == 507333717:
      data_dtype = np.dtype("uint8")  # uint8 does not have a byte order.
    elif magic == 507333716:
      data_dtype = np.dtype("int32").newbyteorder("<")
    else:
      raise ValueError("Invalid magic value for data type!")

    # The second 4 bytes contain an int32 with the number of dimensions of the
    # stored array.
    ndim = int(np.frombuffer(s, dtype=int32_dtype, count=1, offset=4))

    # The next ndim x 4 bytes contain the shape of the array in int32.
    dims = np.frombuffer(s, dtype=int32_dtype, count=ndim, offset=8)

    # If the array has less than three dimensions, three int32 are still used to
    # save the shape info (remaining int32 are simply set to 1). The shape info
    # hence uses max(3, ndim) bytes.
    bytes_used_for_shape_info = max(3, ndim) * 4

    # The remaining bytes are the array.
    data = np.frombuffer(
        s, dtype=data_dtype, offset=8 + bytes_used_for_shape_info)
  return data.reshape(tuple(dims))

def download_dataset():
    import os
    import pickle 
    import urllib.request
    

    
    if not os.path.exists('norb_test.p') or not os.path.exists('norb_train.p'):
        
        filenames = {
            "training_dat": _TRAINING_URL_TEMPLATE.format(type="dat"),
            "training_cat": _TRAINING_URL_TEMPLATE.format(type="cat"),
            "training_info": _TRAINING_URL_TEMPLATE.format(type="info"),
            # "testing_dat": _TESTING_URL_TEMPLATE.format(type="dat"),
            # "testing_cat": _TESTING_URL_TEMPLATE.format(type="cat"),
            # "testing_info": _TESTING_URL_TEMPLATE.format(type="info"),
        }
        for k in filenames:
            urllib.request.urlretrieve(filenames[k], filenames[k][-10:])
            os.system(f'gunzip -f {filenames[k][-10:]}')
        dat_arr, cat_arr, inf_arr = _load_chunk('dat.mat', 'cat.mat', 'nfo.mat')
        res = dat_arr, cat_arr, inf_arr

        with open('norb_train.p', 'wb') as file:
            pickle.dump(res, file)
            
        
        filenames = {
            # "training_dat": _TRAINING_URL_TEMPLATE.format(type="dat"),
            # "training_cat": _TRAINING_URL_TEMPLATE.format(type="cat"),
            # "training_info": _TRAINING_URL_TEMPLATE.format(type="info"),
            "testing_dat": _TESTING_URL_TEMPLATE.format(type="dat"),
            "testing_cat": _TESTING_URL_TEMPLATE.format(type="cat"),
            "testing_info": _TESTING_URL_TEMPLATE.format(type="info"),
        }
        for k in filenames:
            urllib.request.urlretrieve(filenames[k], filenames[k][-10:])
            os.system(f'gunzip -f {filenames[k][-10:]}')
        dat_arr, cat_arr, inf_arr = _load_chunk('dat.mat', 'cat.mat', 'nfo.mat')
        res = dat_arr, cat_arr, inf_arr

        with open('norb_test.p', 'wb') as file:
            pickle.dump(res, file)

    

    
# dat_arr, cat_arr, inf_arr = _load_chunk(filenames['testing_dat'], filenames['testing_cat'], filenames['testing_info'])

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
    

class NorbDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, split):
        download_dataset()
        file_name = 'norb_train.p' if split else 'norb_test.p'
        
        with open(file_name, 'rb') as file:
            dat_arr, cat_arr, inf_arr = pickle.load(file)
            self.dat_arr = np.array(dat_arr[:,0,:,:,0])
            np.squeeze(self.dat_arr)
        
            self.cat_arr = np.array(cat_arr)
            
            
            
            
            # print(self.cat_arr.shape)
            # exit(1)
            
        train_transform = transforms.Compose([
                    transforms.Resize(48),
                    transforms.RandomCrop(32),
                    transforms.ToTensor(),
                ])

        test_transform = transforms.Compose([
                            transforms.Resize(48),
                            transforms.CenterCrop(32),
                            transforms.ToTensor(),
                        ])
    
        self.transform = train_transform if split else test_transform


    def __len__(self):
        return self.cat_arr.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # print(f'!@#!@#{self.dat_arr.shape}')
        sample = Image.fromarray(self.dat_arr[idx])
        # sample = np.expand_dims(sample, 0)

        label = self.cat_arr[idx]

        sample = self.transform(sample)

        return sample, label
    
    
