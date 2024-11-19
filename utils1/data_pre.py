import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F
import cv2


num_classes = 6
ST_COLORMAP = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]]
ST_CLASS = ['nonbd', 'complex bd', "simple bd", "regular bd", "irregular bd", "large-scale bd"]

# UBTD
MEAN_A = np.array([62.239872, 63.211044, 66.932594])
STD_A = np.array([46.32426, 44.79888, 44.923523])
# TBTD
# MEAN_A = [75.4391, 81.203964, 91.874855]
# STD_A = [48.639874, 49.273838, 47.913185]

root = './'


colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    # IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0


def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = (im - MEAN_A) / STD_A
    return im


def normalize_images(imgs, time='A'):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im, time)
    return imgs


# 读取到
def read_RSimages(mode, rescale=False):
    # assert mode in ['train', 'val', 'test']
    img_A_dir = os.path.join(root, mode, 'img1')
    label_A_dir = os.path.join(root, mode, 'label1')
    label_bound_dir = os.path.join(root, mode, 'boundary1')
    # To use rgb labels:
    # label_A_dir = os.path.join(root, mode, 'label1_rgb')
    # label_B_dir = os.path.join(root, mode, 'label2_rgb')

    data_list = os.listdir(img_A_dir)
    imgs_list_A,labels_A,label_mask,label_bound = [], [], [], []
    count = 0
    for it in data_list:
        # print(it)
        if (it[-4:] == '.png'):
            img_A_path = os.path.join(img_A_dir, it)
            label_A_path = os.path.join(label_A_dir, it)
            label_bound_path = os.path.join(label_bound_dir, it)
            imgs_list_A.append(img_A_path)

            label_A = io.imread(label_A_path)
            label_A = np.nan_to_num(label_A)
            label_A_mask = np.zeros_like(label_A)
            label_A_mask[label_A != 0] = 1

            label_bound1 = io.imread(label_bound_path)
            label_bound1 = np.nan_to_num(label_bound1)
            label_bound_mask = np.zeros_like(label_bound1)
            label_bound_mask[label_bound1 !=0 ] = 1

            # for rgb labels:
            # label_A = Color2Index(label_A)
            # label_B = Color2Index(label_B)
            labels_A.append(label_A)
            label_mask.append(label_A_mask)
            label_bound.append(label_bound_mask)

        count += 1
        if not count % 500: print('%d/%d images loaded.' % (count, len(data_list)))

    print(labels_A[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')

    return imgs_list_A, labels_A, label_mask, label_bound


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        self.imgs_list_A, self.labels_A, self.label_mask, self.label_bound = read_RSimages(mode)

    # 返回图片的名称
    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = np.nan_to_num(img_A)
        img_A = normalize_image(img_A, 'A')
        label_A = self.labels_A[idx]
        label_mask = self.label_mask[idx]
        label_bound_mask = self.label_bound[idx]
        # if self.random_flip:
        #     img_A, label_A, label_mask = transforms.rand_rot90_flip_MCD(img_A, label_A, label_mask)
        return self.get_mask_name(idx), F.to_tensor(img_A), torch.from_numpy(label_A), \
            torch.from_numpy(label_mask), torch.from_numpy(label_bound_mask)

    def __len__(self):
        return len(self.imgs_list_A)


class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'img1')
        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:] == '.png'):
                img_A_path = os.path.join(imgA_dir, it)
                self.imgs_A.append(io.imread(img_A_path))
                self.mask_name_list.append(it)
        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_A = normalize_image(img_A, 'A')
        return self.get_mask_name(idx), F.to_tensor(img_A)

    def __len__(self):
        return self.len

