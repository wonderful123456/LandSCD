"""
变化检测数据集
"""

import os
from PIL import Image
import numpy as np

from torch.utils import data
import os
import sys
import cv2 as cv

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from dataset.data_utils import CDDataAugmentation


"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "CD_label"
IMG_LABEL_FOLDER_NAME = "A_label"
IMG_POST_LABEL_FOLDER_NAME = "B_label"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name) + ".png"


def get_img_path(root_dir, img_name):
    # print('img path is',os.path.join(root_dir, IMG_FOLDER_NAME, img_name)+ '.tif')
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name) + ".png"

def get_A_label_path(root_dir, img_name):
    # print('img path is',os.path.join(root_dir, IMG_FOLDER_NAME, img_name)+ '.tif')
    return os.path.join(root_dir, IMG_LABEL_FOLDER_NAME, img_name) + ".png"

def get_B_label_path(root_dir, img_name):
    # print('img path is',os.path.join(root_dir, IMG_FOLDER_NAME, img_name)+ '.tif')
    return os.path.join(root_dir, IMG_POST_LABEL_FOLDER_NAME, img_name) + ".png"

def get_label_path(root_dir, img_name):
    # print('label path is',os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name))
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name) + '.png'


class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir="C:/Users/zyy/Documents/Tencent Files/919688409/FileRecv/regularCultivatedLandDatasetsV1", split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split  # train | train_aug | val
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list_tmp = load_img_name_list(self.list_path)
        self.img_name_list = []

        for line in self.img_name_list_tmp:
            self.img_name_list.append(line)
            self.img_name_list.append(line + "_aug_0")
            self.img_name_list.append(line + "_aug_1")
            self.img_name_list.append(line + "_aug_2")
            self.img_name_list.append(line + "_aug_3")

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        A_label_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir="C:/Users/zyy/Documents/Tencent Files/919688409/FileRecv/regularCultivatedLandDatasetsV1", img_size=256, split='train', is_train=True, label_transform=None,
                 to_tensor=True):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        img_A = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        img_L = cv.imread(L_path)
        imgGray = cv.cvtColor(img_L, cv.COLOR_BGR2GRAY)
        label = np.array(imgGray, dtype=np.uint8)

        L_A_path = get_A_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        img_A_L = cv.imread(L_A_path)
        imgGrayA = cv.cvtColor(img_A_L, cv.COLOR_BGR2GRAY)
        label_A = np.array(imgGrayA, dtype=np.uint8)

        L_B_path = get_B_label_path(self.root_dir, self.img_name_list[index % self.A_size])
        img_B_L = cv.imread(L_B_path)
        imgGrayB = cv.cvtColor(img_B_L, cv.COLOR_BGR2GRAY)
        label_B = np.array(imgGrayB, dtype=np.uint8)

        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255

        # print('label shape is ', label.shape)

        [img, img_B], [label], [label_A], [label_B] = self.augm.transform([img_A, img_B], [label], [label_A], [label_B], to_tensor=self.to_tensor)
        cls_num_list = [0] * 2
        # print(cls_num_list = [0] * 2)
        # print(label.max())
        # print('img shape is ', img.shape)
        return {'img_A': img, 'img_B': img_B, 'label_BCD': label, 'label_SGA': label_A, 'label_SGB': label_B}

