import os
import sys
sys.path.append(os.path.split(os.path.dirname(__file__))[0])

from PIL import Image
from torch.utils.data import Dataset

DEBUG = False
if DEBUG:
    import datatransforms as tr
    import datatransforms_seg as tr_seg
    import datatransforms_single_cd as tr_single_cd
else:
    import dataset.datatransforms as tr
    import dataset.datatransforms_seg as tr_seg
    import dataset.datatransforms_single_cd as tr_single_cd

import numpy as np


def full_path_loader_for_txt(datasets_path, txt_path):
    dataset = {}

    A_path = []
    B_path = []
    label_BCD_path = []
    label_SGA_path = []
    label_SGB_path = []

    with open(txt_path, "r") as f1:
        lines = f1.read().splitlines()



    for index, line in enumerate(lines):
        img_A = os.path.join(datasets_path, "A", line, "png")
        img_B = os.path.join(datasets_path, "B", line, "png")
        label_BCD = os.path.join(datasets_path, "CD_label", line, "png")
        label_SGA = os.path.join(datasets_path, "A_label", line, "png")
        label_SGB = os.path.join(datasets_path, "B_label", line, "png")
        A_path.append(img_A)
        B_path.append(img_B)
        label_BCD_path.append(label_BCD)
        label_SGA_path.append(label_SGA)
        label_SGB_path.append(label_SGB)
        dataset[index] = {'img_A': A_path[index],
                          'img_B': B_path[index],
                          'label_BCD': label_BCD_path[index],
                          'label_SGA': label_SGA_path[index],
                          'label_SGB': label_SGB_path[index]
                          }
    return dataset

class GEPataset(Dataset):
    def __init__(self, args, split='train'):
        self.pretrained = args.pretrained
        self.with_colorjit = args.with_colorjit
        self.split = split
        if split == 'train':
            self.full_load = full_path_loader_for_txt(args.dataset_path,  args.train_txt_path)
        elif split == 'val':
            self.full_load = full_path_loader_for_txt(args.dataset_path, args.val_txt_path)
        elif split == 'test':
            self.full_load = full_path_loader_for_txt(args.dataset_path, args.test_txt_path)

        self.train_transforms = tr.get_train_transforms(seg_ignore_value=9, with_colorjit=self.with_colorjit)
        self.test_transforms = tr.test_transforms
        if not self.pretrained:
            self.train_transforms.transforms[5].mean = (0.5, 0.5, 0.5)
            self.train_transforms.transforms[5].std = (0.5, 0.5, 0.5)

    def __len__(self):
        return len(self.full_load)

    def __getitem__(self, idx):
        img_A_path = self.full_load[idx]['img_A']
        img_B_path = self.full_load[idx]['img_B']
        label_path = self.full_load[idx]['label_BCD']

        img_A = Image.open(img_A_path)
        img_B = Image.open(img_B_path)
        label_array = np.asarray(Image.open(label_path))
        label_SGA = self.convert_numpy_to_Image(label_array, 0)
        label_SGB = self.convert_numpy_to_Image(label_array, 1)
        label_BCD = self.convert_numpy_to_Image(label_array, 2)

        sample = {'img_A': img_A,
                  'img_B': img_B,
                  'label_BCD': label_BCD,
                  'label_SGA': label_SGA,
                  'label_SGB': label_SGB
                  }

        if self.split == 'train':
            sample = self.train_transforms(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            sample = self.test_transforms(sample)
            sample['name'] = img_A_path
        return sample

    def convert_numpy_to_Image(self, numpy_array, channel):
        return Image.fromarray(numpy_array[:, :, channel])
