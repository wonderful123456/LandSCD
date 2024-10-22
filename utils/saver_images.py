import os
import shutil

import cv2
import torch
from collections import OrderedDict
import glob
import numpy as np
import matplotlib.pyplot as plt

# from mypath import Path

# 该函数中定义了一些如何去保存实验结果到run的文件夹下，增强可视化

# 定义了一些颜色的映射
color_mapping = [
    [0, 0, 0],  #0
[150, 250,   0], #1
[250, 150, 150], #2
[200,   0,   0], #3
[200, 150, 150], #4
[  0, 200, 250], #5
[  0,   0, 200], #6
[200,   0, 200], #7
[150, 200, 150], #8
[250,   0, 150],#9
[150,   0, 250],#10
[150, 150, 250],#11
[  0, 150, 200],#12
[  0, 200,   0],#13
[250, 200,   0],#14
[200, 200,   0] #15
]



label_5_pivix_map = [
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 255, 255],
    [255, 255, 0],
    [0, 0, 255],
]

label_7_pivix_map = [
    [0,0,0],#0 未知区域
    [0,255,255],#1, 城市
    [255,255,0], #2 农业地
    [255,0,255],#3 牧场
    [0,255,0],#4 森林
    [0,0,255],#5 水
    [255,255,255],#6 贫瘠地
]

label_6_pivix_map = [
    [0,0,0],#0 未知区域
    [255,255,0],#1, 耕地
    [255,0,0], #2 城市
    [128,0,128],#3 森林
    [0,255,0],#4 草地
    [0,0,255],#5 水
]
label_2_pivix_map = [
    [0,0,0],#0 未知区域
    [255,255,255],#1, 耕地

]



def save_images(matrix, fname_list, epo,batch_size,directory,num_class):
    n, w, h = matrix.shape
    for k in range(n):
        res = []
        for i in range(w):
            tmp = []
            for j in range(h):

                # if self.args.num_class == 16:
                #     if matrix[k, i, j] <= 15:
                #         tmp.append(color_mapping[matrix[k, i, j]])
                #     else:
                #         tmp.append(color_mapping[0])
                # elif self.args.num_class == 7:
                #     if matrix[k, i, j] <= 15:
                #         tmp.append(label_7_pivix_map[matrix[k, i, j]])
                #     else:
                #         tmp.append(label_7_pivix_map[0])
                # else:
                #     if matrix[k, i, j] <= 15:
                #         tmp.append(label_5_pivix_map[matrix[k, i, j]])
                #     else:
                #         tmp.append(label_5_pivix_map[0])
                if num_class == 6:
                    if matrix[k, i, j] < 6:
                        tmp.append(label_6_pivix_map[matrix[k, i, j]])
                    else:
                        tmp.append(label_6_pivix_map[0])
                elif num_class == 2:
                    if matrix[k, i, j] < 2:
                        tmp.append(label_2_pivix_map[matrix[k, i, j]])
                    else:
                        tmp.append(label_2_pivix_map[0])

            res.append(tmp)

        res = np.array(res)
        res = np.flip(res, axis=2)
        dst = os.path.join(directory, "predict_result")
        if not os.path.exists(dst):
            os.makedirs(dst)

        # 模型预测的结果
        cv2.imwrite(os.path.join(dst, fname_list[epo * batch_size + k] + ".png"), res,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 2])
        # # 真实的结果
        # shutil.copyfile(os.path.join(Path.db_root_dir(dataset), "GTImage",
        #                              fname_list[epo * batch_size + k] + ".tif"),
        #                 os.path.join(dst, fname_list[epo * batch_size + k] + "g.png"))

        # shutil.copyfile(os.path.join(Path.db_root_dir(self.args.dataset), "JPEGImages",
        #                              fname_list[epo * self.args.batch_size + k] + "_0.png"),
        #                 os.path.join(dst, fname_list[epo * self.args.batch_size + k] + ".png"))