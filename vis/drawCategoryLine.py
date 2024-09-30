# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
#
# # 定义图像文件夹路径
# folder_path = "C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV1\A_label"
#
# # 初始化一个空的像素值数组
# all_pixel_values = []
#
# # 遍历文件夹中的每个图像
# for filename in os.listdir(folder_path):
#     if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # 选择图像文件
#         image_path = os.path.join(folder_path, filename)
#         # 读取图像并转换为灰度
#         image = Image.open(image_path).convert('L')
#         # 将图像转换为numpy数组并扩展到all_pixel_values
#         pixel_values = np.array(image).flatten()
#         all_pixel_values.extend(pixel_values)
#
# # 将列表转换为numpy数组
# all_pixel_values = np.array(all_pixel_values)
#
# # 统计像素值的频率
# unique, counts = np.unique(all_pixel_values, return_counts=True)
#
# # 绘制曲线图
# plt.figure(figsize=(10, 5))
# plt.plot(unique, counts, color='blue')
# plt.title('Pixel Value Distribution of All Images')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.grid()
# plt.xlim(0, 255)  # 灰度值范围
# # plt.xticks(np.arange(0, 6, 1))  # 设置x轴刻度
# plt.show()

# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
#
# def compute_histogram(image_folder):
#     pixel_values = []
#
#     # 遍历文件夹中的所有图像文件
#     for filename in os.listdir(image_folder):
#         if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 支持的图像格式
#             image_path = os.path.join(image_folder, filename)
#             with Image.open(image_path) as img:
#                 # 确保图像是单通道
#                 img = img.convert('L')  # 转换为灰度图像
#                 # 获取像素值并扩展到列表
#                 pixel_values.extend(list(img.getdata()))
#
#     # 将像素值转换为 NumPy 数组
#     pixel_values = np.array(pixel_values)
#
#     # 计算直方图，范围在 [0, 5] 之间
#     bins = np.arange(0, 6, 1)  # 创建 [0, 1, 2, 3, 4, 5] 的边界
#     histogram, bin_edges = np.histogram(pixel_values, bins=bins, range=(0, 5))
#
#     # 绘制直方图
#     plt.figure(figsize=(10, 6))
#     plt.bar(bin_edges[:-1], histogram, width=0.8, align='center', color='blue', alpha=0.7)
#     plt.xticks(bin_edges[:-1])
#     plt.title('Pixel Value Distribution Histogram')
#     plt.xlabel('Pixel Values')
#     plt.ylabel('Frequency')
#     plt.xlim([0, 5])
#     plt.grid()
#     plt.show()
#
# # 使用示例
# image_folder = 'C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV1\CD_label'  # 替换为你的图像文件夹路径
# compute_histogram(image_folder)



import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_histogram(image_folder):
    pixel_values = []

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 支持的图像格式
            image_path = os.path.join(image_folder, filename)
            with Image.open(image_path) as img:
                # 确保图像是单通道，并转换为二值图像
                # img = img.convert('1')  # 转换为二值图像
                # 获取像素值并扩展到列表
                pixel_values.extend(list(img.getdata()))

    # 将像素值转换为 NumPy 数组
    pixel_values = np.array(pixel_values)

    # 计算直方图
    histogram, bin_edges = np.histogram(pixel_values, bins=[0, 1, 2], range=(0, 1))

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], histogram, width=0.5, align='center', color='blue', alpha=0.7)
    plt.xticks([0, 1])
    plt.title('Pixel Value Distribution Histogram (Binary Images)')
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency')
    plt.xlim([-0.5, 1.5])
    plt.grid()
    plt.show()
    plt.savefig('histogram.png')

# 使用示例
image_folder = 'C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV1\CD_label'  # 替换为你的图像文件夹路径
compute_histogram(image_folder)

