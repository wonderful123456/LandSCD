# import cv2
# import numpy as np
#
# # 加载影像和标签
# image_a = cv2.imread("C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\A\\000004.png")
# image_b = cv2.imread("C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\B\\000004.png")
# mask_a = cv2.imread("C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\A_label\\000004.png", cv2.IMREAD_GRAYSCALE)
# mask_b = cv2.imread("C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\B_label\\000004.png", cv2.IMREAD_GRAYSCALE)
#
# # 特征点匹配与配准
# orb = cv2.ORB_create()
# keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
# keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors_a, descriptors_b)
# matches = sorted(matches, key=lambda x: x.distance)
#
# # 计算匹配点的坐标
# points_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
# points_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#
# # 估计单应性矩阵
# matrix, mask = cv2.findHomography(points_b, points_a, cv2.RANSAC)
#
# # 对影像B进行变换
# aligned_image_b = cv2.warpPerspective(image_b, matrix, (image_a.shape[1], image_a.shape[0]))
# aligned_mask_b = cv2.warpPerspective(mask_b, matrix, (mask_a.shape[1], mask_a.shape[0]))
#
# # 计算重叠区域
# overlap_mask = np.logical_and(mask_a > 0, aligned_mask_b > 0).astype(np.uint8)
#
# # 去除不重叠的部分
# final_image_a = cv2.bitwise_and(image_a, image_a, mask=overlap_mask)
# final_image_b = cv2.bitwise_and(aligned_image_b, aligned_image_b, mask=overlap_mask)
#
# # 找到重叠区域的边界
# x, y, w, h = cv2.boundingRect(overlap_mask)
#
# # 切割到相同大小
# final_image_a_cropped = final_image_a[y:y+h, x:x+w]
# final_image_b_cropped = final_image_b[y:y+h, x:x+w]
# final_mask_a_cropped = mask_a[y:y+h, x:x+w]
# final_mask_b_cropped = aligned_mask_b[y:y+h, x:x+w]
#
# # 保存处理后的影像和标签
# cv2.imwrite('final_image_a.png', final_image_a_cropped)
# cv2.imwrite('final_image_b.png', final_image_b_cropped)
# cv2.imwrite('final_mask_a.png', final_mask_a_cropped)
# cv2.imwrite('final_mask_b.png', final_mask_b_cropped)


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 加载影像和标签
image_a = cv2.imread("C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\A\\000034.png")
image_b = cv2.imread("C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\B\\000034.png")
mask_a = cv2.imread("C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\A_label\\000034.png", cv2.IMREAD_GRAYSCALE)
mask_b = cv2.imread("C:\\Users\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\B_label\\000034.png", cv2.IMREAD_GRAYSCALE)

## 确保输入图像和标签都是256x256
assert image_a.shape == (256, 256, 3)
assert image_b.shape == (256, 256, 3)
assert mask_a.shape == (256, 256)
assert mask_b.shape == (256, 256)

# 初始化最优相似度和最佳窗口位置
max_ssim = -1
best_crop_a = np.zeros((256, 256, 3), dtype=np.uint8)
best_crop_b = np.zeros((256, 256, 3), dtype=np.uint8)

# 窗口大小（确保是小于256的奇数）
window_size = 7  # 可以根据需要调整这个值

# 遍历图像A，计算每个局部窗口与图像B的SSIM
for y in range(0, 256 - window_size + 1):
    for x in range(0, 256 - window_size + 1):
        crop_a = image_a[y:y + window_size, x:x + window_size]
        crop_b = image_b[y:y + window_size, x:x + window_size]

        # 计算局部窗口的SSIM，确保窗口大小小于当前局部块的大小
        current_win_size = min(window_size, crop_a.shape[0], crop_a.shape[1])

        if current_win_size < 7:
            continue  # 跳过小于7x7的窗口

        # 计算SSIM，使用有效的窗口大小
        current_ssim = ssim(crop_a, crop_b, multichannel=True, win_size=current_win_size)

        # 更新最佳相似度和窗口
        if current_ssim > max_ssim:
            max_ssim = current_ssim
            best_crop_a = crop_a
            best_crop_b = crop_b

# 创建256x256的黑色背景图像
final_image_a = np.zeros((256, 256, 3), dtype=np.uint8)
final_image_b = np.zeros((256, 256, 3), dtype=np.uint8)
final_mask_a = np.zeros((256, 256), dtype=np.uint8)
final_mask_b = np.zeros((256, 256), dtype=np.uint8)

# 将最佳窗口放入最终图像中
final_image_a[:window_size, :window_size] = best_crop_a
final_image_b[:window_size, :window_size] = best_crop_b

# 将标签也填充到最终图像中
final_mask_a[:window_size, :window_size] = mask_a[:window_size, :window_size]
final_mask_b[:window_size, :window_size] = mask_b[:window_size, :window_size]

# 保存处理后的影像和标签
cv2.imwrite('final_image_a.png', final_image_a)
cv2.imwrite('final_image_b.png', final_image_b)
cv2.imwrite('final_mask_a.png', final_mask_a)
cv2.imwrite('final_mask_b.png', final_mask_b)



# # 特征点匹配与配准
# orb = cv2.ORB_create()
# keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
# keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors_a, descriptors_b)
# matches = sorted(matches, key=lambda x: x.distance)
#
# # 计算匹配点的坐标
# points_a = np.float32([keypoints_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
# points_b = np.float32([keypoints_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#
# # 估计单应性矩阵
# matrix, mask = cv2.findHomography(points_b, points_a, cv2.RANSAC)
#
# # 对影像B进行变换
# aligned_image_b = cv2.warpPerspective(image_b, matrix, (image_a.shape[1], image_a.shape[0]))
# aligned_mask_b = cv2.warpPerspective(mask_b, matrix, (mask_a.shape[1], mask_a.shape[0]))
#
# # 计算重叠区域
# overlap_mask = np.logical_and(mask_a > 0, aligned_mask_b > 0).astype(np.uint8)
#
# # 找到重叠区域的边界
# x, y, w, h = cv2.boundingRect(overlap_mask)
#
# # 裁剪重叠区域
# cropped_image_a = image_a[y:y+h, x:x+w] if y+h <= image_a.shape[0] and x+w <= image_a.shape[1] else np.zeros((h, w, 3), dtype=np.uint8)
# cropped_image_b = aligned_image_b[y:y+h, x:x+w] if y+h <= aligned_image_b.shape[0] and x+w <= aligned_image_b.shape[1] else np.zeros((h, w, 3), dtype=np.uint8)
#
# # 创建256x256的黑色背景
# final_image_a = np.zeros((256, 256, 3), dtype=np.uint8)
# final_image_b = np.zeros((256, 256, 3), dtype=np.uint8)
#
# # 将裁剪后的图像放入256x256图像中
# final_image_a[0:cropped_image_a.shape[0], 0:cropped_image_a.shape[1]] = cropped_image_a
# final_image_b[0:cropped_image_b.shape[0], 0:cropped_image_b.shape[1]] = cropped_image_b
#
# # 保存处理后的影像
# cv2.imwrite('final_image_a.png', final_image_a)
# cv2.imwrite('final_image_b.png', final_image_b)
