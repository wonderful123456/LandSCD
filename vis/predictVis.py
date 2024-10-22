import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms.functional as TF

from models.SwinTransformerV2UperNetSegV1 import SwinTransformerUperNetBase

class SemanticChangeDetector:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        return self.transform(img).unsqueeze(0)  # 添加一个维度

    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_G_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    def predict(self, img_A, img_B):
        self.model.eval()
        with torch.no_grad():
            img_A_pil = TF.to_pil_image(img_A)
            img_B_pil = TF.to_pil_image(img_B)
            pred_A, pred_B, pred_change = self.model(img_A_pil.to(self.device), img_B_pil.to(self.device))
            return torch.argmax(pred_A, dim=1).cpu(), torch.argmax(pred_B, dim=1).cpu(), torch.argmax(pred_change,
                                                                                                      dim=1).cpu()

    def map_to_color(self, pred, color_map):
        pred = pred.squeeze(0)
        color_image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for color, label in color_map.items():
            color_image[pred == label] = color
        return color_image

    def visualize(self, img_A, img_B, pred_A, pred_B, pred_change):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        color_map = {
            (0, 0, 0): 0,  # 黑色 -> 标签值 0 背景
            (255, 255, 0): 1,  # 黄色 -> 标签值 1 耕地
            (255, 0, 0): 2,  # 红色 -> 标签值 2 建筑
            (128, 0, 128): 3,  # 紫色 -> 标签值 3 森林
            (0, 255, 0): 4,  # 绿色 -> 标签值 4 草地
            (0, 0, 255): 5  # 蓝色 -> 标签值 5 水体
        }

        color_map_ch = {
            (0, 0, 0): 0,  # 黑色 -> 标签值 0 背景
            (255, 255, 255): 1  # 蓝色 -> 标签值 5 水体
        }

        # 映射预测结果到颜色图
        color_pred_A = self.map_to_color(pred_A, color_map)
        color_pred_B = self.map_to_color(pred_B, color_map)
        color_pred_change = self.map_to_color(pred_change, color_map_ch)

        axes[0, 0].imshow(img_A.squeeze(0).permute(1, 2, 0))
        axes[0, 0].set_title("Image A")
        axes[0, 1].imshow(img_B.squeeze(0).permute(1, 2, 0))
        axes[0, 1].set_title("Image B")
        axes[0, 2].imshow(color_pred_A)
        axes[0, 2].set_title("Prediction A")

        axes[1, 0].imshow(color_pred_B)
        axes[1, 0].set_title("Prediction B")
        axes[1, 1].imshow(color_pred_change)
        axes[1, 1].set_title("Change Detection")

        for ax in axes.flatten():
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def process_images_in_folder(self, folder_A, folder_B):
        img_A_paths = sorted(Path(folder_A).glob("*"))  # 获取文件夹中所有图像路径
        img_B_paths = sorted(Path(folder_B).glob("*"))  # 假设两个文件夹中的文件一一对应

        for img_A_path, img_B_path in zip(img_A_paths, img_B_paths):
            img_A = self.load_image(img_A_path)
            img_B = self.load_image(img_B_path)

            # 进行预测
            pred_A, pred_B, pred_change = self.predict(img_A, img_B)

            # 可视化结果
            self.visualize(img_A, img_B, pred_A, pred_B, pred_change)


if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = SwinTransformerUperNetBase(num_classes=6)  # 实例化和加载您训练好的模型
    model.to(device)

    detector = SemanticChangeDetector(model, device)

    # 加载权重
    checkpoint_path = "../checkpoints/SwinTransformerUperNetBase_T3/best_ckpt.pt"
    detector.load_checkpoint(checkpoint_path)

    # 定义图像文件夹路径
    folder_A = "C:\\Users\\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\A"
    folder_B = "C:\\Users\\zyy\Documents\Tencent Files\919688409\FileRecv\\regularCultivatedLandDatasetsV4\B"

    # 处理文件夹中的所有图像
    detector.process_images_in_folder(folder_A, folder_B)
