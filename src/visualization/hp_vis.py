import os
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from typing import Optional
import numpy as np

def heatmaps_vis(batch_idx: int, batch: torch.Tensor, encodings: torch.Tensor, save_path: str, vis_type: str, cmap: Optional[str] = 'viridis'):
    """
    保存编码张量的热度图。

    参数:
    - encodings: 一个形状为 [N, C, H, W] 的编码张量。
    - save_path: 保存图像的路径。
    - vis_type: 可视化的方式
        - 单独 - "I" (for "Individual")
        - 单独（带原图） - "IO" (for "Individual + Original")
        - 并行 - "P" (for "Parallel")
        - 并行（带原图） - "PO" (for "Parallel + Original")
    - cmap: 使用的颜色映射。
    """
    # 确保保存路径存在
    folder_path = os.path.join(save_path, f"batch_idx_{str(batch_idx)}")
    os.makedirs(folder_path, exist_ok=True)

    img1 = batch["transformed_image1"]
    img2 = batch["transformed_image2"]

    # 检查第一个维度是否相等
    assert img1.shape[0] + img2.shape[0] == encodings.shape[0] == 2, f"The first dimensions of transformed_image:{img1.shape[0] + img2.shape[0]} and encodings:{encodings.shape[0]} do not eqaul 2."

    if vis_type == 'IO':
        for i, (img, encoding) in enumerate(zip([img1, img2], encodings)):
            # 将图像数据转移到CPU并转换为numpy数组
            img_numpy = img.squeeze().permute(1, 2, 0).cpu().numpy()

            # 对于浮点数图像数据，确保其范围在 [0, 1] 内
            if img.dtype == torch.float32 and img_numpy.max() > 1:
                img_numpy = img_numpy / img_numpy.max()  # 归一化到 [0, 1]
            img_numpy = np.clip(img_numpy, 0, 1)  # 确保值在 [0, 1] 范围内
            
            # 创建保存图像的位置
            # 创建一个包含两个子图的图形：左侧是原图，右侧是热度图
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # 显示原图在左侧
            axs[0].imshow(img_numpy)  # 直接显示彩色图像
            axs[0].set_title(f"Original Image {i+1}")
            axs[0].axis('off')  # 不显示坐标轴

            # 显示热度图在右侧
            im = axs[1].imshow(encoding.squeeze().cpu().numpy(), cmap=cmap)
            axs[1].set_title(f"Heatmap {i+1}")
            axs[1].axis('off')
            fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

            # 保存图像
            plt.savefig(f"{folder_path}/combined_{i+1}.png")
            plt.close(fig)