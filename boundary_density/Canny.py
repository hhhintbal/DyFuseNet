import cv2
import numpy as np
from typing import List, Tuple
import os

# 定义类别颜色映射
CLASS_COLORS = {
    'road': (255, 255, 255),    # 白色 (RGB)
    'building': (0, 0, 255),    # 蓝色 (RGB)
    'tree': (0, 255, 0),        # 绿色 (RGB)
    'low_vegetation': (255, 255, 0),  # 浅蓝色 (RGB)
    'vehicle': (0, 255, 255),   # 黄色 (RGB)
    'background': (0, 0, 0)     # 背景
}

# 目标类别对
TARGET_PAIRS = [('building', 'low_vegetation'), ('road', 'low_vegetation')]

def calculate_boundary_density(label_path: str) -> float:
    """
    计算单张标注图像的边界密度
    Args:
        label_path: 标注图像路径（PNG格式，RGB通道）
    Returns:
        boundary_density: 边界密度百分比（0-100）
    """
    # 1. 加载标注图像并转为NumPy数组
    if not os.path.exists(label_path):
        print(f"文件不存在: {label_path}")
        return 0.0  # 或者根据需要抛出异常

    label_img = cv2.imread(label_path)  # shape: (H, W, 3), BGR格式
    if label_img is None:
        print(f"无法读取图像: {label_path}")
        return 0.0  # 或者根据需要抛出异常

    print(f"成功读取图像: {label_path}, 尺寸: {label_img.shape}")

    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)  # 转为RGB
    h, w = label_img.shape[:2]
    total_pixels = h * w

    # 2. 生成类别掩码
    class_mask = np.zeros((h, w), dtype=np.uint8)  # 0: 无目标类别，1-6: 对应类别
    for class_name, rgb_color in CLASS_COLORS.items():
        mask = np.all(label_img == rgb_color, axis=-1)
        if class_name in ['building', 'low_vegetation', 'road']:
            class_id = list(CLASS_COLORS.keys()).index(class_name) + 1
            class_mask[mask] = class_id

    # 3. 边缘检测
    target_mask = np.zeros((h, w), dtype=np.uint8)
    for class_name, _ in TARGET_PAIRS:
        if class_name in CLASS_COLORS:
            rgb_color = CLASS_COLORS[class_name]
            mask = np.all(label_img == rgb_color, axis=-1)
            target_mask[mask] = 1

    gray_mask = (target_mask * 255).astype(np.uint8)
    edges = cv2.Canny(gray_mask, threshold1=int(0.1 * 255), threshold2=int(0.3 * 255))

    # 4. 统计边界像素数
    boundary_pixels = np.count_nonzero(edges)

    # 5. 计算边界密度
    boundary_density = (boundary_pixels / total_pixels) * 100
    return boundary_density

def compute_dataset_boundary_density(label_paths: List[str]) -> Tuple[float, float]:
    densities = []
    for path in label_paths:
        density = calculate_boundary_density(path)
        densities.append(density)
        print(f"图像 {os.path.basename(path)} 的边界密度: {density:.2f}%")

    mean_density = np.mean(densities)
    std_density = np.std(densities)
    return mean_density, std_density

if __name__ == "__main__":
    label_paths_vaihingen =  "/root/autodl-tmp/DyFuseNet/data/gts_for_participants/*.tif"
    label_paths_potsdam = "/root/autodl-tmp/DyFuseNet/data/Postdom_original/3_Labels_all/*.tif"

    # 使用glob获取所有匹配的文件路径
    import glob
    vaihingen_paths = sorted(glob.glob(label_paths_vaihingen))
    potsdam_paths = sorted(glob.glob(label_paths_potsdam))

    print(f"Vaihingen 图像数量: {len(vaihingen_paths)}")
    print(f"Potsdam 图像数量: {len(potsdam_paths)}")

    vaihingen_mean, vaihingen_std = compute_dataset_boundary_density(vaihingen_paths)
    potsdam_mean, potsdam_std = compute_dataset_boundary_density(potsdam_paths)

    print(f"Vaihingen: mean boundary density = {vaihingen_mean:.1f}% ± {vaihingen_std:.1f}%")
    print(f"Potsdam: mean boundary density = {potsdam_mean:.1f}% ± {potsdam_std:.1f}%")