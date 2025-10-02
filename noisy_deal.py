import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
from PIL import Image
import glob

# 定义DSM和多光谱图像的路径
DSM_PATH = r"E:\Users\admin\PycharmProjects\pj1\4.15\ASMFNet\predealData\dealed_data\DSM\11\*.png"  # 修改为实际路径
IRRB_PATH = r"E:\Users\admin\PycharmProjects\pj1\4.15\ASMFNet\predealData\dealed_data\RGB\11\*.png"  # 修改为实际路径

# 定义保存加噪图像的路径
SAVE_PATH = './noisy_samples/'
os.makedirs(SAVE_PATH, exist_ok=True)


def load_image_as_tensor(image_path: str, is_dsm: bool = False) -> torch.Tensor:
    """
    从图像路径加载图像并转换为PyTorch张量。

    参数:
        image_path (str): 图像文件路径
        is_dsm (bool): 是否为DSM图像（单通道，高程值）

    返回:
        torch.Tensor: 形状为 (1, H, W) 如果是DSM，或 (C, H, W) 如果是多光谱
    """
    # 使用PIL加载图像
    img = Image.open(image_path)
    img = img.convert('L') if is_dsm else img.convert('RGB')  # DSM为灰度，多光谱为RGB或更多通道

    # 转换为NumPy数组
    img_np = np.array(img)

    if is_dsm:
        # DSM图像应为单通道
        if len(img_np.shape) == 3:
            img_np = img_np[:, :, 0]  # 如果意外加载为RGB，取第一个通道
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        # 多光谱图像，假定为RGB或更多通道
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)  # 如果是灰度，转换为三通道
        img_np = img_np.transpose(2, 0, 1)  # 转为 (C, H, W)
        img_tensor = torch.from_numpy(img_np).float().unsqueeze(0)  # (1, C, H, W)

    return img_tensor


def add_multimodal_noise(
        dsm_tensor: torch.Tensor,  # DSM张量，形状 (B, 1, H, W)
        ms_tensor: torch.Tensor,  # 多光谱张量，形状 (B, C, H, W)
        dsm_sigma: float = 0.5,  # DSM噪声标准差（单位：高程值，如米）
        ms_sigma: float = 5.0,  # 多光谱噪声标准差（单位：辐射值，如DN值）
        save_path: str = None,  # 可选：保存加噪图像的路径（如 './noisy_samples/'）
        sample_idx: int = 0  # 当前处理的样本索引（用于保存多批次中的单张图）
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对DSM和多光谱图像分别添加高斯噪声，并可选保存加噪结果。

    返回:
        noisy_dsm: 加噪后的DSM张量（与输入形状一致）
        noisy_ms: 加噪后的多光谱张量（与输入形状一致）
    """
    # ===== 1. 确保输入为浮点型（避免整数运算溢出） =====
    dsm_tensor = dsm_tensor.float()
    ms_tensor = ms_tensor.float()

    # ===== 2. 处理单样本输入（确保形状为 (B, 1, H, W) 和 (B, C, H, W)） =====
    B, C_dsm, H, W = dsm_tensor.shape
    B_ms, C_ms, H_ms, W_ms = ms_tensor.shape

    assert B == B_ms and H == H_ms and W == W_ms, "DSM和多光谱图像的批次、高度和宽度必须一致"

    # ===== 3. 生成DSM高斯噪声（单位：高程值） =====
    dsm_noise = torch.randn(B, 1, H, W, device=dsm_tensor.device) * dsm_sigma
    noisy_dsm = dsm_tensor + dsm_noise

    # ===== 4. 生成多光谱高斯噪声（单位：辐射值） =====
    ms_noise = torch.randn(B, C_ms, H, W, device=ms_tensor.device) * ms_sigma
    noisy_ms = ms_tensor + ms_noise

    # ===== 5. 可选：保存加噪图像（可视化验证） =====
    if save_path is not None:
        # 保存加噪DSM（单样本示例，取批次第一个）
        dsm_np = noisy_dsm[0, 0].cpu().numpy()  # (H, W)
        plt.imsave(os.path.join(save_path, f'noisy_dsm_sample{sample_idx}.png'), dsm_np, cmap='gray')

        # 保存加噪多光谱（取第一个通道可视化，或合成RGB）
        ms_np = noisy_ms[0].cpu().numpy()  # (C, H, W)
        if C_ms >= 3:  # 若多光谱包含RGB波段（如前3通道）
            rgb_ms = ms_np[:3]  # 取前3通道（假设为RGB）
            rgb_ms = (rgb_ms - rgb_ms.min()) / (rgb_ms.max() - rgb_ms.min())  # 归一化到 [0,1]
            plt.imsave(os.path.join(save_path, f'noisy_ms_rgb_sample{sample_idx}.png'), rgb_ms.transpose(1, 2, 0))
        else:  # 单通道或多通道（保存第一个通道）
            plt.imsave(os.path.join(save_path, f'noisy_ms_sample{sample_idx}.png'), ms_np[0], cmap='viridis')

    return noisy_dsm, noisy_ms


def process_images(dsm_dir: str, ms_dir: str, dsm_sigma: float = 0.5, ms_sigma: float = 5.0,
                   save_path: str = './noisy_samples/'):
    """
    处理指定目录中的所有DSM和多光谱图像，为每对图像添加噪声并保存。

    参数:
        dsm_dir (str): DSM图像目录路径，支持通配符（如 *.png）
        ms_dir (str): 多光谱图像目录路径，支持通配符（如 *.png）
        dsm_sigma (float): DSM噪声标准差
        ms_sigma (float): 多光谱噪声标准差
        save_path (str): 保存加噪图像的路径
    """
    os.makedirs(save_path, exist_ok=True)
    dsm_paths = sorted(glob.glob(dsm_dir))
    ms_paths = sorted(glob.glob(ms_dir))

    if len(dsm_paths) != len(ms_paths):
        print(f"警告：DSM图像数量 ({len(dsm_paths)}) 与多光谱图像数量 ({len(ms_paths)}) 不匹配。")
        min_len = min(len(dsm_paths), len(ms_paths))
        dsm_paths = dsm_paths[:min_len]
        ms_paths = ms_paths[:min_len]

    for idx, (dsm_path, ms_path) in enumerate(zip(dsm_paths, ms_paths)):
        print(f"处理第 {idx + 1} 对图像: DSM={dsm_path}, 多光谱={ms_path}")

        # 加载DSM和多光谱图像
        dsm_tensor = load_image_as_tensor(dsm_path, is_dsm=True)
        ms_tensor = load_image_as_tensor(ms_path, is_dsm=False)

        # 确保DSM和多光谱图像的尺寸一致
        assert dsm_tensor.shape[2:] == ms_tensor.shape[
                                       2:], f"DSM和多光谱图像尺寸不一致: DSM {dsm_tensor.shape[2:]}, 多光谱 {ms_tensor.shape[2:]}"

        # 添加噪声
        noisy_dsm, noisy_ms = add_multimodal_noise(
            dsm_tensor=dsm_tensor,
            ms_tensor=ms_tensor,
            dsm_sigma=dsm_sigma,
            ms_sigma=ms_sigma,
            save_path=save_path,
            sample_idx=idx
        )

        # 如果需要，可以进一步处理或保存noisy_dsm和noisy_ms张量
        # 例如，保存为新的图像文件或进行后续模型输入

    print("所有图像处理完成。")


if __name__ == "__main__":
    # 定义DSM和多光谱图像的路径模式
    DSM_PATTERN = r"E:\Users\admin\PycharmProjects\pj1\4.15\ASMFNet\predealData\dealed_data\DSM\11\*.tif"  # 修改为实际路径
    IRRB_PATTERN = r"E:\Users\admin\PycharmProjects\pj1\4.15\ASMFNet\predealData\dealed_data\RGB\11\*.tif"  # 修改为实际路径

    # 定义噪声参数
    DSM_SIGMA = 0.5  # DSM噪声标准差（米）
    MS_SIGMA = 5.0  # 多光谱噪声标准差（辐射值）

    # 处理图像
    process_images(
        dsm_dir=DSM_PATTERN,
        ms_dir=IRRB_PATTERN,
        dsm_sigma=DSM_SIGMA,
        ms_sigma=MS_SIGMA,
        save_path='./noisy_samples/'
    )