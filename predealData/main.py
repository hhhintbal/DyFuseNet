import os
import numpy as np
from skimage import io
from tqdm import tqdm


def crop_and_save_images(rgb_path, dsm_path, gts_for_participants_path, label_path, output_dir, crop_size=512, stride=512):
    """
    切割并保存图像块
    :param rgb_path: RGB图像路径
    :param dsm_path: DSM高程图路径
    :param label_path: 标签图像路径
    :param output_dir: 输出根目录
    :param crop_size: 切割尺寸（默认512）
    :param stride: 滑动步长（默认512）
    """
    # 创建输出目录
    base_name = os.path.splitext(os.path.basename(rgb_path))[0].replace('top_potsdam_RGB', '')
    os.makedirs(os.path.join(output_dir, 'RGB', base_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'DSM', base_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gts_for_participants', base_name), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Label', base_name), exist_ok=True)

    # 读取图像
    rgb = io.imread(rgb_path)
    dsm = io.imread(dsm_path)
    gts_for_participants = io.imread(gts_for_participants_path)
    label = io.imread(label_path)

    # 获取图像尺寸并计算填充量
    h, w = rgb.shape[:2]
    pad_h = (crop_size - (h % crop_size)) % crop_size
    pad_w = (crop_size - (w % crop_size)) % crop_size

    # 填充图像（边缘镜像填充）
    rgb_padded = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    dsm_padded = np.pad(dsm, ((0, pad_h), (0, pad_w)), mode='constant')
    gts_for_participants_padded = np.pad(gts_for_participants, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    label_padded = np.pad(label, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    # 滑动窗口切割
    count = 0
    for y in tqdm(range(0, h + pad_h, stride), desc=f"Processing {base_name}"):
        for x in range(0, w + pad_w, stride):
            # 计算当前切片范围
            y_end = y + crop_size
            x_end = x + crop_size

            # 提取图像块
            rgb_patch = rgb_padded[y:y_end, x:x_end, :]
            dsm_patch = dsm_padded[y:y_end, x:x_end]
            gts_for_participants_patch = gts_for_participants_padded[y:y_end, x:x_end]
            label_patch = label_padded[y:y_end, x:x_end, :]

            # 仅保存完整尺寸的块
            if rgb_patch.shape[0] == crop_size and rgb_patch.shape[1] == crop_size:
                # 保存RGB
                io.imsave(
                    os.path.join(output_dir, 'RGB', base_name, f'{count}.tif'),
                    rgb_patch,
                    check_contrast=False
                )
                # 保存DSM
                io.imsave(
                    os.path.join(output_dir, 'DSM', base_name, f'{count}.tif'),
                    dsm_patch,
                    check_contrast=False
                )

                # 保存gts_for_participants
                io.imsave(
                    os.path.join(output_dir, 'gts_for_participants', base_name, f'{count}.tif'),
                    gts_for_participants_patch,
                    check_contrast=False
                )


                # 保存Label
                io.imsave(
                    os.path.join(output_dir, 'Label', base_name, f'{count}.tif'),
                    label_patch,
                    check_contrast=False
                )
                count += 1

    print(f"保存完成！共生成 {count} 个图像块")


if __name__ == "__main__":
    # 使用示例
    dataset_root = "/root/autodl-fs/Postdam/Postdam/"
    output_dir = "/root/autodl-tmp/py_testASMF/predealData/dealed_data/"

    # 假设存在以下文件：
    test_ids = ['11']
    count = 0
    for test_id in test_ids:
        rgb_path = os.path.join(dataset_root, f"top/top_potsdam_RGB{test_id}.tif")
        dsm_path = os.path.join(dataset_root, f"dsm/dsm_potsdam_normalized_lastools{test_id}.jpg")
        gts_for_participants_path = os.path.join(dataset_root,
                                  f"gts_for_participants/top_potsdam_label{test_id}.tif")
        label_path = os.path.join(dataset_root,
                                  f"gts_eroded_for_participants/top_potsdam_label_noBoundary{test_id}.tif")

        crop_and_save_images(
            rgb_path=rgb_path,
            dsm_path=dsm_path,
            gts_for_participants_path=gts_for_participants_path,
            label_path=label_path,
            output_dir=output_dir,
            crop_size=512,
            stride=512
        )