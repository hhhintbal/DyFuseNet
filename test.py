# test.py
import os
import numpy as np
from skimage import io
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.swinfusenet.vision_transformer import SwinFuseNet as ViT_seg
from utils import convert_from_color, convert_to_color, sliding_window, count_sliding_window, grouper


# 配置参数
class TestConfig:
    # 硬件设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 路径配置
    dataset_root = "/root/autodl-tmp/py_testASMF/predealData/dealed_data/"  # Postdam测试集路径
    model_path = "results/best_model_0.2988.pth"  # 训练好的模型路径
    output_dir = "res_images/"  # 预测结果保存路径

    # 数据参数
    test_ids = ['0','2560']  # 要测试的区块ID列表
    n_classes = 6
    class_names = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]

    # 推理参数
    window_size = (224, 224)  # 分块大小
    stride = 128  # 滑动步长（影响边缘处理效果）
    batch_size = 8  # 推理批大小
    use_flip_aug = True  # 是否使用翻转增强（TTA）


# 初始化配置
cfg = TestConfig()


def load_model():
    model = ViT_seg(num_classes=cfg.n_classes).to(cfg.device)
    state_dict = torch.load(cfg.model_path, map_location=cfg.device)

    # 处理可能的DataParallel包装
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    # 注入维度检查
    original_forward = model.forward

    def checked_forward(x, dsm):
        # 输入维度验证
        assert x.dim() == 4, f"输入必须是4D张量，实际维度: {x.dim()}"
        assert dsm.dim() == 4, f"DSM必须是4D张量，实际维度: {dsm.dim()}"
        return original_forward(x, dsm)

    model.forward = checked_forward
    model.eval()
    return model


def prepare_data(test_id):
    """加载测试数据"""
    # 数据路径模板
    # data_path = os.path.join(cfg.dataset_root, f"top/top_mosaic_09cm_area{test_id}.tif")
    # dsm_path = os.path.join(cfg.dataset_root, f"dsm/dsm_09cm_matching_area{test_id}.tif")
    # label_path = os.path.join(cfg.dataset_root,
    #                           f"gts_eroded_for_participants/top_mosaic_09cm_area{test_id}_noBoundary.tif")

    data_path = os.path.join(cfg.dataset_root, f"RGB/10/10_0_{test_id}.tif")
    dsm_path = os.path.join(cfg.dataset_root, f"DSM/10/10_0_{test_id}.tif")
    label_path =os.path.join(cfg.dataset_root, f'lable/10/10_0_{test_id}.tif')

    # 加载并预处理数据
    rgb = io.imread(data_path).astype(np.float32) / 255.0
    dsm = io.imread(dsm_path).astype(np.float32)
    label = convert_from_color(io.imread(label_path)) if os.path.exists(label_path) else None

    # 归一化DSM
    dsm = (dsm - dsm.min()) / (dsm.max() - dsm.min())

    # 转换维度 (H, W, C) -> (C, H, W)
    rgb = rgb.transpose(2, 0, 1)
    return rgb, dsm, label

# 修改滑动窗口生成函数（关键修复）
def sliding_window(img, step, window_size):
    """生成有效分块坐标"""
    _, h, w = img.shape  # 输入为(C,H,W)
    win_h, win_w = window_size
    for y in range(0, h, step):
        for x in range(0, w, step):
            # 计算实际分块范围（防止负尺寸）
            y_end = min(y + win_h, h)
            x_end = min(x + win_w, w)
            actual_h = y_end - y
            actual_w = x_end - x
            if actual_h > 0 and actual_w > 0:  # 仅生成有效分块
                yield (y, y_end, x, x_end, actual_h, actual_w)


# 修改inference函数（处理边界分块）
def inference(model, rgb, dsm):
    """执行分块推理"""
    # 初始化预测结果矩阵
    h, w = rgb.shape[1], rgb.shape[2]
    pred = np.zeros((h, w, cfg.n_classes), dtype=np.float32)

    # 分块推理（添加有效性检查）
    total = count_sliding_window(rgb, step=cfg.stride, window_size=cfg.window_size) // cfg.batch_size
    with torch.no_grad():
        for coords in tqdm(grouper(cfg.batch_size, sliding_window(rgb, cfg.stride, cfg.window_size)),
                           total=total, desc="Processing patches"):

            batch_rgb = []
            batch_dsm = []
            valid_coords = []

            for c in coords:
                if c is None:
                    continue
                y_start, y_end, x_start, x_end, h, w = c

                # 跳过无效分块
                if h == 0 or w == 0:
                    continue

                # 提取分块并进行边缘填充
                rgb_patch = np.zeros((3, cfg.window_size[0], cfg.window_size[1]), dtype=np.float32)
                dsm_patch = np.zeros(cfg.window_size, dtype=np.float32)

                actual_rgb = rgb[:, y_start:y_end, x_start:x_end]
                actual_dsm = dsm[y_start:y_end, x_start:x_end]

                rgb_patch[:, :h, :w] = actual_rgb
                dsm_patch[:h, :w] = actual_dsm

                batch_rgb.append(rgb_patch.transpose(1, 2, 0))  # (C,H,W) -> (H,W,C)
                batch_dsm.append(dsm_patch)
                valid_coords.append((y_start, y_end, x_start, x_end))

            if not batch_rgb:
                continue

            # 转换为Tensor（添加维度验证）
            batch_rgb = torch.from_numpy(np.array(batch_rgb)).float().permute(0, 3, 1, 2).to(cfg.device)
            batch_dsm = torch.from_numpy(np.array(batch_dsm)).unsqueeze(1).float().to(cfg.device)

            # 模型推理（添加异常捕获）
            try:
                outputs = model(batch_rgb, batch_dsm)
                outputs = F.softmax(outputs, dim=1).cpu().numpy()
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Input shapes - RGB: {batch_rgb.shape}, DSM: {batch_dsm.shape}")
                continue

            # 填充预测结果（加权处理）
            for out, (y_s, y_e, x_s, x_e) in zip(outputs, valid_coords):
                h = y_e - y_s
                w = x_e - x_s
                out = out.transpose(1, 2, 0)[:h, :w, :]  # 裁剪到实际尺寸
                pred[y_s:y_e, x_s:x_e] += out

                # TTA增强（添加有效性检查）
                if cfg.use_flip_aug and h > 1 and w > 1:
                    pred[y_s:y_e, x_s:x_e] += np.fliplr(out)
                    pred[y_s:y_e, x_s:x_e] += np.flipud(out)

    # 合并预测结果（防止除以零）
    if cfg.use_flip_aug:
        valid_mask = pred.sum(axis=-1) > 0
        pred[valid_mask] /= 3.0

    final_pred = np.argmax(pred, axis=-1)
    return final_pred

def evaluate(pred, label):
    """计算评估指标"""
    from sklearn.metrics import confusion_matrix

    # 过滤无效区域（如有）
    mask = label != 255
    y_true = label[mask].ravel()
    y_pred = pred[mask].ravel()

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(cfg.n_classes))

    # 计算各类IoU
    ious = []
    for i in range(cfg.n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        iou = tp / (tp + fp + fn + 1e-10)
        ious.append(iou)

    # 计算平均指标
    miou = np.nanmean(ious)
    acc = np.diag(cm).sum() / cm.sum()

    return {
        "confusion_matrix": cm,
        "class_iou": dict(zip(cfg.class_names, ious)),
        "mean_iou": miou,
        "overall_accuracy": acc
    }


def save_results(test_id, pred, rgb, label=None):
    """保存可视化结果"""
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 保存预测图
    color_pred = convert_to_color(pred)
    io.imsave(os.path.join(cfg.output_dir, f"pred_{test_id}.png"), color_pred)

    # 保存叠加效果图
    # plt.figure(figsize=(20, 10))
    #
    # plt.subplot(1, 3, 1)
    # plt.imshow(rgb.transpose(1, 2, 0))
    # plt.title("Original Image")
    #
    # plt.subplot(1, 3, 2)
    # plt.imshow(color_pred)
    # plt.title("Prediction")

    if label is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(convert_to_color(label))
        plt.title("Ground Truth")

    # plt.savefig(os.path.join(cfg.output_dir, f"comparison_{test_id}.png"))
    # plt.close()


def main():
    # 加载模型
    model = load_model()

    # 遍历测试样本
    for test_id in cfg.test_ids:
        print(f"\nProcessing test sample {test_id}...")

        # 加载数据
        rgb, dsm, label = prepare_data(test_id)

        # 执行推理
        pred = inference(model, rgb, dsm)

        # 评估结果
        if label is not None:
            metrics = evaluate(pred, label)
            print(f"Overall Accuracy: {metrics['overall_accuracy'] * 100:.2f}%")
            print(f"Mean IoU: {metrics['mean_iou'] * 100:.2f}%")
            for name, iou in metrics['class_iou'].items():
                print(f"{name}: {iou * 100:.2f}%")

        # 保存结果
        save_results(test_id, pred, rgb, label)
        print(f"Saved results for {test_id}")


if __name__ == "__main__":
    main()