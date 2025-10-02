import numpy as np
from skimage import io
from glob import glob
from tqdm.notebook import tqdm
import random
import itertools
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from torch.autograd import Variable
import os
from IPython.display import clear_output
from models.swinfusenet.vision_transformer import SwinFuseNet as ViT_seg

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Parameters
# Parameters
WINDOW_SIZE = (224, 224)  # Patch size
STRIDE = 32  # Stride for testing
IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
# FOLDER = "D:/RA/CUHK-SZ/ISPRS_dataset/ISPRS_semantic_labeling_Vaihingen/"
FOLDER = "/root/autodl-fs/"  # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10  # Number of samples in a mini-batch

LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]  # Label names
N_CLASSES = len(LABELS)  # Number of classes
WEIGHTS = torch.ones(N_CLASSES)  # Weights for class balancing
CACHE = True  # Store the dataset in-memorycd x
DATASET = 'Postdam/'
test_path = 'data/test/'
# MAIN_FOLDER = FOLDER + DATASET
MAIN_FOLDER = '/root/autodl-tmp/py_testASMF/predealData/dealed_data/'
# DATA_FOLDER = MAIN_FOLDER + 'top/top_mosaic_09cm_area{}.tif'
# DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_09cm_matching_area{}.tif'
# LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_mosaic_09cm_area{}.tif'
# ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif'

# Postdam Dataset

# DATA_FOLDER = MAIN_FOLDER + 'top/top_potsdam_RGB{}.tif'
# DSM_FOLDER = MAIN_FOLDER + 'dsm/dsm_potsdam_normalized_lastools{}.jpg'
# LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/top_potsdam_label{}.tif'
# ERODED_FOLDER = MAIN_FOLDER + 'gts_eroded_for_participants/top_potsdam_label_noBoundary{}.tif'


DATA_FOLDER = MAIN_FOLDER + 'RGB/11/{}.tif'
DSM_FOLDER = MAIN_FOLDER + 'DSM/11/{}.tif'
LABEL_FOLDER = MAIN_FOLDER + 'gts_for_participants/11/{}.tif'
ERODED_FOLDER = MAIN_FOLDER + 'Label/11/{}.tif'





class ISPRS_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(ISPRS_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.dsm_files = [DSM_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.dsm_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        # Default epoch size is 10 000 samples
        return 10000

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, i):
        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)

        # If the tile hasn't been loaded yet, put in cache
        if random_idx in self.data_cache_.keys():
            data = self.data_cache_[random_idx]
        else:
            # Data is normalized in [0, 1]
            data = io.imread(self.data_files[random_idx])
            # print("原始数据维度:", data.shape)
            data = 1 / 255 * np.asarray(data.transpose((2, 0, 1)), dtype='float32')
            if self.cache:
                self.data_cache_[random_idx] = data

        if random_idx in self.dsm_cache_.keys():
            dsm = self.dsm_cache_[random_idx]
        else:
            # DSM is normalized in [0, 1]
            dsm = np.asarray(io.imread(self.dsm_files[random_idx]), dtype='float32')

            min = np.min(dsm)
            max = np.max(dsm)
            dsm = (dsm - min) / (max - min)
            if self.cache:
                self.dsm_cache_[random_idx] = dsm

        if random_idx in self.label_cache_.keys():
            label = self.label_cache_[random_idx]
        else:
            # Labels are converted from RGB to their numeric values
            label = np.asarray(convert_from_color(io.imread(self.label_files[random_idx])), dtype='int64')
            if self.cache:
                self.label_cache_[random_idx] = label

        # Get a random patch
        x1, x2, y1, y2 = get_random_pos(data, WINDOW_SIZE)
        data_p = data[:, x1:x2, y1:y2]
        dsm_p = dsm[x1:x2, y1:y2]
        label_p = label[x1:x2, y1:y2]

        # Data augmentation
        data_p, dsm_p, label_p = self.data_augmentation(data_p, dsm_p, label_p)
        # print((torch.from_numpy(dsm_p).shape))
        # print((torch.from_numpy(data_p).shape))
        # Return the torch.Tensor values
        return (torch.from_numpy(data_p),
                torch.from_numpy(dsm_p),
                torch.from_numpy(label_p))


load_path = 'pretrain/swin_tiny_patch4_window7_224.pth'
train_path = 'res/segnet256_epoch10_999_87.55378723144531'
# load_path= '/content/drive/My Drive/SwinFuseNet/pretrain/swin_tiny_patch4_window7_224.pth'
net = ViT_seg(num_classes=6).to(device)
params = 0
for name, param in net.named_parameters():
    params += param.nelement()
print('Params: ', params)
net.load_from(load_path)
# Load the datasets
# net = nn.DataParallel(net)
# train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
# train_ids = ['0','1024','1536','2048','2560','3072','3584','4096','512']

# test_ids = ['1024','2560','0','1024','4608','1536','512','5120']
train_ids = [str(x) for x in range(0, 85)]
test_ids = ['41','42','13','25','58','71','19']
# test_ids = ['49','55','62','3','4','5','6','7','68','9','74','75','89']
print('Tiles for training :', train_ids)
print('Tiles for testing :', test_ids)

train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

base_lr = 0.001
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params': [value], 'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params': [value], 'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
# We define the scheduler
# lr=0.001, momentum=0.9, weight_decay=0.0005
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)


def test(net, test_ids, num1 = None, num2 = None, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (convert_from_color(io.imread(LABEL_FOLDER.format(id))) for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    count = 0
    with torch.no_grad():
        for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids),
                                       leave=False):
            # e1 = 0
            # len1 = len(list(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))))

            pred = np.zeros(img.shape[:2] + (N_CLASSES,))
            total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
            for i, coords in enumerate(
                    tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                         leave=False)):
                # Build the tensor
                image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = Variable(torch.from_numpy(image_patches).to(device))

                min = np.min(dsm)
                max = np.max(dsm)
                dsm = (dsm - min) / (max - min)
                dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                dsm_patches = np.asarray(dsm_patches)
                dsm_patches = Variable(torch.from_numpy(dsm_patches).to(device))

                # Do the inference
                outs = net(image_patches, dsm_patches)
                outs = outs.data.cpu().numpy()

                # Fill in the results array
                for out, (x, y, w, h) in zip(outs, coords):
                    out = out.transpose((1, 2, 0))
                    pred[x:x + w, y:y + h] += out
                del (outs)

            pred = np.argmax(pred, axis=-1)
            clear_output()

            all_preds.append(pred)
            # all_gts.append(gt)
            all_gts.append(gt_e)

            clear_output()
            # Compute some metrics
            # metrics(pred.ravel(), gt_e.ravel())
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                       np.concatenate([p.ravel() for p in all_gts]).ravel())
    # 在测试结束后保存预测结果
    os.makedirs('no_afm_res', exist_ok=True)  # 确保目录存在
    for pred, test_id in zip(all_preds, test_ids):
        pred_color = convert_to_color(pred)
        io.imsave(f'no_afm_res/pred_{test_id}.png', pred_color)
        print(f"Saved prediction for {test_id} to no_afm_res/pred_{test_id}.png")

    return accuracy, all_preds, all_gts
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=2):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.to(device)

    # criterion = nn.NLLLoss2d(weight=weights)
    iter_ = 0
    acc_best = 88.0
    oa = 0
    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()

        for batch_idx, (data, dsm, target) in enumerate(train_loader):
            net.train()
            data, dsm, target = Variable(data.to(device)), Variable(dsm.to(device)), Variable(target.to(device))
            optimizer.zero_grad()
            output = net(data, dsm)
            loss = CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()
            # # 输入：真实标签图 y_true (H, W)，模型预测概率 p (B, H, W, C)
            # edge_mask = canny_edge_detect(y_true)  # 生成二值边缘掩码 (H, W)，边缘像素为1，否则为0
            # for i in range(B * H * W):  # 遍历所有像素
            #     pixel_pos = get_pixel_position(i)  # 获取像素 (x,y) 的坐标
            #     is_edge = edge_mask[pixel_pos]  # 检查是否为边缘像素（0或1）
            #     loss += lambda *is_edge *(p[i] - 0.5) ^ 2  # 仅边缘像素计算该项

            losses[iter_] = loss.data
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                oa = accuracy(pred, gt)
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data, oa))

            iter_ += 1

            # del (data, target, loss)

        if e % 1 == 0:
            acc = test(net, test_ids, all=False, stride=32, num1=e, num2=batch_idx)
            print("acc:",acc)
            oa = 0
            if len(acc)>1:
                if acc[0] > acc_best:
                    torch.save(net.state_dict(), 'res/segnet256_epoch{}_{}_{}'.format(e, batch_idx, acc[0]))
                    # torch.save(net, 'res/complete_model_epoch{}_{}_{}.pth'.format(e, batch_idx, acc))

                    acc_best = acc[0]
                    print("Model saved with accuracy:", acc[0])
            else:
                if acc > acc_best:
                    torch.save(net.state_dict(), 'res/segnet256_epoch{}_{}_{}'.format(e, batch_idx, acc))
                    # torch.save(net, 'res/complete_model_epoch{}_{}_{}.pth'.format(e, batch_idx, acc))
                    acc_best = acc
                    print("Model saved with accuracy:", acc)
    print("Train Done!!")


train(net, optimizer, 30, scheduler)


# ######   test   ####
# acc, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
# acc= test(net, test_ids, all=True, stride=32)

# print("Acc: ", acc)
# for p, id_ in zip(all_preds, test_ids):
    # img = convert_to_color(p)
# #     # plt.imshow(img) and plt.show()
#     io.imsave('res_images/inference9064_tile_{}.png'.format(id_), img)


# import os
# import re
# import glob
# import random
# import numpy as np
# from skimage import io
# import matplotlib
#
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from models.swinfusenet.vision_transformer import SwinFuseNet as ViT_seg
# from utils import convert_from_color, convert_to_color
#
#
# # 配置参数
# class Config:
#     # 硬件设置
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # 路径配置
#     data_root = "/root/autodl-tmp/py_testASMF/predealData/dealed_data/"
#     output_dir = "./results/"
#
#     # 训练参数
#     batch_size = 8
#     num_epochs = 5
#     learning_rate = 1e-3
#     weight_decay = 1e-4
#     num_workers = 4
#
#     # 数据参数
#     window_size = (224, 224)
#     stride = 32
#     num_classes = 6
#     class_names = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
#
#     # 模型参数
#     pretrain_path = "pretrain/swin_tiny_patch4_window7_224.pth"
#
#
# cfg = Config()
#
#
# # 自动获取有效ID
# def get_valid_ids(data_root):
#     """获取所有完整的数据ID"""
#     rgb_files = glob.glob(os.path.join(data_root, "RGB/*/*.tif"))
#     id_pattern = re.compile(r"(\d+)_\d+_\d+\.tif")
#
#     valid_ids = set()
#     for f in rgb_files:
#         match = id_pattern.search(os.path.basename(f))
#         if match:
#             base_id = match.group(1)
#             # 检查其他模态数据是否存在
#             dsm_files = glob.glob(os.path.join(data_root, f"DSM/{base_id}/{base_id}_*.tif"))
#             label_files = glob.glob(os.path.join(data_root, f"gts_for_participants/{base_id}/{base_id}_*.tif"))
#             if dsm_files and label_files:
#                 valid_ids.add(base_id)
#     return sorted(list(valid_ids))
#
#
# # 自定义数据集类
# class PostdamDataset(Dataset):
#     def __init__(self, ids, data_root, cache=True, augmentation=True):
#         super().__init__()
#         self.ids = ids
#         self.data_root = data_root
#         self.cache = cache
#         self.augmentation = augmentation
#         self.samples = []
#
#         # 构建样本列表
#         for base_id in ids:
#             rgb_files = glob.glob(os.path.join(data_root, f"RGB/{base_id}/{base_id}_*.tif"))
#             for rgb_path in rgb_files:
#                 file_id = os.path.basename(rgb_path).replace(".tif", "")
#                 dsm_path = os.path.join(data_root, f"DSM/{base_id}/{file_id}.tif")
#                 label_path = os.path.join(data_root, f"gts_for_participants/{base_id}/{file_id}.tif")
#
#                 if os.path.exists(dsm_path) and os.path.exists(label_path):
#                     self.samples.append({
#                         "rgb": rgb_path,
#                         "dsm": dsm_path,
#                         "label": label_path
#                     })
#
#         # 初始化缓存
#         self.data_cache = {}
#         self.dsm_cache = {}
#         self.label_cache = {}
#
#     def __len__(self):
#         return len(self.samples) * 10  # 数据增强增强因子
#
#     def _augment(self, *arrays):
#         """数据增强"""
#         # 水平翻转
#         if random.random() > 0.5:
#             arrays = [np.fliplr(arr) for arr in arrays]
#         # 垂直翻转
#         if random.random() > 0.5:
#             arrays = [np.flipud(arr) for arr in arrays]
#         return arrays
#
#     def __getitem__(self, idx):
#         sample_idx = idx % len(self.samples)
#         sample = self.samples[sample_idx]
#
#         if sample["rgb"] in self.data_cache:
#             rgb = self.data_cache[sample["rgb"]]
#             dsm = self.dsm_cache[sample["dsm"]]
#             label = self.label_cache[sample["label"]]
#         else:
#             rgb = io.imread(sample["rgb"]).astype(np.float32) / 255.0
#             h_rgb, w_rgb = rgb.shape[0], rgb.shape[1]
#
#             # 读取并调整dsm和label尺寸
#             dsm = io.imread(sample["dsm"]).astype(np.float32)
#             dsm = resize(dsm, (h_rgb, w_rgb), preserve_range=True, anti_aliasing=False)
#
#             label = convert_from_color(io.imread(sample["label"]))
#             label = resize(label, (h_rgb, w_rgb), preserve_range=True, order=0, anti_aliasing=False)
#
#             dsm = (dsm - dsm.min()) / (dsm.max() - dsm.min() + 1e-6)
#             rgb = rgb.transpose(2, 0, 1)  # HWC -> CHW
#
#             if self.cache:
#                 self.data_cache[sample["rgb"]] = rgb
#                 self.dsm_cache[sample["dsm"]] = dsm
#                 self.label_cache[sample["label"]] = label
#
#         h, w = rgb.shape[1], rgb.shape[2]
#         # 确保图像尺寸足够大，否则填充
#         pad_h = max(cfg.window_size[0] - h, 0)
#         pad_w = max(cfg.window_size[1] - w, 0)
#         if pad_h > 0 or pad_w > 0:
#             rgb = np.pad(rgb, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
#             dsm = np.pad(dsm, ((0, pad_h), (0, pad_w)), mode='constant')
#             label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant')
#             h, w = h + pad_h, w + pad_w
#
#         x = random.randint(0, h - cfg.window_size[0])
#         y = random.randint(0, w - cfg.window_size[1])
#
#         rgb_patch = rgb[:, x:x + cfg.window_size[0], y:y + cfg.window_size[1]].copy()
#         dsm_patch = dsm[x:x + cfg.window_size[0], y:y + cfg.window_size[1]].copy()
#         label_patch = label[x:x + cfg.window_size[0], y:y + cfg.window_size[1]].copy()
#
#         if self.augmentation:
#             rgb_patch, dsm_patch, label_patch = self._augment(rgb_patch, dsm_patch, label_patch)
#
#         # 确保增强后尺寸正确（如随机缩放需调整）
#         # 若_augment改变尺寸，需在此处理或修改增强逻辑
#         # 例如：统一应用相同的随机裁剪或缩放
#
#         assert rgb_patch.shape == (3, cfg.window_size[0], cfg.window_size[1]), f"RGB shape error: {rgb_patch.shape}"
#         assert dsm_patch.shape == (cfg.window_size[0], cfg.window_size[1]), f"DSM shape error: {dsm_patch.shape}"
#         assert label_patch.shape == (cfg.window_size[0], cfg.window_size[1]), f"Label shape error: {label_patch.shape}"
#
#         return (
#             torch.from_numpy(rgb_patch).float(),
#             torch.from_numpy(dsm_patch).float(),
#             torch.from_numpy(label_patch).long()
#         )
#
# # 指标计算类
# class SegmentationMetrics:
#     def __init__(self, num_classes):
#         self.num_classes = num_classes
#         self.confusion = np.zeros((num_classes, num_classes))
#
#     def update(self, pred, target):
#         mask = (target >= 0) & (target < self.num_classes)
#         hist = np.bincount(
#             self.num_classes * target[mask].astype(int) + pred[mask],
#             minlength=self.num_classes ** 2
#         ).reshape(self.num_classes, self.num_classes)
#         self.confusion += hist
#
#     def compute(self):
#         tp = np.diag(self.confusion)
#         fp = self.confusion.sum(axis=0) - tp
#         fn = self.confusion.sum(axis=1) - tp
#
#         iou = tp / (tp + fp + fn + 1e-6)
#         acc = tp / (self.confusion.sum(axis=1) + 1e-6)
#
#         return {
#             "mIoU": np.nanmean(iou),
#             "pixel_acc": np.nansum(tp) / (self.confusion.sum() + 1e-6),
#             "class_iou": dict(zip(cfg.class_names, iou))
#         }
#
#
# # 模型初始化
# def init_model():
#     model = ViT_seg(num_classes=cfg.num_classes)
#     if os.path.exists(cfg.pretrain_path):
#         model.load_from(cfg.pretrain_path)
#         print(f"Loaded pretrained weights from {cfg.pretrain_path}")
#     return model.to(cfg.device)
#
#
# # 训练流程
# def train():
#     # 准备数据
#     all_ids = get_valid_ids(cfg.data_root)
#     random.seed(42)
#     random.shuffle(all_ids)
#     split_idx = int(len(all_ids) * 0.8)
#     train_ids, test_ids = all_ids[:split_idx], all_ids[split_idx:]
#
#     train_set = PostdamDataset(train_ids, cfg.data_root)
#     test_set = PostdamDataset(test_ids, cfg.data_root, augmentation=False)
#
#     train_loader = DataLoader(
#         train_set,
#         batch_size=cfg.batch_size,
#         shuffle=True,
#         num_workers=cfg.num_workers,
#         pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_set,
#         batch_size=cfg.batch_size,
#         num_workers=cfg.num_workers,
#         pin_memory=True
#     )
#
#     # 初始化模型
#     model = init_model()
#     optimizer = optim.AdamW(
#         model.parameters(),
#         lr=cfg.learning_rate,
#         weight_decay=cfg.weight_decay
#     )
#     criterion = nn.CrossEntropyLoss()
#
#     # 训练循环
#     best_miou = 0.0
#     for epoch in range(cfg.num_epochs):
#         model.train()
#         progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.num_epochs}")
#
#         for rgb, dsm, labels in progress:
#             rgb = rgb.to(cfg.device)
#             dsm = dsm.to(cfg.device)
#             labels = labels.to(cfg.device)
#
#             # 前向传播
#             outputs = model(rgb, dsm)
#             loss = criterion(outputs, labels)
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             progress.set_postfix({"loss": f"{loss.item():.4f}"})
#
#         # 验证
#         metrics = evaluate(model, test_loader)
#         print(f"Validation mIoU: {metrics['mIoU']:.4f}, Acc: {metrics['pixel_acc']:.4f}")
#
#         # 保存最佳模型
#         if metrics['mIoU'] > best_miou:
#             best_miou = metrics['mIoU']
#             torch.save(model.state_dict(), os.path.join(cfg.output_dir, f"best_model_{best_miou:.4f}.pth"))
#             print(f"Saved new best model with mIoU {best_miou:.4f}")
#
#
# # 评估流程
# def evaluate(model, loader):
#     model.eval()
#     metrics = SegmentationMetrics(cfg.num_classes)
#
#     with torch.no_grad():
#         for rgb, dsm, labels in tqdm(loader, desc="Evaluating"):
#             rgb = rgb.to(cfg.device)
#             dsm = dsm.to(cfg.device)
#             labels = labels.cpu().numpy()
#
#             outputs = model(rgb, dsm)
#             preds = torch.argmax(outputs, dim=1).cpu().numpy()
#
#             for pred, label in zip(preds, labels):
#                 metrics.update(pred.flatten(), label.flatten())
#
#     return metrics.compute()
#
#
# # 主程序
# if __name__ == "__main__":
#     os.makedirs(cfg.output_dir, exist_ok=True)
#     train()
# # 新增的模型导入
# import sys
# from IPython.display import clear_output
# from models.swinfusenet.vision_transformer import SwinFuseNet as ViT_seg
#
# # 其他原有导入
# import os
# import glob
# import random
# import numpy as np
# from skimage import io
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torch import optim
# from torchvision.transforms import functional as F
# from skimage.transform import resize
# from tqdm import tqdm
#
#
# # 配置参数
# class Config:
#     window_size = (224, 224)
#     stride = 32
#     batch_size = 16
#     n_classes = 6
#     cache = True
#     data_root = "/root/autodl-tmp/py_testASMF/predealData/dealed_data/"
#     lr = 0.001
#     epochs = 50
#     num_workers = 4
#
#
# cfg = Config()
#
#
# # 工具函数
# def convert_from_color(arr):
#     return (arr[..., 0] // 50) * 3 + (arr[..., 1] // 50)
#
#
# def get_all_samples(data_root):
#     samples = []
#     rgb_dirs = glob.glob(os.path.join(data_root, "RGB/**/*.tif"), recursive=True)
#
#     for rgb_path in tqdm(rgb_dirs, desc="Scanning files"):
#         # 生成对应路径
#         rel_path = os.path.relpath(rgb_path, start=os.path.join(data_root, "RGB"))
#         dsm_path = os.path.join(data_root, "DSM", rel_path)
#         label_path = os.path.join(data_root, "Label", rel_path)
#
#         # 验证文件存在性
#         if all(os.path.exists(p) for p in [dsm_path, label_path]):
#             samples.append((rgb_path, dsm_path, label_path))
#     return samples
#
#
# # 数据集类
# class ISPRSDataset(Dataset):
#     def __init__(self, samples, augmentation=True):
#         self.samples = samples
#         self.augmentation = augmentation
#         self.data_cache = {}
#         self.dsm_cache = {}
#         self.label_cache = {}
#
#     def __len__(self):
#         return 10000  # 维持原有epoch长度
#
#     def _load_data(self, path, mode='rgb'):
#         if path in self.data_cache:
#             return self.data_cache[path]
#
#         img = io.imread(path)
#         if mode == 'rgb':
#             img = img.astype(np.float32) / 255.0
#             img = img.transpose(2, 0, 1)  # HWC -> CHW
#         elif mode == 'dsm':
#             img = img.astype(np.float32)
#             img = (img - img.min()) / (img.max() - img.min() + 1e-6)
#         elif mode == 'label':
#             img = convert_from_color(img)
#
#         if cfg.cache:
#             if mode == 'rgb':
#                 self.data_cache[path] = img
#             elif mode == 'dsm':
#                 self.dsm_cache[path] = img
#             else:
#                 self.label_cache[path] = img
#         return img
#
#     def _random_crop(self, *arrays):
#         h, w = arrays[0].shape[1], arrays[0].shape[2]
#         pad_h = max(cfg.window_size[0] - h, 0)
#         pad_w = max(cfg.window_size[1] - w, 0)
#
#         padded = []
#         for arr in arrays:
#             if len(arr.shape) == 3:
#                 pad_width = ((0, 0), (0, pad_h), (0, pad_w))
#             else:
#                 pad_width = ((0, pad_h), (0, pad_w))
#             padded.append(np.pad(arr, pad_width, mode='constant'))
#
#         h, w = padded[0].shape[1], padded[0].shape[2]
#         x = np.random.randint(0, h - cfg.window_size[0])
#         y = np.random.randint(0, w - cfg.window_size[1])
#
#         return [arr[:, x:x + cfg.window_size[0], y:y + cfg.window_size[1]] if len(arr.shape) == 3 else
#                 arr[x:x + cfg.window_size[0], y:y + cfg.window_size[1]] for arr in padded]
#
#     def _augment(self, rgb, dsm, label):
#         # 随机水平翻转
#         if np.random.rand() > 0.5:
#             rgb = np.flip(rgb, axis=2).copy()
#             dsm = np.flip(dsm, axis=1).copy()
#             label = np.flip(label, axis=1).copy()
#
#         # 随机垂直翻转
#         if np.random.rand() > 0.5:
#             rgb = np.flip(rgb, axis=1).copy()
#             dsm = np.flip(dsm, axis=0).copy()
#             label = np.flip(label, axis=0).copy()
#
#         return rgb, dsm, label
#
#     def __getitem__(self, idx):
#         sample_idx = idx % len(self.samples)
#         rgb_path, dsm_path, label_path = self.samples[sample_idx]
#
#         # 加载数据
#         rgb = self._load_data(rgb_path, 'rgb')
#         dsm = self._load_data(dsm_path, 'dsm')
#         label = self._load_data(label_path, 'label')
#
#         # 统一尺寸
#         target_shape = (rgb.shape[1], rgb.shape[2])
#         if dsm.shape != target_shape:
#             dsm = resize(dsm, target_shape, order=1, preserve_range=True)
#         if label.shape != target_shape:
#             label = resize(label, target_shape, order=0, preserve_range=True)
#
#         # 随机裁剪
#         rgb_patch, dsm_patch, label_patch = self._random_crop(rgb, dsm, label)
#
#         # 数据增强
#         if self.augmentation:
#             rgb_patch, dsm_patch, label_patch = self._augment(rgb_patch, dsm_patch, label_patch)
#
#         return (
#             torch.from_numpy(rgb_patch.copy()).float(),
#             torch.from_numpy(dsm_patch.copy()).float().unsqueeze(0),
#             torch.from_numpy(label_patch.copy()).long()
#         )
#
#
# # 训练流程
# def train():
#     # 初始化数据集
#     all_samples = get_all_samples(cfg.data_root)
#     random.shuffle(all_samples)
#     split = int(0.8 * len(all_samples))
#     train_samples = all_samples[:split]
#     test_samples = all_samples[split:]
#
#     train_set = ISPRSDataset(train_samples)
#     test_set = ISPRSDataset(test_samples, augmentation=False)
#
#     train_loader = DataLoader(train_set, batch_size=cfg.batch_size,
#                               shuffle=True, num_workers=cfg.num_workers,
#                               pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=cfg.batch_size,
#                              num_workers=cfg.num_workers, pin_memory=True)
#
#     # 初始化模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = ViT_seg(num_classes=cfg.n_classes).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     best_miou = 0
#     for epoch in range(cfg.epochs):
#         # 训练阶段
#         model.train()
#         train_loss = 0
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [Train]")
#         for rgb, dsm, labels in pbar:
#             rgb = rgb.to(device)
#             dsm = dsm.to(device)
#             labels = labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(rgb, dsm)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             train_loss += loss.item()
#             pbar.set_postfix(loss=loss.item())
#
#         # 验证阶段
#         model.eval()
#         total = 0
#         correct = 0
#         test_loss = 0
#         with torch.no_grad():
#             for rgb, dsm, labels in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [Val]"):
#                 rgb = rgb.to(device)
#                 dsm = dsm.to(device)
#                 labels = labels.to(device)
#
#                 outputs = model(rgb, dsm)
#                 loss = criterion(outputs, labels)
#                 test_loss += loss.item()
#
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.nelement()
#                 correct += (predicted == labels).sum().item()
#
#         # 计算指标
#         scheduler.step()
#         miou = correct / total
#         print(f"Epoch {epoch + 1} | "
#               f"Train Loss: {train_loss / len(train_loader):.4f} | "
#               f"Val Loss: {test_loss / len(test_loader):.4f} | "
#               f"mIoU: {miou:.4f}")
#
#         # 保存最佳模型
#         if miou > best_miou:
#             best_miou = miou
#             torch.save(model.state_dict(), f"best_model_{miou:.4f}.pth")
#             print(f"Saved new best model with mIoU {miou:.4f}")
#
#
# if __name__ == "__main__":
#     train()

