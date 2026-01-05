import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torchvision.transforms as transforms
from openstl.datasets.utils import create_loader


class CIKMDataset(Dataset):
    """CIKM视频预测数据集（支持动态序列长度和分布式训练）

    参数说明：
    root_dir: 数据集根目录
    mode: 数据集模式 [train|test|validation]
    pre_seq_length: 输入序列长度（默认5帧）
    aft_seq_length: 预测序列长度（默认10帧）
    image_size: 统一输出图像尺寸
    use_augment: 是否启用数据增强
    """

    def __init__(self, root_dir, mode='train', pre_seq_length=5, aft_seq_length=10,
                 image_size=128, use_augment=False,data_name='cikm'):
        self.sample_dirs = [
            os.path.join(root_dir, mode, f"sample_{i + 1}")
            for i in range(self._get_sample_count(mode))
        ]

        # 序列配置
        self.pre_seq = pre_seq_length
        self.aft_seq = aft_seq_length
        self.total_frames = 15  # 每个样本固定15帧
        self.mean = 0
        self.std = 1
        # 数据预处理流水线
        self.transform = self._build_transform(use_augment, mode)
        #self.normalize = transforms.Lambda(lambda x: x / 255.0)  # 添加归一化

    def _get_sample_count(self, mode):
        return {
            'train': 8000,
            'test': 4000,
            'validation': 2000
        }[mode]

    def _build_transform(self, use_augment, mode):
        """构建图像预处理流程"""
        transform_list = [
            transforms.Grayscale(num_output_channels=1),    # 确保单通道
            #transforms.Resize((104, 104)),
            # 数据增强应在转换为Tensor前应用
            transforms.ToTensor(),
            # 新增归一化层：执行 p=2*(p/255-0.5) 公式
            # 替代lambda的标准化方法
            #transforms.Normalize(mean=[0.5], std=[0.5])  # 等价于 2*(x-0.5)
        ]

        # 训练集增强策略
        if use_augment and mode == 'train':
            transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.3))
            transform_list.insert(2, transforms.RandomRotation(5))

        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]

        # 加载所有帧（优化内存管理）
        frames = []
        for i in range(1, self.total_frames + 1):
            img_path = os.path.join(sample_path, f"img_{i}.png")
            img = Image.open(img_path)
            tensor = self.transform(img)  # 单次归一化
            frames.append(tensor)  # 直接添加Tensor

        # 划分输入输出序列
        input_seq = torch.stack(frames[:self.pre_seq], dim=0)  # [5,1,100,100]
        target_seq = torch.stack(frames[self.pre_seq:self.pre_seq + self.aft_seq], dim=0)  # [10,1,100,100]

        return input_seq, target_seq


def load_data(batch_size, val_batch_size, data_root, num_workers=4, data_name='cikm',
              pre_seq_length=5, aft_seq_length=10, in_shape=[5, 1, 128, 128],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):
    """工业级数据加载流水线

    参数说明：
    batch_size: 训练集batch大小
    val_batch_size: 验证/测试集batch大小
    data_root: 数据集根路径
    num_workers: 数据加载线程数
    pre_seq_length: 输入帧数（5帧）
    aft_seq_length: 预测帧数（10帧）
    in_shape: 输入张量形状 [T,C,H,W]
    distributed: 是否启用分布式训练
    use_augment: 是否使用数据增强
    """

    # # 分布式配置
    # if distributed:
    #     rank = torch.distributed.get_rank()
    #     world_size = torch.distributed.get_world_size()
    # else:
    #     rank = 0
    #     world_size = 1
    image_size = in_shape[-1] if in_shape is not None else 128
    # 初始化数据集
    train_set = CIKMDataset(
        data_root, 'train',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        image_size=image_size,
        use_augment=use_augment,
        data_name=data_name
    )

    val_set = CIKMDataset(
        data_root, 'validation',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        image_size=image_size,
        use_augment=False,
        data_name=data_name
    )

    test_set = CIKMDataset(
        data_root, 'test',
        pre_seq_length=pre_seq_length,
        aft_seq_length=aft_seq_length,
        image_size=image_size,
        use_augment=False,
        data_name=data_name
    )

    # # 分布式采样器
    # train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank) if distributed else None
    # val_sampler = DistributedSampler(val_set, shuffle=False) if distributed else None

    # 创建DataLoader
    dataloader_train = create_loader(
        train_set,
        batch_size=batch_size,
        shuffle=True, is_training=True,
        pin_memory=True, drop_last=True,
        num_workers=num_workers,
        distributed=distributed, use_prefetcher=use_prefetcher
    )

    dataloader_val = create_loader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False, is_training=False,
        pin_memory=True, drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed, use_prefetcher=use_prefetcher
    )

    dataloader_test = create_loader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False, is_training=False,
        pin_memory=True, drop_last=drop_last,
        num_workers=num_workers,
        distributed=distributed, use_prefetcher=use_prefetcher
    )

    return dataloader_train, dataloader_val, dataloader_test


# 测试代码
if __name__ == '__main__':
    # 初始化分布式环境（示例）
    #torch.distributed.init_process_group(backend='nccl', init_method='env://')
    from openstl.utils import init_dist
    os.environ['LOCAL_RANK'] = str(0)
    os.environ['RANK'] = str(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist_params = dict(launcher='pytorch', backend='nccl', init_method='env://', world_size=1)
    init_dist(**dist_params)
    train_loader, val_loader, test_loader = load_data(
        batch_size=8,
        val_batch_size=8,
        data_root='/home/tq_qwj/VMRNN/data/cikm_128',
        pre_seq_length=5,
        aft_seq_length=10,
        in_shape=[5, 1, 128, 128],
        distributed=True,
        use_prefetcher=False
    )

    #验证数据形状
    print(len(train_loader), len(test_loader))
    for item in train_loader:
        print(item[0].shape, item[1].shape)
        break
    for item in test_loader:
        print(item[0].shape, item[1].shape)
        break
    sample_input, sample_target = next(iter(train_loader))
    print(f"输入形状: {sample_input.shape}")  # 应为[16,5,1,100,100]
    print(f"目标形状: {sample_target.shape}")  # 应为[16,10,1,100,100]
    print(f"数据类型: {sample_input.dtype}")  # 应显示torch.float32
    print(f"像素范围: {sample_input.min():.2f}~{sample_input.max():.2f}")  # 应为-1.00~1.00