# dataloaders/sampler.py
import torch
import random
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
from torch.utils.data import Dataset # 仅用于类型提示

class StratifiedBatchSampler(Sampler[list[int]]):
    """
    一个分层的批次采样器 (Stratified Batch Sampler)。
    
    此采样器实现了 (P, K) 策略，确保每个批次包含：
    - P 个不同的类别。
    - 每个类别 K 个样本。
    
    总批次大小为 P * K。
    """
    
    def __init__(self, dataset: Dataset, num_classes_per_batch: int, num_samples_per_class: int):
        """
        Args:
            dataset: 必须是 EmotionDataset 实例。
                     我们依赖它来访问 `dataset.dataframe['emotion']`。
            num_classes_per_batch (P): 每个批次中唯一的类别数。
            num_samples_per_class (K): 每个类别要采样的样本数。
        """
        
        # --- [思考过程 1: 必须了解数据集的结构] ---
        # "我需要知道数据集中每个样本的标签是什么，以及它在数据集中的索引。"
        
        # 从 dataset.dataframe 中提取所有标签
        try:
            # 假设 dataset 是 EmotionDataset 实例
            labels_series = dataset.dataframe['emotion']
            self.labels = labels_series.values
            self.dataset_len = len(self.labels)
        except AttributeError:
            raise ValueError("传入的 dataset 必须拥有一个 `dataframe['emotion']` 属性。")

        self.num_classes_per_batch = num_classes_per_batch # P
        self.num_samples_per_class = num_samples_per_class # K
        self.batch_size = self.num_classes_per_batch * self.num_samples_per_class

        # --- [思考过程 2: 必须按类别对索引进行分组] ---
        # "为了能按类别采样，我需要一个快速查找'所有高兴样本的索引'的方法。"
        # "字典 (哈希图) 是最快的方式：{ 'happy': [0, 5, 12], 'sad': [1, 3, 8], ... }"
        
        self.class_indices = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.class_indices[label].append(index)
            
        # 获取所有唯一的类别键 (e.g., ['happy', 'sad', 'angry', ...])
        self.unique_classes = list(self.class_indices.keys())

        # 确保 P 不大于总类别数
        if self.num_classes_per_batch > len(self.unique_classes):
            print(f"警告: P (num_classes_per_batch={self.num_classes_per_batch}) 大于 "
                  f"数据集中唯一的类别数 ({len(self.unique_classes)})。"
                  f"将 P 自动设置为 {len(self.unique_classes)}")
            self.num_classes_per_batch = len(self.unique_classes)

        print(f"分层采样器已初始化 (StratifiedBatchSampler):")
        print(f"  - {len(self.unique_classes)} 个唯一类别。")
        print(f"  - P = {self.num_classes_per_batch} 个类别/批次。")
        print(f"  - K = {self.num_samples_per_class} 个样本/类别。")
        print(f"  - 总批次大小 = {self.batch_size}。")

    def __iter__(self):
        """
        迭代器，在每个 epoch 开始时被调用，用于生成一个批次的索引列表。
        """
        
        # --- [思考过程 3: 如何构建一个 epoch 的批次？] ---
        # "我需要生成 N 个批次，直到数据大致被遍历一遍。"
        # "N 约等于 (总样本数 / 批次大小)。"
        
        num_batches_per_epoch = self.dataset_len // self.batch_size
        
        for _ in range(num_batches_per_epoch):
            
            # --- [思考过程 4: 如何构建 *一个* 批次？] ---
            # "1. 从所有类别中，随机选择 P 个类别。"
            # "2. 对于这 P 个类别中的 *每一个*，从该类别的索引列表中，随机选择 K 个索引。"
            # "3. 把这 P*K 个索引组合起来，打乱顺序，然后 'yield' (交出) 它们。"
            
            batch_indices = []

            # 步骤 1: 随机选择 P 个类别
            # (np.random.choice 更快，且能确保 P <= 总类别数时无放回)
            selected_classes = np.random.choice(
                self.unique_classes, 
                self.num_classes_per_batch, 
                replace=False # 确保我们得到 P 个 *不同* 的类别
            )

            # 步骤 2: 为每个类别选择 K 个样本
            for class_label in selected_classes:
                
                # 获取该类别的所有索引 (e.g., [0, 5, 12, ...])
                indices_for_this_class = self.class_indices[class_label]
                
                # 从该列表中随机选择 K 个索引
                # (replace=True 意味着如果 K 大于该类的样本总数，可以重复采样)
                selected_indices = np.random.choice(
                    indices_for_this_class,
                    self.num_samples_per_class,
                    replace=True 
                )
                
                batch_indices.extend(selected_indices)

            # 步骤 3: (可选但推荐) 打乱批次内的顺序
            # "为了防止模型学到某种顺序 (例如，批次中总是先 happy 后 sad)"
            random.shuffle(batch_indices)
            
            yield batch_indices

    def __len__(self):
        """
        返回一个 epoch 中的批次数。
        """
        # --- [思考过程 5: tqdm 和 DataLoader 需要知道何时停止] ---
        # "我必须告诉 DataLoader 一个 epoch 有多长 (有多少个批次)。"
        return self.dataset_len // self.batch_size