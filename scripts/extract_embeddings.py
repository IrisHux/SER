# scripts/extract_embeddings.py

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from core.config import CONFIG
from contrastive.model import MemoryOptimizedContrastiveModel



def load_model(model_weights_name, device):
    """加载训练好的模型并设置为评估模式"""
    print(f"--- 正在加载配置 ---")
    CONFIG.load_config("config.yaml")
    
    num_labels = len(CONFIG.dataset_emotions(CONFIG.training_dataset_name()))
    model_path = os.path.join(CONFIG.saved_ckpt_location(), model_weights_name)

    print(f"--- 正在实例化模型 MemoryOptimizedContrastiveModel ---")
    model = MemoryOptimizedContrastiveModel(num_labels=num_labels).to(device)
    
    print(f"--- 正在加载模型权重 {model_path} ---")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 关键：设置为评估模式
    model.eval()
    print("--- 模型加载成功并已设置为评估模式 (model.eval()) ---")
    return model

def extract_embeddings(dataloader, model, device):
    """遍历dataloader，提取声学嵌入和标签"""
    all_embeddings = []
    all_labels = []

    # 关键：不计算梯度
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"提取特征中..."):
            # 1. 将数据移动到设备
            audio_input = batch['audio_input_values'].to(device)
            text_input = batch['text_input_ids'].to(device)
            mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 2. 前向传播 (我们只需要 acoustic_embedding 和 logits)
            # 根据 model.py，
            # acoustic_embedding 是投影头的输出
            acoustic_embedding, _, _, _ = model(
                audio_input_values=audio_input,
                text_input_ids=text_input,
                text_attention_mask=mask
            )

            # 3. 收集结果（转移到CPU以防显存溢出）
            all_embeddings.append(acoustic_embedding.cpu())
            all_labels.append(labels.cpu())

    # 将列表中的所有批次张量连接成一个大张量
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    return all_embeddings_tensor, all_labels_tensor

def get_metadata(dataloader):
    """从Dataloader的Dataset中获取元数据DataFrame"""
    # 根据 dataset.py，
    # .dataset 属性包含 EmotionDataset 实例，它持有 .dataframe
    if hasattr(dataloader.dataset, 'dataframe'):
        return dataloader.dataset.dataframe.copy()
    else:
        # 处理 Subset 的情况 (如果您的 get_dataloaders 像 tuning.py 中那样使用了 Subset)
        if hasattr(dataloader.dataset, 'dataset'):
             # 获取原始数据集的 dataframe
            original_df = dataloader.dataset.dataset.dataframe
            # 获取子集的索引
            indices = dataloader.dataset.indices
            # 返回索引对应的子集 dataframe
            return original_df.iloc[indices].reset_index(drop=True)
        else:
            raise ValueError("无法从Dataloader中定位DataFrame元数据")


