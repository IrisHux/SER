# scripts/extract_lgca_embeddings.py

import sys
import os
# 将项目根目录添加到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from core.config import CONFIG, device
from contrastive.model import MemoryOptimizedContrastiveModel
from audio.baseline_model import AudioBaselineModel
from scripts.get_dataloaders import get_dataloaders, get_contrastive_dataloaders

MODEL_WEIGHTS_NAME = "Contrastive_LGCA_model.pt" 
OUTPUT_EMBEDDINGS_FILE = "lgca_classification_embeddings.pt"
OUTPUT_METADATA_FILE = "lgca_metadata.pkl"

def load_lgca_model(model_weights_name, device):
    """加载训练好的模型并设置为评估模式"""
    print(f"--- 正在加载配置 ---")
    CONFIG.load_config("config.yaml")
    
    num_labels = len(CONFIG.dataset_emotions(CONFIG.training_dataset_name()))
    model_path = os.path.join(CONFIG.saved_models_location(), model_weights_name)

    print(f"--- 正在实例化模型 MemoryOptimizedContrastiveModel ---")
    model = MemoryOptimizedContrastiveModel(num_labels=num_labels).to(device)
    
    print(f"--- 正在加载模型权重 {model_path} ---")
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 关键：设置为评估模式
    model.eval()
    print("--- 模型加载成功并已设置为评估模式 (model.eval()) ---")
    return model

def extract_lgca_features(dataloader, model, device):
    """遍历dataloader，提取声学嵌入、分类特征和标签"""
    all_embeddings = []  # 对比学习的声学嵌入 (用于对比损失)
    all_classification_features = []  # 门控融合后的分类特征 (用于情绪分类可视化)
    all_labels = []

    # 关键：不计算梯度
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"提取特征中..."):
            # 1. 将数据移动到设备
            audio_input = batch['audio_input_values'].to(device)
            text_input = batch['text_input_ids'].to(device)
            mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 2. 前向传播
            # 根据更新后的 model.py，forward 现在返回4个值：
            # acoustic_embedding: 对比学习的声学嵌入 (投影头输出)
            # text_embedding: 对比学习的文本嵌入 (投影头输出)
            # final_logits: 分类logits
            # pooled_fused_features: 门控融合后的分类特征 (用于情绪分类可视化)
            _, _, _, pooled_fused_features = model(
                audio_input_values=audio_input,
                text_input_ids=text_input,
                text_attention_mask=mask,
                use_text_modality=False  # 评估时使用单模态 (纯声学)
            )

            # 3. 收集结果（转移到CPU以防显存溢出）
            all_embeddings.append(pooled_fused_features.cpu())
            all_labels.append(labels.cpu())

    # 将列表中的所有批次张量连接成一个大张量
    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    return all_embeddings_tensor, all_labels_tensor

def main():

    model = load_lgca_model(MODEL_WEIGHTS_NAME, device)

    # 1. 加载 IEMOCAP 验证集
    print("--- 正在加载 IEMOCAP (LGCA Dataloader) ---")
    iemocap_loaders = get_contrastive_dataloaders(CONFIG.training_dataset_name(), use_audio_augmentation=False)
    iemocap_val_loader = iemocap_loaders['validation']

    # 2. 加载 CREMA-D 测试集
    print("--- 正在加载 CREMA-D (LGCA Dataloader) ---")
    cremad_loaders = get_contrastive_dataloaders(CONFIG.evaluation_dataset_name(), use_audio_augmentation=False)
    cremad_eval_loader = cremad_loaders['evaluation']

    # 3. 提取特征
    iem_embeds, iem_labels = extract_lgca_features(iemocap_val_loader, model, device)
    crema_embeds, crema_labels = extract_lgca_features(cremad_eval_loader, model, device)

    # 4. 提取元数据
    # get_contrastive_dataloaders 返回的 dataset 不是 Subset，可以直接访问 .dataframe
    iem_meta_df = iemocap_val_loader.dataset.dataframe.copy()
    iem_meta_df['dataset_source'] = 'IEMOCAP'
    
    crema_meta_df = cremad_eval_loader.dataset.dataframe.copy()
    crema_meta_df['dataset_source'] = 'CREMA-D'

    # 5. 合并数据
    print("--- 正在合并 IEMOCAP 和 CREMA-D 数据 ---")
    all_embeddings = torch.cat([iem_embeds, crema_embeds], dim=0)
    all_metadata_df = pd.concat([iem_meta_df, crema_meta_df], ignore_index=True)
    
    assert len(all_embeddings) == len(all_metadata_df), "数据不匹配！"
    print(f"--- 数据校验通过：共 {len(all_embeddings)} 个样本 ---")

    # 6. 保存到文件
    print(f"--- 正在保存嵌入向量到 {OUTPUT_EMBEDDINGS_FILE} ---")
    torch.save(all_embeddings, OUTPUT_EMBEDDINGS_FILE)
    
    print(f"--- 正在保存元数据到 {OUTPUT_METADATA_FILE} ---")
    all_metadata_df.to_pickle(OUTPUT_METADATA_FILE)
    
    print("--- LGCA 特征提取完成！---")

if __name__ == "__main__":
    main()

