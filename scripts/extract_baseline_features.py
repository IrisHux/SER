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
# import os

from core.config import CONFIG, device
from audio.baseline_model import AudioBaselineModel
from scripts.get_dataloaders import get_dataloaders # 注意：使用不同的 dataloader

# --- 配置 ---
MODEL_WEIGHTS_NAME = "Audio_Baseline_trained_model.pt" # 来自 main_audio.py
OUTPUT_EMBEDDINGS_FILE = "baseline_classification_embeddings.pt"
OUTPUT_METADATA_FILE = "baseline_metadata.pkl"

def load_baseline_model(model_weights_name, device):
    """加载训练好的 Baseline 模型并设置为评估模式"""
    print(f"--- 正在加载配置 ---")
    CONFIG.load_config("config.yaml")
    
    model_path = os.path.join(CONFIG.saved_models_location(), model_weights_name) # 注意：路径不同
    
    print(f"--- 正在加载模型权重 {model_path} ---")
    # main_audio.py 保存的是一个字典
    checkpoint = torch.load(model_path, map_location=device)
    
    num_labels = checkpoint['num_labels']
    
    print(f"--- 正在实例化模型 AudioBaselineModel ---")
    model = AudioBaselineModel(num_labels=num_labels).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    print("--- Baseline 模型加载成功并已设置为评估模式 (model.eval()) ---")
    return model

def extract_baseline_features(dataloader, model, device):
    """
    遍历dataloader，提取 Baseline 的 *分类* 特征
    (WavLM 主干网络的池化输出)
    """
    all_embeddings = []
    all_labels = []
    
    # 关键：我们不直接使用 model.forward()，
    # 而是访问其内部的主干网络
    feature_extractor = model.wavlm.wavlm.to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"提取 Baseline 分类特征中..."):
            audio_input = batch['audio_input_values'].to(device)
            labels = batch['labels'].cpu()

            # --- 关键修正 ---
            # 1. 将输入喂给主干网络
            outputs = feature_extractor(input_values=audio_input)
            
            # 2. 提取 last_hidden_state 并进行平均池化 (这才是分类器的输入)
            classification_features = torch.mean(outputs.last_hidden_state, dim=1)
            # --- 修正结束 ---

            all_embeddings.append(classification_features.cpu())
            all_labels.append(labels)

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    return all_embeddings_tensor, all_labels_tensor

def main():
    model = load_baseline_model(MODEL_WEIGHTS_NAME, device)

    # 1. 加载 IEMOCAP 验证集
    print("--- 正在加载 IEMOCAP (Baseline Dataloader) ---")
    # 注意：使用 get_dataloaders
    iemocap_loaders = get_dataloaders(CONFIG.training_dataset_name())
    iemocap_val_loader = iemocap_loaders['validation']

    # 2. 加载 CREMA-D 测试集
    print("--- 正在加载 CREMA-D (Baseline Dataloader) ---")
    cremad_loaders = get_dataloaders(CONFIG.evaluation_dataset_name())
    cremad_eval_loader = cremad_loaders['evaluation']

    # 3. 提取特征
    iem_embeds, iem_labels = extract_baseline_features(iemocap_val_loader, model, device)
    crema_embeds, crema_labels = extract_baseline_features(cremad_eval_loader, model, device)

    # 4. 提取元数据
    # get_dataloaders 返回的 dataset 是 AudioDataset，它内部分割了
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
    
    print("--- Baseline 特征提取完成！---")

if __name__ == "__main__":
    main()