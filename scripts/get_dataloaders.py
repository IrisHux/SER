# scripts/get_dataloaders.py

import os
import gc
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, AutoProcessor, Wav2Vec2FeatureExtractor, AutoTokenizer # 导入 AutoProcessor
from typing import Dict  # 需要添加这个导入

from core.config import CONFIG # 保持不变
from dataloaders.dataset import EmotionDataset # 导入新的Dataset
from dataloaders.collator import AudioDataCollator # 导入新的Collator
from contrastive.collator import ContrastiveDataCollator

def get_contrastive_dataloaders(dataset_name: str) -> Dict[str, torch.utils.data.DataLoader]:
    """
    为对比学习框架获取双模态的 Dataloaders。
    """
    print(f"--- 正在为 '{dataset_name}' 准备双模态 Dataloaders ---")

    emotions = CONFIG.dataset_emotions(dataset_name)
    dataloader_params = CONFIG.dataloader_dict()
    preprocessed_dir = CONFIG.dataset_preprocessed_dir_path(dataset_name)

    base_name = dataset_name.split('_')[0].lower()
    raw_df_path = os.path.join(preprocessed_dir, f"{base_name}_raw.pkl")

    try:
        dataframe = pd.read_pickle(raw_df_path)
        print(f"[INFO] 已从 {raw_df_path} 加载原始数据信息。")
    except FileNotFoundError as e:
        print(f"[ERROR] 找不到文件 {raw_df_path}。")
        raise e

    # 2. 初始化音频 Processor 和文本 Tokenizer
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG.audio_encoder_name())
    text_tokenizer = AutoTokenizer.from_pretrained(CONFIG.text_encoder_name())
    print("[INFO] 音频和文本处理器初始化完成。")

    # 3. 实例化新的 ContrastiveDataCollator
    contrastive_collator = ContrastiveDataCollator(
        audio_processor=audio_processor,
        text_tokenizer=text_tokenizer
    )

    # 4. 创建 Dataloaders
    dataloaders = {}
    if dataset_name == CONFIG.training_dataset_name():
        splits = ["train", "validation"]
    else:
        splits = ["evaluation"]

    for split in splits:
        dataset = EmotionDataset(dataframe, dataset_name, emotions, split)
        dataloaders[split] = DataLoader(
            dataset,
            collate_fn=contrastive_collator, # <-- 使用新的双模态实时处理collator
            **dataloader_params
        )

    return dataloaders
# 修改函数，使其适应新的流程
def get_dataloaders(dataset_name: str) -> Dict[str, DataLoader]:
    print(f"--- 正在为数据集 '{dataset_name}' 准备Dataloaders (实时处理模式) ---")

    # 1. 加载包含音频路径的原始DataFrame (_raw.pkl)
    emotions = CONFIG.dataset_emotions(dataset_name)
    dataloader_params = CONFIG.dataloader_dict()
    preprocessed_dir = CONFIG.dataset_preprocessed_dir_path(dataset_name)

    base_name = dataset_name.split('_')[0].lower()
    raw_df_path = os.path.join(preprocessed_dir, f"{base_name}_raw.pkl")

    try:
        dataframe = pd.read_pickle(raw_df_path)
        print(f"[INFO] 已从以下路径加载原始数据信息: {raw_df_path}")
    except FileNotFoundError as e:
        print(f"[ERROR] 找不到原始数据文件 {raw_df_path}。请先运行 process_raw_data_to_pickle。")
        raise e

    # 2. 初始化 Processor 和我们新的 Collator
    # processor = Wav2Vec2Processor.from_pretrained(CONFIG.audio_encoder_name()) # Old code
    # try:
    #     # 尝试直接加载完整的 processor
    #     processor = Wav2Vec2Processor.from_pretrained(CONFIG.audio_encoder_name())
    # except Exception as e:
    #     print(f"[WARNING] 无法加载完整 Wav2Vec2Processor: {e}")
    #     # 如果失败，手动创建（虽然对于纯音频任务，tokenizer 可能为空）
    #     feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG.audio_encoder_name())
    #     processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=None)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG.audio_encoder_name())
    print(f"[INFO] 使用的 processor 类型: {type(processor)}")
    audio_collator = AudioDataCollator(processor=processor)

    # 3. 根据数据集类型，创建Dataloaders
    # 注意：现在将整个dataframe传给Dataset，由它内部分割
    training_dataset_name = CONFIG.training_dataset_name()
    evaluation_dataset_name = CONFIG.evaluation_dataset_name()

    dataloaders = {}
    if dataset_name == training_dataset_name:
        # 创建训练集 DataLoader
        train_dataset = EmotionDataset(dataframe, dataset_name, emotions, "train")
        dataloaders["train"] = DataLoader(
            train_dataset,
            collate_fn=audio_collator, # 使用新的 collator
            **dataloader_params
        )
        # 创建验证集 DataLoader
        val_dataset = EmotionDataset(dataframe, dataset_name, emotions, "validation")
        dataloaders["validation"] = DataLoader(
            val_dataset,
            collate_fn=audio_collator, # 使用新的 collator
            **dataloader_params
        )

    elif dataset_name == evaluation_dataset_name:
        # 创建评估集 DataLoader
        eval_dataset = EmotionDataset(dataframe, dataset_name, emotions, "evaluation")
        dataloaders["evaluation"] = DataLoader(
            eval_dataset,
            collate_fn=audio_collator, # 使用新的 collator
            **dataloader_params
        )

    # 确保 num_workers > 0 以启用多进程实时加载
    if dataloader_params.get('num_workers', 0) == 0:
        print("[WARNING] num_workers=0, 实时处理将非常缓慢。建议在 config.yaml 中设置为大于0的值。")

    return dataloaders