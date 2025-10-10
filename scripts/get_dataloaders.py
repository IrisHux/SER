# scripts/get_dataloaders.py

import os
import gc
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor, AutoProcessor, Wav2Vec2FeatureExtractor, AutoTokenizer # 导入 AutoProcessor
from typing import Dict  # 需要添加这个导入

from core.config import CONFIG
from dataloaders.dataset import EmotionDataset, EmotionDatasetAblation
from dataloaders.collator import AudioDataCollator, AblationCollator
from contrastive.collator import ContrastiveDataCollator

def get_contrastive_dataloaders(dataset_name: str, use_audio_augmentation: bool = False)  -> Dict[str, torch.utils.data.DataLoader]:
    """
    为对比学习框架获取双模态的 Dataloaders。
    """
    print(f"--- 正在为 '{dataset_name}' 准备双模态 Dataloaders ---")
    if use_audio_augmentation:
        print("[INFO] 本次将为训练集启用音频数据增强。")
    # 1. 加载数据集的基础信息
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
        # 仅在训练集上启用数据增强
        should_augment = (split == "train") and use_audio_augmentation
 
        dataset = EmotionDataset(dataframe, dataset_name, emotions, split, use_audio_augmentation=should_augment)
        if should_augment:
            print(f"[INFO] 已为 '{split}' 划分启用数据增强并创建 Dataset。")

        dataloaders[split] = DataLoader(
            dataset,
            collate_fn=contrastive_collator, # <-- 使用新的双模态实时处理collator
            **dataloader_params
        )

    return dataloaders


def get_ablation_no_text_dataloaders(dataset_name: str) -> Dict[str, torch.utils.data.DataLoader]:
    """
    [消融实验B: w/o Text Anchor] 的专用 Dataloader 创建函数。

    该函数构建了一个完全独立的、用于纯声学监督对比学习的数据管道：
    1. 使用 `EmotionDatasetAblation` 来加载音频并应用两种不同的数据增强。
    2. 使用 `AblationCollator` 来分别处理这两个增强后的音频视图。
    """
    print(f"--- [消融实验] 正在为 '{dataset_name}' 准备纯声学对比学习 Dataloaders ---")

    # 步骤 1: 加载数据集的基础信息 (这部分与原函数完全相同，直接复用)
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

    # 步骤 2: 初始化音频处理器 (Collator 需要它)
    # 只需要音频的 Feature Extractor，不需要文本的 Tokenizer
    audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG.audio_encoder_name())
    print("[INFO] 音频特征提取器初始化完成。")

    # 步骤 3: 实例化两种 Collator
    ablation_collator = AblationCollator(audio_processor=audio_processor)
    # 这个 collator 用于处理单音频输入的验证/评估集
    std_audio_collator = AudioDataCollator(processor=audio_processor)

    # 步骤 4: 创建 Dataloaders，并确保使用新的 Dataset 和 Collator
    dataloaders = {}
    if dataset_name == CONFIG.training_dataset_name():
        splits = ["train", "validation"]
    else:
        # 消融实验通常也需要在评估集上测试，所以保留 evaluation
        splits = ["evaluation"]

    for split in splits:
        # *** 实例化 EmotionDatasetAblation ***
        if split == "train":
            dataset = EmotionDatasetAblation(dataframe, dataset_name, emotions, split)
            collator_to_use = ablation_collator
            print(f"[INFO] 已为 '{split}' 划分创建 Dataloader (使用 EmotionDatasetAblation 和 AblationCollator)。")
        else:
            # 验证集和评估集使用标准的 Dataset 和 Collator
            # 注意：这里我们复用 EmotionDataset，但需要确保它不返回文本，
            dataset = EmotionDataset(dataframe, dataset_name, emotions, split)
            collator_to_use = std_audio_collator
            print(f"[INFO] 已为 '{split}' 划分创建 Dataloader (使用标准的 EmotionDataset 和 AudioDataCollator)。")

        dataloaders[split] = DataLoader(
            dataset,
            collate_fn=collator_to_use, # 动态选择collator
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