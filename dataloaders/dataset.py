# dataloaders/dataset.py

import logging
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from sklearn.model_selection import train_test_split
from core.config import CONFIG # 保持不变


logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    # 修改__init__，现在只接收一个包含原始信息的DataFrame
    def __init__(self, dataframe: pd.DataFrame, dataset_name: str, emotions: list, split: str):
        self._emotions = np.array(emotions)

        # IEMOCAP需要按session和说话人性别进行划分
        # 这里简化为按情感类别进行分层抽样，与你之前的逻辑保持一致
        if split in ["train", "validation"]:
            train_indices, val_indices = train_test_split(
                dataframe.index,
                test_size=0.2,
                random_state=42,
                stratify=dataframe['emotion']
            )
            indices_to_use = train_indices if split == "train" else val_indices
            self.dataframe = dataframe.loc[indices_to_use].reset_index(drop=True)
        elif split == "evaluation":
            self.dataframe = dataframe.reset_index(drop=True)
        else:
             raise ValueError(f"未知的划分: '{split}'")

        logger.info(f"已加载 '{dataset_name}' 数据集用于 '{split}' 划分。大小: {len(self)}")

    def __getitem__(self, index: int):
        # 1. 获取音频路径和标签
        audio_path = self.dataframe.loc[index, 'audio_path']
        emotion_label = self.dataframe.loc[index, 'emotion']
        emotion_index = torch.tensor(np.where(self._emotions == emotion_label)[0][0])

        # 2. 加载原始音频波形 (这是核心变化)
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # 3. 重采样到16kHz (WavLM的期望采样率，非常重要)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            # 4. 如果是双声道，转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 返回一个字典，包含原始波形和标签，便于 collator 处理
            return {
                "waveform": waveform.squeeze(0), # 移除通道维度，变为 (T,)
                "label": emotion_index
            }
        except Exception as e:
            logger.error(f"加载或处理音频文件 {audio_path} 时出错: {e}")
            return None # 返回None，让collator可以过滤掉它

    def __len__(self):
        return len(self.dataframe)
    
class EmotionDataset(Dataset):
    # 修改__init__，现在只接收一个包含原始信息的DataFrame
    def __init__(self, dataframe: pd.DataFrame, dataset_name: str, emotions: list, split: str):
        self._emotions = np.array(emotions)

        # IEMOCAP需要按session和说话人性别进行划分
        # 这里简化为按情感类别进行分层抽样，与你之前的逻辑保持一致
        if split in ["train", "validation"]:
            train_indices, val_indices = train_test_split(
                dataframe.index,
                test_size=0.2,
                random_state=42,
                stratify=dataframe['emotion']
            )
            indices_to_use = train_indices if split == "train" else val_indices
            self.dataframe = dataframe.loc[indices_to_use].reset_index(drop=True)
        elif split == "evaluation":
            self.dataframe = dataframe.reset_index(drop=True)
        else:
             raise ValueError(f"未知的划分: '{split}'")

        logger.info(f"已加载 '{dataset_name}' 数据集用于 '{split}' 划分。大小: {len(self)}")


    def __getitem__(self, index: int):
        # 1. 获取音频路径和标签
        row = self.dataframe.loc[index]
        audio_path = row['audio_path']
        text_content = row['text'] # <-- 新增：获取文本，如果是纯声学模型，则把这一行注释掉
        emotion_label = row['emotion']
        emotion_index = torch.tensor(np.where(self._emotions == emotion_label)[0][0])

        # 2. 加载原始音频波形 (这是核心变化)
        try:
            waveform, sample_rate = torchaudio.load(audio_path)  # 更新成torchcodec

            # 3. 重采样到16kHz (WavLM的期望采样率，非常重要)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            # 4. 如果是双声道，转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 准备返回的字典，包含原始波形和标签，便于 collator 处理
            item = {
                "waveform": waveform.squeeze(0), # 移除通道维度，变为 (T,)
                "text": text_content, # <-- 新增：返回文本，如果是纯声学模型，则把这一行注释掉
                "label": emotion_index
            }
            return item
        except Exception as e:
            logger.error(f"加载或处理音频文件 {audio_path} 时出错: {e}")
            return None

    def __len__(self):
        return len(self.dataframe)
    
class EmotionDatasetAblation(EmotionDataset):
    """
    [消融实验B] 的数据集：继承自 EmotionDataset，但重写了 __getitem__ 方法。
    - 目标：为纯声学对比学习生成两个不同的音频增强视图。
    - 复用：完全复用父类的 __init__ 方法，无需重写。
    """
    def __init__(self, dataframe: pd.DataFrame, dataset_name: str, emotions: list, split: str):
        # 1. 调用父类的 __init__，完美复用所有数据加载和划分逻辑
        super().__init__(dataframe, dataset_name, emotions, split)

        # 2. 定义两种不同的数据增强流程
        #    p=0.5 表示有50%的概率应用该增强
        self.augment_v1 = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5)
        ])

        self.augment_v2 = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # 确保两种增强流程不完全相同
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.5),
        ])

        logger.info(f"EmotionDatasetAblation 初始化完成，已配置音频数据增强。")

    def __getitem__(self, index: int):
        # 1. 获取音频路径和标签 (这部分逻辑与父类相同)
        row = self.dataframe.loc[index]
        audio_path = row['audio_path']
        emotion_label = row['emotion']
        emotion_index = torch.tensor(np.where(self._emotions == emotion_label)[0][0])

        # 2. 加载和预处理原始音频 (这部分逻辑也与父类相同)
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # *** 核心修改点：应用数据增强 ***
            # 将 PyTorch 张量 (C, T) -> (T,) -> NumPy 数组，以用于增强
            waveform_np = waveform.squeeze().numpy()

            # 应用两种不同的增强
            augmented_waveform_1 = self.augment_v1(samples=waveform_np, sample_rate=16000)
            augmented_waveform_2 = self.augment_v2(samples=waveform_np, sample_rate=16000)

            # 将增强后的 NumPy 数组转回 PyTorch 张量
            waveform_1_tensor = torch.from_numpy(augmented_waveform_1)
            waveform_2_tensor = torch.from_numpy(augmented_waveform_2)

            # *** 核心修改点：返回值的结构改变 ***
            # 不再返回 "text"，而是返回两个增强后的 "waveform"
            return {
                "waveform_1": waveform_1_tensor,
                "waveform_2": waveform_2_tensor,
                "label": emotion_index
            }

        except Exception as e:
            logger.error(f"为消融实验加载或增强音频 {audio_path} 时出错: {e}")
            return None