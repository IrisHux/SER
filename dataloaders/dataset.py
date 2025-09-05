# dataloaders/dataset.py

import logging
import numpy as np
import pandas as pd
import torch
import torchaudio # 导入 torchaudio
# import torchcodec
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from core.config import CONFIG # 保持不变

logger = logging.getLogger(__name__)

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
            
            # # 1. 使用 TorchCodec 加载原始音频波形 (这是核心变化)
            # decoder = torchcodec.decoders.AudioDecoder()
            # media_info, waveforms = decoder.decode(filepath=audio_path)
            
            # # 检查是否成功解码出音频流
            # if not waveforms:
            #     logger.error(f"使用TorchCodec无法解码音频文件: {audio_path}")
            #     return None 

            # # 获取波形和采样率
            # waveform = waveforms[0] # 取出第一个音频流
            # sample_rate = media_info.audio_streams[0].sample_rate


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
                "text": text_content, # <-- 新增：返回文本，如果是纯声学模型，则把这一行注释掉
                "label": emotion_index
            }
        except Exception as e:
            logger.error(f"加载或处理音频文件 {audio_path} 时出错: {e}")
            return None # 返回None，让collator可以过滤掉它

    def __len__(self):
        return len(self.dataframe)