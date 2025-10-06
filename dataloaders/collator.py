# dataloaders/collator.py

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
from typing import List, Dict, Tuple

class AudioDataCollator:
    """
    接收原始音频波形列表，使用Wav2Vec2Processor进行实时处理。
    - 特征提取
    - 动态填充 (padding) 到批次内最长
    """
    # Ensure processor is of type Wav2Vec2Processor
    def __init__(self, processor: Wav2Vec2Processor, sampling_rate: int = 16000):
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 过滤掉加载失败的样本 (返回None的)
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}

        # 2. 提取波形和标签列表
        waveforms_tensors = [item["waveform"] for item in batch]
        labels = [item["label"] for item in batch]

        # ----> 核心修复：在这里将Tensor列表转换为NumPy数组列表 <----
        waveforms_np = [w.numpy() for w in waveforms_tensors]

        # 3. 使用processor进行实时处理，传入NumPy数组列表
        inputs = self.processor(
            waveforms_np, # 使用转换后的 numpy 列表
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000 * 6 # 限制最大长度为6秒，防止OOM
        )

        # 4. 将标签列表转换为张量
        labels_batch = torch.stack(labels)

        # 5. 以模型和训练器期望的字典格式返回
        return {
            "audio_input_values": inputs.input_values,
            "labels": labels_batch
        }


class AblationCollator:
    """
    [消融实验B] 的数据整理器。
    - 接收一个包含 'waveform_1' 和 'waveform_2' 的批次。
    - 分别对这两个波形列表进行实时处理和填充。
    """
    def __init__(self, audio_processor: Wav2Vec2FeatureExtractor):
        self.audio_processor = audio_processor

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 过滤掉加载失败的样本
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}

        # 2. 分别提取两个波形视图和标签的列表
        waveforms_1 = [item["waveform_1"].numpy() for item in batch]
        waveforms_2 = [item["waveform_2"].numpy() for item in batch]
        labels = [item["label"] for item in batch]

        # 3. 分别处理两个波形列表
        audio_inputs_1 = self.audio_processor(
            waveforms_1,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000 * 6
        )

        audio_inputs_2 = self.audio_processor(
            waveforms_2,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000 * 6
        )

        # 4. 将标签列表转换为张量
        labels_batch = torch.stack(labels)

        # 5. 组合成最终的模型输入字典
        return {
            "audio_input_1": audio_inputs_1.input_values,
            "audio_input_2": audio_inputs_2.input_values,
            "labels": labels_batch
        }