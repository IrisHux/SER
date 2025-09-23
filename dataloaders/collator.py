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
