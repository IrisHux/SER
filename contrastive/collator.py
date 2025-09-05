# contrastive/collator.py

import torch
from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer
from typing import List, Dict

class ContrastiveDataCollator:
    """
    为双模态对比学习进行实时数据处理。
    - 接收原始音频波形和原始文本字符串。
    - 使用音频 processor 对波形进行特征提取和填充。
    - 使用文本 tokenizer 对文本进行编码和填充。
    """
    def __init__(self, audio_processor: Wav2Vec2FeatureExtractor, text_tokenizer: AutoTokenizer):
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1. 过滤掉加载失败的样本
        batch = [item for item in batch if item is not None]
        if not batch:
            return {}

        # 2. 提取波形、文本和标签列表
        waveforms = [item["waveform"].numpy() for item in batch] # Processor 接收 numpy 数组
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        # 3. 实时处理音频
        audio_inputs = self.audio_processor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16000 * 15 # 限制最大长度为15秒
        )

        # 4. 实时处理文本
        text_inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512 # 设定一个合理的文本最大长度
        )

        # 5. 将标签列表转换为张量
        labels_batch = torch.stack(labels)

        # 6. 组合成最终的模型输入字典
        return {
            "audio_input_values": audio_inputs.input_values,
            "text_input_ids": text_inputs.input_ids,
            "text_attention_mask": text_inputs.attention_mask,
            "labels": labels_batch
        }