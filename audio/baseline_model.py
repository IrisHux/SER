import safetensors
import torch.nn as nn
from transformers import WavLMForSequenceClassification

from core.config import CONFIG

class AudioBaselineModel(nn.Module):
    """
    纯声学基线模型。
    这个类封装了Hugging Face的WavLMForSequenceClassification模型，
    以便与我们的训练器框架集成。
    """
    def __init__(self, num_labels: int):
        super().__init__()
        # 从Hugging Face加载预训练的WavLM模型，并指定情感类别的数量
        self.wavlm = WavLMForSequenceClassification.from_pretrained(
            CONFIG.audio_encoder_name(),  # "microsoft/wavlm-base-plus"
            num_labels=num_labels,
            use_safetensors = True
        )

    def forward(self, audio_input_values):
        """
        模型的前向传播。
        它接收音频处理器的输出，并返回模型的logits。
        """
        # 直接将输入传递给WavLM模型
        outputs = self.wavlm(input_values=audio_input_values)
        return outputs.logits