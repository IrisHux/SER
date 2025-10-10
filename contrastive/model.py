# contrastive/model.py
import os
import safetensors
import torch
import torch.nn as nn


from core.config import CONFIG, device
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc
from transformers import (
    WavLMModel, AutoModel,
    get_linear_schedule_with_warmup,
    AutoTokenizer
)

# ===============================
# 1. 全局内存优化设置
# ===============================
def setup_memory_optimization():
    """设置全局内存优化参数"""
    # 设置CUDA内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # 启用内存池优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # 设置内存碎片整理
    torch.cuda.empty_cache()

    print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("内存优化设置完成")


# ===============================
# 2. 内存优化的模型配置
# ===============================
class MemoryOptimizedContrastiveModel(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()

        # 使用更小的模型或者冻结部分层
        self.audio_encoder = WavLMModel.from_pretrained(
            CONFIG.audio_encoder_name(),
            use_safetensors = True
        )
        self.text_encoder = AutoModel.from_pretrained(
            CONFIG.text_encoder_name(),
            use_safetensors = True
        )

        # 冻结部分编码器层以减少训练时的计算量和内存占用
        # 这里我们冻结 WavLM 的 CNN 特征提取器和 DeBERTa 的前6层 Transformer 层
        self._freeze_model_parts(num_text_layers_to_freeze=6)

        # # 启用梯度检查点以节省内存
        # if use_gradient_checkpointing:
        #     self.audio_encoder.gradient_checkpointing_enable()
        #     self.text_encoder.gradient_checkpointing_enable()
        #     print("已启用梯度检查点以节省内存")

        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size

        # --- 2. 为每个模态定义投影头 (Projection Head) ---
        # 投影头是一个简单的多层感知机 (MLP)，它将编码器输出的高维特征，
        # 映射到一个统一的、较低维度的嵌入空间，用于计算对比损失。
        # 投影头的维度配置从 config.yaml 文件中读取。
        proj_config = CONFIG.projection_bridge_config()
        projection_dims = proj_config['hidden_dims']

        self.audio_projection_head = nn.Sequential(
            nn.Linear(audio_hidden_size, projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1])
        )

        self.text_projection_head = nn.Sequential(
            nn.Linear(text_hidden_size, projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1])
        )

        self.audio_classifier = nn.Linear(audio_hidden_size, num_labels)

    def _freeze_model_parts(self, num_text_layers_to_freeze: int):
        """
        冻结编码器的特定部分。
        """
        # --- 1. 冻结 WavLM 的 CNN 特征提取器 ---
        print("--- 正在冻结 WavLM 的 CNN Feature Extractor ---")
        for param in self.audio_encoder.feature_extractor.parameters():
            param.requires_grad = False

        # --- 2. 冻结 DeBERTa 的 Embedding 层 ---
        print("--- 正在冻结 DeBERTa 的 Embeddings ---")
        for param in self.text_encoder.embeddings.parameters():
            param.requires_grad = False

        # --- 3. 冻结 DeBERTa 的底层 Transformer ---
        print(f"--- 正在冻结 DeBERTa 的 Encoder 前 {num_text_layers_to_freeze} 层 ---")
        total_layers = len(self.text_encoder.encoder.layer)
        layers_to_freeze = min(num_text_layers_to_freeze, total_layers)

        if layers_to_freeze < num_text_layers_to_freeze:
                print(f"[警告] 想要冻结 {num_text_layers_to_freeze} 层, "
                      f"但模型只有 {total_layers} 层。将只冻结 {layers_to_freeze} 层。")

        for i, layer in enumerate(self.text_encoder.encoder.layer):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

        print("--- 编码器冻结操作完成 ---")

    def masked_mean(self, tensor, mask):
        # x: [B, L, D], mask: [B, L]
        mask = mask.unsqueeze(-1).type_as(tensor)
        return (tensor * mask).sum(1) / mask.sum(1).clamp_min(1e-6)

    def forward(self, audio_input_values, text_input_ids, text_attention_mask, augmented_audio_input_values=None):
        # 使用半精度计算
        with torch.amp.autocast('cuda'):
            # 音频分支 - 使用更小的序列长度
            audio_outputs = self.audio_encoder(input_values=audio_input_values)
            # 使用池化而不是平均，减少计算量
            acoustic_features = torch.mean(audio_outputs.last_hidden_state, dim=1)
            # 投影和分类
            acoustic_embedding = self.audio_projection_head(acoustic_features)
            audio_logits = self.audio_classifier(acoustic_features)

            # --- 增强音频分支 (新增逻辑) ---
            augmented_acoustic_embedding = None
            if augmented_audio_input_values is not None:
                # 使用相同的编码器和投影头
                aug_audio_outputs = self.audio_encoder(input_values=augmented_audio_input_values)
                aug_acoustic_features = torch.mean(aug_audio_outputs.last_hidden_state, dim=1)
                augmented_acoustic_embedding = self.audio_projection_head(aug_acoustic_features)
                # 注意：这里没有分类头，因为增强音频主要用于对比学习
            # 文本分支
            text_embedding = None
            if text_input_ids is not None:
                text_outputs = self.text_encoder(
                    input_ids=text_input_ids,
                    attention_mask=text_attention_mask
                )
                text_features = self.masked_mean(text_outputs.last_hidden_state, text_attention_mask)
                # text_features = torch.mean(text_outputs.last_hidden_state, dim=1)
                text_embedding = self.text_projection_head(text_features)


        return acoustic_embedding, text_embedding, audio_logits, augmented_acoustic_embedding

class ContrastiveModel(nn.Module):
    """
    双模态对比学习模型 (LGCA框架的核心)。
    该模型包含一个声学编码器 (WavLM) 和一个文本编码器 (DeBERTa)，
    并为两者分别附加了投影头，用于生成对比学习所需的嵌入向量。
    """
    def __init__(self, num_labels: int):
        super().__init__()

        # --- 1. 实例化声学和文本编码器的基础模型 (Backbone) ---
        # 我们从Hugging Face加载预训练的WavLM和DeBERTa模型。
        # 注意：这里使用的是 WavLMModel 和 AutoModel，而不是 WavLMForSequenceClassification。
        # 这是因为我们需要的是模型输出的特征表示 (hidden states)，而不是最终的分类结果。
        self.audio_encoder = WavLMModel.from_pretrained(CONFIG.audio_encoder_name(), use_safetensors = True)
        self.text_encoder = AutoModel.from_pretrained(CONFIG.text_encoder_name(), use_safetensors = True)

        # 冻结部分编码器层以减少训练时的计算量和内存占用
        # 这里我们冻结 WavLM 的 CNN 特征提取器和 DeBERTa 的前6层 Transformer 层
        self._freeze_model_parts(num_text_layers_to_freeze=6)

        # 获取编码器的输出维度，通常对于 "base" 模型是 768
        audio_hidden_size = self.audio_encoder.config.hidden_size
        text_hidden_size = self.text_encoder.config.hidden_size

        # --- 2. 为每个模态定义投影头 (Projection Head) ---
        # 投影头是一个简单的多层感知机 (MLP)，它将编码器输出的高维特征，
        # 映射到一个统一的、较低维度的嵌入空间，用于计算对比损失。
        # 投影头的维度配置从 config.yaml 文件中读取。
        proj_config = CONFIG.projection_bridge_config()
        projection_dims = proj_config['hidden_dims'] # 例如：[512, 256]

        self.audio_projection_head = nn.Sequential(
            nn.Linear(audio_hidden_size, projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1])
        )

        self.text_projection_head = nn.Sequential(
            nn.Linear(text_hidden_size, projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1])
        )

        # --- 3. (为后续步骤准备) 为声学模型添加一个线性分类头 ---
        # 这个分类头将在对比学习训练后，用于进行单模态的语音情感识别评估。
        # 它的输入是声学编码器的原始特征 (投影之前)。
        self.audio_classifier = nn.Linear(audio_hidden_size, num_labels)

    def _freeze_model_parts(self, num_text_layers_to_freeze: int):
        """
        [内部方法] 冻结编码器的特定部分。
        这个方法会直接修改 self.audio_encoder 和 self.text_encoder。
        (此逻辑基于你原始的 __init__ 方法)
        """
        
        # --- 1. 冻结 WavLM 的 CNN 特征提取器 ---
        print("--- [修改] 正在冻结 WavLM 的 CNN Feature Extractor ---")
        if hasattr(self.audio_encoder, 'feature_extractor'):
            for param in self.audio_encoder.feature_extractor.parameters():
                param.requires_grad = False
        else:
            print("[警告] audio_encoder 没有 'feature_extractor' 属性，跳过冻结。")

        # --- 2. 冻结 DeBERTa 的 Embedding 层 ---
        print("--- [修改] 正在冻结 DeBERTa 的 Embeddings ---")
        if hasattr(self.text_encoder, 'embeddings'):
            for param in self.text_encoder.embeddings.parameters():
                param.requires_grad = False
        else:
            print("[警告] text_encoder 没有 'embeddings' 属性，跳过冻结。")

        # --- 3. 冻结 DeBERTa 的底层 Transformer ---
        print(f"--- [修改] 正在冻结 DeBERTa 的 Encoder 前 {num_text_layers_to_freeze} 层 ---")
        if (hasattr(self.text_encoder, 'encoder') and 
            hasattr(self.text_encoder.encoder, 'layer') and 
            len(self.text_encoder.encoder.layer) > 0):
            
            total_layers = len(self.text_encoder.encoder.layer)
            layers_to_freeze = min(num_text_layers_to_freeze, total_layers)

            if layers_to_freeze < num_text_layers_to_freeze:
                 print(f"[警告] 想要冻结 {num_text_layers_to_freeze} 层, "
                       f"但模型只有 {total_layers} 层。将只冻结 {layers_to_freeze} 层。")

            for i, layer in enumerate(self.text_encoder.encoder.layer):
                if i < layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
        else:
            print("[警告] text_encoder 没有 'encoder.layer' 结构，跳过底层 Transformer 冻结。")

        print("--- 编码器冻结操作完成 ---")

    def forward(self, audio_input_values, text_input_ids=None, text_attention_mask=None):
        """
        模型的前向传播逻辑。

        Args:
            audio_input_values (torch.Tensor): 从音频预处理得到的输入张量。
            text_input_ids (torch.Tensor): 从文本tokenizer得到的输入ID。
            text_attention_mask (torch.Tensor): 文本输入的注意力掩码。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - acoustic_embedding: 声学模态经过投影头后的嵌入向量。
                - text_embedding: 文本模态经过投影头后的嵌入向量。
                - audio_logits: 声学模态经过分类头后的logits (用于交叉熵损失)。
        """
        # --- 声学分支 ---
        # 1. 通过WavLM编码器获取音频特征
        audio_outputs = self.audio_encoder(input_values=audio_input_values)
        # 2. 从最后一层隐藏状态中提取特征。我们通过在时间维度上取平均值来获得一个固定长度的句子级表示。
        #    audio_outputs.last_hidden_state 的形状是 (batch_size, sequence_length, hidden_size)
        acoustic_features = torch.mean(audio_outputs.last_hidden_state, dim=1)

        # # --- 文本分支 ---
        # # 1. 通过DeBERTa编码器获取文本特征
        # text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        # # 2. 同样，对最后一层隐藏状态在序列长度维度上取平均，获得句子级表示。
        # text_features = torch.mean(text_outputs.last_hidden_state, dim=1)

        # --- 文本分支 (增加判断) ---
        text_embedding = None
        if text_input_ids is not None:
            text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
            text_features = torch.mean(text_outputs.last_hidden_state, dim=1)
            text_embedding = self.text_projection_head(text_features)
        
        # --- 投影和分类 ---
        # 3. 将声学和文本特征分别送入各自的投影头，生成用于对比学习的嵌入向量
        acoustic_embedding = self.audio_projection_head(acoustic_features)
        # text_embedding = self.text_projection_head(text_features)

        # 4. 将声学特征送入分类头，生成用于分类任务的logits
        audio_logits = self.audio_classifier(acoustic_features)

        return acoustic_embedding, text_embedding, audio_logits
    

class AcousticSupConModel(nn.Module):
    """
    消融模型B: LGCA w/o Text Anchor
    一个纯声学的监督对比学习模型。
    """
    def __init__(self, num_labels: int):
        super().__init__()

        # --- 1. 只实例化声学编码器 ---
        self.audio_encoder = WavLMModel.from_pretrained(
            CONFIG.audio_encoder_name(),
            use_safetensors=True
        )

        # (可选但推荐) 冻结特征提取器
        if hasattr(self.audio_encoder, 'feature_extractor'):
            for param in self.audio_encoder.feature_extractor.parameters():
                param.requires_grad = False

        audio_hidden_size = self.audio_encoder.config.hidden_size

        # --- 2. 只需要一个声学投影头 ---
        proj_config = CONFIG.projection_bridge_config()
        projection_dims = proj_config['hidden_dims']
        self.audio_projection_head = nn.Sequential(
            nn.Linear(audio_hidden_size, projection_dims[0]),
            nn.ReLU(),
            nn.Linear(projection_dims[0], projection_dims[1])
        )

        # --- 3. 仍然需要分类头 ---
        self.audio_classifier = nn.Linear(audio_hidden_size, num_labels)

    def forward(self, audio_input_1, audio_input_2):
        """
        前向传播接收两个增强版本的音频输入。
        """
        # --- 分别处理两个音频输入 ---
        # 它们共享同一个编码器和投影头
        outputs_1 = self.audio_encoder(input_values=audio_input_1)
        features_1 = torch.mean(outputs_1.last_hidden_state, dim=1)
        embedding_1 = self.audio_projection_head(features_1)
        # --- 计算分类 Logits ---
        # 通常只使用一个视图（例如第一个）的特征来做分类，以保持稳定性
        logits = self.audio_classifier(features_1)

        if audio_input_2 is not None: # 仅在训练时
            outputs_2 = self.audio_encoder(input_values=audio_input_2)
            features_2 = torch.mean(outputs_2.last_hidden_state, dim=1)
            embedding_2 = self.audio_projection_head(features_2)
            return embedding_1, embedding_2, logits
        else:
            # --- 评估模式 ---
            # 只返回 logits，其他两个返回 None 以匹配训练时的输出元组结构
            return None, None, logits